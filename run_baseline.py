#!/usr/bin/env python3
"""
Single-Agent Baseline — Homogeneous & Random Assignments

Each MMLU question is routed to its domain role (Historian, Scientist, Economist)
based on subject. The assigned model answers directly with the role-specific
system prompt. One LLM call per question, no chain, no FinalNode.

Modes:
  - homogeneous: same model for all roles (6 baselines, one per model)
  - random: random model→role assignment per trial
  - both: run homogeneous + random

Usage:
    python run_baseline.py --mode both --split test --limit 500 --concurrency 64
    python run_baseline.py --mode homogeneous --split test --limit 500
    python run_baseline.py --mode random --n-trials 10 --split test --limit 500
"""

import argparse
import csv
import json
import os
import random
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LLM_PROFILE = PROJECT_ROOT / "config" / "llm_profile_full.json"
DEFAULT_ROLE_SUBJECTS = PROJECT_ROOT / "config" / "role_subjects.yaml"
DEFAULT_MODEL_COSTS = PROJECT_ROOT / "config" / "model_costs.json"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "Datasets" / "MMLU" / "data"

MMLU_COLUMNS = ["question", "A", "B", "C", "D", "correct_answer"]

# Roles directory for loading role JSON files
ROLES_DIR = PROJECT_ROOT / "MAR" / "Roles" / "Commonsense"

# Output format prompt for "Answer" (from MAR/Prompts/output_format.py)
ANSWER_FORMAT_PROMPT = (
    "The last line of your output must contain only the final result "
    "without any units or redundant explanation,"
    "for example: The answer is 140\n"
    "If it is a multiple choice question, please output the options. "
    "For example: The answer is A.\n"
    "However, The answer is 140$ or The answer is Option A "
    "or The answer is A.140 is not allowed.\n"
)

ROLES = ["Historian", "Scientist", "Economist"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_role_subjects(path: Path = DEFAULT_ROLE_SUBJECTS) -> Dict[str, List[str]]:
    """Load role -> subject mapping from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_subject_questions(
    split: str, subjects: List[str], data_root: Path = DEFAULT_DATA_ROOT,
) -> pd.DataFrame:
    """Load MMLU questions for a list of subjects."""
    data_path = data_root / split
    dfs = []
    for subject in subjects:
        csv_path = data_path / f"{subject}.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path, header=None, names=MMLU_COLUMNS, encoding="utf-8")
        df["subject"] = subject
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=MMLU_COLUMNS + ["subject"])
    return pd.concat(dfs, ignore_index=True)


def load_llm_profile(path: Path) -> Tuple[List[Dict], Dict[str, str]]:
    """Load model list and base_urls from llm_profile_full.json."""
    with open(path) as f:
        data = json.load(f)
    return data.get("models", []), data.get("model_base_urls", {})


def load_role_system_prompt(role: str) -> str:
    """Load role description from JSON and combine with Answer output format."""
    role_json_path = ROLES_DIR / f"{role}.json"
    with open(role_json_path) as f:
        role_data = json.load(f)
    return f"{role_data['Description']}\n\n{ANSWER_FORMAT_PROMPT}"


def load_model_costs(path: Path = DEFAULT_MODEL_COSTS) -> Dict[str, Dict[str, float]]:
    """Load per-model token costs from JSON."""
    with open(path) as f:
        data = json.load(f)
    return data.get("models", {})


def build_subject_to_role(role_subjects: Dict[str, List[str]]) -> Dict[str, str]:
    """Invert role_subjects mapping: subject -> role."""
    mapping = {}
    for role, subjects in role_subjects.items():
        for subject in subjects:
            mapping[subject] = role
    return mapping


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def format_question(row: pd.Series) -> str:
    """Format an MMLU row into a question string."""
    return (
        f"{row['question']}\n"
        f"A: {row['A']}\n"
        f"B: {row['B']}\n"
        f"C: {row['C']}\n"
        f"D: {row['D']}"
    )


def extract_answer(response: str) -> str:
    """Extract A/B/C/D from model response."""
    if not response:
        return ""
    boxed = re.findall(r"\\boxed\{([A-Da-d])\}", response)
    if boxed:
        return boxed[-1].upper()
    ans_match = re.search(r"answer is[:\s]*([A-Da-d])", response, re.IGNORECASE)
    if ans_match:
        return ans_match.group(1).upper()
    last_match = re.findall(r"\b([A-D])\b", response)
    if last_match:
        return last_match[-1]
    return ""


def query_model(
    client: OpenAI, model: str, system_prompt: str, question: str,
    max_tokens: int = 512, temperature: float = 0.0,
) -> Tuple[str, int, int]:
    """Send a question to the model and return (response_text, input_tokens, output_tokens).

    Falls back to folding system prompt into user message if model
    doesn't support the system role (e.g. Gemma).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    try:
        response = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
    except Exception as e:
        if "System role not supported" in str(e):
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{question}"}]
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature,
                )
                usage = response.usage
                return (
                    response.choices[0].message.content or "",
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                )
            except Exception as e2:
                print(f"  ERROR [{model}]: {e2}")
                return "", 0, 0
        print(f"  ERROR [{model}]: {e}")
        return "", 0, 0


def compute_cost(model_name: str, input_tokens: int, output_tokens: int,
                 model_costs: Dict[str, Dict[str, float]]) -> float:
    """Compute USD cost for a single LLM call."""
    costs = model_costs.get(model_name, {})
    input_rate = costs.get("input_per_million", 0.0)
    output_rate = costs.get("output_per_million", 0.0)
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


# ---------------------------------------------------------------------------
# Single question runner
# ---------------------------------------------------------------------------

def run_one_question(
    client: OpenAI, model: str, role: str, system_prompt: str,
    row: pd.Series, max_tokens: int, temperature: float,
    model_costs: Dict[str, Dict[str, float]],
) -> Dict:
    """Run a single question through one model with role prompt. Thread-safe."""
    question_text = format_question(row)
    start = time.perf_counter()
    response_text, input_tokens, output_tokens = query_model(
        client, model, system_prompt, question_text, max_tokens, temperature,
    )
    elapsed = time.perf_counter() - start
    predicted = extract_answer(response_text)
    is_correct = predicted == row["correct_answer"]
    cost = compute_cost(model, input_tokens, output_tokens, model_costs)

    return {
        "role": role,
        "model": model,
        "subject": row["subject"],
        "question": row["question"][:200],
        "gold": row["correct_answer"],
        "pred": predicted,
        "correct": int(is_correct),
        "latency_sec": round(elapsed, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 8),
    }


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(
    assignment: Dict[str, str],
    clients: Dict[str, OpenAI],
    role_prompts: Dict[str, str],
    role_questions: Dict[str, pd.DataFrame],
    trial_name: str,
    output_path: Path,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    model_costs: Dict[str, Dict[str, float]],
) -> Dict:
    """Run all questions through a role->model assignment.

    Args:
        assignment: {role: model_name} mapping
    """
    assignment_str = ", ".join(
        f"{role}={model.split('/')[-1]}" for role, model in assignment.items()
    )
    print(f"\n[{trial_name}] Assignment: {assignment_str}")

    n_questions = sum(len(df) for role, df in role_questions.items() if role in assignment)
    all_rows = []
    correct_count = 0
    total_count = 0
    lock = threading.Lock()

    def on_result(row):
        nonlocal correct_count, total_count
        with lock:
            all_rows.append(row)
            correct_count += row["correct"]
            total_count += 1
            if total_count % 100 == 0 or total_count == n_questions:
                acc = correct_count / total_count if total_count else 0
                print(f"  [{trial_name}] [{total_count}/{n_questions}] acc={acc:.4f}")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for role, model in assignment.items():
            if role not in role_questions:
                continue
            client = clients[model]
            system_prompt = role_prompts[role]
            df = role_questions[role]
            for _, row in df.iterrows():
                future = pool.submit(
                    run_one_question,
                    client, model, role, system_prompt, row,
                    max_tokens, temperature, model_costs,
                )
                futures[future] = (model, role)

        for future in as_completed(futures):
            on_result(future.result())

    # Write per-trial CSV sorted by subject then question
    all_rows.sort(key=lambda r: (r["role"], r["subject"], r["question"]))
    fieldnames = [
        "role", "model", "subject", "question", "gold", "pred", "correct",
        "latency_sec", "input_tokens", "output_tokens", "cost_usd",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    accuracy = correct_count / total_count if total_count else 0
    total_cost = sum(r["cost_usd"] for r in all_rows)
    total_input = sum(r["input_tokens"] for r in all_rows)
    total_output = sum(r["output_tokens"] for r in all_rows)
    print(f"  [{trial_name}] Final: {correct_count}/{total_count} = {accuracy:.4f} | Cost: ${total_cost:.6f}")

    return {
        "trial": trial_name,
        "correct": correct_count,
        "total": total_count,
        "accuracy": round(accuracy, 4),
        "cost_usd": round(total_cost, 6),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "assignment": json.dumps({r: m.split("/")[-1] for r, m in assignment.items()}),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Single-agent baseline with random/homogeneous assignments")
    parser.add_argument("--mode", choices=["random", "homogeneous", "both"], default="both")
    parser.add_argument("--n-trials", type=int, default=5, help="Number of random trials")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=500, help="Stratified sample limit (0 = all)")
    parser.add_argument("--output-dir", default="results/baseline")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--llm-profile", default=str(DEFAULT_LLM_PROFILE))
    parser.add_argument("--role-subjects", default=str(DEFAULT_ROLE_SUBJECTS))
    parser.add_argument("--model-costs", default=str(DEFAULT_MODEL_COSTS))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    # Setup
    job_id = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S"))
    output_dir = Path(args.output_dir) / f"job_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    models, base_urls = load_llm_profile(Path(args.llm_profile))
    model_costs = load_model_costs(Path(args.model_costs))
    role_subjects = load_role_subjects(Path(args.role_subjects))
    subject_to_role = build_subject_to_role(role_subjects)
    model_names = [m["Name"] for m in models]

    print(f"Models ({len(model_names)}): {[m.split('/')[-1] for m in model_names]}")
    print(f"Roles: {ROLES}")
    print(f"Split: {args.split} | Limit: {args.limit} | Concurrency: {args.concurrency}")
    print(f"Mode: {args.mode} | Random trials: {args.n_trials}")

    # Create one OpenAI client per model
    clients: Dict[str, OpenAI] = {}
    for name in model_names:
        url = base_urls.get(name)
        if not url:
            print(f"WARNING: No base_url for {name}, skipping")
            continue
        clients[name] = OpenAI(base_url=url, api_key="EMPTY")

    # Load role system prompts
    role_prompts: Dict[str, str] = {}
    for role in ROLES:
        role_prompts[role] = load_role_system_prompt(role)

    # Load questions per role with stratified sampling
    all_subjects = []
    for subs in role_subjects.values():
        all_subjects.extend(subs)
    all_subjects = sorted(set(all_subjects))

    # Load all questions, then stratified sample
    role_questions: Dict[str, pd.DataFrame] = {}
    for role in ROLES:
        subjects = role_subjects.get(role, [])
        df = load_subject_questions(args.split, subjects, Path(DEFAULT_DATA_ROOT))
        role_questions[role] = df
        print(f"  {role}: {len(df)} questions (full split)")

    total_full = sum(len(df) for df in role_questions.values())
    print(f"Total questions (full): {total_full}")

    # Stratified sampling across all roles
    if args.limit > 0 and total_full > args.limit:
        # Sample proportionally from each role
        ratio = args.limit / total_full
        for role in ROLES:
            df = role_questions[role]
            n_sample = max(1, int(len(df) * ratio))
            # Stratify within role by subject
            sampled = df.groupby("subject", group_keys=False).apply(
                lambda x: x.sample(n=max(1, int(len(x) * ratio)), random_state=42)
            ).reset_index(drop=True)
            role_questions[role] = sampled
            print(f"  {role}: sampled {len(sampled)} questions")

    total_sampled = sum(len(df) for df in role_questions.values())
    print(f"Total questions (sampled): {total_sampled}")
    print()

    all_results = []

    # --- Homogeneous baselines ---
    if args.mode in ("homogeneous", "both"):
        print("=" * 70)
        print("HOMOGENEOUS BASELINES")
        print("=" * 70)
        for model in model_names:
            if model not in clients:
                continue
            trial_name = f"homo_{model.split('/')[-1]}"
            assignment = {role: model for role in ROLES}
            result = run_trial(
                assignment=assignment,
                clients=clients,
                role_prompts=role_prompts,
                role_questions=role_questions,
                trial_name=trial_name,
                output_path=output_dir / f"{trial_name}.csv",
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                model_costs=model_costs,
            )
            all_results.append(result)

    # --- Random assignments ---
    if args.mode in ("random", "both"):
        print()
        print("=" * 70)
        print(f"RANDOM ASSIGNMENTS ({args.n_trials} trials)")
        print("=" * 70)
        rng = random.Random(args.seed)
        for trial_idx in range(args.n_trials):
            trial_name = f"random_{trial_idx:03d}"
            assignment = {role: rng.choice(model_names) for role in ROLES}
            result = run_trial(
                assignment=assignment,
                clients=clients,
                role_prompts=role_prompts,
                role_questions=role_questions,
                trial_name=trial_name,
                output_path=output_dir / f"{trial_name}.csv",
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                model_costs=model_costs,
            )
            all_results.append(result)

    # --- Summary ---
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary_path = output_dir / f"summary_{args.split}.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "trial", "accuracy", "correct", "total",
            "cost_usd", "input_tokens", "output_tokens", "assignment",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    homo_results = [r for r in all_results if r["trial"].startswith("homo_")]
    random_results = [r for r in all_results if r["trial"].startswith("random_")]

    if homo_results:
        print(f"\n{'Trial':<45} {'Accuracy':>10} {'Cost($)':>12}")
        print("-" * 70)
        for r in homo_results:
            print(f"{r['trial']:<45} {r['accuracy']:>10.4f} {r['cost_usd']:>12.6f}")

    if random_results:
        accs = [r["accuracy"] for r in random_results]
        costs = [r["cost_usd"] for r in random_results]
        mean_acc = sum(accs) / len(accs)
        mean_cost = sum(costs) / len(costs)
        std_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5
        print(f"\nRandom ({len(random_results)} trials):")
        print(f"  Mean accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
        print(f"  Min: {min(accs):.4f}  Max: {max(accs):.4f}")
        print(f"  Mean cost: ${mean_cost:.6f}")
        for r in random_results:
            print(f"  {r['trial']}: acc={r['accuracy']:.4f} cost=${r['cost_usd']:.6f} {r['assignment']}")

    print(f"\nSummary saved to: {summary_path}")
    print(f"Per-trial CSVs in: {output_dir}/")
    print(f"Job ID: {job_id}")


if __name__ == "__main__":
    main()
