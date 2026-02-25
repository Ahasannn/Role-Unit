#!/usr/bin/env python3
"""
Role Unit Test Runner — Parallel All-Model Execution

Deploys all models (from llm_profile_full.json) and tests every (model × role)
combination concurrently. Each model is evaluated on role-specific MMLU subjects
using the actual role system prompt (not a generic prompt).

Usage:
    python unit_tests/run_unit_tests.py \
        --split validation \
        --concurrency 32 \
        --llm-profile config/llm_profile_full.json \
        --role-subjects config/role_subjects.yaml

    # Quick smoke test:
    python unit_tests/run_unit_tests.py --split dev --concurrency 4
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLM_PROFILE = PROJECT_ROOT / "config" / "llm_profile_full.json"
DEFAULT_ROLE_SUBJECTS = PROJECT_ROOT / "config" / "role_subjects.yaml"
DEFAULT_MODEL_COSTS = PROJECT_ROOT / "config" / "model_costs.json"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "Datasets" / "MMLU" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "unit_tests" / "results"

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_role_subjects(path: Path = DEFAULT_ROLE_SUBJECTS) -> Dict[str, List[str]]:
    """Load role -> subject mapping from YAML."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_subject_questions(
    split: str,
    subjects: List[str],
    data_root: Path = DEFAULT_DATA_ROOT,
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
    """Load model list and base_urls from llm_profile_full.json.

    Returns:
        models: list of model dicts (Name, vllm_config, etc.)
        base_urls: dict of model_name -> base_url
    """
    with open(path, "r") as f:
        data = json.load(f)
    models = data.get("models", [])
    base_urls = data.get("model_base_urls", {})
    return models, base_urls


def load_role_system_prompt(role: str) -> str:
    """Load role description from JSON and combine with Answer output format."""
    role_json_path = ROLES_DIR / f"{role}.json"
    with open(role_json_path, "r") as f:
        role_data = json.load(f)
    description = role_data["Description"]
    return f"{description}\n\n{ANSWER_FORMAT_PROMPT}"


def load_model_costs(path: Path = DEFAULT_MODEL_COSTS) -> Dict[str, Dict[str, float]]:
    """Load per-model token costs from JSON.

    Returns dict: model_name -> {"input_per_million": float, "output_per_million": float}
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("models", {})


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
    # Try boxed format first: \boxed{A}
    boxed = re.findall(r"\\boxed\{([A-Da-d])\}", response)
    if boxed:
        return boxed[-1].upper()

    # Try "answer is X" pattern
    ans_match = re.search(r"answer is[:\s]*([A-Da-d])", response, re.IGNORECASE)
    if ans_match:
        return ans_match.group(1).upper()

    # Fallback: last standalone A/B/C/D
    last_match = re.findall(r"\b([A-D])\b", response)
    if last_match:
        return last_match[-1]

    return ""


def query_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    question: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Tuple[str, int, int]:
    """Send a question to the model and return (response_text, input_tokens, output_tokens).

    Falls back to prepending system prompt into user message if the model
    doesn't support the system role (e.g. Gemma).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
    except Exception as e:
        if "System role not supported" in str(e):
            # Fold system prompt into user message
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{question}"},
            ]
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
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


# ---------------------------------------------------------------------------
# Single question task (for thread pool)
# ---------------------------------------------------------------------------

def process_question(
    client: OpenAI,
    model: str,
    role: str,
    system_prompt: str,
    row: pd.Series,
    max_tokens: int,
    temperature: float,
) -> Dict:
    """Process a single (model, role, question) triple. Thread-safe."""
    question_text = format_question(row)
    response_text, input_tokens, output_tokens = query_model(
        client, model, system_prompt, question_text, max_tokens, temperature
    )
    predicted = extract_answer(response_text)
    is_correct = predicted == row["correct_answer"]
    return {
        "model": model,
        "role": role,
        "subject": row["subject"],
        "question": row["question"][:100],
        "correct_answer": row["correct_answer"],
        "predicted": predicted,
        "is_correct": is_correct,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Role Unit Test Runner — Parallel All-Model")
    parser.add_argument("--split", default="validation", choices=["dev", "validation", "test"],
                        help="MMLU split (default: validation)")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Max parallel requests (default: 32)")
    parser.add_argument("--llm-profile", default=str(DEFAULT_LLM_PROFILE),
                        help="Path to llm_profile_full.json")
    parser.add_argument("--role-subjects", default=str(DEFAULT_ROLE_SUBJECTS),
                        help="Path to role_subjects.yaml")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                        help="Path to MMLU data root")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: unit_tests/results/job_<SLURM_JOB_ID>)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max questions per role (0 = all)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--roles", nargs="*", default=None,
                        help="Specific roles to test (default: all)")
    parser.add_argument("--model-costs", default=str(DEFAULT_MODEL_COSTS),
                        help="Path to model_costs.json")
    args = parser.parse_args()

    # --- Load configs ---
    models, base_urls = load_llm_profile(Path(args.llm_profile))
    model_costs = load_model_costs(Path(args.model_costs))
    role_subjects = load_role_subjects(Path(args.role_subjects))
    roles_to_test = args.roles or list(role_subjects.keys())

    model_names = [m["Name"] for m in models]
    print(f"Models ({len(model_names)}): {model_names}")
    print(f"Roles  ({len(roles_to_test)}): {roles_to_test}")
    print(f"Split: {args.split} | Concurrency: {args.concurrency}")
    print()

    # --- Setup output directory ---
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        output_dir = DEFAULT_OUTPUT_DIR / f"job_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Create one OpenAI client per model ---
    clients: Dict[str, OpenAI] = {}
    for model_name in model_names:
        url = base_urls.get(model_name)
        if not url:
            print(f"WARNING: No base_url for {model_name}, skipping")
            continue
        clients[model_name] = OpenAI(base_url=url, api_key="EMPTY")

    # --- Load role system prompts ---
    role_prompts: Dict[str, str] = {}
    for role in roles_to_test:
        role_prompts[role] = load_role_system_prompt(role)
        print(f"Loaded system prompt for {role} ({len(role_prompts[role])} chars)")

    # --- Load questions per role ---
    role_questions: Dict[str, pd.DataFrame] = {}
    for role in roles_to_test:
        subjects = role_subjects.get(role, [])
        if not subjects:
            print(f"WARNING: No subjects for role '{role}', skipping")
            continue
        df = load_subject_questions(args.split, subjects, Path(args.data_root))
        if args.limit > 0 and len(df) > args.limit:
            df = df.sample(n=args.limit, random_state=42).reset_index(drop=True)
        role_questions[role] = df
        print(f"  {role}: {len(df)} questions across {len(subjects)} subjects")

    total_questions = sum(len(df) for df in role_questions.values())
    total_tasks = total_questions * len(clients)
    print(f"\nTotal role-relevant questions: {total_questions}")
    print(f"Total tasks: {total_tasks} ({len(clients)} models x {total_questions} questions)")
    print()

    # --- Submit all (model, role, question) tasks ---
    all_details: List[Dict] = []
    start_time = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        for model_name, client in clients.items():
            for role in roles_to_test:
                if role not in role_questions:
                    continue
                df = role_questions[role]
                system_prompt = role_prompts[role]
                for _, row in df.iterrows():
                    future = executor.submit(
                        process_question,
                        client, model_name, role, system_prompt, row,
                        args.max_tokens, args.temperature,
                    )
                    futures[future] = (model_name, role)

        for future in as_completed(futures):
            result = future.result()
            all_details.append(result)
            completed += 1
            if completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                print(f"  [{completed}/{total_tasks}] {rate:.1f} q/s | {elapsed:.0f}s elapsed")

    total_elapsed = time.time() - start_time
    print(f"\nAll tasks complete: {completed} in {total_elapsed:.1f}s ({completed/total_elapsed:.1f} q/s)")

    # --- Compute cost per detail row ---
    details_df = pd.DataFrame(all_details)

    def compute_cost(row):
        costs = model_costs.get(row["model"], {})
        input_rate = costs.get("input_per_million", 0.0)
        output_rate = costs.get("output_per_million", 0.0)
        return (row["input_tokens"] * input_rate + row["output_tokens"] * output_rate) / 1_000_000

    details_df["cost_usd"] = details_df.apply(compute_cost, axis=1)

    # --- Compute score matrix ---
    score_rows = []
    for model_name in model_names:
        if model_name not in clients:
            continue
        for role in roles_to_test:
            if role not in role_questions:
                continue
            mask = (details_df["model"] == model_name) & (details_df["role"] == role)
            subset = details_df[mask]
            total = len(subset)
            correct = int(subset["is_correct"].sum())
            accuracy = correct / total if total > 0 else 0.0
            total_cost = float(subset["cost_usd"].sum())
            total_input = int(subset["input_tokens"].sum())
            total_output = int(subset["output_tokens"].sum())
            score_rows.append({
                "model": model_name,
                "role": role,
                "correct": correct,
                "total": total,
                "accuracy": round(accuracy, 4),
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cost_usd": round(total_cost, 6),
            })

    score_df = pd.DataFrame(score_rows)

    # --- Print score matrix ---
    print(f"\n{'='*90}")
    print("SCORE MATRIX")
    print(f"{'='*90}")
    print(f"{'Model':<45} {'Role':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10} {'Cost($)':>10}")
    print("-" * 90)
    for _, row in score_df.iterrows():
        print(f"{row['model']:<45} {row['role']:<12} {row['correct']:>8} {row['total']:>8} {row['accuracy']:>10.4f} {row['cost_usd']:>10.6f}")

    # --- Print pivot table ---
    print(f"\n{'='*90}")
    print("PIVOT: Model (rows) x Role (columns) = Accuracy")
    print(f"{'='*90}")
    pivot = score_df.pivot(index="model", columns="role", values="accuracy")
    pivot["mean"] = pivot.mean(axis=1)
    print(pivot.to_string(float_format="%.4f"))

    # Cost pivot
    print(f"\n{'='*90}")
    print("COST: Model (rows) x Role (columns) = USD")
    print(f"{'='*90}")
    cost_pivot = score_df.pivot(index="model", columns="role", values="cost_usd")
    cost_pivot["total"] = cost_pivot.sum(axis=1)
    print(cost_pivot.to_string(float_format="%.6f"))

    # --- Save outputs ---
    score_path = output_dir / "score_matrix.csv"
    score_df.to_csv(score_path, index=False)
    print(f"\nScore matrix saved to: {score_path}")

    details_path = output_dir / "details.csv"
    details_df.to_csv(details_path, index=False)
    print(f"Details saved to: {details_path}")

    # Run config CSV — captures full metadata for reproducibility
    config_path = output_dir / "run_config.csv"
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    config_rows = [{
        "job_id": job_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": args.split,
        "models": ";".join(sorted(clients.keys())),
        "roles": ";".join(roles_to_test),
        "total_questions": total_questions,
        "total_tasks": total_tasks,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "elapsed_s": round(total_elapsed, 1),
        "throughput_qps": round(completed / total_elapsed, 1),
        "llm_profile": args.llm_profile,
        "role_subjects": args.role_subjects,
    }]
    pd.DataFrame(config_rows).to_csv(config_path, index=False)
    print(f"Run config saved to: {config_path}")

    # Summary text
    total_cost = float(details_df["cost_usd"].sum())
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Role Unit Test Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Job ID: {job_id}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Models: {len(clients)}\n")
        f.write(f"Roles: {roles_to_test}\n")
        f.write(f"Total questions: {total_questions}\n")
        f.write(f"Total tasks: {total_tasks}\n")
        f.write(f"Concurrency: {args.concurrency}\n")
        f.write(f"Total time: {total_elapsed:.1f}s\n")
        f.write(f"Throughput: {completed/total_elapsed:.1f} q/s\n")
        f.write(f"Total cost (OpenRouter equiv): ${total_cost:.6f}\n")
        f.write(f"\nAccuracy Matrix:\n")
        f.write(pivot.to_string(float_format="%.4f"))
        f.write(f"\n\nCost Matrix (USD):\n")
        f.write(cost_pivot.to_string(float_format="%.6f"))
        f.write(f"\n\nPer-model summary:\n")
        for model_name in pivot.index:
            model_cost = float(cost_pivot.loc[model_name, "total"])
            f.write(f"  {model_name}: accuracy={pivot.loc[model_name, 'mean']:.4f}, cost=${model_cost:.6f}\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
