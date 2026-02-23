"""LLM-as-a-Judge scoring for InfraMind Phase 1 CSV data.

Reads a Phase 1 exploration CSV, sends each agent response to a judge LLM
with a rubric (1-10 scale), and writes an output CSV with ``judge_score``
column added.

Usage:
    python Experiments/judge_responses.py \
        --input-csv logs/inframind_explore.csv \
        --output-csv logs/inframind_explore_scored.csv \
        --judge-model meta-llama/Llama-3.3-70B-Instruct \
        --judge-url http://localhost:8010/v1 \
        --max-concurrent 20
"""

import argparse
import asyncio
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger

import json

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth_map(dataset_root: str, split: str = "train") -> Dict[str, str]:
    """Load ground truth answers from MATH dataset.

    Returns mapping from item_id (e.g. 'Prealgebra/89') to reference answer string.
    The answer string includes both the final answer and the reference solution.
    """
    gt_map: Dict[str, str] = {}
    root = Path(dataset_root) / split
    if not root.is_dir():
        logger.warning("MATH dataset root not found: {}", root)
        return gt_map
    for json_path in root.rglob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            category = json_path.parent.name
            stem = json_path.stem
            item_id = f"{category}/{stem}"
            answer = data.get("answer", "")
            solution = data.get("solution", "")
            if answer:
                gt_map[item_id] = f"Final answer: {answer}\nReference solution: {solution}"
        except Exception:
            continue
    logger.info("Loaded {} ground truth answers from {}", len(gt_map), root)
    return gt_map


# ---------------------------------------------------------------------------
# Judge rubric prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for a multi-agent math-solving system. Your task is \
to score an agent's response on how much it contributes toward reaching the \
correct answer.

You will be given:
1. The math problem and the agent's role/instructions.
2. The REFERENCE ANSWER (ground truth) — use this to evaluate correctness.
3. The agent's RESPONSE.

IMPORTANT: This is a step in a multi-agent pipeline. The agent may not produce \
the final answer alone — it may contribute partial reasoning that later agents \
build upon. Score based on the QUALITY OF CONTRIBUTION, not whether a neat \
final answer appears.

IMPORTANT: Response length is IRRELEVANT. A long chain-of-thought that makes \
correct progress is just as valuable as a short direct answer. Do NOT penalize \
verbose responses. Do NOT reward brevity. Only the mathematical correctness and \
reasoning quality matter.

## Scoring Rubric (1-10)

### Correctness & Reasoning (sole criteria)
- **10**: Arrives at the correct answer with sound reasoning.
- **9**: Correct answer; reasoning is sound but has minor notation or formatting \
issues.
- **8**: Correct answer; reasoning has small gaps or redundant steps but core \
logic holds.
- **7**: Reasoning is correct and heading toward the right answer, but the \
response is incomplete (truncated, unfinished). Award 7 if the mathematical \
steps shown so far are correct and aligned with the reference solution.
- **6**: Sound mathematical approach that would lead to the correct answer, but \
has minor arithmetic or algebraic errors (sign error, miscalculation).
- **5**: Identifies the right approach/method but makes significant execution \
errors leading to a wrong answer.
- **4**: Shows awareness of the relevant topic but the approach has fundamental \
flaws or incorrect assumptions.
- **3**: Mostly wrong reasoning, but contains at least one relevant mathematical \
insight or step.
- **2**: Almost entirely wrong; minimal mathematical relevance to the problem.
- **1**: Completely wrong, empty, refuses to answer, or irrelevant.

### Co-Agent Context Usage (adjust up or down by 1 point)
- If hints or solutions from co-agents appear in the prompt, evaluate whether \
the agent appropriately incorporated, verified, or built upon them.
- Reward agents that correctly identify and fix errors from co-agent hints.
- Penalize blind copying without verification.

Output ONLY a single integer from 1 to 10. No explanation, no extra text."""

JUDGE_USER_TEMPLATE = """\
## Problem & Agent Instructions
{prompt_base}

## Reference Answer (Ground Truth)
{reference_answer}

## Agent Response
{response}

Score (1-10):"""


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def parse_score(text: str) -> Optional[int]:
    """Extract integer score from judge response."""
    text = text.strip()
    # Try first line
    first_line = text.split("\n")[0].strip()
    # Direct integer
    if first_line.isdigit():
        score = int(first_line)
        if 1 <= score <= 10:
            return score
    # Regex fallback: find first number in range 1-10
    match = re.search(r"\b(\d{1,2})\b", first_line)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 10:
            return score
    # Search entire response
    match = re.search(r"\b(\d{1,2})\s*/\s*10\b", text)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 10:
            return score
    match = re.search(r"(?:score|rating)\s*[:=]\s*(\d{1,2})", text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 10:
            return score
    return None


async def judge_single(
    client: "AsyncOpenAI",
    model: str,
    prompt_base: str,
    strategy_name: str,
    response: str,
    reference_answer: str = "",
    max_retries: int = 3,
    timeout: float = 300.0,
) -> Optional[int]:
    """Call the judge LLM and return a 1-10 score."""
    user_msg = JUDGE_USER_TEMPLATE.format(
        prompt_base=prompt_base,
        reference_answer=reference_answer or "(not available)",
        response=response,
    )
    for attempt in range(max_retries):
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=64,
                    temperature=0.0,
                ),
                timeout=timeout,
            )
            text = resp.choices[0].message.content or ""
            score = parse_score(text)
            if score is not None:
                return score
            logger.warning("Could not parse score from judge response: {}", text[:100])
        except asyncio.TimeoutError:
            logger.warning("Judge timeout (attempt {}/{})", attempt + 1, max_retries)
        except Exception as e:
            logger.warning("Judge API error (attempt {}/{}): {}", attempt + 1, max_retries, e)
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_input_csv(path: str) -> List[Dict[str, str]]:
    """Load CSV and filter to role_step rows that have response data."""
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_scored_map(output_path: str) -> Dict[str, int]:
    """Load already-scored row identifiers with their scores for resume.

    Only rows with a non-empty ``judge_score`` are considered scored.
    Returns a mapping from row key to score.
    """
    if not os.path.isfile(output_path):
        return {}
    scored = {}
    with open(output_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score_str = row.get("judge_score", "").strip()
            if score_str:
                try:
                    scored[_row_key(row)] = int(score_str)
                except ValueError:
                    pass
    return scored


def _row_key(row: Dict[str, str]) -> str:
    """Unique identifier for a row (episode + step + arrival rate)."""
    return f"{row.get('run_id', '')}|{row.get('episode_index', '')}|{row.get('step_index', '')}|{row.get('arrival_rate', '')}|{row.get('record_type', '')}"


async def run_judge_pipeline(args: argparse.Namespace) -> None:
    if AsyncOpenAI is None:
        logger.error("openai package not installed. Run: pip install openai")
        return

    client = AsyncOpenAI(
        base_url=args.judge_url,
        api_key=args.api_key or "EMPTY",
    )

    # Load ground truth answers (if dataset root provided)
    gt_map: Dict[str, str] = {}
    if args.dataset_root:
        gt_map = load_ground_truth_map(args.dataset_root, split=args.dataset_split)

    # Load input
    all_rows = load_input_csv(args.input_csv)
    logger.info("Loaded {} total rows from {}", len(all_rows), args.input_csv)

    # Determine output fieldnames (add judge_score)
    with open(args.input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
    if "judge_score" not in fieldnames:
        fieldnames.append("judge_score")

    # Resume support: load existing scores (only rows with non-empty judge_score)
    existing_scores: Dict[str, int] = {}
    if args.resume and os.path.isfile(args.output_csv):
        existing_scores = load_scored_map(args.output_csv)
        logger.info("Resuming: {} rows already scored, will skip those", len(existing_scores))

    # Separate rows: role_step rows with response get judged, others pass through
    to_judge = []
    passthrough = []
    for row in all_rows:
        key = _row_key(row)
        if key in existing_scores:
            # Already scored — carry the score forward, treat as passthrough
            row["judge_score"] = str(existing_scores[key])
            passthrough.append(row)
            continue
        record_type = row.get("record_type", "")
        response = row.get("response", "").strip()
        prompt_base = row.get("prompt_base", "").strip()
        if record_type == "role_step" and response and prompt_base:
            to_judge.append(row)
        else:
            passthrough.append(row)

    logger.info("{} rows to judge, {} passthrough rows (incl. {} already scored)",
                len(to_judge), len(passthrough), len(existing_scores))

    # Always rewrite the full output file (fresh write with all rows)
    out_f = open(args.output_csv, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    # Write passthrough rows immediately (episode rows + already-scored rows)
    for row in passthrough:
        row.setdefault("judge_score", "")
        writer.writerow(row)
    out_f.flush()

    # Concurrent judging with semaphore
    semaphore = asyncio.Semaphore(args.max_concurrent)
    write_lock = asyncio.Lock()
    scored_count = 0
    failed_count = 0
    start_time = time.time()

    async def judge_and_write(row: Dict[str, str]) -> None:
        nonlocal scored_count, failed_count
        # Look up ground truth by item_id
        item_id = row.get("item_id", "").strip()
        reference_answer = gt_map.get(item_id, "")
        async with semaphore:
            score = await judge_single(
                client,
                model=args.judge_model,
                prompt_base=row.get("prompt_base", ""),
                strategy_name=row.get("strategy_name", ""),
                response=row.get("response", ""),
                reference_answer=reference_answer,
            )
        async with write_lock:
            if score is not None:
                row["judge_score"] = str(score)
                scored_count += 1
            else:
                row["judge_score"] = ""
                failed_count += 1
            writer.writerow(row)
            total = scored_count + failed_count
            if total % 100 == 0:
                out_f.flush()
                elapsed = time.time() - start_time
                rate = total / max(elapsed, 0.01)
                remaining = (len(to_judge) - total) / max(rate, 0.01)
                logger.info(
                    "Progress: {}/{} scored ({} failed), {:.1f} rows/s, ETA {:.0f}s",
                    scored_count, len(to_judge), failed_count, rate, remaining,
                )

    # Run all judge tasks
    tasks = [judge_and_write(row) for row in to_judge]
    await asyncio.gather(*tasks)

    out_f.flush()
    out_f.close()
    elapsed = time.time() - start_time
    logger.info(
        "Done: {}/{} scored, {} failed, {:.1f}s elapsed. Output: {}",
        scored_count, len(to_judge), failed_count, elapsed, args.output_csv,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge scoring for InfraMind Phase 1 exploration data.",
    )
    parser.add_argument(
        "--input-csv", type=str, required=True,
        help="Path to Phase 1 exploration CSV with response column.",
    )
    parser.add_argument(
        "--output-csv", type=str, required=True,
        help="Output CSV path (input CSV + judge_score column).",
    )
    parser.add_argument(
        "--judge-model", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model name for the judge LLM.",
    )
    parser.add_argument(
        "--judge-url", type=str, default="http://localhost:8010/v1",
        help="OpenAI-compatible API base URL for the judge LLM.",
    )
    parser.add_argument(
        "--api-key", type=str, default="",
        help="API key for the judge endpoint (default: EMPTY).",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=20,
        help="Maximum concurrent judge requests.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from partial output (append to existing output CSV).",
    )
    parser.add_argument(
        "--dataset-root", type=str, default="",
        help="Path to MATH dataset root (contains train/test dirs with JSON files). "
             "When provided, ground truth answers are included in the judge prompt.",
    )
    parser.add_argument(
        "--dataset-split", type=str, default="train",
        help="Dataset split to load ground truth from (default: train).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_judge_pipeline(args))
