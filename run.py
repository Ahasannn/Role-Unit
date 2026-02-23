"""
Role-Unit: Run MMLU questions through a fixed chain of LLM roles and measure accuracy.

Usage:
    python run.py [--config config/mmlu_config.yaml] [--limit N] [--split test]

Each role is assigned a specific LLM. The pipeline runs all roles sequentially,
then a FinalNode aggregates their outputs into a single answer (A/B/C/D).
Results are saved as CSV in the results/ directory.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from Datasets.mmlu_dataset import MMLUDataset
from MAR.Utils.utils import get_kwargs

# Import agent/graph machinery
import MAR.LLM.gpt_chat  # noqa: F401 — registers LLM classes
import MAR.Agent.agent    # noqa: F401 — registers Agent / FinalRefer


def parse_args():
    parser = argparse.ArgumentParser(description="Role-Unit MMLU Benchmark")
    parser.add_argument("--config", default="config/mmlu_config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Override dataset limit")
    parser.add_argument("--split", type=str, default=None, help="Override dataset split (dev/val/test)")
    parser.add_argument("--output", type=str, default=None, help="Override CSV output path")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_answer(text: str) -> str:
    """Extract A/B/C/D from model output."""
    if not text:
        return ""
    # Look for boxed answer first: \boxed{A}
    m = re.search(r"\\boxed\{([ABCD])\}", text)
    if m:
        return m.group(1)
    # Look for "the answer is X"
    m = re.search(r"answer\s+is\s+([ABCD])\b", text, re.IGNORECASE)
    if m:
        return m.group(1)
    # Take first letter A-D in the response
    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1)
    return text.strip()[:1].upper() if text.strip() else ""


def build_graph(role_configs: List[Dict], final_node_cfg: Dict, topology: str, num_rounds: int, domain: str):
    """Build a MAR Graph from the role configuration."""
    from MAR.Graph.graph import Graph
    from MAR.Utils.utils import get_kwargs

    agent_names = [r["role"] for r in role_configs]
    llm_names = [r["llm"] for r in role_configs]
    N = len(agent_names)
    kwargs = get_kwargs(topology, N)

    graph = Graph(
        domain=domain,
        llm_names=llm_names,
        agent_names=agent_names,
        decision_method="FinalRefer",
        reasoning_name="IO",
        prompt_file=final_node_cfg["prompt_file"],
        **kwargs,
    )
    # Override the decision node's LLM if specified
    from MAR.LLM.llm_registry import LLMRegistry
    graph.decision_node.llm = LLMRegistry.get(final_node_cfg["llm"])
    graph.decision_node.llm_name = final_node_cfg["llm"]
    return graph, kwargs.get("num_rounds", num_rounds)


def run_benchmark(cfg: Dict) -> str:
    """Run the full MMLU benchmark and return the CSV output path."""
    ds_cfg = cfg.get("dataset", {})
    pipe_cfg = cfg.get("pipeline", {})
    role_cfgs = cfg.get("roles", [])
    final_cfg = cfg.get("final_node", {})
    results_cfg = cfg.get("results", {})

    split = ds_cfg.get("split", "test")
    data_root = ds_cfg.get("data_root") or None
    limit = ds_cfg.get("limit", 0)
    topology = pipe_cfg.get("topology", "Chain")
    num_rounds = pipe_cfg.get("num_rounds", 1)
    domain = pipe_cfg.get("domain", "Commonsense")

    logger.info("Loading MMLU dataset (split={}, limit={})...", split, limit)
    dataset = MMLUDataset(split, data_root=data_root, stratified_limit=limit)
    logger.info("Loaded {} questions.", len(dataset))

    # Output CSV
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(results_cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_cfg.get("csv_file") or str(out_dir / f"mmlu_{ts}.csv")

    fieldnames = ["item_id", "question", "gold", "pred", "correct", "latency_sec",
                  "roles", "llms"]

    correct_total = 0
    total = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx in range(len(dataset)):
            record = dataset[idx]
            inp = MMLUDataset.record_to_input(record)
            gold = MMLUDataset.record_to_target_answer(record)

            # Build a fresh graph for each question (resets node state)
            graph, effective_rounds = build_graph(role_cfgs, final_cfg, topology, num_rounds, domain)

            start = time.perf_counter()
            try:
                answers, _ = graph.run(
                    {"query": inp["task"], "task": inp["task"]},
                    num_rounds=effective_rounds,
                )
                raw = answers[0] if answers else ""
            except Exception as e:
                logger.warning("Question {} failed: {}", idx, e)
                raw = ""
            elapsed = time.perf_counter() - start

            pred = extract_answer(raw)
            is_correct = pred.strip().upper() == gold.strip().upper()
            correct_total += int(is_correct)
            total += 1

            writer.writerow({
                "item_id": idx,
                "question": inp["task"][:200],
                "gold": gold,
                "pred": pred,
                "correct": int(is_correct),
                "latency_sec": round(elapsed, 3),
                "roles": json.dumps([r["role"] for r in role_cfgs]),
                "llms": json.dumps([r["llm"] for r in role_cfgs]),
            })
            f.flush()

            if (total % 10) == 0 or total == len(dataset):
                acc = correct_total / total if total else 0.0
                logger.info("[{}/{}] Accuracy so far: {:.3f}", total, len(dataset), acc)

    acc = correct_total / total if total else 0.0
    logger.info("Final accuracy: {:.4f} ({}/{})  →  {}", acc, correct_total, total, csv_path)
    return csv_path


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.limit is not None:
        cfg.setdefault("dataset", {})["limit"] = args.limit
    if args.split is not None:
        cfg.setdefault("dataset", {})["split"] = args.split
    if args.output is not None:
        cfg.setdefault("results", {})["csv_file"] = args.output

    run_benchmark(cfg)
