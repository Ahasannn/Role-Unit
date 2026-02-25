#!/usr/bin/env python3
"""
Generate result plots for Role-Unit paper.

Reads validation analysis + baseline test results and produces:
  1. Pareto frontier: ILP-optimal vs baselines (accuracy vs cost)
  2. Fitness heatmap: model x role validation accuracy matrix
  3. Strategy comparison bar chart
  4. Cost breakdown: per-role cost across Pareto points

Usage:
    python visualization/plot_results.py \
        --val-scores unit_tests/results/job_25626312/score_matrix_validation.csv \
        --pareto unit_tests/results/job_25626312/pareto_frontier.csv \
        --baseline-summary results/baseline/job_25626312/summary_test.csv \
        --output-dir visualization/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

USD_TO_CENTS = 100


def short_name(model: str) -> str:
    name = model.split("/")[-1]
    mapping = {
        "Mistral-Small-24B-Instruct-2501": "Mistral-24B",
        "Qwen2.5-Coder-14B-Instruct": "Qwen-Coder-14B",
        "Qwen2.5-7B-Instruct": "Qwen-7B",
        "gemma-2-9b-it": "Gemma-9B",
        "Llama-3.1-8B-Instruct": "Llama-8B",
        "Llama-3.2-3B-Instruct": "Llama-3B",
    }
    return mapping.get(name, name)


def _build_val_lookups(val_df):
    """Build common lookups from val_df."""
    fitness_lookup = {(row["model"], row["role"]): row["accuracy"]
                      for _, row in val_df.iterrows()}
    cost_lookup = {(row["model"], row["role"]): row["cost_usd"]
                   for _, row in val_df.iterrows()}
    roles = sorted(val_df["role"].unique())
    models = sorted(val_df["model"].unique())
    role_weights = {}
    for role in roles:
        role_weights[role] = int(val_df[val_df["role"] == role]["total"].values[0])
    total_w = sum(role_weights.values())
    return fitness_lookup, cost_lookup, roles, models, role_weights, total_w


def plot_pareto_frontier(pareto_df, val_df, baseline_df, output_dir):
    """Plot 1: Pareto frontier with baselines overlaid."""
    fig, ax = plt.subplots(figsize=(10, 7))

    fitness_lookup, cost_lookup, roles, models, role_weights, total_w = _build_val_lookups(val_df)

    # --- Pareto frontier ---
    pareto_costs = pareto_df["val_cost_usd"].values * USD_TO_CENTS
    pareto_accs = pareto_df["val_accuracy"].values * 100

    ax.plot(pareto_costs, pareto_accs, "-o", linewidth=2, markersize=8,
            label="ILP-Optimal (Pareto)", zorder=5, color="#2563EB")
    ax.fill_between(pareto_costs, pareto_accs, alpha=0.08, color="#2563EB")

    # Annotate key Pareto points
    for i, row in pareto_df.iterrows():
        assignment = json.loads(row["assignment"])
        models_used = set(assignment.values())
        if len(pareto_df) <= 6 or i in [0, len(pareto_df) // 2, len(pareto_df) - 1]:
            label = "/".join(sorted(set(short_name(m) for m in models_used)))
            ax.annotate(label,
                        (row["val_cost_usd"] * USD_TO_CENTS, row["val_accuracy"] * 100),
                        textcoords="offset points", xytext=(8, -12),
                        fontsize=7, color="#1E40AF", style="italic")

    # --- Homogeneous baselines ---
    homo_colors = sns.color_palette("Set2", len(models))
    for idx, model in enumerate(models):
        model_data = val_df[val_df["model"] == model]
        acc = model_data["accuracy"].mean() * 100
        cost = model_data["cost_usd"].sum() * USD_TO_CENTS
        sname = short_name(model)
        ax.scatter(cost, acc, s=120, marker="s", color=homo_colors[idx],
                   edgecolor="black", linewidth=0.8, zorder=4, label=f"Homo: {sname}")

    # --- Baseline test results (if provided) ---
    if baseline_df is not None:
        random_trials = baseline_df[baseline_df["trial"].str.startswith("random_")]
        if not random_trials.empty:
            rand_accs = random_trials["accuracy"].values * 100
            rand_costs = random_trials["cost_usd"].values * USD_TO_CENTS
            ax.scatter(rand_costs, rand_accs, s=60, marker="x", color="#DC2626",
                       linewidth=1.5, zorder=3, alpha=0.7, label="Random trials (test)")
            ax.scatter(rand_costs.mean(), rand_accs.mean(), s=150, marker="X",
                       color="#DC2626", edgecolor="black", linewidth=1, zorder=4,
                       label=f"Random mean ({rand_accs.mean():.1f}%)")

    # --- All 216 assignments (gray cloud) ---
    from itertools import product
    all_accs_rand = []
    all_costs_rand = []
    for combo in product(models, repeat=len(roles)):
        assignment = dict(zip(roles, combo))
        acc = sum(role_weights[r] / total_w * fitness_lookup.get((m, r), 0)
                  for r, m in assignment.items()) * 100
        cst = sum(cost_lookup.get((m, r), 0) for r, m in assignment.items()) * USD_TO_CENTS
        all_accs_rand.append(acc)
        all_costs_rand.append(cst)

    ax.scatter(all_costs_rand, all_accs_rand, s=15, marker=".", color="#9CA3AF",
               alpha=0.3, zorder=1, label=f"All {len(all_accs_rand)} assignments")

    ax.set_xlabel("Cost (cents)", fontsize=13)
    ax.set_ylabel("Weighted Accuracy (%)", fontsize=13)
    ax.set_title("ILP-Optimal Pareto Frontier vs Baselines", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_dir / "pareto_frontier.png", bbox_inches="tight")
    fig.savefig(output_dir / "pareto_frontier.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: pareto_frontier.png/pdf")


def plot_fitness_heatmap(val_df, output_dir):
    """Plot 2: Fitness heatmap — model x role accuracy matrix."""
    pivot = val_df.pivot(index="model", columns="role", values="accuracy")
    pivot.index = [short_name(m) for m in pivot.index]

    pivot["Mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("Mean", ascending=False)
    mean_col = pivot.pop("Mean")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5),
                                     gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05})

    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.4, vmax=1.0,
                linewidths=1, ax=ax1, cbar_kws={"label": "Accuracy", "shrink": 0.8})
    ax1.set_title("Validation Fitness Matrix", fontsize=14, fontweight="bold")
    ax1.set_ylabel("")
    ax1.set_xlabel("")

    mean_df = mean_col.to_frame("Mean")
    sns.heatmap(mean_df, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.4, vmax=1.0,
                linewidths=1, ax=ax2, cbar=False, yticklabels=False)
    ax2.set_title("Mean", fontsize=12)
    ax2.set_xlabel("")

    plt.tight_layout()
    fig.savefig(output_dir / "fitness_heatmap.png", bbox_inches="tight")
    fig.savefig(output_dir / "fitness_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fitness_heatmap.png/pdf")


def plot_cost_heatmap(val_df, output_dir):
    """Plot 3: Cost heatmap — model x role cost matrix in cents."""
    pivot = val_df.pivot(index="model", columns="role", values="cost_usd")
    pivot.index = [short_name(m) for m in pivot.index]
    pivot = pivot * USD_TO_CENTS  # convert to cents

    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False)
    total_col = pivot.pop("Total")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5),
                                     gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05})

    def cents_annot(data):
        return data.map(lambda v: f"{v:.2f}\u00a2")

    sns.heatmap(pivot, annot=cents_annot(pivot), fmt="", cmap="YlOrRd", linewidths=1,
                ax=ax1, cbar_kws={"label": "Cost (cents)", "shrink": 0.8})
    ax1.set_title("Validation Cost Matrix (cents)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("")
    ax1.set_xlabel("")

    total_df = total_col.to_frame("Total")
    sns.heatmap(total_df, annot=cents_annot(total_df), fmt="", cmap="YlOrRd", linewidths=1,
                ax=ax2, cbar=False, yticklabels=False)
    ax2.set_title("Total", fontsize=12)
    ax2.set_xlabel("")

    plt.tight_layout()
    fig.savefig(output_dir / "cost_heatmap.png", bbox_inches="tight")
    fig.savefig(output_dir / "cost_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: cost_heatmap.png/pdf")


def plot_strategy_comparison(pareto_df, val_df, baseline_df, output_dir):
    """Plot 4: Bar chart comparing key strategies."""
    fitness_lookup, cost_lookup, roles, models, role_weights, total_w = _build_val_lookups(val_df)

    rows = []

    # Key Pareto points
    if len(pareto_df) >= 3:
        mid = len(pareto_df) // 2
        key_pareto = [
            (0, "ILP: Min Cost"),
            (mid, "ILP: Balanced"),
            (len(pareto_df) - 1, "ILP: Max Acc"),
        ]
    else:
        key_pareto = [(i, f"ILP-P{i}") for i in range(len(pareto_df))]

    for idx, label in key_pareto:
        row = pareto_df.iloc[idx]
        rows.append({"Strategy": label, "Accuracy": row["val_accuracy"] * 100,
                      "Cost": row["val_cost_usd"] * USD_TO_CENTS, "type": "ilp"})

    # Homogeneous baselines
    for model in models:
        acc = sum(role_weights[r] / total_w * fitness_lookup.get((model, r), 0)
                  for r in roles) * 100
        cost = sum(cost_lookup.get((model, r), 0) for r in roles) * USD_TO_CENTS
        rows.append({"Strategy": f"Homo: {short_name(model)}", "Accuracy": acc,
                      "Cost": cost, "type": "homo"})

    # Random baseline from test
    if baseline_df is not None:
        random_trials = baseline_df[baseline_df["trial"].str.startswith("random_")]
        if not random_trials.empty:
            rows.append({
                "Strategy": "Random (test mean)",
                "Accuracy": random_trials["accuracy"].mean() * 100,
                "Cost": random_trials["cost_usd"].mean() * USD_TO_CENTS,
                "type": "random",
            })

    df = pd.DataFrame(rows)

    colors = []
    for _, row in df.iterrows():
        if row["type"] == "ilp":
            colors.append("#2563EB")
        elif row["type"] == "random":
            colors.append("#DC2626")
        else:
            colors.append("#6B7280")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy bars
    bars1 = ax1.barh(df["Strategy"], df["Accuracy"], color=colors, edgecolor="white")
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_title("Accuracy Comparison", fontweight="bold")
    ax1.set_xlim(50, 95)
    for bar, val in zip(bars1, df["Accuracy"]):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=9)

    # Cost bars
    bars2 = ax2.barh(df["Strategy"], df["Cost"], color=colors, edgecolor="white")
    ax2.set_xlabel("Cost (cents)")
    ax2.set_title("Cost Comparison", fontweight="bold")
    for bar, val in zip(bars2, df["Cost"]):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}\u00a2", va="center", fontsize=9)

    ax1.invert_yaxis()
    ax2.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_dir / "strategy_comparison.png", bbox_inches="tight")
    fig.savefig(output_dir / "strategy_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: strategy_comparison.png/pdf")


def plot_pareto_savings(pareto_df, output_dir):
    """Plot 5: Cost savings vs accuracy tradeoff along Pareto frontier."""
    if len(pareto_df) < 2:
        return

    max_cost = pareto_df["val_cost_usd"].max()

    fig, ax = plt.subplots(figsize=(8, 5))

    accs = pareto_df["val_accuracy"].values * 100
    savings = (1 - pareto_df["val_cost_usd"].values / max_cost) * 100

    ax.plot(accs, savings, "o-", color="#2563EB", linewidth=2, markersize=10, zorder=3)

    for i, (a, s) in enumerate(zip(accs, savings)):
        ax.annotate(f"P{i}", (a, s), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="#1E40AF")

    # Highlight best efficiency point
    ratios = pareto_df["val_accuracy"] / pareto_df["val_cost_usd"]
    best_eff_idx = ratios.idxmax()
    ax.scatter(accs[best_eff_idx], savings[best_eff_idx], s=200, marker="*",
               color="#F59E0B", edgecolor="black", linewidth=1, zorder=5,
               label="Best cost-efficiency")

    ax.set_xlabel("Accuracy (%)", fontsize=13)
    ax.set_ylabel("Cost Savings vs Max (%)", fontsize=13)
    ax.set_title("Accuracy-Cost Tradeoff (Pareto Frontier)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "pareto_savings.png", bbox_inches="tight")
    fig.savefig(output_dir / "pareto_savings.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: pareto_savings.png/pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate result plots")
    parser.add_argument("--val-scores", required=True)
    parser.add_argument("--pareto", required=True, help="pareto_frontier.csv")
    parser.add_argument("--baseline-summary", default=None, help="Baseline summary_test.csv")
    parser.add_argument("--output-dir", default="visualization/plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_csv(args.val_scores)
    pareto_df = pd.read_csv(args.pareto)
    baseline_df = pd.read_csv(args.baseline_summary) if args.baseline_summary else None

    print("Generating plots...")

    plot_pareto_frontier(pareto_df, val_df, baseline_df, output_dir)
    plot_fitness_heatmap(val_df, output_dir)
    plot_cost_heatmap(val_df, output_dir)
    plot_strategy_comparison(pareto_df, val_df, baseline_df, output_dir)
    plot_pareto_savings(pareto_df, output_dir)

    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
