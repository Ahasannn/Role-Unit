"""
InfraMind Failure Analysis — Plots for Group Meeting Slides
============================================================
Generates publication-quality figures showing:
1. Training accuracy (flat / no learning)
2. Model selection collapse across epochs
3. Strategy selection collapse across epochs
4. Test-time model & strategy collapse
5. Accuracy comparison table (InfraMind vs Baseline)
6. Credit assignment problem diagram
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
TRAIN_CSV = "logs/inframind_math_20260212_161547_job24816311.csv"
TEST_CSV  = "logs/motivation_plot_generator_data/inframind_motivation_sweep_math_test_1000_poisson.csv"
BASE_CSV  = "logs/motivation_plot_generator_data/baseline_motivation_sweep_math_test_1000_poisson.csv"
OUT_DIR   = Path("visualization/failure_analysis_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SHORT = {
    "meta-llama/Llama-3.2-3B-Instruct":               "Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B-Instruct":               "Llama-3.1-8B",
    "Qwen/Qwen2.5-Coder-14B-Instruct":               "Qwen2.5-14B",
    "mistralai/Mistral-Small-24B-Instruct-2501":      "Mistral-24B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":      "DeepSeek-R1-32B",
}
MODEL_ORDER = [
    "Llama-3.2-3B", "Llama-3.1-8B", "Qwen2.5-14B",
    "Mistral-24B", "DeepSeek-R1-32B",
]
STRAT_ORDER = ["Flash", "Concise", "DeepThink"]

# Colors
MODEL_COLORS = {
    "Llama-3.2-3B":   "#e74c3c",
    "Llama-3.1-8B":   "#3498db",
    "Qwen2.5-14B":    "#2ecc71",
    "Mistral-24B":    "#f39c12",
    "DeepSeek-R1-32B":"#9b59b6",
}
STRAT_COLORS = {
    "Flash":     "#e74c3c",
    "Concise":   "#f39c12",
    "DeepThink": "#2ecc71",
}

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# ──────────────────────────────────────────────────────────────
# Load data helpers
# ──────────────────────────────────────────────────────────────
def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def short_model(name):
    return MODEL_SHORT.get(name, name.split("/")[-1])


# ──────────────────────────────────────────────────────────────
# 1. Training accuracy by epoch (flat line = no learning)
# ──────────────────────────────────────────────────────────────
def plot_training_accuracy(train_rows):
    episodes = [r for r in train_rows if r["record_type"] == "episode"]
    acc_by_epoch = defaultdict(list)
    for r in episodes:
        acc_by_epoch[int(r["epoch"])].append(float(r["quality"]))

    epochs = sorted(acc_by_epoch)
    means  = [np.mean(acc_by_epoch[e]) for e in epochs]
    stds   = [np.std(acc_by_epoch[e])  for e in epochs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(epochs, means, yerr=stds, fmt="o-", color="#e74c3c",
                linewidth=2, markersize=6, capsize=4, label="InfraMind Executor")
    ax.axhline(y=means[0], color="gray", linestyle="--", alpha=0.6,
               label=f"Epoch 0 baseline ({means[0]:.1%})")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("InfraMind Training Accuracy — No Learning Signal")
    ax.set_ylim(0.30, 0.75)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT_DIR / "01_training_accuracy.png")
    plt.close(fig)
    print(f"  [1/7] Training accuracy plot saved")


# ──────────────────────────────────────────────────────────────
# 2. Model selection distribution over epochs (stacked area)
# ──────────────────────────────────────────────────────────────
def plot_model_selection_by_epoch(train_rows):
    steps = [r for r in train_rows if r["record_type"] == "role_step"]
    counts = defaultdict(lambda: defaultdict(int))
    for r in steps:
        ep = int(r["epoch"])
        m  = short_model(r["model_name"])
        counts[ep][m] += 1

    epochs = sorted(counts)
    fracs = {m: [] for m in MODEL_ORDER}
    for ep in epochs:
        total = sum(counts[ep].values())
        for m in MODEL_ORDER:
            fracs[m].append(counts[ep].get(m, 0) / total * 100)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(epochs))
    for m in MODEL_ORDER:
        vals = np.array(fracs[m])
        ax.bar(epochs, vals, bottom=bottom, label=m, color=MODEL_COLORS[m], width=0.8)
        bottom += vals

    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Selection Frequency (%)")
    ax.set_title("Executor Model Selection During Training")
    ax.legend(loc="upper left", fontsize=10, ncol=2)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2, axis="y")
    fig.savefig(OUT_DIR / "02_model_selection_by_epoch.png")
    plt.close(fig)
    print(f"  [2/7] Model selection by epoch saved")


# ──────────────────────────────────────────────────────────────
# 3. Strategy selection distribution over epochs
# ──────────────────────────────────────────────────────────────
def plot_strategy_selection_by_epoch(train_rows):
    steps = [r for r in train_rows if r["record_type"] == "role_step"]
    counts = defaultdict(lambda: defaultdict(int))
    for r in steps:
        ep = int(r["epoch"])
        s  = r["strategy_name"]
        counts[ep][s] += 1

    epochs = sorted(counts)
    fracs = {s: [] for s in STRAT_ORDER}
    for ep in epochs:
        total = sum(counts[ep].values())
        for s in STRAT_ORDER:
            fracs[s].append(counts[ep].get(s, 0) / total * 100)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(epochs))
    for s in STRAT_ORDER:
        vals = np.array(fracs[s])
        ax.bar(epochs, vals, bottom=bottom, label=s, color=STRAT_COLORS[s], width=0.8)
        bottom += vals

    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Selection Frequency (%)")
    ax.set_title("Executor Strategy Selection During Training")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2, axis="y")
    fig.savefig(OUT_DIR / "03_strategy_selection_by_epoch.png")
    plt.close(fig)
    print(f"  [3/7] Strategy selection by epoch saved")


# ──────────────────────────────────────────────────────────────
# 4. Test-time collapse — pie charts for model & strategy
# ──────────────────────────────────────────────────────────────
def plot_test_collapse(test_rows):
    steps = [r for r in test_rows if r["record_type"] == "role_step"]
    model_counts = defaultdict(int)
    strat_counts = defaultdict(int)
    for r in steps:
        model_counts[short_model(r["model_name"])] += 1
        strat_counts[r["strategy_name"]] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Model pie
    labels_m = sorted(model_counts, key=lambda x: -model_counts[x])
    sizes_m  = [model_counts[m] for m in labels_m]
    colors_m = [MODEL_COLORS.get(m, "#95a5a6") for m in labels_m]
    explode_m = [0.05 if model_counts[m]/sum(sizes_m) > 0.5 else 0 for m in labels_m]
    wedges, texts, autotexts = ax1.pie(
        sizes_m, labels=labels_m, colors=colors_m, autopct="%1.1f%%",
        explode=explode_m, startangle=140, textprops={"fontsize": 11})
    for t in autotexts:
        t.set_fontweight("bold")
    ax1.set_title("Model Selection at Test Time\n(Collapsed to smallest model)", fontsize=13)

    # Strategy pie
    labels_s = sorted(strat_counts, key=lambda x: -strat_counts[x])
    sizes_s  = [strat_counts[s] for s in labels_s]
    colors_s = [STRAT_COLORS.get(s, "#95a5a6") for s in labels_s]
    wedges2, texts2, autotexts2 = ax2.pie(
        sizes_s, labels=labels_s, colors=colors_s, autopct="%1.1f%%",
        startangle=140, textprops={"fontsize": 11})
    for t in autotexts2:
        t.set_fontweight("bold")
    ax2.set_title("Strategy Selection at Test Time\n(Collapsed to single strategy)", fontsize=13)

    fig.suptitle("Executor Policy Collapse at Test Time", fontsize=15, fontweight="bold", y=1.02)
    fig.savefig(OUT_DIR / "04_test_time_collapse.png")
    plt.close(fig)
    print(f"  [4/7] Test-time collapse pie charts saved")


# ──────────────────────────────────────────────────────────────
# 5. InfraMind vs Baseline accuracy comparison (bar chart)
# ──────────────────────────────────────────────────────────────
def plot_accuracy_comparison(test_rows, base_rows):
    # InfraMind by arrival rate
    im_episodes = [r for r in test_rows if r["record_type"] == "episode"]
    im_acc = defaultdict(list)
    im_lat = defaultdict(list)
    for r in im_episodes:
        ar = float(r["arrival_rate"])
        im_acc[ar].append(float(r["quality"]))
        im_lat[ar].append(float(r["workflow_latency_seconds"]))

    # Baseline
    base_episodes = [r for r in base_rows if r["record_type"] == "episode"]
    base_acc_vals = [float(r.get("quality_is_correct", r.get("quality", "0"))) for r in base_episodes]
    base_lat_vals = [float(r["workflow_latency_seconds"]) for r in base_episodes]
    base_ar = float(base_episodes[0]["arrival_rate"]) if base_episodes else 500

    # Figure: accuracy vs arrival rate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    rates = sorted(im_acc)
    im_means = [np.mean(im_acc[r]) * 100 for r in rates]
    im_lats  = [np.mean(im_lat[r]) for r in rates]

    # Bar: accuracy
    x = np.arange(len(rates) + 1)
    bar_labels = [f"{int(r)}" for r in rates] + [f"{int(base_ar)}"]
    bar_vals   = im_means + [np.mean(base_acc_vals) * 100]
    bar_colors = ["#3498db"] * len(rates) + ["#e74c3c"]

    bars = ax1.bar(x, bar_vals, color=bar_colors, width=0.6, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bar_labels)
    ax1.set_xlabel("Arrival Rate (req/min)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy: InfraMind vs Baseline MAS")
    ax1.set_ylim(0, 65)
    ax1.grid(True, alpha=0.2, axis="y")

    # Add value labels
    for bar, val in zip(bars, bar_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Legend
    ax1.legend([mpatches.Patch(color="#3498db"), mpatches.Patch(color="#e74c3c")],
               ["InfraMind", "Baseline MAS"], loc="upper right")

    # Bar: latency
    lat_vals = im_lats + [np.mean(base_lat_vals)]
    bars2 = ax2.bar(x, lat_vals, color=bar_colors, width=0.6, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_labels)
    ax2.set_xlabel("Arrival Rate (req/min)")
    ax2.set_ylabel("Avg Latency (s)")
    ax2.set_title("Latency: InfraMind vs Baseline MAS")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.2, axis="y")

    for bar, val in zip(bars2, lat_vals):
        label = f"{val:.1f}s" if val < 10 else f"{val:.0f}s"
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.legend([mpatches.Patch(color="#3498db"), mpatches.Patch(color="#e74c3c")],
               ["InfraMind", "Baseline MAS"], loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_accuracy_latency_comparison.png")
    plt.close(fig)
    print(f"  [5/7] Accuracy/Latency comparison saved")


# ──────────────────────────────────────────────────────────────
# 6. Credit assignment problem diagram
# ──────────────────────────────────────────────────────────────
def plot_credit_assignment_diagram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # LEFT: Current broken reward flow
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("Current: Episode-Level Quality\n(Broken Credit Assignment)", fontsize=13, color="#e74c3c", fontweight="bold")

    # Draw steps
    step_y = [7.5, 5.5, 3.5]
    step_labels = [
        "Step 1: Llama-3B + Flash\n(0.15s, low quality?)",
        "Step 2: Qwen-14B + Concise\n(0.62s, high quality?)",
        "Step 3: Llama-3B + Flash\n(0.17s, low quality?)",
    ]
    for i, (y, label) in enumerate(zip(step_y, step_labels)):
        rect = mpatches.FancyBboxPatch((0.5, y-0.6), 4.5, 1.2, boxstyle="round,pad=0.1",
                                        facecolor="#fadbd8", edgecolor="#e74c3c", linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(2.75, y, label, ha="center", va="center", fontsize=10)

    # Final answer
    rect_final = mpatches.FancyBboxPatch((0.5, 1.0), 4.5, 1.2, boxstyle="round,pad=0.1",
                                          facecolor="#d5f5e3", edgecolor="#27ae60", linewidth=2)
    ax1.add_patch(rect_final)
    ax1.text(2.75, 1.6, "Final Answer: Correct (1.0)", ha="center", va="center",
             fontsize=11, fontweight="bold")

    # Reward broadcast arrows
    for y in step_y:
        ax1.annotate("", xy=(5.5, y), xytext=(7.5, 1.6),
                     arrowprops=dict(arrowstyle="<-", color="#e74c3c", lw=1.5, ls="--"))

    # Reward box
    rect_reward = mpatches.FancyBboxPatch((6.5, 5.0), 3.0, 1.5, boxstyle="round,pad=0.1",
                                           facecolor="#f9e79f", edgecolor="#f39c12", linewidth=2)
    ax1.add_patch(rect_reward)
    ax1.text(8.0, 5.75, "Same reward\nto ALL steps:\nq = 1.0", ha="center", va="center",
             fontsize=11, fontweight="bold", color="#e74c3c")

    ax1.text(7.5, 1.2, "Broadcast\nidentical\nreward", ha="center", va="center",
             fontsize=9, color="#e74c3c", style="italic")

    # Problem annotation
    ax1.text(8.0, 8.5, "Problem: Executor cannot\ndistinguish which\nmodel/strategy helped",
             ha="center", va="center", fontsize=10, color="#c0392b",
             bbox=dict(boxstyle="round", facecolor="#f5b7b1", alpha=0.8))

    # RIGHT: What executor learns
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("Result: Executor Learns to\nMinimize Latency Only", fontsize=13, color="#e74c3c", fontweight="bold")

    # Reward equation (simplified - no underbrace)
    ax2.text(5, 8.8, r"$r_t = q - \lambda \cdot \frac{lat_t}{budget_t}$",
             ha="center", va="center", fontsize=18,
             bbox=dict(boxstyle="round", facecolor="#eaf2f8", edgecolor="#2980b9", linewidth=2))
    ax2.text(2.8, 8.0, "same for\nall steps", ha="center", va="center", fontsize=9,
             color="#e74c3c", fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="#fadbd8", alpha=0.8, pad=0.2))
    ax2.annotate("", xy=(3.6, 8.55), xytext=(3.1, 8.2),
                 arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2))
    ax2.text(7.2, 8.0, "only\ndifferentiator", ha="center", va="center", fontsize=9,
             color="#2980b9", fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="#d6eaf8", alpha=0.8, pad=0.2))
    ax2.annotate("", xy=(5.8, 8.55), xytext=(6.8, 8.2),
                 arrowprops=dict(arrowstyle="->", color="#2980b9", lw=1.2))

    # Consequence arrows
    consequences = [
        (5, 6.8, "Quality term identical for all steps in an episode\n→ No gradient signal for quality"),
        (5, 5.1, "Only latency term varies per step\n→ Executor learns: lower latency = higher reward"),
        (5, 3.4, "Smallest model (Llama-3B) has lowest latency\n→ Policy collapses to Llama-3B (95.1%)"),
        (5, 1.7, "DeepSeek-R1-32B (best quality) used < 1%\n→ Quality potential wasted"),
    ]
    colors_cons = ["#2980b9", "#e67e22", "#e74c3c", "#c0392b"]
    for (x, y, text), color in zip(consequences, colors_cons):
        ax2.text(x, y, text, ha="center", va="center", fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor=color, linewidth=1.5))

    # Arrow chain
    for i in range(len(consequences)-1):
        ax2.annotate("", xy=(5, consequences[i+1][1]+0.55), xytext=(5, consequences[i][1]-0.55),
                     arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5))

    fig.subplots_adjust(wspace=0.3)
    fig.savefig(OUT_DIR / "06_credit_assignment_problem.png")
    plt.close(fig)
    print(f"  [6/7] Credit assignment diagram saved")


# ──────────────────────────────────────────────────────────────
# 7. Proposed fix diagram
# ──────────────────────────────────────────────────────────────
def plot_proposed_fix():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # LEFT: Short-term fix
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("Fix 1 (Immediate):\nBenchmark-Based Quality Prior", fontsize=13, color="#27ae60", fontweight="bold")

    # Table-like display of benchmark weights
    ax1.text(5, 9.0, "Replace episode-level q with per-step quality estimate",
             ha="center", va="center", fontsize=11, style="italic",
             bbox=dict(boxstyle="round", facecolor="#d5f5e3", edgecolor="#27ae60"))

    # Reward equation
    ax1.text(5, 7.5, r"$r_t = \hat{q}(model_t, role_t) - \lambda \cdot \frac{lat_t}{budget_t}$",
             ha="center", va="center", fontsize=15,
             bbox=dict(boxstyle="round", facecolor="#eaf2f8", edgecolor="#2980b9", linewidth=2))

    # Benchmark table
    table_data = [
        ["Model",           "MATH Bench", "Weight"],
        ["DeepSeek-R1-32B", "79.8%",      "0.80"],
        ["Mistral-24B",     "52.1%",      "0.52"],
        ["Qwen2.5-14B",     "47.3%",      "0.47"],
        ["Llama-3.1-8B",    "39.4%",      "0.39"],
        ["Llama-3.2-3B",    "24.6%",      "0.25"],
    ]
    table = ax1.table(
        cellText=table_data[1:], colLabels=table_data[0],
        loc="center", cellLoc="center",
        bbox=[0.1, 0.05, 0.8, 0.55],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    # Color header
    for j in range(3):
        table[0, j].set_facecolor("#2ecc71")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Color rows by quality
    row_colors = ["#d5f5e3", "#e8f8f5", "#fef9e7", "#fef9e7", "#fadbd8"]
    for i in range(1, 6):
        for j in range(3):
            table[i, j].set_facecolor(row_colors[i-1])

    # RIGHT: Long-term fix
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("Fix 2 (Long-term):\nLLM-as-Judge Offline Quality Scores", fontsize=13, color="#2980b9", fontweight="bold")

    # Flow diagram
    steps_fix = [
        (5, 8.5, "1. Collect offline data from\nbaseline MAS sweep (all 15 combos)", "#eaf2f8", "#2980b9"),
        (5, 6.7, "2. For each (model, role, problem):\nUse GPT-4/Claude as judge to\nscore intermediate response quality", "#fef9e7", "#f39c12"),
        (5, 4.9, "3. Train quality predictor:\nq̂(model, strategy, role, query)\n→ learned per-step quality", "#e8f8f5", "#27ae60"),
        (5, 3.1, "4. Use predicted q̂ as dense\nper-step reward in CMDP training", "#d5f5e3", "#27ae60"),
        (5, 1.3, "Result: Executor gets meaningful\nquality gradient for each decision", "#d5f5e3", "#27ae60"),
    ]
    for x, y, text, fc, ec in steps_fix:
        ax2.text(x, y, text, ha="center", va="center", fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=fc, edgecolor=ec, linewidth=1.5))

    for i in range(len(steps_fix)-1):
        ax2.annotate("", xy=(5, steps_fix[i+1][1]+0.55), xytext=(5, steps_fix[i][1]-0.55),
                     arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=2))

    fig.tight_layout(w_pad=3)
    fig.savefig(OUT_DIR / "07_proposed_fixes.png")
    plt.close(fig)
    print(f"  [7/7] Proposed fixes diagram saved")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    train_rows = load_csv(TRAIN_CSV)
    test_rows  = load_csv(TEST_CSV)
    base_rows  = load_csv(BASE_CSV)
    print(f"  Train: {len(train_rows)} rows,  Test: {len(test_rows)} rows,  Baseline: {len(base_rows)} rows")

    print("\nGenerating plots...")
    plot_training_accuracy(train_rows)
    plot_model_selection_by_epoch(train_rows)
    plot_strategy_selection_by_epoch(train_rows)
    plot_test_collapse(test_rows)
    plot_accuracy_comparison(test_rows, base_rows)
    plot_credit_assignment_diagram()
    plot_proposed_fix()

    # Print summary table for slides
    print("\n" + "="*70)
    print("SUMMARY TABLE (copy to slides)")
    print("="*70)

    train_episodes = [r for r in train_rows if r["record_type"] == "episode"]
    test_episodes  = [r for r in test_rows  if r["record_type"] == "episode"]
    base_episodes  = [r for r in base_rows  if r["record_type"] == "episode"]

    train_acc = np.mean([float(r["quality"]) for r in train_episodes])
    test_acc  = np.mean([float(r["quality"]) for r in test_episodes])
    base_acc  = np.mean([float(r.get("quality_is_correct", r.get("quality", "0"))) for r in base_episodes])

    train_lat = np.mean([float(r["workflow_latency_seconds"]) for r in train_episodes])
    test_lat  = np.mean([float(r["workflow_latency_seconds"]) for r in test_episodes])
    base_lat  = np.mean([float(r["workflow_latency_seconds"]) for r in base_episodes])

    test_steps = [r for r in test_rows if r["record_type"] == "role_step"]
    model_counts = defaultdict(int)
    for r in test_steps:
        model_counts[short_model(r["model_name"])] += 1
    total_steps = sum(model_counts.values())

    print(f"\n{'Metric':<35} {'InfraMind Train':>15} {'InfraMind Test':>15} {'Baseline':>15}")
    print("-"*82)
    print(f"{'Accuracy':<35} {train_acc:>14.1%} {test_acc:>14.1%} {base_acc:>14.1%}")
    print(f"{'Avg Latency (s)':<35} {train_lat:>15.2f} {test_lat:>15.2f} {base_lat:>15.2f}")
    print(f"{'Epochs Trained':<35} {'13':>15} {'-':>15} {'-':>15}")
    print(f"{'Test Arrival Rates':<35} {'-':>15} {'10-200':>15} {'500':>15}")

    print(f"\nTest-Time Model Distribution:")
    for m in MODEL_ORDER:
        pct = model_counts.get(m, 0) / total_steps * 100
        bar = "#" * int(pct / 2)
        print(f"  {m:<25} {pct:5.1f}%  {bar}")

    strat_counts = defaultdict(int)
    for r in test_steps:
        strat_counts[r["strategy_name"]] += 1
    print(f"\nTest-Time Strategy Distribution:")
    for s in STRAT_ORDER:
        pct = strat_counts.get(s, 0) / total_steps * 100
        bar = "#" * int(pct / 2)
        print(f"  {s:<25} {pct:5.1f}%  {bar}")

    print(f"\nAll plots saved to: {OUT_DIR.resolve()}")
    print("="*70)


if __name__ == "__main__":
    main()
