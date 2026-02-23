#!/usr/bin/env python3
"""
Generate publication-ready comparison figures for InfraMind vs Baseline MAS Router.

Produces:
1. Queue Balance: How InfraMind distributes load better across models
2. Quality-Latency Pareto: InfraMind achieves better quality-latency tradeoffs
3. SLO Adherence: InfraMind meets budget constraints better under load
4. Budget Sensitivity: InfraMind quality vs budget at fixed arrival rate
"""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME_MAP = {
    'Qwen/Qwen2.5-Coder-14B-Instruct': 'Qwen-14B',
    'mistralai/Mistral-Small-24B-Instruct-2501': 'Mistral-24B',
    'meta-llama/Llama-3.2-3B-Instruct': 'Llama-3B',
    'meta-llama/Llama-3.1-8B-Instruct': 'Llama-8B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B': 'DeepSeek-32B',
}

SIZE_GROUPS = {
    'Small (3-8B)': {'Llama-3B', 'Llama-8B'},
    'Large (14-32B)': {'Qwen-14B', 'Mistral-24B', 'DeepSeek-32B'},
}

COLORS = {
    'baseline': '#D6604D',      # Red/coral
    'inframind': '#4393C3',     # Blue
    'Small (3-8B)': '#2166AC',
    'Large (14-32B)': '#D6604D',
}


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def load_csv(path: Path, system_name: str) -> pd.DataFrame:
    """Load and preprocess CSV with system label."""
    df = pd.read_csv(path)
    df['system'] = system_name

    # Map model names
    if 'llm_name' in df.columns:
        df['model_short'] = df['llm_name'].map(MODEL_NAME_MAP)
    elif 'model_name' in df.columns:
        df['model_short'] = df['model_name'].map(MODEL_NAME_MAP)

    return df


def assign_size_group(model_short: str) -> str:
    """Assign model to size group."""
    for group, models in SIZE_GROUPS.items():
        if model_short in models:
            return group
    return None


def plot_queue_balance(baseline_df: pd.DataFrame, inframind_df: pd.DataFrame,
                       output_path: Path):
    """Figure 1: Queue balance comparison across arrival rates."""
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Process both systems
    for df, system, color in [(baseline_df, 'Baseline', COLORS['baseline']),
                               (inframind_df, 'InfraMind', COLORS['inframind'])]:

        # Use step/role_step records
        step_df = df[df['record_type'].isin(['step', 'role_step'])].copy()

        # Convert to numeric
        for col in ['llm_running', 'llm_waiting', 'arrival_rate']:
            if col in step_df.columns:
                step_df[col] = pd.to_numeric(step_df[col], errors='coerce')

        # Compute queue depth
        step_df['queue_depth'] = step_df['llm_running'].fillna(0) + step_df['llm_waiting'].fillna(0)

        # Assign size groups
        step_df = step_df[step_df['model_short'].notna()].copy()
        step_df['size_group'] = step_df['model_short'].apply(assign_size_group)
        step_df = step_df[step_df['size_group'].notna()].copy()

        # Compute imbalance metric: std of queue depth across models per arrival rate
        stats = []
        for rate in sorted(step_df['arrival_rate'].unique()):
            rate_data = step_df[step_df['arrival_rate'] == rate]

            # Queue depth std across models (higher = more imbalanced)
            model_queues = rate_data.groupby('model_short')['queue_depth'].mean()
            queue_std = model_queues.std()
            queue_mean = model_queues.mean()

            # Coefficient of variation
            cv = queue_std / queue_mean if queue_mean > 0 else 0

            stats.append({
                'arrival_rate': rate,
                'queue_std': queue_std,
                'queue_cv': cv,
                'system': system,
            })

        stats_df = pd.DataFrame(stats)

        # Plot
        ax.plot(stats_df['arrival_rate'], stats_df['queue_cv'],
                marker='o', linewidth=2.5, markersize=9,
                label=system, color=color, markeredgecolor='white',
                markeredgewidth=1.5)

    ax.set_xlabel('Request Arrival Rate (req/min)', fontweight='medium')
    ax.set_ylabel('Queue Imbalance (CV)', fontweight='medium')
    ax.set_title('Queue Load Balance Across Models', fontweight='bold', pad=15)
    ax.legend(loc='upper left', frameon=True, fancybox=False,
              edgecolor='#cccccc', facecolor='white')
    ax.set_ylim(bottom=0)

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


def plot_quality_latency_pareto(baseline_df: pd.DataFrame, inframind_df: pd.DataFrame,
                                 output_path: Path):
    """Figure 2: Quality-Latency Pareto frontier."""
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Process episode-level data
    for df, system, color, marker in [(baseline_df, 'Baseline', COLORS['baseline'], 'o'),
                                       (inframind_df, 'InfraMind', COLORS['inframind'], 's')]:

        episode_df = df[df['record_type'] == 'episode'].copy()

        for col in ['workflow_latency_seconds', 'quality_is_correct', 'quality_is_solved', 'arrival_rate']:
            if col in episode_df.columns:
                episode_df[col] = pd.to_numeric(episode_df[col], errors='coerce')

        # Use quality_is_correct (baseline) or quality_is_solved (inframind)
        if 'quality_is_solved' in episode_df.columns:
            quality_col = 'quality_is_solved'
        else:
            quality_col = 'quality_is_correct'

        # Compute mean quality and latency per arrival rate
        stats = episode_df.groupby('arrival_rate').agg({
            quality_col: 'mean',
            'workflow_latency_seconds': 'mean',
        }).reset_index()

        stats.columns = ['arrival_rate', 'quality', 'latency']

        # Plot with labels
        for _, row in stats.iterrows():
            ax.scatter(row['latency'], row['quality'],
                      s=150, marker=marker, color=color, alpha=0.7,
                      edgecolor='white', linewidth=1.5, zorder=5)
            ax.annotate(f"{int(row['arrival_rate'])}",
                       xy=(row['latency'], row['quality']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color=color, fontweight='bold')

        # Connect points with line
        stats = stats.sort_values('latency')
        ax.plot(stats['latency'], stats['quality'],
               linewidth=2, color=color, alpha=0.4, linestyle='--',
               label=f'{system} (arrival rates)')

    ax.set_xlabel('Average Latency (seconds)', fontweight='medium')
    ax.set_ylabel('Correctness Rate', fontweight='medium')
    ax.set_title('Quality-Latency Pareto Frontier', fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, fancybox=False,
              edgecolor='#cccccc', facecolor='white')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


def plot_slo_adherence(baseline_df: pd.DataFrame, inframind_df: pd.DataFrame,
                       output_path: Path):
    """Figure 3: SLO adherence (budget violations) across arrival rates."""
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Baseline: no explicit budget, use median latency at low load as implicit SLO
    baseline_episodes = baseline_df[baseline_df['record_type'] == 'episode'].copy()
    baseline_episodes['workflow_latency_seconds'] = pd.to_numeric(
        baseline_episodes['workflow_latency_seconds'], errors='coerce')
    baseline_episodes['arrival_rate'] = pd.to_numeric(
        baseline_episodes['arrival_rate'], errors='coerce')

    # Baseline SLO = median latency at lowest arrival rate
    low_rate = baseline_episodes['arrival_rate'].min()
    baseline_slo = baseline_episodes[baseline_episodes['arrival_rate'] == low_rate]['workflow_latency_seconds'].median()

    baseline_stats = []
    for rate in sorted(baseline_episodes['arrival_rate'].unique()):
        rate_data = baseline_episodes[baseline_episodes['arrival_rate'] == rate]
        violation_rate = (rate_data['workflow_latency_seconds'] > baseline_slo * 1.5).mean() * 100
        baseline_stats.append({'arrival_rate': rate, 'violation_pct': violation_rate})

    baseline_stats_df = pd.DataFrame(baseline_stats)

    # InfraMind: explicit budget_total, check violations
    inframind_episodes = inframind_df[inframind_df['record_type'] == 'episode'].copy()
    for col in ['workflow_latency_seconds', 'budget_total', 'arrival_rate']:
        if col in inframind_episodes.columns:
            inframind_episodes[col] = pd.to_numeric(inframind_episodes[col], errors='coerce')

    inframind_stats = []
    for rate in sorted(inframind_episodes['arrival_rate'].unique()):
        rate_data = inframind_episodes[inframind_episodes['arrival_rate'] == rate]
        # SLO violation = latency > budget_total
        violations = (rate_data['workflow_latency_seconds'] > rate_data['budget_total']).mean() * 100
        inframind_stats.append({'arrival_rate': rate, 'violation_pct': violations})

    inframind_stats_df = pd.DataFrame(inframind_stats)

    # Plot
    ax.plot(baseline_stats_df['arrival_rate'], baseline_stats_df['violation_pct'],
           marker='o', linewidth=2.5, markersize=9,
           label='Baseline (implicit SLO)', color=COLORS['baseline'],
           markeredgecolor='white', markeredgewidth=1.5)

    ax.plot(inframind_stats_df['arrival_rate'], inframind_stats_df['violation_pct'],
           marker='s', linewidth=2.5, markersize=9,
           label='InfraMind (budget-aware)', color=COLORS['inframind'],
           markeredgecolor='white', markeredgewidth=1.5)

    ax.set_xlabel('Request Arrival Rate (req/min)', fontweight='medium')
    ax.set_ylabel('SLO Violation Rate (%)', fontweight='medium')
    ax.set_title('SLO Adherence Under Load', fontweight='bold', pad=15)
    ax.legend(loc='upper left', frameon=True, fancybox=False,
              edgecolor='#cccccc', facecolor='white')
    ax.set_ylim(bottom=0)
    ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='10% threshold')

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


def plot_budget_sensitivity(inframind_df: pd.DataFrame, output_path: Path):
    """Figure 4: InfraMind quality vs budget at fixed arrival rate."""
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    episode_df = inframind_df[inframind_df['record_type'] == 'episode'].copy()

    for col in ['budget_total', 'quality_is_solved', 'workflow_latency_seconds', 'arrival_rate']:
        if col in episode_df.columns:
            episode_df[col] = pd.to_numeric(episode_df[col], errors='coerce')

    # Pick a middle arrival rate for analysis
    rates = sorted(episode_df['arrival_rate'].unique())
    if len(rates) >= 2:
        target_rate = rates[1]  # Second rate (e.g., 50 or 100)
    else:
        target_rate = rates[0]

    rate_data = episode_df[episode_df['arrival_rate'] == target_rate]

    # Group by budget
    stats = rate_data.groupby('budget_total').agg({
        'quality_is_solved': 'mean',
        'workflow_latency_seconds': 'mean',
    }).reset_index()

    stats.columns = ['budget', 'quality', 'latency']
    stats = stats.sort_values('budget')

    # Plot quality vs budget
    ax.plot(stats['budget'], stats['quality'],
           marker='o', linewidth=2.5, markersize=9, color=COLORS['inframind'],
           markeredgecolor='white', markeredgewidth=1.5)

    ax.set_xlabel('Budget (seconds)', fontweight='medium')
    ax.set_ylabel('Correctness Rate', fontweight='medium')
    ax.set_title(f'Quality vs Budget at {int(target_rate)} req/min', fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate comparison figures for InfraMind vs Baseline MAS Router.'
    )
    parser.add_argument(
        '--baseline-csv', required=True,
        help='Path to baseline CSV (e.g., logs/paper_results/math/baseline_math_test_10_poisson.csv)'
    )
    parser.add_argument(
        '--inframind-csv', required=True,
        help='Path to InfraMind CSV (e.g., logs/paper_results/math/inframind_math_test_10_poisson.csv)'
    )
    parser.add_argument(
        '--output-dir', default='visualization/output_plots/paper_results',
        help='Output directory for generated figures'
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline_csv)
    inframind_path = Path(args.inframind_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading baseline: {baseline_path}')
    baseline_df = load_csv(baseline_path, 'Baseline')
    print(f'  → {len(baseline_df)} rows')

    print(f'Loading InfraMind: {inframind_path}')
    inframind_df = load_csv(inframind_path, 'InfraMind')
    print(f'  → {len(inframind_df)} rows')

    print('\nGenerating figures...')

    # Figure 1: Queue Balance
    plot_queue_balance(baseline_df, inframind_df, output_dir / 'queue_balance.png')

    # Figure 2: Quality-Latency Pareto
    plot_quality_latency_pareto(baseline_df, inframind_df, output_dir / 'quality_latency_pareto.png')

    # Figure 3: SLO Adherence
    plot_slo_adherence(baseline_df, inframind_df, output_dir / 'slo_adherence.png')

    # Figure 4: Budget Sensitivity (InfraMind only)
    plot_budget_sensitivity(inframind_df, output_dir / 'budget_sensitivity.png')

    print('\n' + '='*70)
    print(f'All figures saved to: {output_dir}')
    print('='*70)


if __name__ == '__main__':
    main()
