#!/usr/bin/env python3
"""
Assignment Strategy Analysis — ILP Optimizer + Baselines

Reads the validation score matrix (fitness + cost), formulates optimal
model→role assignment as an Integer Linear Program (Multiple-Choice
Knapsack Problem), and compares against baseline strategies.

ILP Formulation:
    maximize    Σ_r  w_r × Σ_m  x[m,r] × f[m,r]
    subject to:
        Σ_m x[m,r] = 1              ∀r    (one model per role)
        Σ_r Σ_m x[m,r] × c[m,r] ≤ B      (total cost ≤ budget)
        x[m,r] ∈ {0, 1}

    where:
        f[m,r] = validation fitness (accuracy) of model m on role r
        c[m,r] = empirical cost of model m on role r (from validation)
        w_r    = role weight (proportion of questions for role r)
        B      = cost budget (swept to produce Pareto frontier)

Strategies compared:
    1. ILP-optimal (Pareto frontier across budget levels)
    2. Fitness-guided (best val accuracy per role — ignores cost)
    3. Homogeneous (same model all roles)
    4. Random (expected value over all M^R combinations)
    5. Cheapest (lowest cost model per role)

Usage:
    python unit_tests/analyze.py \
        --val-scores unit_tests/results/job_XXX/score_matrix_validation.csv \
        --output unit_tests/results/job_XXX/analysis.csv

    # With test scores for final evaluation:
    python unit_tests/analyze.py \
        --val-scores unit_tests/results/job_XXX/score_matrix_validation.csv \
        --test-scores results/baseline/job_XXX/score_matrix_test.csv \
        --output unit_tests/results/analysis.csv
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pulp

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_score_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_lookup(df: pd.DataFrame, col: str = "accuracy") -> Dict[Tuple[str, str], float]:
    """Build (model, role) -> value lookup from score_matrix."""
    return {(row["model"], row["role"]): row[col] for _, row in df.iterrows()}


# ---------------------------------------------------------------------------
# ILP Optimizer
# ---------------------------------------------------------------------------

def solve_ilp(
    models: List[str],
    roles: List[str],
    fitness: Dict[Tuple[str, str], float],
    cost: Dict[Tuple[str, str], float],
    role_weights: Dict[str, float],
    budget: float,
) -> Optional[Dict[str, str]]:
    """Solve the budget-constrained assignment ILP.

    maximize    Σ_r  w_r × Σ_m  x[m,r] × f[m,r]
    subject to:
        Σ_m x[m,r] = 1              ∀r
        Σ_r Σ_m x[m,r] × c[m,r] ≤ B
        x[m,r] ∈ {0, 1}

    Returns:
        assignment dict {role: model} or None if infeasible.
    """
    prob = pulp.LpProblem("RoleAssignment", pulp.LpMaximize)

    # Decision variables: x[m,r] ∈ {0,1}
    x = {}
    for m in models:
        for r in roles:
            x[m, r] = pulp.LpVariable(f"x_{m}_{r}", cat=pulp.LpBinary)

    # Objective: maximize weighted accuracy
    total_w = sum(role_weights.values())
    prob += pulp.lpSum(
        (role_weights[r] / total_w) * fitness.get((m, r), 0.0) * x[m, r]
        for m in models for r in roles
    )

    # Constraint: exactly one model per role
    for r in roles:
        prob += pulp.lpSum(x[m, r] for m in models) == 1, f"one_model_{r}"

    # Constraint: total cost ≤ budget
    prob += pulp.lpSum(
        cost.get((m, r), 0.0) * x[m, r] for m in models for r in roles
    ) <= budget, "budget"

    # Solve silently
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != pulp.constants.LpStatusOptimal:
        return None

    # Extract assignment
    assignment = {}
    for r in roles:
        for m in models:
            if pulp.value(x[m, r]) > 0.5:
                assignment[r] = m
                break
    return assignment


def compute_pareto_frontier(
    models: List[str],
    roles: List[str],
    fitness: Dict[Tuple[str, str], float],
    cost: Dict[Tuple[str, str], float],
    role_weights: Dict[str, float],
    n_points: int = 20,
) -> List[Dict]:
    """Sweep budget from min to max cost, solve ILP at each level.

    Returns list of Pareto-optimal points: [{budget, accuracy, cost, assignment}]
    """
    # Find cost range
    min_cost_per_role = {r: min(cost.get((m, r), 0.0) for m in models) for r in roles}
    max_cost_per_role = {r: max(cost.get((m, r), 0.0) for m in models) for r in roles}
    total_min = sum(min_cost_per_role.values())
    total_max = sum(max_cost_per_role.values())

    # Sweep budget levels
    budgets = [total_min + (total_max - total_min) * i / (n_points - 1) for i in range(n_points)]

    frontier = []
    seen_assignments = set()

    for budget in budgets:
        assignment = solve_ilp(models, roles, fitness, cost, role_weights, budget)
        if assignment is None:
            continue

        # De-duplicate identical assignments
        key = tuple(sorted(assignment.items()))
        if key in seen_assignments:
            continue
        seen_assignments.add(key)

        total_w = sum(role_weights.values())
        acc = sum(
            (role_weights[r] / total_w) * fitness.get((assignment[r], r), 0.0)
            for r in roles
        )
        actual_cost = sum(cost.get((assignment[r], r), 0.0) for r in roles)

        frontier.append({
            "budget": round(budget, 6),
            "accuracy": round(acc, 4),
            "cost": round(actual_cost, 6),
            "assignment": {r: m for r, m in assignment.items()},
        })

    # Filter to true Pareto-optimal (remove dominated points)
    pareto = []
    for point in sorted(frontier, key=lambda p: p["cost"]):
        if not pareto or point["accuracy"] > pareto[-1]["accuracy"]:
            pareto.append(point)

    return pareto


# ---------------------------------------------------------------------------
# Baseline strategies
# ---------------------------------------------------------------------------

def compute_assignment_accuracy(
    assignment: Dict[str, str],
    lookup: Dict[Tuple[str, str], float],
    role_weights: Dict[str, float],
) -> float:
    total_w = sum(role_weights.values())
    weighted = sum(
        lookup.get((model, role), 0.0) * role_weights[role]
        for role, model in assignment.items()
    )
    return weighted / total_w if total_w > 0 else 0.0


def compute_assignment_cost(
    assignment: Dict[str, str],
    cost_lookup: Dict[Tuple[str, str], float],
) -> float:
    return sum(cost_lookup.get((model, role), 0.0) for role, model in assignment.items())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ILP Assignment Optimizer + Baseline Comparison")
    parser.add_argument("--val-scores", required=True, help="Validation score_matrix.csv")
    parser.add_argument("--test-scores", default=None, help="Test score_matrix.csv (optional, for final eval)")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--pareto-points", type=int, default=20,
                        help="Number of budget levels for Pareto sweep (default: 20)")
    args = parser.parse_args()

    # --- Load validation data (used for optimization) ---
    val_df = load_score_matrix(Path(args.val_scores))
    val_fitness = build_lookup(val_df, "accuracy")
    val_cost = build_lookup(val_df, "cost_usd")

    roles = sorted(val_df["role"].unique())
    models = sorted(val_df["model"].unique())

    # Role weights from validation question counts
    role_weights = {}
    for role in roles:
        totals = val_df[val_df["role"] == role]["total"].values
        role_weights[role] = int(totals[0]) if len(totals) > 0 else 0

    print(f"Models ({len(models)}):")
    for m in models:
        print(f"  {m}")
    print(f"Roles: {roles}")
    print(f"Role weights: {role_weights}")
    print(f"Total questions: {sum(role_weights.values())}")
    print()

    # --- Print validation fitness matrix ---
    print("=" * 80)
    print("VALIDATION FITNESS MATRIX")
    print("=" * 80)
    val_pivot = val_df.pivot(index="model", columns="role", values="accuracy")
    val_pivot["mean"] = val_pivot.mean(axis=1)
    print(val_pivot.to_string(float_format="%.4f"))
    print()

    # --- Print validation cost matrix ---
    print("=" * 80)
    print("VALIDATION COST MATRIX (USD)")
    print("=" * 80)
    cost_pivot = val_df.pivot(index="model", columns="role", values="cost_usd")
    cost_pivot["total"] = cost_pivot.sum(axis=1)
    print(cost_pivot.to_string(float_format="%.6f"))
    print()

    results = []

    # =====================================================================
    # STRATEGY 1: ILP-OPTIMAL (Pareto frontier)
    # =====================================================================
    print("=" * 80)
    print("ILP-OPTIMAL ASSIGNMENTS (Pareto Frontier)")
    print("=" * 80)
    print()
    print("Formulation:")
    print("  maximize    Σ_r w_r × Σ_m x[m,r] × f[m,r]")
    print("  subject to  Σ_m x[m,r] = 1  ∀r  (one model per role)")
    print("              Σ cost × x ≤ B       (budget constraint)")
    print("              x[m,r] ∈ {0,1}")
    print()

    pareto = compute_pareto_frontier(
        models, roles, val_fitness, val_cost, role_weights,
        n_points=args.pareto_points,
    )

    print(f"Pareto-optimal solutions ({len(pareto)} points):")
    print(f"{'Point':<8} {'Accuracy':>10} {'Cost($)':>12}  Assignment")
    print("-" * 80)
    for i, pt in enumerate(pareto):
        short_assign = {r: m.split("/")[-1] for r, m in pt["assignment"].items()}
        print(f"  P{i:<5} {pt['accuracy']:>10.4f} {pt['cost']:>12.6f}  {json.dumps(short_assign)}")
        results.append({
            "strategy": f"ilp-optimal-P{i}",
            "val_accuracy": pt["accuracy"],
            "val_cost_usd": pt["cost"],
            "assignment": json.dumps(short_assign),
        })

    # Highlight key operating points
    if pareto:
        cheapest_pt = pareto[0]
        best_pt = pareto[-1]
        # Best cost-efficiency: highest accuracy/cost ratio
        efficient_pt = max(pareto, key=lambda p: p["accuracy"] / p["cost"] if p["cost"] > 0 else 0)

        print()
        print("Key operating points:")
        print(f"  MAX ACCURACY:  acc={best_pt['accuracy']:.4f}  cost=${best_pt['cost']:.6f}")
        print(f"  MIN COST:      acc={cheapest_pt['accuracy']:.4f}  cost=${cheapest_pt['cost']:.6f}")
        print(f"  BEST EFF:      acc={efficient_pt['accuracy']:.4f}  cost=${efficient_pt['cost']:.6f}")
    print()

    # =====================================================================
    # STRATEGY 2: FITNESS-GUIDED (best val accuracy per role, no cost)
    # =====================================================================
    fitness_assignment = {}
    for role in roles:
        fitness_assignment[role] = max(models, key=lambda m: val_fitness.get((m, role), 0.0))

    fitness_acc = compute_assignment_accuracy(fitness_assignment, val_fitness, role_weights)
    fitness_cost = compute_assignment_cost(fitness_assignment, val_cost)

    print("=" * 80)
    print("BASELINE STRATEGIES (evaluated on validation)")
    print("=" * 80)

    print(f"\n1. FITNESS-GUIDED (best val accuracy per role, ignores cost):")
    for role, model in fitness_assignment.items():
        acc = val_fitness.get((model, role), 0)
        print(f"   {role}: {model.split('/')[-1]} (acc={acc:.4f})")
    print(f"   Weighted accuracy: {fitness_acc:.4f} | Cost: ${fitness_cost:.6f}")

    results.append({
        "strategy": "fitness-guided",
        "val_accuracy": round(fitness_acc, 4),
        "val_cost_usd": round(fitness_cost, 6),
        "assignment": json.dumps({r: m.split("/")[-1] for r, m in fitness_assignment.items()}),
    })

    # =====================================================================
    # STRATEGY 3: HOMOGENEOUS (same model all roles)
    # =====================================================================
    print(f"\n2. HOMOGENEOUS (same model all roles):")
    for model in models:
        homo_assignment = {role: model for role in roles}
        homo_acc = compute_assignment_accuracy(homo_assignment, val_fitness, role_weights)
        homo_cost = compute_assignment_cost(homo_assignment, val_cost)
        short = model.split("/")[-1]
        print(f"   {short:<40} acc={homo_acc:.4f}  cost=${homo_cost:.6f}")
        results.append({
            "strategy": f"homogeneous ({short})",
            "val_accuracy": round(homo_acc, 4),
            "val_cost_usd": round(homo_cost, 6),
            "assignment": json.dumps({r: short for r in roles}),
        })

    # =====================================================================
    # STRATEGY 4: RANDOM (expected value over all M^R combos)
    # =====================================================================
    n_combos = len(models) ** len(roles)
    all_accs = []
    all_costs = []
    for combo in itertools.product(models, repeat=len(roles)):
        assignment = dict(zip(roles, combo))
        all_accs.append(compute_assignment_accuracy(assignment, val_fitness, role_weights))
        all_costs.append(compute_assignment_cost(assignment, val_cost))

    random_mean_acc = sum(all_accs) / n_combos
    random_mean_cost = sum(all_costs) / n_combos
    random_std = (sum((a - random_mean_acc) ** 2 for a in all_accs) / n_combos) ** 0.5

    print(f"\n3. RANDOM (expected over {n_combos} combinations):")
    print(f"   Mean accuracy: {random_mean_acc:.4f} +/- {random_std:.4f}")
    print(f"   Min: {min(all_accs):.4f}  Max: {max(all_accs):.4f}")
    print(f"   Mean cost: ${random_mean_cost:.6f}")

    results.append({
        "strategy": "random (expected)",
        "val_accuracy": round(random_mean_acc, 4),
        "val_cost_usd": round(random_mean_cost, 6),
        "assignment": "random",
    })

    # =====================================================================
    # STRATEGY 5: CHEAPEST (lowest cost per role)
    # =====================================================================
    cheapest_assignment = {}
    for role in roles:
        cheapest_assignment[role] = min(models, key=lambda m: val_cost.get((m, role), float("inf")))

    cheapest_acc = compute_assignment_accuracy(cheapest_assignment, val_fitness, role_weights)
    cheapest_cost = compute_assignment_cost(cheapest_assignment, val_cost)

    print(f"\n4. CHEAPEST (lowest cost model per role):")
    for role, model in cheapest_assignment.items():
        acc = val_fitness.get((model, role), 0)
        print(f"   {role}: {model.split('/')[-1]} (acc={acc:.4f})")
    print(f"   Weighted accuracy: {cheapest_acc:.4f} | Cost: ${cheapest_cost:.6f}")

    results.append({
        "strategy": "cheapest",
        "val_accuracy": round(cheapest_acc, 4),
        "val_cost_usd": round(cheapest_cost, 6),
        "assignment": json.dumps({r: m.split("/")[-1] for r, m in cheapest_assignment.items()}),
    })

    # =====================================================================
    # SUMMARY TABLE
    # =====================================================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':<40} {'Val Acc':>10} {'Cost($)':>12} {'Acc/Cost':>12}")
    print("-" * 80)
    for r in results:
        acc = r["val_accuracy"]
        cst = r["val_cost_usd"]
        ratio = acc / cst if cst > 0 else 0
        print(f"{r['strategy']:<40} {acc:>10.4f} {cst:>12.6f} {ratio:>12.1f}")

    # =====================================================================
    # SAVE
    # =====================================================================
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.val_scores).parent / "analysis.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save main results
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # Save Pareto frontier separately
    pareto_path = out_path.parent / "pareto_frontier.csv"
    pareto_rows = []
    for i, pt in enumerate(pareto):
        pareto_rows.append({
            "point": i,
            "val_accuracy": pt["accuracy"],
            "val_cost_usd": pt["cost"],
            "budget": pt["budget"],
            "assignment": json.dumps({r: m.split("/")[-1] for r, m in pt["assignment"].items()}),
        })
    pd.DataFrame(pareto_rows).to_csv(pareto_path, index=False)
    print(f"Pareto frontier saved to: {pareto_path}")


if __name__ == "__main__":
    main()
