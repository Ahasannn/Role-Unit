#!/bin/bash
# inframind_test_sweep_math.sh
# Inference-only sweep for InfraMind (CMDP Router) on MATH test set.
# Runs a cross-product of arrival rates × budgets with --skip-training
# and --deterministic so weights are never updated.
#
# Environment variables (set by SLURM wrapper or manually):
#   TEST_LIMIT   — number of test items (default: 1000)
#   CONCURRENCY  — max concurrent requests (default: 1000)
#   BUDGET_SWEEP — comma-separated budgets in seconds (default: 30,60,120)
#
# Expects vLLM servers already running.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

# ===== Load shared dataset config =====
DATASET_CONFIG="${REPO_ROOT}/Experiments/dataset_config.json"
DATASET_NAME="math"

ARRIVAL_PATTERN=$(python3 -c "import json; print(json.load(open('${DATASET_CONFIG}'))['${DATASET_NAME}']['arrival_pattern'])")
ARRIVAL_RATES_CSV=$(python3 -c "import json; print(','.join(str(r) for r in json.load(open('${DATASET_CONFIG}'))['${DATASET_NAME}']['arrival_rates']))")

# ===== Configurable via env =====
TEST_LIMIT="${TEST_LIMIT:-1000}"
CONCURRENCY="${CONCURRENCY:-1000}"
BUDGET_SWEEP="${BUDGET_SWEEP:-30,60,120}"

# ===== Paths =====
BLUE_STORAGE="${BLUE_STORAGE:-/blue/qi855292.ucf/ah872032.ucf}"

CHECKPOINT="${BLUE_STORAGE}/checkpoints/inframind/inframind_math.pt"
LATENCY_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/latency_estimator.pth"
LENGTH_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/length_estimator.pth"

MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"

# ===== Output =====
OUTPUT_DIR="logs/paper_results/math"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_CSV="${OUTPUT_DIR}/inframind_math_test_${TEST_LIMIT}_${ARRIVAL_PATTERN}.csv"

# ===== Validation =====
for f in "$CHECKPOINT" "$LATENCY_PREDICTOR" "$LENGTH_PREDICTOR"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done
if [[ ! -d "${MATH_DATASET_ROOT}/test" ]]; then
    echo "ERROR: MATH test dataset not found at ${MATH_DATASET_ROOT}/test"
    exit 1
fi

# ===== Environment =====
export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

echo "========================================="
echo "InfraMind — MATH Test Sweep"
echo "========================================="
echo "Checkpoint:       ${CHECKPOINT}"
echo "Latency pred:     ${LATENCY_PREDICTOR}"
echo "Length pred:       ${LENGTH_PREDICTOR}"
echo "Test limit:       ${TEST_LIMIT}"
echo "Concurrency:      ${CONCURRENCY}"
echo "Arrival rates:    ${ARRIVAL_RATES_CSV}"
echo "Arrival pattern:  ${ARRIVAL_PATTERN}"
echo "Budget sweep:     ${BUDGET_SWEEP}"
echo "Output CSV:       ${OUTPUT_CSV}"
echo "========================================="
echo ""

# ===== Run sweep =====
# --skip-training: no weight updates, no checkpoint saves
# --deterministic: argmax actions (reproducible)
# --split test: use MATH test split
# --epochs 1: single pass over data per sweep config

python -m MAR.InfraMind.training \
    --dataset math \
    --dataset-root "${MATH_DATASET_ROOT}" \
    --split test \
    --limit "${TEST_LIMIT}" \
    --epochs 1 \
    --arrival-rates "${ARRIVAL_RATES_CSV}" \
    --arrival-pattern "${ARRIVAL_PATTERN}" \
    --budget-sweep "${BUDGET_SWEEP}" \
    --concurrency "${CONCURRENCY}" \
    --latency-predictor "${LATENCY_PREDICTOR}" \
    --length-predictor "${LENGTH_PREDICTOR}" \
    --checkpoint-path "${CHECKPOINT}" \
    --deterministic \
    --skip-training \
    --telemetry-csv "${OUTPUT_CSV}"

echo ""
echo "========================================="
echo "InfraMind sweep complete: ${OUTPUT_CSV}"
echo "========================================="
