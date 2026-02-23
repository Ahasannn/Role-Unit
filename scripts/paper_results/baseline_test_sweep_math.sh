#!/bin/bash
# baseline_test_sweep_math.sh
# Inference-only sweep for Baseline MAS Router on MATH test set.
# Runs at each arrival rate from dataset_config.json, appending all results
# to a single CSV for later comparison with InfraMind.
#
# Environment variables (set by SLURM wrapper or manually):
#   TEST_LIMIT   — number of test items (default: 1000)
#   CONCURRENCY  — max concurrent requests (default: 1000)
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
ARRIVAL_RATES=$(python3 -c "import json; print(' '.join(str(r) for r in json.load(open('${DATASET_CONFIG}'))['${DATASET_NAME}']['arrival_rates']))")

# ===== Configurable via env =====
TEST_LIMIT="${TEST_LIMIT:-1000}"
CONCURRENCY="${CONCURRENCY:-1000}"

# ===== Paths =====
BLUE_STORAGE="${BLUE_STORAGE:-/blue/qi855292.ucf/ah872032.ucf}"
CHECKPOINT="${BLUE_STORAGE}/checkpoints/mas_router/mas_math_train_519_cost700.pth"

export MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"

# ===== Output =====
OUTPUT_DIR="logs/paper_results/math"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_CSV="${OUTPUT_DIR}/baseline_math_test_${TEST_LIMIT}_${ARRIVAL_PATTERN}.csv"

# ===== Validation =====
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: Baseline checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

echo "========================================="
echo "Baseline MAS Router — MATH Test Sweep"
echo "========================================="
echo "Checkpoint:  ${CHECKPOINT}"
echo "Test limit:  ${TEST_LIMIT}"
echo "Concurrency: ${CONCURRENCY}"
echo "Rates:       ${ARRIVAL_RATES}"
echo "Pattern:     ${ARRIVAL_PATTERN}"
echo "Output CSV:  ${OUTPUT_CSV}"
echo "========================================="
echo ""

# ===== Run sweep =====
for arrival_rate in $ARRIVAL_RATES; do
    echo "--- rate=${arrival_rate} pattern=${ARRIVAL_PATTERN} concurrency=${CONCURRENCY} ---"

    python Experiments/run_math.py \
        --epochs 0 \
        --checkpoint "${CHECKPOINT}" \
        --test_limit "${TEST_LIMIT}" \
        --concurrency "${CONCURRENCY}" \
        --arrival-rate "${arrival_rate}" \
        --arrival-pattern "${ARRIVAL_PATTERN}" \
        --test-telemetry-csv "${OUTPUT_CSV}"

    echo "Completed rate=${arrival_rate}"
    echo ""
done

echo "========================================="
echo "Baseline sweep complete: ${OUTPUT_CSV}"
echo "========================================="
