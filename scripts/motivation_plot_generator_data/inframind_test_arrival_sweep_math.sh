#!/bin/bash
# InfraMind Test Arrival Rate Sweep for MATH
# Mirrors the baseline motivation sweep but uses the InfraMind (CMDP) router.
# Generates data for side-by-side comparison with baseline motivation figure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

# Blue storage configuration
BLUE_STORAGE="${BLUE_STORAGE:-/blue/qi855292.ucf/ah872032.ucf}"

# Dataset
MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"

# InfraMind checkpoint (latest trained)
INFRAMIND_CHECKPOINT="${BLUE_STORAGE}/checkpoints/inframind/inframind_math_20260212_161547_job24816311.pt"

# Predictor checkpoints
LATENCY_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/latency_estimator.pth"
LENGTH_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/length_estimator.pth"

# Match baseline settings exactly
TEST_LIMIT=1000
CONCURRENCY=1000
ARRIVAL_PATTERN="poisson"
ARRIVAL_RATES="10 50 100 200"

# Budget: use 60s (medium) — tight enough for interesting behavior,
# generous enough to not cut episodes short artificially.
# The baseline has no budget, so this is the closest comparison.
BUDGET=60

# Output CSV
OUTPUT_CSV="logs/motivation_plot_generator_data/inframind_motivation_sweep_math_test_${TEST_LIMIT}_${ARRIVAL_PATTERN}.csv"
mkdir -p "$(dirname "$OUTPUT_CSV")"

# Environment
export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

# Validation
for f in "$INFRAMIND_CHECKPOINT" "$LATENCY_PREDICTOR" "$LENGTH_PREDICTOR"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done
if [[ ! -d "${MATH_DATASET_ROOT}/test" ]]; then
    echo "ERROR: MATH dataset not found at ${MATH_DATASET_ROOT}"
    exit 1
fi

echo "========================================="
echo "InfraMind Motivation Sweep — MATH"
echo "========================================="
echo "Checkpoint:  ${INFRAMIND_CHECKPOINT}"
echo "Test limit:  ${TEST_LIMIT}"
echo "Rates:       ${ARRIVAL_RATES}"
echo "Pattern:     ${ARRIVAL_PATTERN}"
echo "Budget:      ${BUDGET}s"
echo "Concurrency: ${CONCURRENCY}"
echo "Output:      ${OUTPUT_CSV}"
echo "========================================="
echo ""

for rate in $ARRIVAL_RATES; do
    echo "========================================="
    echo "Running with arrival_rate=${rate}, budget=${BUDGET}s"
    echo "========================================="

    python Experiments/train_inframind_math.py \
        --dataset-root "${MATH_DATASET_ROOT}" \
        --limit "${TEST_LIMIT}" \
        --epochs 1 \
        --arrival-rates "${rate}" \
        --arrival-pattern "${ARRIVAL_PATTERN}" \
        --budget-sweep "${BUDGET}" \
        --concurrency "${CONCURRENCY}" \
        --latency-predictor "${LATENCY_PREDICTOR}" \
        --length-predictor "${LENGTH_PREDICTOR}" \
        --checkpoint-path "${INFRAMIND_CHECKPOINT}" \
        --resume-checkpoint \
        --skip-training \
        --deterministic \
        --telemetry-csv "${OUTPUT_CSV}"

    echo "Completed run with arrival_rate=${rate}"
    echo ""
done

echo "========================================="
echo "All sweeps completed!"
echo "Results saved to: ${OUTPUT_CSV}"
echo "========================================="
