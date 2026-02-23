#!/bin/bash
# Baseline MAS Test Arrival Rate Sweep for MATH (Sustained High Load)
# This script tests with sustained arrival pattern at high request rate
# Arrival rates loaded from shared dataset config.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Blue storage configuration
BLUE_STORAGE="${BLUE_STORAGE:-/blue/qi855292.ucf/ah872032.ucf}"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"

# Dataset directory (offline)
export MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"

# ===== Load shared dataset config =====
DATASET_CONFIG="${REPO_ROOT}/Experiments/dataset_config.json"
DATASET_NAME="math"

ARRIVAL_PATTERN=$(python3 -c "import json; print(json.load(open('${DATASET_CONFIG}'))['${DATASET_NAME}']['arrival_pattern'])")
ARRIVAL_RATES=$(python3 -c "import json; print(' '.join(str(r) for r in json.load(open('${DATASET_CONFIG}'))['${DATASET_NAME}']['arrival_rates']))")

# Test-specific settings (separate from training config)
TEST_LIMIT=1000
CONCURRENCY=1000

# Build run configs from shared dataset config
# Key insight: Shared connection pool prevents connection explosion.
# Requests queue inside httpx pool → flow into vLLM scheduler's waiting queue (CPU memory).
# - With 5 models x 32 max_num_seqs = 160 running; rest wait in vLLM queues
RUN_CONFIGS=()
for rate in $ARRIVAL_RATES; do
    RUN_CONFIGS+=("${ARRIVAL_PATTERN} ${rate} ${CONCURRENCY}")
done

# Output CSV file
OUTPUT_CSV="logs/motivation_plot_generator_data/baseline_motivation_sweep_math_test_${TEST_LIMIT}_${ARRIVAL_PATTERN}.csv"
mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "========================================"
echo "MATH Arrival Rate Sweep (Sustained)"
echo "Rates:   ${ARRIVAL_RATES}"
echo "Pattern: ${ARRIVAL_PATTERN}"
echo "Output:  ${OUTPUT_CSV}"
echo "========================================"
echo ""

# Loop through each configuration
for pair in "${RUN_CONFIGS[@]}"; do
    read -r pattern arrival_rate concurrency <<< "$pair"

    echo "========================================"
    echo "Running with arrival_rate=$arrival_rate, concurrency=$concurrency, pattern=$pattern"
    echo "========================================"

    python Experiments/run_math.py \
        --concurrency "$concurrency" \
        --arrival-rate "$arrival_rate" \
        --arrival-pattern "$pattern" \
        --checkpoint "${CHECKPOINT_DIR}/mas_math_train_full.pth" \
        --test-telemetry-csv "$OUTPUT_CSV" \
        --epochs 0 \
        --test_limit "$TEST_LIMIT"

    echo "Completed run with arrival_rate=$arrival_rate, concurrency=$concurrency, pattern=$pattern"
    echo ""
done

echo "========================================"
echo "All sweeps completed!"
echo "Results saved to: $OUTPUT_CSV"
echo "========================================"
