#!/bin/bash
# Baseline MAS Test Arrival Rate Sweep for HumanEval (Sustained High Load)
# This script tests with sustained arrival pattern at high request rate

# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ah872032.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"

# HumanEval is loaded from HuggingFace - no offline path needed

# Output CSV file
OUTPUT_CSV="logs/motivation_plot_generator_data/baseline_motivation_sweep_humaneval_test_300_poisson.csv"

# Arrival patterns, rates, and concurrency tuples: (pattern, arrival_rate, concurrency)
# Note: HumanEval has only 164 total examples, so 300 might be too many
# Using 150 as the test limit to be safe
RUN_CONFIGS=(
    "poisson 2 1000"
    "poisson 5 1000"
    "poisson 100 1000"
    "poisson 200 1000"
    "poisson 300 1000"
)

# Loop through each configuration
for pair in "${RUN_CONFIGS[@]}"; do
    read -r pattern arrival_rate concurrency <<< "$pair"

    echo "========================================"
    echo "Running with arrival_rate=$arrival_rate, concurrency=$concurrency, pattern=$pattern"
    echo "========================================"

    python Experiments/run_humaneval.py \
        --concurrency "$concurrency" \
        --arrival-rate "$arrival_rate" \
        --arrival-pattern "$pattern" \
        --checkpoint "${CHECKPOINT_DIR}/mas_humaneval_train_full.pth" \
        --test-telemetry-csv "$OUTPUT_CSV" \
        --epochs 0 \
        --test_limit 150

    echo "Completed run with arrival_rate=$arrival_rate, concurrency=$concurrency, pattern=$pattern"
    echo ""
done

echo "========================================"
echo "All sweeps completed!"
echo "Results saved to: $OUTPUT_CSV"
echo "========================================"
