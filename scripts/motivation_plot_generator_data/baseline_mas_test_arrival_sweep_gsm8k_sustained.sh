#!/bin/bash
# Baseline MAS Test Arrival Rate Sweep for GSM8K (Sustained High Load)
# This script tests with sustained arrival pattern at high request rate

# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ah872032.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"

# GSM8K is loaded from HuggingFace - no offline path needed

# Output CSV file
OUTPUT_CSV="logs/motivation_plot_generator_data/baseline_motivation_sweep_gsm8k_test_300_poisson.csv"

# Arrival patterns, rates, and concurrency tuples: (pattern, arrival_rate, concurrency)
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

    python Experiments/run_gsm8k.py \
        --concurrency "$concurrency" \
        --arrival-rate "$arrival_rate" \
        --arrival-pattern "$pattern" \
        --checkpoint "${CHECKPOINT_DIR}/mas_gsm8k_train_full.pth" \
        --test-telemetry-csv "$OUTPUT_CSV" \
        --epochs 0 \
        --test_limit 300

    echo "Completed run with arrival_rate=$arrival_rate, concurrency=$concurrency, pattern=$pattern"
    echo ""
done

echo "========================================"
echo "All sweeps completed!"
echo "Results saved to: $OUTPUT_CSV"
echo "========================================"
