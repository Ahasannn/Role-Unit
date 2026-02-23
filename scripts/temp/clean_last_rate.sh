#!/bin/bash
# Quick wrapper to remove the last (highest) arrival rate from CSV files
#
# Usage:
#   bash scripts/temp/clean_last_rate.sh <csv_file>
#
# Example:
#   bash scripts/temp/clean_last_rate.sh logs/motivation_plot_generator_data/baseline_motivation_sweep_math_test_4000_poisson.csv

if [ $# -eq 0 ]; then
    echo "Usage: $0 <csv_file>"
    echo ""
    echo "This removes the highest arrival rate from the CSV file."
    echo "Output will be saved as: <input>_cleaned.csv"
    exit 1
fi

CSV_FILE="$1"

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: File not found: $CSV_FILE"
    exit 1
fi

python3 scripts/temp/clean_arrival_rate_csv.py "$CSV_FILE" --remove-last
