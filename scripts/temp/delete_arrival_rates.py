#!/usr/bin/env python3
"""
Delete rows with specific arrival rates from a CSV file.
Set the variables below and run the script.
"""

import pandas as pd

# =============================================================================
# CONFIGURATION - Set these values
# =============================================================================

CSV_FILE = "logs/motivation_plot_generator_data/baseline_motivation_sweep_math_test_4000_poisson.csv"

ARRIVAL_RATES_TO_DELETE = [2000.0]

# =============================================================================
# SCRIPT - Do not modify below
# =============================================================================

def main():
    print(f"Loading: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)

    print(f"Total rows before: {len(df)}")
    print(f"Unique arrival_rate values: {sorted(df['arrival_rate'].unique())}")
    print(f"Arrival rates to delete: {ARRIVAL_RATES_TO_DELETE}")
    print()

    # Count rows to be deleted
    mask = df['arrival_rate'].isin(ARRIVAL_RATES_TO_DELETE)
    rows_to_delete = mask.sum()
    print(f"Rows to delete: {rows_to_delete}")

    # Filter out the rows
    df = df[~mask]

    print(f"Total rows after: {len(df)}")
    print()

    # Save
    df.to_csv(CSV_FILE, index=False)
    print(f"Saved to: {CSV_FILE}")

    print()
    print("Remaining arrival rates:")
    print(df['arrival_rate'].value_counts().sort_index())

if __name__ == "__main__":
    main()
