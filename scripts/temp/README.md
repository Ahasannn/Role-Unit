# Temporary Utility Scripts

This folder contains utility scripts for data processing and analysis.

## CSV Cleaning Scripts

### `clean_arrival_rate_csv.py`

Remove rows with problematic arrival rates from telemetry CSV files.

**Quick Usage:**
```bash
# Remove the last (highest) arrival rate
python3 scripts/temp/clean_arrival_rate_csv.py logs/data.csv --remove-last

# Or use the wrapper script
bash scripts/temp/clean_last_rate.sh logs/data.csv
```

**Advanced Usage:**
```bash
# Remove specific arrival rate
python3 scripts/temp/clean_arrival_rate_csv.py logs/data.csv --remove-rate 2000.0

# Remove multiple rates
python3 scripts/temp/clean_arrival_rate_csv.py logs/data.csv --remove-rate 1000.0 --remove-rate 2000.0

# Auto-detect and remove rates with timeout issues (avg latency > 1500s)
python3 scripts/temp/clean_arrival_rate_csv.py logs/data.csv --auto-detect-timeout

# Auto-detect with custom threshold
python3 scripts/temp/clean_arrival_rate_csv.py logs/data.csv --auto-detect-timeout --timeout-threshold 1200

# Specify output file
python3 scripts/temp/clean_arrival_rate_csv.py logs/data.csv --remove-last --output logs/data_fixed.csv

# Dry run (see what would be removed)
python3 scripts/temp/clean_arrival_rate_csv.py logs/data.csv --remove-last --dry-run
```

**What it does:**
1. Analyzes the CSV file to find all unique arrival rates
2. Shows statistics for each rate (row count, avg/max latency)
3. Removes rows matching the specified criteria
4. Saves cleaned data to `<input>_cleaned.csv` (or custom output path)

**Features:**
- **--remove-last**: Remove the highest arrival rate (useful when last rate had timeouts)
- **--remove-rate**: Remove specific rate(s) by value
- **--auto-detect-timeout**: Automatically detect problematic rates based on:
  - High average latency (default: >1500s)
  - High empty result rate (>30%)
- **--dry-run**: Preview what would be removed without modifying files

### `clean_last_rate.sh`

Simple wrapper script that removes the last (highest) arrival rate.

**Usage:**
```bash
bash scripts/temp/clean_last_rate.sh logs/motivation_plot_generator_data/baseline_motivation_sweep_math_test_4000_poisson.csv
```

Output will be saved as: `<input>_cleaned.csv`

## Examples

### Common Workflow

1. **Run experiment with multiple arrival rates**
2. **Check if last rate had issues:**
   ```bash
   python3 scripts/temp/clean_arrival_rate_csv.py logs/data.csv --auto-detect-timeout --dry-run
   ```
3. **Clean if needed:**
   ```bash
   bash scripts/temp/clean_last_rate.sh logs/data.csv
   ```
4. **Use cleaned file for analysis/plotting**

### Batch Cleaning Multiple Files

```bash
for csv in logs/motivation_plot_generator_data/*.csv; do
    echo "Cleaning $csv"
    python3 scripts/temp/clean_arrival_rate_csv.py "$csv" --remove-last
done
```
