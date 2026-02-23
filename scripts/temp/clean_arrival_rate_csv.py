#!/usr/bin/env python3
"""
Clean CSV telemetry files by removing rows with problematic arrival rates.

Usage:
    python scripts/temp/clean_arrival_rate_csv.py <input_csv> [options]

Options:
    --remove-last              Remove the highest arrival rate
    --remove-rate RATE         Remove specific arrival rate (can be used multiple times)
    --auto-detect-timeout      Auto-detect and remove rates with timeout issues
    --output OUTPUT            Output file path (default: <input>_cleaned.csv)
    --timeout-threshold SECS   Latency threshold for timeout detection (default: 1500)

Examples:
    # Remove the last (highest) arrival rate
    python scripts/temp/clean_arrival_rate_csv.py logs/data.csv --remove-last

    # Remove specific rate
    python scripts/temp/clean_arrival_rate_csv.py logs/data.csv --remove-rate 2000.0

    # Auto-detect problematic rates
    python scripts/temp/clean_arrival_rate_csv.py logs/data.csv --auto-detect-timeout
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Set


def analyze_arrival_rates(csv_path: str) -> dict:
    """Analyze each arrival rate for statistics."""
    stats = defaultdict(lambda: {
        'count': 0,
        'avg_latency': 0.0,
        'max_latency': 0.0,
        'empty_results': 0,
        'total_latency': 0.0,
    })

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rate = row.get('arrival_rate', '0.0')
            if not rate:
                continue

            rate_stats = stats[rate]
            rate_stats['count'] += 1

            # Check latency
            try:
                latency = float(row.get('workflow_latency_seconds', 0.0))
                rate_stats['total_latency'] += latency
                rate_stats['max_latency'] = max(rate_stats['max_latency'], latency)
            except (ValueError, TypeError):
                pass

            # Check for empty/error indicators
            if row.get('record_type') == 'summary':
                if not row.get('quality_pred') or row.get('quality_pred') == '0':
                    rate_stats['empty_results'] += 1

    # Calculate averages
    for rate, data in stats.items():
        if data['count'] > 0:
            data['avg_latency'] = data['total_latency'] / data['count']

    return dict(stats)


def detect_timeout_rates(stats: dict, threshold: float = 1500.0) -> Set[str]:
    """Detect arrival rates that have timeout issues."""
    problematic_rates = set()

    for rate, data in stats.items():
        # Check if average latency is very high
        if data['avg_latency'] > threshold:
            problematic_rates.add(rate)
            print(f"  ⚠️  Rate {rate}: High avg latency ({data['avg_latency']:.1f}s)")

        # Check if many empty results
        if data['count'] > 0:
            empty_ratio = data['empty_results'] / data['count']
            if empty_ratio > 0.3:  # More than 30% empty
                problematic_rates.add(rate)
                print(f"  ⚠️  Rate {rate}: High empty rate ({empty_ratio:.1%})")

    return problematic_rates


def clean_csv(
    input_path: str,
    output_path: str,
    rates_to_remove: Set[str],
    verbose: bool = True
) -> int:
    """Remove rows with specified arrival rates."""
    removed_count = 0
    kept_count = 0

    with open(input_path, 'r') as fin, open(output_path, 'w', newline='') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            rate = row.get('arrival_rate', '')
            if rate in rates_to_remove:
                removed_count += 1
            else:
                writer.writerow(row)
                kept_count += 1

    if verbose:
        print(f"\n✅ Cleaned CSV saved to: {output_path}")
        print(f"   Kept: {kept_count} rows")
        print(f"   Removed: {removed_count} rows")

    return removed_count


def main():
    parser = argparse.ArgumentParser(
        description='Clean CSV telemetry files by removing problematic arrival rates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input_csv', help='Input CSV file path')
    parser.add_argument('--remove-last', action='store_true',
                        help='Remove the highest arrival rate')
    parser.add_argument('--remove-rate', action='append', dest='remove_rates',
                        help='Remove specific arrival rate (can be used multiple times)')
    parser.add_argument('--auto-detect-timeout', action='store_true',
                        help='Auto-detect and remove rates with timeout issues')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--timeout-threshold', type=float, default=1500.0,
                        help='Latency threshold for timeout detection (default: 1500s)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be removed without actually doing it')

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"❌ Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_cleaned.csv"

    print(f"📊 Analyzing: {input_path}")

    # Analyze arrival rates
    stats = analyze_arrival_rates(str(input_path))

    print(f"\n📈 Found {len(stats)} arrival rates:")
    for rate in sorted(stats.keys(), key=float):
        data = stats[rate]
        print(f"  • Rate {rate}: {data['count']} rows, "
              f"avg latency {data['avg_latency']:.1f}s, "
              f"max latency {data['max_latency']:.1f}s")

    # Determine which rates to remove
    rates_to_remove = set()

    if args.remove_rates:
        rates_to_remove.update(args.remove_rates)
        print(f"\n🎯 Manually specified rates to remove: {args.remove_rates}")

    if args.remove_last:
        last_rate = max(stats.keys(), key=float)
        rates_to_remove.add(last_rate)
        print(f"\n🎯 Removing last (highest) arrival rate: {last_rate}")

    if args.auto_detect_timeout:
        print(f"\n🔍 Auto-detecting problematic rates (threshold: {args.timeout_threshold}s):")
        detected = detect_timeout_rates(stats, args.timeout_threshold)
        rates_to_remove.update(detected)

    if not rates_to_remove:
        print("\n⚠️  No rates specified for removal. Use --remove-last, --remove-rate, or --auto-detect-timeout")
        sys.exit(0)

    print(f"\n🗑️  Rates to remove: {sorted(rates_to_remove, key=float)}")

    # Calculate what will be removed
    total_to_remove = sum(stats[rate]['count'] for rate in rates_to_remove if rate in stats)
    total_rows = sum(data['count'] for data in stats.values())
    print(f"   Will remove {total_to_remove:,} / {total_rows:,} rows ({100*total_to_remove/total_rows:.1f}%)")

    if args.dry_run:
        print("\n🔍 Dry run - no files modified")
        sys.exit(0)

    # Perform cleaning
    print(f"\n🧹 Cleaning CSV...")
    clean_csv(str(input_path), str(output_path), rates_to_remove)


if __name__ == '__main__':
    main()
