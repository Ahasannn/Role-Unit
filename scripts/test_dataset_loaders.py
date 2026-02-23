#!/usr/bin/env python3
"""
Test script to verify dataset loaders with limit parameters work correctly.
Run this locally to test before submitting SLURM jobs.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_mbpp():
    print("\n" + "="*60)
    print("Testing MBPP Dataset Loader")
    print("="*60)
    try:
        from Datasets.mbpp_dataset import MbppDataset

        # Test without limit
        print("Loading full test dataset...")
        dataset_full = MbppDataset('test', limit=0)
        print(f"✓ Full dataset: {len(dataset_full)} items")

        # Test with limit
        print("Loading with limit=50...")
        dataset_limited = MbppDataset('test', limit=50)
        print(f"✓ Limited dataset: {len(dataset_limited)} items")

        # Verify limit works
        assert len(dataset_limited) == 50, f"Expected 50 items, got {len(dataset_limited)}"
        print("✓ MBPP loader test PASSED")

    except Exception as e:
        print(f"✗ MBPP loader test FAILED: {e}")
        return False
    return True

def test_gsm8k():
    print("\n" + "="*60)
    print("Testing GSM8K Dataset Loader")
    print("="*60)
    try:
        from Datasets.gsm8k_dataset import gsm_data_process
        from datasets import load_dataset

        # Load from HuggingFace
        print("Loading GSM8K from HuggingFace...")
        hf_test = load_dataset("openai/gsm8k", "main", split="test")

        # Test without limit
        print("Processing full test dataset...")
        dataset_full = gsm_data_process(hf_test, limit=0)
        print(f"✓ Full dataset: {len(dataset_full)} items")

        # Test with limit
        print("Processing with limit=50...")
        dataset_limited = gsm_data_process(hf_test, limit=50)
        print(f"✓ Limited dataset: {len(dataset_limited)} items")

        # Verify limit works
        assert len(dataset_limited) == 50, f"Expected 50 items, got {len(dataset_limited)}"
        print("✓ GSM8K loader test PASSED")

    except Exception as e:
        print(f"✗ GSM8K loader test FAILED: {e}")
        return False
    return True

def test_humaneval():
    print("\n" + "="*60)
    print("Testing HumanEval Dataset Loader")
    print("="*60)
    try:
        from Datasets.humaneval_dataset import HumanEvalDataset

        # Test without limit
        print("Loading full test dataset...")
        dataset_full = HumanEvalDataset('test', limit=0)
        print(f"✓ Full dataset: {len(dataset_full)} items")

        # Test with limit
        print("Loading with limit=50...")
        dataset_limited = HumanEvalDataset('test', limit=50)
        print(f"✓ Limited dataset: {len(dataset_limited)} items")

        # Verify limit works
        assert len(dataset_limited) == 50, f"Expected 50 items, got {len(dataset_limited)}"
        print("✓ HumanEval loader test PASSED")

    except Exception as e:
        print(f"✗ HumanEval loader test FAILED: {e}")
        return False
    return True

def test_mmlu():
    print("\n" + "="*60)
    print("Testing MMLU Dataset Loader")
    print("="*60)
    try:
        from Datasets.mmlu_dataset import MMLUDataset

        # Test without limit
        print("Loading full test dataset...")
        dataset_full = MMLUDataset('test', stratified_limit=0)
        print(f"✓ Full dataset: {len(dataset_full)} items")

        # Test with stratified limit
        print("Loading with stratified_limit=100...")
        dataset_limited = MMLUDataset('test', stratified_limit=100)
        print(f"✓ Limited dataset: {len(dataset_limited)} items")

        # Verify limit is approximately correct (may not be exact due to rounding)
        assert 90 <= len(dataset_limited) <= 110, f"Expected ~100 items, got {len(dataset_limited)}"
        print("✓ MMLU loader test PASSED")

    except Exception as e:
        print(f"✗ MMLU loader test FAILED: {e}")
        return False
    return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Dataset Loader Tests")
    print("="*60)
    print("This script tests all dataset loaders with limit parameters")
    print("to ensure they work correctly before submitting SLURM jobs.")
    print("="*60)

    results = []

    # Run tests (some may fail if datasets not available)
    results.append(("MBPP", test_mbpp()))
    results.append(("GSM8K", test_gsm8k()))
    results.append(("HumanEval", test_humaneval()))
    results.append(("MMLU", test_mmlu()))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:15s}: {status}")

    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)

    sys.exit(0 if passed == total else 1)
