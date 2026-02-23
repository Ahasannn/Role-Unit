#!/bin/bash
# Baseline MAS training for MATH

# Repo root (script is in scripts/baseline_train)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ah872032.ucf"

# Read train_limit from dataset config
CONFIG_FILE="${REPO_ROOT}/Experiments/dataset_config.json"
TRAIN_LIMIT=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['math']['train_limit'])")

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"
mkdir -p "${CHECKPOINT_DIR}"

# Checkpoint and CSV paths
CHECKPOINT_PATH="${CHECKPOINT_DIR}/mas_math_train_${TRAIN_LIMIT}.pth"
TELEMETRY_DIR="logs/baseline_mas_training/math"
TELEMETRY_CSV="${TELEMETRY_DIR}/mas_train_math_${TRAIN_LIMIT}.csv"

# Create logs directory if needed
mkdir -p logs "${TELEMETRY_DIR}"

# Dataset root (override with MATH_DATASET_ROOT)
MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"
if [[ ! -d "${MATH_DATASET_ROOT}/train" || ! -d "${MATH_DATASET_ROOT}/test" ]]; then
    echo "MATH dataset not found at ${MATH_DATASET_ROOT}."
    echo "Expected ${MATH_DATASET_ROOT}/train and ${MATH_DATASET_ROOT}/test."
    echo "Download and extract the dataset there, or set MATH_DATASET_ROOT."
    exit 1
fi

# vLLM API configuration (EMPTY means no authentication required)
export KEY="EMPTY"

# Cost-quality tradeoff coefficient (override with COST_RATE env var)
COST_RATE="${COST_RATE:-700.0}"

# Build command with conditional checkpoint loading
CMD="python Experiments/run_math.py \
  --train_limit ${TRAIN_LIMIT} \
  --epochs 2 \
  --batch_size 32 \
  --lr 0.01 \
  --cost_rate ${COST_RATE} \
  --dataset-root ${MATH_DATASET_ROOT} \
  --train-telemetry-csv ${TELEMETRY_CSV} \
  --skip-test \
  --save-checkpoint ${CHECKPOINT_PATH}"

# If checkpoint exists, resume from it
if [ -f "${CHECKPOINT_PATH}" ]; then
    echo "Found existing checkpoint: ${CHECKPOINT_PATH}"
    echo "Resuming training..."
    CMD="${CMD} --checkpoint ${CHECKPOINT_PATH}"
else
    echo "No existing checkpoint found. Starting fresh training..."
fi

# Run training
echo "Running: ${CMD}"
eval ${CMD}
