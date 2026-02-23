#!/bin/bash
# Temporary script to train baseline MAS with multiple cost_rate values sequentially
# This will create: mas_math_train_519_cost100.pth and mas_math_train_519_cost400.pth

# Repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ah872032.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"
mkdir -p "${CHECKPOINT_DIR}"

# Dataset configuration
TRAIN_LIMIT=519
MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"

if [[ ! -d "${MATH_DATASET_ROOT}/train" || ! -d "${MATH_DATASET_ROOT}/test" ]]; then
    echo "ERROR: MATH dataset not found at ${MATH_DATASET_ROOT}"
    exit 1
fi

# vLLM API configuration
export KEY="EMPTY"

# Training parameters
EPOCHS=3
BATCH_SIZE=32
LR=0.01

# Cost rates to train (in order)
COST_RATES=(100 400)

echo "=========================================="
echo "Baseline MAS Cost Rate Sweep Training"
echo "=========================================="
echo "Training ${TRAIN_LIMIT} samples for ${EPOCHS} epochs each"
echo "Cost rates: ${COST_RATES[@]}"
echo ""

for COST_RATE in "${COST_RATES[@]}"; do
    echo "=========================================="
    echo "Training with cost_rate=${COST_RATE}"
    echo "=========================================="

    # Checkpoint and telemetry paths
    CHECKPOINT_PATH="${CHECKPOINT_DIR}/mas_math_train_${TRAIN_LIMIT}_cost${COST_RATE}.pth"
    TELEMETRY_DIR="logs/baseline_mas_training/math"
    TELEMETRY_CSV="${TELEMETRY_DIR}/mas_train_math_${TRAIN_LIMIT}_cost${COST_RATE}.csv"

    mkdir -p "${TELEMETRY_DIR}"

    # Build training command
    CMD="python Experiments/run_math.py \
      --train_limit ${TRAIN_LIMIT} \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --lr ${LR} \
      --cost_rate ${COST_RATE} \
      --dataset-root ${MATH_DATASET_ROOT} \
      --train-telemetry-csv ${TELEMETRY_CSV} \
      --skip-test \
      --save-checkpoint ${CHECKPOINT_PATH}"

    echo "Command: ${CMD}"
    echo ""

    # Run training
    eval ${CMD}

    if [ $? -eq 0 ]; then
        echo "✓ Successfully trained cost_rate=${COST_RATE}"
        echo "✓ Checkpoint saved: ${CHECKPOINT_PATH}"
    else
        echo "✗ Training failed for cost_rate=${COST_RATE}"
        exit 1
    fi

    echo ""
done

echo "=========================================="
echo "All trainings completed successfully!"
echo "=========================================="
echo "Created checkpoints:"
for COST_RATE in "${COST_RATES[@]}"; do
    CHECKPOINT_PATH="${CHECKPOINT_DIR}/mas_math_train_${TRAIN_LIMIT}_cost${COST_RATE}.pth"
    echo "  - ${CHECKPOINT_PATH}"
done
echo ""
