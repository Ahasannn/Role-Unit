#!/bin/bash
# Temporary script to resume training for 2 MORE epochs on existing checkpoints
# This will bring total epochs from 3 → 5 for each cost_rate

# Repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ah872032.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"

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
EPOCHS=2  # 2 MORE epochs (total will be 5)
BATCH_SIZE=32
LR=0.01

# Cost rates to resume training
COST_RATES=(100 400 700)

echo "=========================================="
echo "Baseline MAS Resume Training (2 more epochs)"
echo "=========================================="
echo "Training ${TRAIN_LIMIT} samples for ${EPOCHS} additional epochs each"
echo "Cost rates: ${COST_RATES[@]}"
echo "Total epochs after completion: 5"
echo ""

for COST_RATE in "${COST_RATES[@]}"; do
    echo "=========================================="
    echo "Resuming training: cost_rate=${COST_RATE}"
    echo "=========================================="

    # Checkpoint and telemetry paths
    CHECKPOINT_PATH="${CHECKPOINT_DIR}/mas_math_train_${TRAIN_LIMIT}_cost${COST_RATE}.pth"
    TELEMETRY_DIR="logs/baseline_mas_training/math"
    TELEMETRY_CSV="${TELEMETRY_DIR}/mas_train_math_${TRAIN_LIMIT}_cost${COST_RATE}_resume.csv"

    mkdir -p "${TELEMETRY_DIR}"

    # Verify checkpoint exists
    if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
        echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
        exit 1
    fi

    echo "Loading checkpoint: ${CHECKPOINT_PATH}"

    # Build training command with checkpoint resumption
    CMD="python Experiments/run_math.py \
      --train_limit ${TRAIN_LIMIT} \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --lr ${LR} \
      --cost_rate ${COST_RATE} \
      --dataset-root ${MATH_DATASET_ROOT} \
      --train-telemetry-csv ${TELEMETRY_CSV} \
      --skip-test \
      --checkpoint ${CHECKPOINT_PATH} \
      --save-checkpoint ${CHECKPOINT_PATH}"

    echo "Command: ${CMD}"
    echo ""

    # Run training
    eval ${CMD}

    if [ $? -eq 0 ]; then
        echo "✓ Successfully trained cost_rate=${COST_RATE} for ${EPOCHS} more epochs"
        echo "✓ Total epochs: 5"
        echo "✓ Checkpoint updated: ${CHECKPOINT_PATH}"
    else
        echo "✗ Training failed for cost_rate=${COST_RATE}"
        exit 1
    fi

    echo ""
done

echo "=========================================="
echo "All resume trainings completed successfully!"
echo "=========================================="
echo "All checkpoints now have 5 total epochs of training:"
for COST_RATE in "${COST_RATES[@]}"; do
    CHECKPOINT_PATH="${CHECKPOINT_DIR}/mas_math_train_${TRAIN_LIMIT}_cost${COST_RATE}.pth"
    echo "  - ${CHECKPOINT_PATH}"
done
echo ""
