#!/bin/bash
# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ah872032.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/system_router"
mkdir -p "${CHECKPOINT_DIR}"

# vLLM API configuration (EMPTY means no authentication required)
export KEY="EMPTY"

python Experiments/train_system_router_humaneval.py \
  --dataset humaneval \
  --split train \
  --split-ratio 0.2 \
  --limit 32 \
  --epochs 3 \
  --request-timeout 600.0 \
  --seed 42 \
  --log-episodes 1 \
  --checkpoint-path "${CHECKPOINT_DIR}/system_humaneval_train.pt" \
  --checkpoint-every 10 \
  --telemetry-csv logs/system_humaneval_train.csv
