#!/bin/bash
# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ah872032.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/system_router"

# vLLM API configuration (EMPTY means no authentication required)
export KEY="EMPTY"

python Experiments/train_system_router_humaneval.py \
  --dataset humaneval \
  --split test \
  --split-ratio 0.2 \
  --limit 131 \
  --epochs 0 \
  --request-timeout 600.0 \
  --seed 42 \
  --arrival-rates "10,30,50,70,100,150,200" \
  --arrival-pattern poisson \
  --concurrency 70 \
  --checkpoint-path "${CHECKPOINT_DIR}/system_humaneval_train.pt" \
  --resume-checkpoint \
  --telemetry-csv logs/system_humaneval_test.csv
