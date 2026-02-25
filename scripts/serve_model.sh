#!/bin/bash
#SBATCH --job-name=roleunit-vllm
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --account=qi855292.ucf
#SBATCH --output=logs/vllm-serve-%j.log
#SBATCH --error=logs/vllm-serve-%j.err

# Serve all 6 models on 2 B200 GPUs for MAS pipeline evaluation.
# GPU 0: Mistral-24B, Qwen-Coder-14B (~48% utilization)
# GPU 1: Gemma-9B, Llama-8B, Qwen-7B, Llama-3B (~44% utilization)
#
# Usage:
#   sbatch scripts/serve_model.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs

source .venv/bin/activate
bash scripts/serve_full_pool.sh

echo "All 6 models serving on 2 GPUs. Waiting for job cancellation or time limit..."
sleep infinity
