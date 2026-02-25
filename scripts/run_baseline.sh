#!/bin/bash
#SBATCH --job-name=roleunit-baseline
#SBATCH --output=logs/test/baseline/baseline-%j.log
#SBATCH --error=logs/test/baseline/baseline-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1

# Deploy all 6 models on 2 GPUs, run baseline experiments.
#
# Usage:
#   sbatch scripts/run_baseline.sh                              # full: validation split, homogeneous + 10 random
#   sbatch scripts/run_baseline.sh dev 0 homogeneous 0          # smoke test

# Print job info
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="
echo ""

# Repository root
REPO_ROOT="/home/ah872032.ucf/Role-Unit"
cd "$REPO_ROOT" || exit 1

# Parse arguments
SPLIT=${1:-"test"}
LIMIT=${2:-500}
MODE=${3:-"both"}        # random | homogeneous | both
N_TRIALS=${4:-5}
CONCURRENCY=${5:-64}     # questions in parallel

# Load centralized HPC environment configuration
echo "Loading HPC environment configuration..."
source scripts/setup_hpc_env.sh

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo ""

echo "HF Cache: ${HF_HOME}"
echo ""

# Create output directories
mkdir -p logs/test/baseline results/baseline

# Environment variables for LLM routing
export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

# Function to cleanup vLLM servers on exit
cleanup_vllm() {
    echo ""
    echo "========================================="
    echo "Cleaning up vLLM servers..."
    echo "========================================="

    local vllm_log_dir="logs/vllm/job_${SLURM_JOB_ID}"

    if ls "${vllm_log_dir}"/*.pid >/dev/null 2>&1; then
        for pidfile in "${vllm_log_dir}"/*.pid; do
            if [ -f "$pidfile" ]; then
                pid=$(cat "$pidfile")
                name=$(basename "$pidfile" .pid)
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Stopping $name (PID $pid)"
                    kill -TERM "$pid" 2>/dev/null
                    sleep 2
                    if kill -0 "$pid" 2>/dev/null; then
                        kill -KILL "$pid" 2>/dev/null
                    fi
                fi
                rm -f "$pidfile"
            fi
        done
    else
        echo "No PID files found in ${vllm_log_dir}"
    fi

    pkill -u $USER -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

    echo "Cleanup complete"
    echo "========================================="
}

# Register cleanup on script exit
trap cleanup_vllm EXIT INT TERM

# Start vLLM model pool
echo "========================================="
echo "Starting vLLM model pool..."
echo "========================================="
bash scripts/serve_full_pool.sh || {
    echo "ERROR: Failed to start vLLM model pool"
    echo "Check logs in logs/vllm/*.log"
    exit 1
}

echo ""
echo "========================================="
echo "vLLM servers are ready!"
echo "========================================="
echo ""

# Set MODEL_BASE_URLS from the JSON so LLMRegistry routes correctly
export MODEL_BASE_URLS="${REPO_ROOT}/config/llm_profile_full.json"

# Run baseline
echo "========================================="
echo "Running baseline experiments..."
echo "Split: ${SPLIT} | Limit: ${LIMIT}"
echo "Mode: ${MODE} | Trials: ${N_TRIALS} | Concurrency: ${CONCURRENCY}"
echo "========================================="
echo ""

set +e  # Don't exit on error, we want cleanup to run
python run_baseline.py \
    --mode "${MODE}" \
    --n-trials "${N_TRIALS}" \
    --split "${SPLIT}" \
    --limit "${LIMIT}" \
    --concurrency "${CONCURRENCY}" \
    --output-dir "results/baseline" \
    --role-subjects "config/role_subjects.yaml"
RUN_EXIT_CODE=$?
set -e

echo ""
echo "========================================="
echo "Baseline completed with exit code: $RUN_EXIT_CODE"
echo "End Time: $(date)"
echo "========================================="

# Cleanup will happen automatically via trap
exit $RUN_EXIT_CODE
