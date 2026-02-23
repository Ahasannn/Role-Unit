#!/bin/bash
#SBATCH --job-name=sysrouter_math_dry
#SBATCH --output=logs/inframind_training/math/slurm-dryrun-%j.out
#SBATCH --error=logs/inframind_training/math/slurm-dryrun-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

# Print job info
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="
echo ""

# Repository root
REPO_ROOT="/home/ah872032.ucf/system-aware-mas"
cd "$REPO_ROOT" || exit 1

# Load centralized HPC environment configuration
echo "Loading HPC environment configuration..."
source scripts/setup_hpc_env.sh

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo ""

# Blue storage configuration (imported from setup_hpc_env.sh)
echo "Blue Storage: ${BLUE_STORAGE}"
echo "HF Cache: ${HF_HOME}"
echo ""

# Create log directories
mkdir -p logs/inframind_training/math

# Function to cleanup vLLM servers on exit
cleanup_vllm() {
    echo ""
    echo "========================================="
    echo "Cleaning up vLLM servers..."
    echo "========================================="

    # Use job-specific log directory to avoid killing other jobs' vLLM servers
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
    fi

    echo "Cleanup complete"
    echo "========================================="
}

# Register cleanup on script exit
trap cleanup_vllm EXIT INT TERM

# Start vLLM model pool
echo "========================================="
echo "Starting vLLM model pool..."
echo "========================================="
bash scripts/vllm/serve_full_pool.sh || {
    echo "ERROR: Failed to start vLLM model pool"
    echo "Check logs in logs/vllm/*.log"
    exit 1
}

echo ""
echo "========================================="
echo "vLLM servers are ready!"
echo "========================================="
echo ""

# Verify servers are healthy
bash scripts/check_vllm_status.sh
echo ""

# ===== Paths =====
BLUE_STORAGE="${BLUE_STORAGE:-/blue/qi855292.ucf/ah872032.ucf}"
MAS_CHECKPOINT="${BLUE_STORAGE}/checkpoints/mas_router/mas_math_train_519_cost100.pth"
LATENCY_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/latency_estimator.pth"
LENGTH_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/length_estimator.pth"
MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"

# Telemetry (separate from real training)
TELEMETRY_DIR="logs/inframind_training/math"
TELEMETRY_CSV="${TELEMETRY_DIR}/inframind_dryrun_math.csv"

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

echo "========================================="
echo "InfraMind DRY RUN — MATH (20 items, rate=100, budget=1)"
echo "========================================="
echo ""

# Build command
CMD="python Experiments/train_inframind_math.py \
  --dataset-root ${MATH_DATASET_ROOT} \
  --limit 210 \
  --epochs 1 \
  --arrival-rates 200 \
  --arrival-pattern constant \
  --budget-sweep 50 \
  --concurrency 1000 \
  --latency-predictor ${LATENCY_PREDICTOR} \
  --length-predictor ${LENGTH_PREDICTOR} \
  --checkpoint-every 0 \
  --training-batch-size 20 \
  --telemetry-csv ${TELEMETRY_CSV}"

# Load pretrained MAS planner weights if available
if [[ -f "${MAS_CHECKPOINT}" ]]; then
    echo "Loading MAS planner weights: ${MAS_CHECKPOINT}"
    CMD="${CMD} --mas-checkpoint ${MAS_CHECKPOINT}"
else
    echo "WARNING: MAS checkpoint not found: ${MAS_CHECKPOINT}"
    echo "Training planner from scratch."
fi

echo "Command: ${CMD}"
echo ""
$CMD

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================="
echo "Dry run completed with exit code: $TRAIN_EXIT_CODE"
echo "End Time: $(date)"
echo "========================================="

exit $TRAIN_EXIT_CODE
