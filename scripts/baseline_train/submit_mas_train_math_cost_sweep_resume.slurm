#!/bin/bash
#SBATCH --job-name=mas_resume
#SBATCH --output=logs/baseline_mas_training/math/slurm-%j-resume.out
#SBATCH --error=logs/baseline_mas_training/math/slurm-%j-resume.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

# Temporary SLURM script to resume training on all 3 cost_rate checkpoints
# Adds 2 MORE epochs to each (bringing total from 3 → 5 epochs)

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
echo "HF Token: $(if [[ -n "${HF_TOKEN}" ]]; then echo "✓ Configured"; else echo "✗ Missing"; fi)"
echo ""

# Set environment variables
export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

# Create log directories
mkdir -p logs/baseline_mas_training/math logs/vllm

# Function to cleanup vLLM servers on exit
cleanup_vllm() {
    echo ""
    echo "========================================="
    echo "Cleaning up vLLM servers..."
    echo "========================================="

    # Use job-specific log directory
    local vllm_log_dir="logs/vllm/job_${SLURM_JOB_ID}"

    # Kill all vLLM processes from PID files in job-specific directory
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

    # Extra safety: kill any vLLM processes spawned by this job
    # Only kill processes owned by current user to avoid affecting other jobs
    pkill -u $USER -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

    echo "Cleanup complete"
    echo "========================================="
}

# Register cleanup on script exit
trap cleanup_vllm EXIT INT TERM

# Start vLLM model pool using existing script
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

# Run sequential resume training
echo "========================================="
echo "Starting resume training (2 more epochs each)..."
echo "========================================="

# Run with proper error handling
set +e  # Don't exit on error, we want cleanup to run
bash scripts/baseline_train/mas_train_math_cost_sweep_resume.sh
TRAIN_EXIT_CODE=$?
set -e

echo ""
echo "========================================="
echo "Resume training completed with exit code: $TRAIN_EXIT_CODE"
echo "End Time: $(date)"
echo "========================================="

if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ All resume trainings completed successfully!"
    echo ""
    echo "Updated checkpoints (now 5 total epochs each):"
    echo "  - ${BLUE_STORAGE}/checkpoints/mas_router/mas_math_train_519_cost100.pth"
    echo "  - ${BLUE_STORAGE}/checkpoints/mas_router/mas_math_train_519_cost400.pth"
    echo "  - ${BLUE_STORAGE}/checkpoints/mas_router/mas_math_train_519_cost700.pth"
fi

# Cleanup will happen automatically via trap
exit $TRAIN_EXIT_CODE
