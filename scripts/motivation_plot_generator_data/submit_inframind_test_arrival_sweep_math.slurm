#!/bin/bash
#SBATCH --job-name=inframind_sweep_math
#SBATCH --output=logs/motivation_plot_generator_data/slurm-inframind-%j.out
#SBATCH --error=logs/motivation_plot_generator_data/slurm-inframind-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=09:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
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

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo ""

# Set environment variables
export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ah872032.ucf"
export MATH_DATASET_ROOT="${BLUE_STORAGE}/datasets/MATH"

# Ensure log directory exists
mkdir -p logs/motivation_plot_generator_data

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

# Run the arrival rate sweep
echo "========================================="
echo "Starting InfraMind MATH arrival rate sweep..."
echo "========================================="
echo ""

set +e
bash scripts/motivation_plot_generator_data/inframind_test_arrival_sweep_math.sh
SWEEP_EXIT_CODE=$?
set -e

echo ""
echo "========================================="
echo "Sweep completed with exit code: $SWEEP_EXIT_CODE"
echo "End Time: $(date)"
echo "========================================="

exit $SWEEP_EXIT_CODE
