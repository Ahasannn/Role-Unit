#!/bin/bash
#SBATCH --job-name=roleunit-test
#SBATCH --output=logs/test/role-unit/unit-test-%j.log
#SBATCH --error=logs/test/role-unit/unit-test-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=00:45:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1

# Deploy all 6 models on 2 GPUs and run unit tests on the VALIDATION split.
# The TEST split is reserved for final evaluation only.
#
# Usage:
#   sbatch scripts/run_unit_tests.sh                  # full run via sbatch
#   bash scripts/run_unit_tests.sh                    # inside an srun session (models deployed separately)
#   bash scripts/run_unit_tests.sh --no-deploy        # skip model deployment (models already running)

# Print job info
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Job Name: ${SLURM_JOB_NAME:-manual}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Start Time: $(date)"
echo "========================================="
echo ""

# Repository root
REPO_ROOT="/home/ah872032.ucf/Role-Unit"
cd "$REPO_ROOT" || exit 1

# Parse flags
NO_DEPLOY=false
CONCURRENCY=32
for arg in "$@"; do
    case "$arg" in
        --no-deploy) NO_DEPLOY=true ;;
        --concurrency=*) CONCURRENCY="${arg#*=}" ;;
    esac
done

# Load centralized HPC environment configuration
echo "Loading HPC environment configuration..."
source scripts/setup_hpc_env.sh

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo ""

# Create output directories
JOB_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="unit_tests/results/job_${JOB_ID}"
mkdir -p logs/test/role-unit "${OUTPUT_DIR}"

# Environment variables
export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

# --- Deploy models (unless --no-deploy) ---
if [[ "$NO_DEPLOY" == "false" ]]; then
    # Cleanup function
    cleanup_vllm() {
        echo ""
        echo "========================================="
        echo "Cleaning up vLLM servers..."
        echo "========================================="

        local vllm_log_dir="logs/vllm/job_${JOB_ID}"
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
        pkill -u $USER -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
        echo "Cleanup complete"
        echo "========================================="
    }
    trap cleanup_vllm EXIT INT TERM

    echo "========================================="
    echo "Starting vLLM model pool..."
    echo "========================================="
    bash scripts/serve_full_pool.sh || {
        echo "ERROR: Failed to start vLLM model pool"
        exit 1
    }
    echo ""
    echo "vLLM servers are ready!"
    echo ""
else
    echo "Skipping model deployment (--no-deploy)"
    echo "Assuming models are already serving."
    echo ""
fi

# --- Run unit tests on VALIDATION split ---
echo "========================================="
echo "Running unit tests on VALIDATION split..."
echo "Concurrency: ${CONCURRENCY}"
echo "========================================="
echo ""

set +e
python unit_tests/run_unit_tests.py \
    --split validation \
    --concurrency "${CONCURRENCY}" \
    --output-dir "${OUTPUT_DIR}"
VAL_EXIT=$?

if [[ $VAL_EXIT -ne 0 ]]; then
    echo "ERROR: Validation split failed (exit $VAL_EXIT)"
    exit $VAL_EXIT
fi

set -e

echo ""
echo "========================================="
echo "All done!"
echo "Results in: ${OUTPUT_DIR}/"
echo "End Time: $(date)"
echo "========================================="

exit 0
