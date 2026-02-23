#!/bin/bash
#SBATCH --job-name=paper_math
#SBATCH --output=logs/paper_results/slurm-%j.out
#SBATCH --error=logs/paper_results/slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

# ===== Full paper results for MATH =====
# Runs BOTH baseline and InfraMind inference on the MATH test set.
# Enhanced configuration for publication-quality figures.
#
# Baseline: 6 arrival rates × 1000 test items = 6,000 episodes
# InfraMind: 6 arrival rates × 8 budgets × 1000 test items = 48,000 episodes
# Total: 54,000 episodes
#
# Arrival rates: [10, 30, 50, 100, 150, 200] req/min
# Budgets: [10, 20, 30, 50, 70, 100, 150, 200] seconds
#
# Output CSVs:
#   logs/paper_results/math/baseline_math_test_1000_poisson.csv
#   logs/paper_results/math/inframind_math_test_1000_poisson.csv

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="
echo ""

REPO_ROOT="/home/ah872032.ucf/system-aware-mas"
cd "$REPO_ROOT" || exit 1

# Source HPC env
source scripts/setup_hpc_env.sh

# Activate virtual environment
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo ""

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"
export CODE_EXECUTION_TIMEOUT="10"
export MATH_DATASET_ROOT="${BLUE_STORAGE}/datasets/MATH"

# Ensure log directory exists
mkdir -p logs/paper_results/math

# ===== Full run settings =====
export TEST_LIMIT=1000
export CONCURRENCY=1000
export BUDGET_SWEEP="10,20,30,50,70,100,150,200"

# ===== Cleanup function =====
cleanup_vllm() {
    echo ""
    echo "Cleaning up vLLM servers..."

    # Use job-specific log directory to avoid killing other jobs\' vLLM servers
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
}
trap cleanup_vllm EXIT INT TERM

# ===== Start vLLM =====
echo "Starting vLLM model pool..."
bash scripts/vllm/serve_full_pool.sh || { echo "ERROR: Failed to start vLLM"; exit 1; }
echo "vLLM servers ready!"
bash scripts/check_vllm_status.sh
echo ""

set +e  # Don't exit on error; we want cleanup to run

# ===== Phase 1: Baseline =====
echo "========================================="
echo "Phase 1: Baseline MAS Router"
echo "========================================="
echo ""
bash scripts/paper_results/baseline_test_sweep_math.sh
BASELINE_EXIT=$?
echo "Baseline exit code: $BASELINE_EXIT"
echo ""

# ===== Phase 2: InfraMind =====
echo "========================================="
echo "Phase 2: InfraMind"
echo "========================================="
echo ""
bash scripts/paper_results/inframind_test_sweep_math.sh
INFRAMIND_EXIT=$?
echo "InfraMind exit code: $INFRAMIND_EXIT"
echo ""

set -e

# ===== Summary =====
echo "========================================="
echo "Paper Results Complete"
echo "========================================="
echo "Baseline CSV: logs/paper_results/math/baseline_math_test_${TEST_LIMIT}_poisson.csv"
echo "InfraMind CSV: logs/paper_results/math/inframind_math_test_${TEST_LIMIT}_poisson.csv"
echo "Baseline exit: $BASELINE_EXIT"
echo "InfraMind exit: $INFRAMIND_EXIT"
echo "End Time: $(date)"
echo "========================================="

if [[ $BASELINE_EXIT -ne 0 || $INFRAMIND_EXIT -ne 0 ]]; then
    exit 1
fi
exit 0
