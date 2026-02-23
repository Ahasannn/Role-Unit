#!/bin/bash
#SBATCH --job-name=paper_math_dry
#SBATCH --output=logs/paper_results/slurm-%j-dryrun.out
#SBATCH --error=logs/paper_results/slurm-%j-dryrun.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

# ===== Dry-run paper results for MATH =====
# Runs BOTH baseline and InfraMind inference with 10 test items
# at arrival rates [10, 200] req/min and budgets [10s, 100s].
# Baseline: 2 rates × 10 items = 20 episodes
# InfraMind: 2 rates × 2 budgets × 10 items = 40 episodes
# Use this to validate the pipeline before the full run.

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="
echo ""

REPO_ROOT="/home/ah872032.ucf/system-aware-mas"
cd "$REPO_ROOT" || exit 1

# Source HPC env (sets BLUE_STORAGE, HF_HOME, cache dirs, etc.)
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

# ===== Dry-run overrides =====
export TEST_LIMIT=10
export CONCURRENCY=10
export BUDGET_SWEEP="10,100"

# Override arrival rates to just one for dry run
# (The shell scripts read from dataset_config.json which has [10,50,100,200],
#  but for dry run we temporarily override the config reading.)

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

# ===== Temporary config override for dry run =====
# Create a temp config with rates=[10,200] for the dry run
DRY_CONFIG=$(mktemp)
python3 -c "
import json
cfg = json.load(open('${REPO_ROOT}/Experiments/dataset_config.json'))
cfg['math']['arrival_rates'] = [10, 200]
json.dump(cfg, open('${DRY_CONFIG}', 'w'), indent=2)
"
# Swap config temporarily
cp "${REPO_ROOT}/Experiments/dataset_config.json" "${REPO_ROOT}/Experiments/dataset_config.json.bak"
cp "${DRY_CONFIG}" "${REPO_ROOT}/Experiments/dataset_config.json"
rm -f "${DRY_CONFIG}"

restore_config() {
    if [[ -f "${REPO_ROOT}/Experiments/dataset_config.json.bak" ]]; then
        mv "${REPO_ROOT}/Experiments/dataset_config.json.bak" "${REPO_ROOT}/Experiments/dataset_config.json"
    fi
}
# Restore config on exit (before vllm cleanup)
trap 'restore_config; cleanup_vllm' EXIT INT TERM

set +e  # Don't exit on error; we want cleanup to run

# ===== Phase 1: Baseline =====
echo "========================================="
echo "Phase 1: Baseline MAS Router (dry run)"
echo "========================================="
echo ""
bash scripts/paper_results/baseline_test_sweep_math.sh
BASELINE_EXIT=$?
echo "Baseline exit code: $BASELINE_EXIT"
echo ""

# ===== Phase 2: InfraMind =====
echo "========================================="
echo "Phase 2: InfraMind (dry run)"
echo "========================================="
echo ""
bash scripts/paper_results/inframind_test_sweep_math.sh
INFRAMIND_EXIT=$?
echo "InfraMind exit code: $INFRAMIND_EXIT"
echo ""

set -e

# ===== Summary =====
echo "========================================="
echo "Dry Run Complete"
echo "========================================="
echo "Baseline CSV: logs/paper_results/math/baseline_math_test_${TEST_LIMIT}_poisson.csv"
echo "InfraMind CSV: logs/paper_results/math/inframind_math_test_${TEST_LIMIT}_poisson.csv"
echo "Baseline exit: $BASELINE_EXIT"
echo "InfraMind exit: $INFRAMIND_EXIT"
echo "End Time: $(date)"
echo "========================================="

# Exit with combined status
if [[ $BASELINE_EXIT -ne 0 || $INFRAMIND_EXIT -ne 0 ]]; then
    exit 1
fi
exit 0
