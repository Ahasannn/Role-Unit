#!/bin/bash
#SBATCH --job-name=paper_im_math_p1
#SBATCH --output=logs/paper_results/slurm-%j-inframind-part1.out
#SBATCH --error=logs/paper_results/slurm-%j-inframind-part1.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1

# ===== InfraMind — Paper Results for MATH (Part 1 of 2) =====
# Inference-only (no training). Produces CSV for paper result tables/figures.
#
# Part 1 arrival rates: [10, 30, 50] req/min
# Budgets:              [10, 20, 30, 50, 70, 100, 150, 200] seconds
# Test items:           1000
#
# Total: 3 rates × 8 budgets × 1000 items = 24,000 episodes
#
# Output CSV:
#   logs/paper_results/math/inframind_math_test_1000_poisson_part1.csv
#
# Combine with part2 CSV later:
#   head -1 part1.csv > combined.csv
#   tail -n+2 part1.csv >> combined.csv
#   tail -n+2 part2.csv >> combined.csv

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

# ===== Run settings =====
TEST_LIMIT=1000
CONCURRENCY=1000
BUDGET_SWEEP="10,20,30,50,70,100,150,200"

# Part 1: low arrival rates
ARRIVAL_RATES_CSV="10,30,50"
ARRIVAL_PATTERN="poisson"

# ===== Paths =====
CHECKPOINT="${BLUE_STORAGE}/checkpoints/inframind/inframind_math.pt"
LATENCY_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/latency_estimator.pth"
LENGTH_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/length_estimator.pth"

# ===== Output =====
OUTPUT_CSV="logs/paper_results/math/inframind_math_test_${TEST_LIMIT}_${ARRIVAL_PATTERN}_part1.csv"

# ===== Validation =====
for f in "$CHECKPOINT" "$LATENCY_PREDICTOR" "$LENGTH_PREDICTOR"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done
if [[ ! -d "${MATH_DATASET_ROOT}/test" ]]; then
    echo "ERROR: MATH test dataset not found at ${MATH_DATASET_ROOT}/test"
    exit 1
fi

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

echo "========================================="
echo "InfraMind — MATH Test Sweep (Part 1)"
echo "========================================="
echo "Checkpoint:      ${CHECKPOINT}"
echo "Latency pred:    ${LATENCY_PREDICTOR}"
echo "Length pred:      ${LENGTH_PREDICTOR}"
echo "Test limit:      ${TEST_LIMIT}"
echo "Concurrency:     ${CONCURRENCY}"
echo "Arrival rates:   ${ARRIVAL_RATES_CSV}"
echo "Arrival pattern: ${ARRIVAL_PATTERN}"
echo "Budget sweep:    ${BUDGET_SWEEP}"
echo "Output CSV:      ${OUTPUT_CSV}"
echo "========================================="
echo ""

set +e  # Don't exit on error; we want cleanup to run

# Each (arrival_rate × budget) combination runs sequentially inside the
# Python training module's sweep loop.  --skip-training ensures no weight
# updates; --deterministic uses argmax actions for reproducibility.
python -m MAR.InfraMind.training \
    --dataset math \
    --dataset-root "${MATH_DATASET_ROOT}" \
    --split test \
    --limit "${TEST_LIMIT}" \
    --epochs 1 \
    --arrival-rates "${ARRIVAL_RATES_CSV}" \
    --arrival-pattern "${ARRIVAL_PATTERN}" \
    --budget-sweep "${BUDGET_SWEEP}" \
    --concurrency "${CONCURRENCY}" \
    --latency-predictor "${LATENCY_PREDICTOR}" \
    --length-predictor "${LENGTH_PREDICTOR}" \
    --checkpoint-path "${CHECKPOINT}" \
    --deterministic \
    --skip-training \
    --telemetry-csv "${OUTPUT_CSV}"

INFRAMIND_EXIT=$?

set -e

# ===== Summary =====
echo "========================================="
echo "InfraMind Part 1 Complete"
echo "========================================="
echo "Output CSV:  ${OUTPUT_CSV}"
echo "Exit status: ${INFRAMIND_EXIT}"
echo "End Time:    $(date)"
echo "========================================="

exit $INFRAMIND_EXIT
