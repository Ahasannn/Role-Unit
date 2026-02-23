#!/bin/bash
#SBATCH --job-name=temp_baseline_rerun
#SBATCH --output=logs/paper_results/slurm-%j-baseline-rerun.out
#SBATCH --error=logs/paper_results/slurm-%j-baseline-rerun.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1

# ===== TEMP: Rerun missing cost=400 and cost=700 configs =====
# Appends to existing CSV: logs/paper_results/math/baseline_math_test_1000_poisson.csv

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="
echo ""

REPO_ROOT="/home/ah872032.ucf/system-aware-mas"
cd "$REPO_ROOT" || exit 1

source scripts/setup_hpc_env.sh
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo ""

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"
export CODE_EXECUTION_TIMEOUT="10"
export MATH_DATASET_ROOT="${BLUE_STORAGE}/datasets/MATH"

mkdir -p logs/paper_results/math

# ===== Run settings =====
TEST_LIMIT=1000
CONCURRENCY=1000

# ===== Load arrival rates from shared config =====
DATASET_CONFIG="${REPO_ROOT}/Experiments/dataset_config.json"
ARRIVAL_PATTERN=$(python3 -c "import json; print(json.load(open('${DATASET_CONFIG}'))['math']['arrival_pattern'])")
ARRIVAL_RATES=$(python3 -c "import json; print(' '.join(str(r) for r in json.load(open('${DATASET_CONFIG}'))['math']['arrival_rates']))")

# ===== RERUN: Only cost=400 and cost=700 =====
COST_RATES=(400 700)
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"

# ===== Output (append to existing) =====
OUTPUT_CSV="logs/paper_results/math/baseline_math_test_${TEST_LIMIT}_${ARRIVAL_PATTERN}.csv"

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

# ===== Validation =====
for cost in "${COST_RATES[@]}"; do
    CKPT="${CHECKPOINT_DIR}/mas_math_train_519_cost${cost}.pth"
    if [[ ! -f "${CKPT}" ]]; then
        echo "ERROR: Checkpoint not found: ${CKPT}"
        exit 1
    fi
done

echo "========================================="
echo "TEMP Rerun: cost=400, cost=700"
echo "========================================="
echo "Cost rates:    ${COST_RATES[*]}"
echo "Arrival rates: ${ARRIVAL_RATES}"
echo "Pattern:       ${ARRIVAL_PATTERN}"
echo "Test limit:    ${TEST_LIMIT}"
echo "Concurrency:   ${CONCURRENCY}"
echo "Output CSV:    ${OUTPUT_CSV} (append)"
echo "========================================="
echo ""

set +e

OVERALL_EXIT=0

for cost in "${COST_RATES[@]}"; do
    CHECKPOINT="${CHECKPOINT_DIR}/mas_math_train_519_cost${cost}.pth"

    for arrival_rate in $ARRIVAL_RATES; do
        echo "--- cost_rate=${cost}  arrival_rate=${arrival_rate}  pattern=${ARRIVAL_PATTERN} ---"

        python Experiments/run_math.py \
            --epochs 0 \
            --checkpoint "${CHECKPOINT}" \
            --cost_rate "${cost}" \
            --test_limit "${TEST_LIMIT}" \
            --concurrency "${CONCURRENCY}" \
            --arrival-rate "${arrival_rate}" \
            --arrival-pattern "${ARRIVAL_PATTERN}" \
            --test-telemetry-csv "${OUTPUT_CSV}"

        EXIT_CODE=$?
        if [[ $EXIT_CODE -ne 0 ]]; then
            echo "WARNING: cost_rate=${cost} arrival_rate=${arrival_rate} exited with code ${EXIT_CODE}"
            OVERALL_EXIT=1
        fi

        echo "Completed cost_rate=${cost} arrival_rate=${arrival_rate}"
        echo ""
    done
done

set -e

# ===== Summary =====
echo "========================================="
echo "TEMP Rerun Complete"
echo "========================================="
echo "Output CSV:  ${OUTPUT_CSV}"
echo "Exit status: ${OVERALL_EXIT}"
echo "End Time:    $(date)"
echo "========================================="

exit $OVERALL_EXIT
