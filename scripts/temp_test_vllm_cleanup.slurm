#!/bin/bash
#SBATCH --job-name=temp_vllm_cleanup_test
#SBATCH --output=logs/paper_results/slurm-%j-cleanup-test.out
#SBATCH --error=logs/paper_results/slurm-%j-cleanup-test.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

# ===== TEMP: Test job-isolated vLLM cleanup =====
# Starts all vLLM servers, waits for healthy, then cleans up using
# job-specific PID files only. Verifies no stray processes remain.

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="
echo ""

REPO_ROOT="/home/ah872032.ucf/system-aware-mas"
cd "$REPO_ROOT" || exit 1

source scripts/setup_hpc_env.sh
source .venv/bin/activate || exit 1

export KEY="EMPTY"

mkdir -p logs/paper_results

# ===== Cleanup function (job-isolated) =====
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
    else
        echo "No PID files found in ${vllm_log_dir}"
    fi

    echo "Cleanup complete"
    echo "========================================="
}
trap cleanup_vllm EXIT INT TERM

# ===== Start vLLM =====
echo "Starting vLLM model pool..."
bash scripts/vllm/serve_full_pool.sh || { echo "ERROR: Failed to start vLLM"; exit 1; }
echo ""
echo "All vLLM servers healthy!"
bash scripts/check_vllm_status.sh
echo ""

# ===== Show PID files =====
echo "========================================="
echo "PID files for this job:"
echo "========================================="
ls -la "logs/vllm/job_${SLURM_JOB_ID}"/*.pid 2>/dev/null || echo "No PID files found!"
echo ""

# ===== Show vLLM processes BEFORE cleanup =====
echo "========================================="
echo "vLLM processes BEFORE cleanup:"
echo "========================================="
ps aux | grep "vllm.entrypoints.openai.api_server" | grep -v grep || echo "None found"
echo ""

# ===== Trigger cleanup =====
echo "Triggering cleanup now..."
cleanup_vllm
echo ""

# ===== Show vLLM processes AFTER cleanup =====
sleep 3
echo "========================================="
echo "vLLM processes AFTER cleanup:"
echo "========================================="
ps aux | grep "vllm.entrypoints.openai.api_server" | grep -v grep || echo "None found - cleanup worked!"
echo ""

echo "========================================="
echo "Test complete: $(date)"
echo "========================================="
