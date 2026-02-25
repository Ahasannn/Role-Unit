#!/usr/bin/env bash
# Stop all vLLM servers started by serve_full_pool.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    LOG_DIR="${ROOT_DIR}/logs/vllm/job_${SLURM_JOB_ID}"
else
    LOG_DIR="${ROOT_DIR}/logs/vllm/job_local"
fi

echo "Stopping vLLM servers (log dir: ${LOG_DIR})..."

for pidfile in "${LOG_DIR}"/*.pid; do
    [[ -f "${pidfile}" ]] || continue
    pid=$(cat "${pidfile}")
    name=$(basename "${pidfile}" .pid)
    if kill -0 "${pid}" 2>/dev/null; then
        echo "  Stopping ${name} (pid ${pid})"
        kill "${pid}" 2>/dev/null || true
    else
        echo "  ${name} already stopped"
    fi
    rm -f "${pidfile}"
done

echo "Done."
