#!/usr/bin/env bash
set -euo pipefail

# serve_full_pool.sh
# Serves all models from llm_profile_full.json on 2 B200 GPUs.
# Models are started SEQUENTIALLY per GPU with health checks between each,
# so vLLM correctly sizes KV cache around already-loaded models.
# Adapted from system-aware-mas/scripts/vllm/serve_full_pool.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load HPC environment
if [[ -f "${ROOT_DIR}/scripts/setup_hpc_env.sh" ]]; then
    source "${ROOT_DIR}/scripts/setup_hpc_env.sh"
else
    echo "[ERROR] setup_hpc_env.sh not found"
    exit 1
fi

# Job-specific log directory
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    LOG_DIR="${ROOT_DIR}/logs/vllm/job_${SLURM_JOB_ID}"
else
    LOG_DIR="${ROOT_DIR}/logs/vllm/job_local"
fi
mkdir -p "${LOG_DIR}"

echo "[Setup] HF Cache:    ${HF_HOME}"
echo "[Setup] Logs:        ${LOG_DIR}"

# Load CUDA if needed
if ! command -v nvcc &> /dev/null; then
    echo "[Setup] Loading CUDA module..."
    module load cuda/12.8.1
fi

# Python interpreter
VLLM_PYTHON="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${VLLM_PYTHON}" ]]; then
    echo "[ERROR] Python venv not found at ${VLLM_PYTHON}"
    exit 1
fi
export PATH="${ROOT_DIR}/.venv/bin:${PATH}"

LLM_PROFILE_JSON="${ROOT_DIR}/config/llm_profile_full.json"
VLLM_HOST="0.0.0.0"
VLLM_ENTRYPOINT=("${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server)

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
get_model_count() {
    "${VLLM_PYTHON}" -c "
import json; data = json.load(open('${LLM_PROFILE_JSON}'))
print(len(data.get('models', [])))
"
}

get_model_field() {
    local index="$1" field="$2"
    "${VLLM_PYTHON}" -c "
import json; data = json.load(open('${LLM_PROFILE_JSON}'))
models = data.get('models', [])
if ${index} < len(models):
    m = models[${index}]
    print(m.get('${field}', m.get('vllm_config', {}).get('${field}', '')))
else:
    print('')
"
}

get_default_max_model_len() {
    "${VLLM_PYTHON}" -c "
import json; data = json.load(open('${LLM_PROFILE_JSON}'))
print(data.get('global_settings', {}).get('default_max_model_len', 4096))
"
}

# ---------------------------------------------------------------------------
# Crash detection — check vLLM logs for fatal initialization errors
# ---------------------------------------------------------------------------
detect_startup_failure() {
    local logfile="$1"

    if [[ ! -f "${logfile}" ]]; then
        return 1  # No log file, can't detect
    fi

    if tail -100 "${logfile}" 2>/dev/null | grep -qE "(ERROR.*EngineCore failed|ValueError.*cache blocks|RuntimeError.*initialization failed|Available KV cache memory: -|EngineCore failed to start)"; then
        return 0  # Crash detected
    fi

    return 1  # No crash detected
}

# ---------------------------------------------------------------------------
# Health check with crash detection (from system-aware-mas)
# ---------------------------------------------------------------------------
wait_for_health() {
    local name="$1" port="$2"
    local pidfile="${LOG_DIR}/${name}.pid"
    local logfile="${LOG_DIR}/${name}.log"
    local timeout_s="${VLLM_STARTUP_TIMEOUT_SECONDS:-7200}"
    local start_s=$(date +%s)

    while true; do
        # Check if process died
        if [[ -f "${pidfile}" ]] && ! kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
            if detect_startup_failure "${logfile}"; then
                echo "[vLLM] ${name} crashed during initialization (see ${logfile})"
                return 2  # Special code: needs retry
            else
                echo "[vLLM] ${name} failed to start (see ${logfile})"
                return 1  # Permanent failure
            fi
        fi

        # Check if healthy
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            echo "[vLLM] ${name} healthy on port ${port}"
            return 0
        fi

        # Check timeout
        if (( $(date +%s) - start_s > timeout_s )); then
            echo "[vLLM] ${name} did not become healthy within ${timeout_s}s (see ${logfile})"
            return 1
        fi

        sleep 2
    done
}

# ---------------------------------------------------------------------------
# Start a single model server
# ---------------------------------------------------------------------------
start_server() {
    local index="$1"
    local model_name=$(get_model_field "$index" "Name")
    local port=$(get_model_field "$index" "port")
    local gpu_device=$(get_model_field "$index" "gpu_device")
    local gpu_mem=$(get_model_field "$index" "gpu_memory_utilization")
    local max_model_len=$(get_model_field "$index" "MaxModelLen")
    local dtype=$(get_model_field "$index" "dtype")
    local trust_remote=$(get_model_field "$index" "trust_remote_code")
    local enforce_eager=$(get_model_field "$index" "enforce_eager")
    local tp_size=$(get_model_field "$index" "tensor_parallel_size")

    max_model_len="${max_model_len:-$(get_default_max_model_len)}"
    dtype="${dtype:-bfloat16}"
    tp_size="${tp_size:-1}"

    local server_name=$(echo "$model_name" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_')
    local logfile="${LOG_DIR}/${server_name}.log"
    local pidfile="${LOG_DIR}/${server_name}.pid"

    # Skip if already running
    if [[ -f "${pidfile}" ]] && kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
        echo "[vLLM] ${server_name} already running (pid $(cat "${pidfile}"))"
        return 0
    fi

    echo "[vLLM] Starting ${server_name}"
    echo "  Model: ${model_name}"
    echo "  GPU:   ${gpu_device}"
    echo "  Port:  ${port}"
    echo "  Memory Utilization: ${gpu_mem}"
    echo "  Max Model Len: ${max_model_len}"
    echo "  Log File: ${logfile}"

    local extra_flags=()
    if [[ "${trust_remote}" == "true" || "${trust_remote}" == "True" ]]; then
        extra_flags+=(--trust-remote-code)
    fi
    if [[ "${enforce_eager}" == "true" || "${enforce_eager}" == "True" ]]; then
        extra_flags+=(--enforce-eager)
    fi

    # VLLM_USE_V1=0: Force v0 engine — v1 counts ALL GPU memory (including other
    # processes) when sizing KV cache, causing OOM when multiple models share a GPU.
    # v0 only counts memory used by THIS process since init.
    VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES="${gpu_device}" nohup "${VLLM_ENTRYPOINT[@]}" \
        --host "${VLLM_HOST}" \
        --port "${port}" \
        --model "${model_name}" \
        --served-model-name "${model_name}" \
        --dtype "${dtype}" \
        --gpu-memory-utilization "${gpu_mem}" \
        --max-model-len "${max_model_len}" \
        --tensor-parallel-size "${tp_size}" \
        --no-enable-prefix-caching \
        --block-size 32 \
        --max-num-seqs 16 \
        --swap-space 16 \
        --disable-log-requests \
        --uvicorn-log-level warning \
        "${extra_flags[@]}" \
        >"${logfile}" 2>&1 &

    echo "$!" > "${pidfile}"
    echo "[vLLM] ${server_name} pid $! (log: ${logfile})"
}

# ---------------------------------------------------------------------------
# Main: Start all models SEQUENTIALLY with retry logic
# ---------------------------------------------------------------------------
MODEL_COUNT=$(get_model_count)
echo "[vLLM] Found ${MODEL_COUNT} models to serve"
echo ""

MAX_RETRIES=3

for (( i=0; i<MODEL_COUNT; i++ )); do
    model_name="$(get_model_field "${i}" "Name")"
    server_name="$(echo "${model_name}" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_')"
    port="$(get_model_field "${i}" "port")"

    retry_count=0
    success=0

    while (( retry_count < MAX_RETRIES )); do
        # Start the server
        start_server "${i}"

        # Wait for health — blocks until healthy, crashed, or timeout
        wait_for_health "${server_name}" "${port}"
        health_status=$?

        if [[ ${health_status} -eq 0 ]]; then
            success=1
            break
        elif [[ ${health_status} -eq 2 ]]; then
            # Crash during init (likely OOM) — retry
            retry_count=$((retry_count + 1))
            if (( retry_count < MAX_RETRIES )); then
                wait_time=$((retry_count * 5))
                echo "[vLLM] ${server_name} retry ${retry_count}/${MAX_RETRIES} — waiting ${wait_time}s for GPU memory cleanup..."

                # Kill the crashed process
                pidfile="${LOG_DIR}/${server_name}.pid"
                if [[ -f "${pidfile}" ]]; then
                    pid=$(cat "${pidfile}")
                    kill -9 "${pid}" 2>/dev/null || true
                    rm -f "${pidfile}"
                fi

                sleep "${wait_time}"
                echo "[vLLM] Retrying ${server_name}..."
            fi
        else
            # Permanent failure
            echo "[vLLM] ERROR: ${server_name} failed permanently"
            exit 1
        fi
    done

    if [[ ${success} -eq 0 ]]; then
        echo "[vLLM] ERROR: ${server_name} failed after ${MAX_RETRIES} retries"
        echo "[vLLM] Log: ${LOG_DIR}/${server_name}.log"
        exit 1
    fi

    echo ""
done

echo ""
echo "[vLLM] All ${MODEL_COUNT} models serving!"
echo ""
echo "[Model endpoints]"
"${VLLM_PYTHON}" -c "
import json
data = json.load(open('${LLM_PROFILE_JSON}'))
for name, url in data.get('model_base_urls', {}).items():
    print(f'  {name}: {url}')
"
