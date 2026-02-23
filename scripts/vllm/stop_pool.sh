#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/vllm"

stop_server() {
  local name="$1"
  local pidfile="${LOG_DIR}/${name}.pid"

  if [[ ! -f "${pidfile}" ]]; then
    echo "[vLLM] ${name}: no pidfile (${pidfile})"
    return 0
  fi

  local pid
  pid="$(cat "${pidfile}")"

  if kill -0 "${pid}" 2>/dev/null; then
    echo "[vLLM] Stopping ${name} (pid ${pid})"
    kill "${pid}"
  else
    echo "[vLLM] ${name}: process not running (pid ${pid})"
  fi

  rm -f "${pidfile}"
}

stop_server "deepseek_r1_distill_qwen_32b_"
stop_server "mistral_small_24b_instruct_2501_"
stop_server "qwen2_5_coder_14b_instruct_"
stop_server "llama_3_1_8b_instruct_"
stop_server "llama_3_2_3b_instruct_"

