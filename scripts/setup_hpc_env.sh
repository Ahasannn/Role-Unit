#!/bin/bash
# setup_hpc_env.sh
# Centralized HPC environment configuration for all SLURM jobs
# Source this at the beginning of any sbatch script:
#   source scripts/setup_hpc_env.sh

# ===== STORAGE CONFIGURATION =====
export BLUE_STORAGE="/blue/qi855292.ucf/ah872032.ucf"
export ORANGE_STORAGE="/orange/qi855292.ucf/ah872032.ucf"

# ===== HUGGING FACE CONFIGURATION =====
# All model/cache data goes to orange storage (more space, avoids blue congestion)
export HF_HOME="${ORANGE_STORAGE}/huggingface_cache"

# Read token from orange storage (primary) or fallback to home directory
if [[ -f "${HF_HOME}/token" ]]; then
    export HF_TOKEN=$(cat "${HF_HOME}/token")
elif [[ -f ~/.cache/huggingface/token ]]; then
    export HF_TOKEN=$(cat ~/.cache/huggingface/token)
elif [[ -f ~/.huggingface/token ]]; then
    export HF_TOKEN=$(cat ~/.huggingface/token)
fi

if [[ -z "${HF_TOKEN}" ]]; then
    echo "WARNING: HF_TOKEN not found in ${HF_HOME}/token or ~/.cache/huggingface/token"
fi

# ===== PYTORCH / TORCH CACHE CONFIGURATION =====
export TORCH_HOME="${ORANGE_STORAGE}/torch_cache"
export TRITON_CACHE_DIR="${ORANGE_STORAGE}/triton_cache"
export TRITON_HOME="${ORANGE_STORAGE}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${ORANGE_STORAGE}/torchinductor_cache"
export TORCH_EXTENSIONS_DIR="${ORANGE_STORAGE}/torch_extensions"

# ===== TEMPORARY DIRECTORIES =====
# Critical: Override TMPDIR to prevent writes to /scratch/local
export TMPDIR="${ORANGE_STORAGE}/tmp"
export TEMP="${ORANGE_STORAGE}/tmp"
export TMP="${ORANGE_STORAGE}/tmp"

# ===== GENERAL CACHE LOCATIONS =====
export XDG_CACHE_HOME="${ORANGE_STORAGE}/cache"

# ===== CREATE DIRECTORIES =====
mkdir -p "${HF_HOME}" \
         "${TORCH_HOME}" \
         "${TRITON_CACHE_DIR}" \
         "${TORCHINDUCTOR_CACHE_DIR}" \
         "${TORCH_EXTENSIONS_DIR}" \
         "${TMPDIR}" \
         "${XDG_CACHE_HOME}"

# ===== HUGGING FACE SETTINGS =====
export TOKENIZERS_PARALLELISM="false"

# ===== DISPLAY CONFIGURATION =====
if [[ "${VERBOSE_ENV:-0}" == "1" ]]; then
    echo "========================================="
    echo "HPC Environment Configuration"
    echo "========================================="
    echo "Blue Storage:           ${BLUE_STORAGE}"
    echo "Orange Storage:         ${ORANGE_STORAGE}"
    echo "HF_HOME:                ${HF_HOME}"
    echo "HF_TOKEN:               $(if [[ -n "${HF_TOKEN}" ]]; then echo "✓ Set"; else echo "✗ Not found"; fi)"
    echo "TORCH_HOME:             ${TORCH_HOME}"
    echo "TRITON_CACHE_DIR:       ${TRITON_CACHE_DIR}"
    echo "TMPDIR:                 ${TMPDIR}"
    echo "XDG_CACHE_HOME:         ${XDG_CACHE_HOME}"
    echo "========================================="
fi
