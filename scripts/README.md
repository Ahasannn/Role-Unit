# Scripts Directory

This directory contains SLURM job scripts, bash utilities, and dataset processing scripts for the MasRouter project.

## Directory Structure

```
scripts/
├── baseline_train/              # Training SLURM scripts
│   ├── submit_mas_train_math.slurm
│   ├── submit_mas_train_mbpp.slurm
│   ├── submit_mas_train_gsm8k.slurm
│   ├── submit_mas_train_humaneval.slurm
│   └── submit_mas_train_mmlu.slurm
│
├── motivation_plot_generator_data/  # Arrival rate sweep scripts
│   ├── baseline_mas_test_arrival_sweep_math_sustained.sh
│   ├── baseline_mas_test_arrival_sweep_mbpp_sustained.sh
│   ├── baseline_mas_test_arrival_sweep_gsm8k_sustained.sh
│   ├── baseline_mas_test_arrival_sweep_humaneval_sustained.sh
│   ├── baseline_mas_test_arrival_sweep_mmlu_sustained.sh
│   ├── submit_baseline_mas_test_arrival_sweep_math.slurm
│   ├── submit_baseline_mas_test_arrival_sweep_mbpp.slurm
│   ├── submit_baseline_mas_test_arrival_sweep_gsm8k.slurm
│   ├── submit_baseline_mas_test_arrival_sweep_humaneval.slurm
│   └── submit_baseline_mas_test_arrival_sweep_mmlu.slurm
│
├── vllm/                       # vLLM server management
│   ├── serve_full_pool.sh
│   └── stop_pool.sh
│
├── setup_hpc_env.sh           # HPC environment configuration
├── setup_jupyter_kernel.sh    # Jupyter kernel setup
├── check_vllm_status.sh       # vLLM health check
└── test_dataset_loaders.py    # Dataset loader testing script
```

## Quick Start

### 1. Train a MAS Router

Train a MAS router on a specific dataset:

```bash
# MATH dataset (12 hours)
sbatch scripts/baseline_train/submit_mas_train_math.slurm

# MBPP dataset (8 hours)
sbatch scripts/baseline_train/submit_mas_train_mbpp.slurm

# GSM8K dataset (8 hours)
sbatch scripts/baseline_train/submit_mas_train_gsm8k.slurm

# HumanEval dataset (6 hours)
sbatch scripts/baseline_train/submit_mas_train_humaneval.slurm

# MMLU dataset (10 hours)
sbatch scripts/baseline_train/submit_mas_train_mmlu.slurm
```

**Output:**
- Checkpoint: `/blue/qi855292.ucf/ah872032.ucf/checkpoints/mas_router/mas_{dataset}_train_full.pth`
- Telemetry: `logs/baseline_mas_training/{dataset}/mas_train_{dataset}_full.csv`
- SLURM logs: `logs/baseline_mas_training/{dataset}/slurm-{jobid}.out`

### 2. Run Arrival Rate Sweeps

Test trained routers under different arrival rates:

```bash
# MATH dataset
sbatch scripts/motivation_plot_generator_data/submit_baseline_mas_test_arrival_sweep_math.slurm

# MBPP dataset
sbatch scripts/motivation_plot_generator_data/submit_baseline_mas_test_arrival_sweep_mbpp.slurm

# GSM8K dataset
sbatch scripts/motivation_plot_generator_data/submit_baseline_mas_test_arrival_sweep_gsm8k.slurm

# HumanEval dataset
sbatch scripts/motivation_plot_generator_data/submit_baseline_mas_test_arrival_sweep_humaneval.slurm

# MMLU dataset
sbatch scripts/motivation_plot_generator_data/submit_baseline_mas_test_arrival_sweep_mmlu.slurm
```

**Output:**
- Telemetry: `logs/motivation_plot_generator_data/baseline_motivation_sweep_{dataset}_test_300_poisson.csv`
- SLURM logs: `logs/motivation_plot_generator_data/{dataset}/slurm-{jobid}.out`

### 3. Monitor Jobs

```bash
# Check job queue
squeue -u $USER

# Monitor real-time logs
tail -f logs/baseline_mas_training/{dataset}/slurm-{jobid}.out

# Check vLLM server status
bash scripts/check_vllm_status.sh
```

## Script Details

### Training Scripts (`baseline_train/`)

Each training script:
1. Loads HPC environment (`setup_hpc_env.sh`)
2. Activates virtual environment
3. Starts vLLM model pool (5 models on ports 8001-8005)
4. Verifies server health
5. Runs training with checkpointing
6. Auto-resumes from checkpoint if exists
7. Cleans up vLLM servers on exit

**Training Parameters:**
- Epochs: 2
- Batch size: 32
- Learning rate: 0.01
- Test limit: 16 (for validation during training)
- Cost rates:
  - MATH: 700.0
  - MBPP, GSM8K, HumanEval, MMLU: 100.0

**Resources:**
- GPUs: 2x B200
- Memory: 64GB
- CPUs: 16
- Time limits: 6-12 hours (dataset dependent)

### Sweep Scripts (`motivation_plot_generator_data/`)

Each sweep script:
1. Loads trained checkpoint
2. Tests at multiple arrival rates (2, 5, 100, 200, 300 req/min)
3. Uses Poisson arrival pattern with concurrency=1000
4. Appends results to single CSV file
5. Runs in test-only mode (--epochs 0)

**Arrival Rate Configurations:**
- Pattern: Poisson
- Rates: 2, 5, 100, 200, 300 requests/minute
- Concurrency: 1000 concurrent requests

**Test Limits:**
- MATH, MBPP, GSM8K, MMLU: 300 samples
- HumanEval: 150 samples (only 164 total available)

### Utility Scripts

#### `setup_hpc_env.sh`
Centralized HPC environment configuration:
- Sets BLUE_STORAGE path
- Configures HF_HOME for HuggingFace cache
- Exports dataset environment variables
- Loads HF_TOKEN from .env file

**Usage:**
```bash
source scripts/setup_hpc_env.sh
echo $BLUE_STORAGE
echo $HF_HOME
```

#### `check_vllm_status.sh`
Health check for vLLM servers:
- Tests all 5 model endpoints (ports 8001-8005)
- Verifies model availability
- Reports response times

**Usage:**
```bash
bash scripts/check_vllm_status.sh
```

#### `vllm/serve_full_pool.sh`
Starts all 5 vLLM model servers:
- Qwen2.5-Coder-7B-Instruct (port 8001)
- Qwen2.5-32B-Instruct (port 8002)
- Qwen2.5-3B-Instruct (port 8003)
- Qwen2.5-0.5B-Instruct (port 8004)
- Qwen2.5-1.5B-Instruct (port 8005)

**Usage:**
```bash
bash scripts/vllm/serve_full_pool.sh
```

#### `vllm/stop_pool.sh`
Stops all vLLM servers gracefully.

**Usage:**
```bash
bash scripts/vllm/stop_pool.sh
```

#### `test_dataset_loaders.py`
Tests all dataset loaders with limit parameters:
- Verifies loaders work correctly
- Tests with and without limits
- Useful before submitting SLURM jobs

**Usage:**
```bash
# Activate venv first
source .venv/bin/activate

# Run tests
python scripts/test_dataset_loaders.py
```

## Dataset-Specific Notes

### MATH
- **Path**: `/blue/qi855292.ucf/ah872032.ucf/datasets/MATH`
- **Sampling**: Stratified by category
- **Argument**: `--dataset-root`
- **Training time**: ~12 hours

### MBPP
- **Path**: `/blue/qi855292.ucf/ah872032.ucf/datasets/mbpp/full` (or HuggingFace)
- **Sampling**: Deterministic slicing
- **Argument**: None (auto-detected via MBPP_DATASET_PATH env var)
- **Training time**: ~8 hours

### GSM8K
- **Path**: HuggingFace (online)
- **Sampling**: Deterministic slicing
- **Argument**: None (loads from HuggingFace)
- **Training time**: ~8 hours

### HumanEval
- **Path**: HuggingFace (online)
- **Sampling**: Deterministic slicing
- **Argument**: None (loads from HuggingFace)
- **Training time**: ~6 hours
- **Note**: Only 164 total examples (use smaller test limits)

### MMLU
- **Path**: `/blue/qi855292.ucf/ah872032.ucf/datasets/MMLU/data`
- **Sampling**: Stratified by topic
- **Argument**: `--dataset-root`
- **Training time**: ~10 hours

## Troubleshooting

### Common Issues

#### 1. vLLM Servers Won't Start
```bash
# Check if ports are in use
netstat -tuln | grep "800[1-5]"

# Kill any stray vLLM processes
pkill -f "vllm.entrypoints.openai.api_server"

# Check logs
tail logs/vllm/*.log
```

#### 2. Checkpoint Not Found
```bash
# Verify checkpoint exists
ls -lh /blue/qi855292.ucf/ah872032.ucf/checkpoints/mas_router/

# Check permissions
stat /blue/qi855292.ucf/ah872032.ucf/checkpoints/mas_router/mas_math_train_full.pth
```

#### 3. Dataset Not Found
```bash
# Check dataset paths
echo $MATH_DATASET_ROOT
echo $MMLU_DATASET_ROOT
echo $MBPP_DATASET_PATH

# Verify directories exist
ls -lh /blue/qi855292.ucf/ah872032.ucf/datasets/
```

#### 4. Out of Memory
- Reduce batch size in SLURM script
- Check vLLM GPU memory usage
- Ensure no other jobs are running on same node

#### 5. SLURM Job Fails Immediately
```bash
# Check SLURM logs
cat logs/baseline_mas_training/{dataset}/slurm-{jobid}.err

# Verify account and partition
sinfo -p hpg-b200
sacctmgr show user $USER -s
```

### Debug Mode

To run training locally (not recommended for full training):

```bash
# Activate environment
source .venv/bin/activate
source scripts/setup_hpc_env.sh

# Start vLLM pool
bash scripts/vllm/serve_full_pool.sh

# Run training with small limits
python Experiments/run_mbpp.py \
  --epochs 1 \
  --batch_size 4 \
  --lr 0.01 \
  --train_limit 16 \
  --test_limit 8 \
  --save-checkpoint /tmp/test_checkpoint.pth

# Cleanup
bash scripts/vllm/stop_pool.sh
```

## Best Practices

1. **Always check vLLM status** before starting experiments
2. **Monitor blue storage quota** to avoid running out of space
3. **Use checkpoint resumption** to recover from failures
4. **Test with small limits** before running full training
5. **Clean up vLLM processes** after failed jobs
6. **Archive old logs** to avoid clutter

## Log Organization

```
logs/
├── baseline_mas_training/
│   ├── math/
│   │   ├── mas_train_math_full.csv        # Training telemetry
│   │   └── slurm-*.{out,err}              # SLURM logs
│   ├── mbpp/
│   ├── gsm8k/
│   ├── humaneval/
│   └── mmlu/
│
├── motivation_plot_generator_data/
│   ├── baseline_motivation_sweep_math_test_300_poisson.csv
│   ├── baseline_motivation_sweep_mbpp_test_300_poisson.csv
│   ├── baseline_motivation_sweep_gsm8k_test_300_poisson.csv
│   ├── baseline_motivation_sweep_humaneval_test_300_poisson.csv
│   ├── baseline_motivation_sweep_mmlu_test_300_poisson.csv
│   └── {dataset}/slurm-*.{out,err}
│
└── vllm/
    ├── qwen2.5-coder-7b.{log,pid}
    ├── qwen2.5-32b.{log,pid}
    ├── qwen2.5-3b.{log,pid}
    ├── qwen2.5-0.5b.{log,pid}
    └── qwen2.5-1.5b.{log,pid}
```

## Related Documentation

- [BASELINE_MAS_IMPLEMENTATION_SUMMARY.md](../BASELINE_MAS_IMPLEMENTATION_SUMMARY.md) - Implementation details
- [CLAUDE.md](../CLAUDE.md) - Project overview
- [README.md](../README.md) - Repository documentation

## Contributing

When adding new scripts:
1. Follow the existing naming convention
2. Add proper error handling and cleanup
3. Include descriptive comments
4. Update this README
5. Test on development node before submitting
