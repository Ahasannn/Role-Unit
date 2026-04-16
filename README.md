# Role-Unit

A training-free framework for assigning LLMs to roles in multi-agent systems. Instead of training a router, Role-Unit measures each model's fitness on each role's domain, then solves the assignment as an Integer Linear Program (Multiple-Choice Knapsack Problem) to find accuracy-cost tradeoffs.

**How it works:**
1. Run every candidate model on every role's domain-specific questions (fitness testing)
2. Build a fitness matrix (accuracy) and cost matrix (inference cost)
3. Solve the ILP at different budget levels to get a Pareto frontier of assignments

No gradient training. Adding a new model = one validation pass + sub-second re-solve.

## Results

Evaluated on MMLU with 6 open-weight models (3B to 24B parameters) across 3 domain roles and compared against learned routers:

| Method | Accuracy | Training Cost |
|--------|----------|---------------|
| CARROT | 75.5% | $0.042 |
| GraphRouter | 76.6% | $42.13 |
| RouteLLM | 78.7% | $4.21 |
| **Role-Unit P7** | 77.7% | **$0.042** |
| **Role-Unit P10** | **78.7%** | **$0.042** |

Role-Unit P10 matches the best learned router at 100x lower training cost. P7 is one point behind but cuts inference cost by 35%.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --frozen

cp template.env .env
# Edit .env: set URL and KEY for your OpenAI-compatible LLM backend
```

MMLU data goes under `Datasets/MMLU/data/{split}/` (e.g., `Datasets/MMLU/data/test/*.csv`).

## Running

**Run the multi-agent pipeline** with a fixed role-to-LLM assignment:

```bash
python run.py                                      # defaults from config/mmlu_config.yaml
python run.py --limit 100 --split dev              # quick test
python run.py --output results/my_run.csv          # custom output path
```

**Run baselines** (homogeneous and random assignments):

```bash
python run_baseline.py --mode both --split test --limit 500 --concurrency 64
python run_baseline.py --mode homogeneous --split test --limit 500
python run_baseline.py --mode random --n-trials 10 --split test --limit 500
```

**Run unit tests** (per-role fitness evaluation):

```bash
python unit_tests/run_unit_tests.py
```

**Generate plots:**

```bash
python visualization/plot_results.py
```

## Configuration

`config/mmlu_config.yaml` controls the pipeline:

```yaml
roles:
  - role: Historian
    llm: gpt-4o-mini
  - role: Scientist
    llm: gpt-4o-mini
  - role: Economist
    llm: gpt-4o-mini

final_node:
  llm: gpt-4o-mini
  prompt_file: MAR/Roles/FinalNode/mmlu.json
```

`config/role_subjects.yaml` maps MMLU subjects to domain roles (Historian, Scientist, Economist).

## Project Structure

```
Role-Unit/
├── run.py                  # Main pipeline entry point
├── run_baseline.py         # Homogeneous & random baseline runner
├── config/
│   ├── mmlu_config.yaml    # Pipeline config (role→LLM assignments)
│   ├── role_subjects.yaml  # MMLU subject→role mapping
│   ├── llm_profile_full.json  # Model pool config (vLLM serving)
│   └── model_costs.json    # Token pricing
├── Datasets/
│   └── mmlu_dataset.py     # MMLU data loader
├── MAR/
│   ├── Agent/              # Agent + FinalRefer (aggregator)
│   ├── Graph/              # Graph execution engine
│   ├── LLM/                # OpenAI-compatible LLM clients
│   ├── Prompts/            # Prompt construction utilities
│   ├── Roles/              # Role definitions (JSON)
│   └── Utils/              # Logging, cost tracking, telemetry
├── unit_tests/             # Per-role fitness evaluation
├── visualization/          # Result plotting
└── scripts/                # SLURM job scripts (HPC)
```
