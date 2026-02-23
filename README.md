# Role-Unit

**Role-Unit** benchmarks whether LLMs are fit for specific functional roles in multi-agent systems, using the MMLU dataset.

## What it does

Run MMLU multiple-choice questions through a fixed sequential pipeline of LLM roles:

```
Question → KnowledgeExpert → Reflector → Critic → Historian
         → WikiSearcher → Scientist → Economist → FinalNode → Answer (A/B/C/D)
```

Each role is assigned a specific LLM (any OpenAI-compatible model). The FinalNode aggregates all role outputs and selects the final answer. Results are saved as CSV with per-question accuracy.

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --frozen
cp template.env .env
# Edit .env: set URL and KEY for your LLM backend
```

## Running

```bash
# Run with defaults (config/mmlu_config.yaml, test split, all questions)
python run.py

# Limit to 100 questions from the dev split
python run.py --limit 100 --split dev

# Custom config
python run.py --config config/mmlu_config.yaml

# Save to specific CSV
python run.py --output results/my_run.csv
```

## Configuration

Edit `config/mmlu_config.yaml` to assign different LLMs to roles:

```yaml
roles:
  - role: KnowledgeExpert
    llm: gpt-4o-mini
  - role: Reflector
    llm: gpt-4o        # stronger model for reflection
  - role: Critic
    llm: gpt-4o-mini
  # ...

final_node:
  llm: gpt-4o
  prompt_file: MAR/Roles/FinalNode/mmlu.json
```

Topology options: `Chain`, `FullConnected`, `Debate`.

## Roles

Role definitions live in `MAR/Roles/Commonsense/`:

| Role | Description |
|------|-------------|
| `KnowledgeExpert` | Broad knowledge QA |
| `Reflector` | Reflects on previous answers |
| `Critic` | Finds errors in reasoning |
| `Historian` | Historical context |
| `WikiSearcher` | Wikipedia keyword search |
| `Scientist` | Scientific reasoning |
| `Economist` | Economic reasoning |

## Output

Results saved to `results/mmlu_<timestamp>.csv`:

| Column | Description |
|--------|-------------|
| `item_id` | Question index |
| `question` | Question text (truncated) |
| `gold` | Ground truth answer (A/B/C/D) |
| `pred` | Predicted answer |
| `correct` | 1 if correct, 0 otherwise |
| `latency_sec` | Wall-clock time for pipeline |
| `roles` | Role names (JSON list) |
| `llms` | LLM names (JSON list) |

## Structure

```
role-unit/
├── config/
│   └── mmlu_config.yaml      # Pipeline configuration
├── Datasets/
│   └── mmlu_dataset.py       # MMLU loader (auto-downloads from HuggingFace)
├── MAR/
│   ├── Agent/                # Agent and FinalRefer implementations
│   ├── Graph/                # Graph execution engine
│   ├── LLM/                  # OpenAI-compatible LLM wrappers
│   ├── Prompts/              # Prompt utilities
│   ├── Roles/
│   │   ├── Commonsense/      # 7 MMLU role JSON definitions
│   │   └── FinalNode/        # mmlu.json aggregator prompt
│   └── Utils/                # Logging, cost tracking
├── results/                  # CSV output
└── run.py                    # Main entry point
```

## MMLU Data

Place MMLU CSV files under `Datasets/MMLU/data/{split}/` (e.g., `Datasets/MMLU/data/test/*.csv`).
The dataset auto-downloads from HuggingFace on first use if not present locally.
