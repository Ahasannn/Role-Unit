# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**Role-Unit** is a benchmark that tests whether LLMs are fit for specific functional roles in multi-agent systems, using the MMLU dataset.

MMLU questions run through a fixed chain of LLM roles (KnowledgeExpert, Reflector, Critic, Historian, WikiSearcher, Scientist, Economist) followed by a FinalNode that aggregates outputs. Accuracy is measured per-question. There is NO routing/training — role→LLM assignments are fixed in config.

## Build & Environment Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --frozen
cp template.env .env  # Add API keys
```

## Running

```bash
python run.py                         # defaults from config/mmlu_config.yaml
python run.py --limit 100 --split dev # quick test
python run.py --config config/mmlu_config.yaml
python run.py --output results/my_run.csv
```

## Architecture

### `run.py`
Main entry point. Loads config, dataset, builds a Graph per question, runs it, collects results to CSV.

### `config/mmlu_config.yaml`
Single config file: dataset split/limit, topology, role→LLM assignments, output paths.

### `Datasets/mmlu_dataset.py`
MMLU loader with stratified sampling. Loads from `Datasets/MMLU/data/{split}/`.

### `MAR/` — Core Execution Framework

**MAR/Graph/** — Execution engine
- `node.py`: Base Node class (spatial/temporal connections, execute/async_execute)
- `graph.py`: Graph class — builds topology, runs nodes in topological order

**MAR/Agent/**
- `agent.py`: `Agent` (calls LLM with role prompt) and `FinalRefer` (aggregates outputs)
- `agent_registry.py`: Registry for agent types
- `reasoning_profile.py`: Topology reasoning descriptions

**MAR/LLM/**
- `gpt_chat.py`: OpenAI-compatible LLM clients (`ALLChat`, `DSChat`, `GroqChat`, `OpenRouterChat`)
- `llm_registry.py`: LLM instance cache
- `llm_profile_full.py` + `llm_profile_full.json`: Model context length / token limits
- `price.py`: Token counting and cost tracking

**MAR/Roles/**
- `Commonsense/`: 7 MMLU role JSON files (Name, Description, OutputFormat, PostProcess)
- `FinalNode/mmlu.json`: FinalNode system + user prompt
- `role_registry.py`: Loads role JSON by domain + role name

**MAR/Prompts/**
- `message_aggregation.py`: Aggregates predecessor node outputs for the user prompt
- `output_format.py`: Format instruction strings keyed by format name
- `post_process.py`: Post-processing (only "None", "Wiki", "Reflection" used for MMLU)
- `reasoning.py`: Reasoning mode prompts

**MAR/Utils/**
- `globals.py`: Singleton counters for cost/tokens
- `telemetry.py`: LLMUsageTracker and CsvTelemetryWriter
- `utils.py`: `get_kwargs()` for building topology masks, `find_mode()`, etc.
- `log.py`: Loguru configuration and ProgressTracker

## Key Concepts

### Fixed Chain (no routing)
`get_kwargs("Chain", N)` produces a sequential spatial mask so nodes execute in order: node 0 → node 1 → ... → node N-1 → FinalNode.

### Role JSON format
```json
{
  "Name": "KnowledgeExpert",
  "MessageAggregation": "Normal",
  "Description": "You are a knowledgeable expert...",
  "OutputFormat": "Answer",
  "PostProcess": "None",
  "PostDescription": "None",
  "PostOutputFormat": "None"
}
```

### Adding a new role
1. Create `MAR/Roles/Commonsense/MyRole.json`
2. Add `{role: MyRole, llm: model-name}` to `config/mmlu_config.yaml`

### Changing LLM backend
Set `URL` and `KEY` in `.env`, or use `MODEL_BASE_URLS` as a JSON dict for per-model URLs.

## File Locations

- Results: `results/mmlu_<timestamp>.csv`
- MMLU data: `Datasets/MMLU/data/{split}/*.csv`
- Logs: `logs/` (auto-created by loguru)
