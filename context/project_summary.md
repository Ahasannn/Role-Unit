# Role-Unit: Project Summary (from Claude Web Chat)

## Problem
Multi-Agent Systems assign LLMs to functional roles (Critic, Scientist, Reflector, etc.). Today, the only way to know if a model is fit for a role is to run the full expensive pipeline. There's no cheap, fast way to test role fitness beforehand.

## Idea
Build a discriminative (MCQ-based) unit test for each role. Instead of asking models to generate answers, we test whether they can identify correct answers among distractors — making evaluation fast, deterministic, and cheap. Crucially, each role's test measures that role's cognitive skill (e.g., Critic = flaw detection, Reflector = error correction), not just general knowledge.

## Validation
Run the same LLMs in actual MAS pipelines on MMLU, measure per-role performance, and check if our unit test scores correlate with real pipeline performance.

## Scope
- MMLU dataset, 7 roles from MasRouter (KnowledgeExpert, Reflector, Critic, Historian, WikiSearcher, Scientist, Economist + FinalNode)
- 4 LLMs from MasRouter's model pool
- Codebase forked from MasRouter, stripped to MMLU-only

## Needs
- Finalizing system design and experimental plan
- Pipeline architecture
- How to measure per-role performance
- Exam design strategy
- Evaluation metrics
- Experiment structure
