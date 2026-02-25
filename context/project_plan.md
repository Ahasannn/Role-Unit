# Role-Unit: Project Plan (v3)

## 1. Problem Statement

Multi-Agent Systems (MAS) assign LLMs to functional roles (Critic, Scientist, Reflector, etc.), but there is no cheap, fast way to predict whether a given LLM is fit for a given role. We propose **Role-Unit**: domain-knowledge MCQ tests that measure role-specific competence, enabling informed role→LLM assignment without running the full pipeline.

---

## 2. Key Design Decisions

### 2.1 Domain Roles First (Milestone)
Roles split into two categories:

| Category | Roles | How to Test | Status |
|----------|-------|-------------|--------|
| **Domain roles** | Historian, Scientist, Economist, KnowledgeExpert | MMLU subjects filtered by domain — no MCQ generation needed | **Milestone** |
| **Cognitive roles** | Critic, Reflector, WikiSearcher, FinalNode | Need custom-generated MCQs testing reasoning processes | **Paper (future)** |

**Why domain first**: Clean 1-to-1 mapping between roles and MMLU subject groups. Simple, defensible, and we already have the data.

### 2.2 No MAS for Ground Truth
- Unit tests ARE the role-fitness signal
- MAS pipeline is only the downstream evaluation
- Compare: random assignment vs unit-test-guided assignment vs homogeneous baselines

### 2.3 Infrastructure
- Open-source models only, no API
- 1× B200 GPU (180GB), 2× if needed for LLM-as-judge
- sbatch (SLURM) for experiments

---

## 3. MMLU Subject → Role Mapping

57 MMLU subjects mapped to 4 domain roles:

### Historian
- high_school_european_history
- high_school_us_history
- high_school_world_history
- prehistory
- world_religions
- high_school_geography
- high_school_government_and_politics
- us_foreign_policy
- international_law
- security_studies
- sociology

### Scientist
- astronomy
- anatomy
- college_biology
- college_chemistry
- college_physics
- conceptual_physics
- high_school_biology
- high_school_chemistry
- high_school_physics
- virology
- medical_genetics
- clinical_knowledge
- college_medicine
- professional_medicine
- nutrition
- human_aging

### Economist
- econometrics
- high_school_macroeconomics
- high_school_microeconomics
- business_ethics
- professional_accounting
- management
- marketing
- public_relations

### KnowledgeExpert (General / Unassigned)
- abstract_algebra
- college_computer_science
- college_mathematics
- computer_security
- electrical_engineering
- elementary_mathematics
- formal_logic
- global_facts
- high_school_computer_science
- high_school_mathematics
- high_school_psychology
- high_school_statistics
- human_sexuality
- jurisprudence
- logical_fallacies
- machine_learning
- miscellaneous
- moral_disputes
- moral_scenarios
- philosophy
- professional_law
- professional_psychology

**NOTE**: This mapping is a first draft. Some subjects could arguably go to multiple roles. We should finalize and justify in the paper.

---

## 4. Experimental Design

### 4.1 Unit Test = Subject-Filtered MMLU
- For each domain role, the "unit test" is MMLU questions from that role's subject group
- Score = accuracy on those subject-filtered questions
- Use **validation split** (2041q) or **test split** (18738q) for scoring
- Use **dev split** (428q) for development/debugging

### 4.2 Models (Open-Source, fit on B200)
Pick 4-6 spanning sizes and families:
- Llama-3.1-70B-Instruct
- Llama-3.1-8B-Instruct
- Qwen2.5-72B-Instruct
- Mixtral-8x7B-Instruct
- Phi-3-medium-14B
- Mistral-7B-Instruct

*(Finalize based on what's available/downloadable)*

### 4.3 Experiments

#### Experiment 1: Unit Test Scoring
- Run each model on each role's subject-filtered MMLU questions
- **Output**: Score matrix S[model][role] (4-6 models × 4 roles)
- This is cheap — just single-turn MCQ answering

#### Experiment 2: Assignment Strategies
Using the score matrix, create assignments:
- **Random**: Randomly assign models to roles (repeat N times, average)
- **Unit-Test-Based**: For each role, assign the model with highest score on that role's subjects
- **Homogeneous**: Each model fills ALL roles (one config per model, serves as individual baselines)

#### Experiment 3: MAS Pipeline Evaluation
- Run the MAS pipeline on MMLU with each assignment from Experiment 2
- Compare final pipeline accuracy across strategies
- **Hypothesis**: Unit-test-based assignment ≥ best homogeneous baseline > random

### 4.4 Evaluation Metrics
- **Unit test scores**: Per-model per-role accuracy on subject-filtered MMLU
- **Pipeline accuracy**: Final answer accuracy on MMLU for each assignment strategy
- **Score discrimination**: Variance of scores across models per role (do unit tests differentiate?)
- **Assignment quality**: Does best-per-role assignment outperform homogeneous?

---

## 5. Milestone (Due ~Feb 27, 2026)

### 5.1 Report Structure (Professor's Requirements)
1. **Title**: Clear and descriptive
2. **Abstract**: Problem, approach, preliminary results (max 150 words)
3. **Introduction & Problem Statement**: Motivation, significance, objectives
4. **Related Work**: MAS role assignment, LLM evaluation, MMLU benchmarks (ACM citations)
5. **Methodology**: Subject-role mapping, unit test scoring, assignment strategies
6. **Preliminary Results**: Score matrix for 2+ models, initial pipeline comparison
7. **References**: ACM style

### 5.2 Minimum Deliverables
- [ ] Subject→role mapping finalized (this document)
- [ ] Unit test runner script (filter MMLU by subject, run model, compute accuracy)
- [ ] Score matrix for at least 2-3 models × 4 roles
- [ ] At least homogeneous pipeline baselines (1-2 models)
- [ ] Written milestone report

### 5.3 Stretch for Milestone
- [ ] Full 4-6 model score matrix
- [ ] Random vs unit-test-based assignment comparison
- [ ] Visualization (heatmap of score matrix)

---

## 6. Implementation Roadmap (4 days)

### Day 1: Unit Test Infrastructure
- [ ] Build script to partition MMLU subjects by role
- [ ] Build script to run a model on subject-filtered MMLU and compute accuracy
- [ ] Set up vLLM serving on B200 + sbatch job scripts
- [ ] Test with 1 model on 1 role (smoke test)

### Day 2: Run Unit Tests
- [ ] Serve 2-3 models on B200
- [ ] Run all models on all 4 role subject groups
- [ ] Produce score matrix

### Day 3: Pipeline Evaluation + Analysis
- [ ] Run MAS pipeline with homogeneous assignments (each model in all roles)
- [ ] Run MAS pipeline with unit-test-based assignment
- [ ] Compare results, generate plots

### Day 4: Write Milestone Report
- [ ] Write report following professor's structure
- [ ] Include score matrix, pipeline comparison, analysis

---

## 7. Future Work (Paper)

- Cognitive roles (Critic, Reflector, WikiSearcher, FinalNode) — custom MCQ generation
- More models (10+)
- More datasets beyond MMLU
- Cross-dataset validation
- Deeper analysis of when unit test scores predict vs fail to predict pipeline performance
- Human validation of subject→role mapping

---

## 8. File Structure

```
Role-Unit/
├── context/                    # Project docs
│   ├── project_plan.md         # This file (v3)
│   ├── project_summary.md      # Initial idea summary
│   └── milestone_report.md     # Milestone writeup
├── config/
│   ├── mmlu_config.yaml        # Pipeline config
│   └── role_subjects.yaml      # NEW — subject→role mapping
├── MAR/                        # Existing MAS framework
├── Datasets/MMLU/data/         # MMLU data (dev/validation/test)
├── unit_tests/                 # NEW — role unit test framework
│   ├── run_unit_tests.py       # Score models on subject-filtered MMLU
│   ├── subject_mapping.py      # MMLU subject → role mapping
│   └── results/                # Score matrices (CSV/JSON)
├── scripts/                    # NEW — sbatch job scripts
│   ├── serve_model.sh          # vLLM model serving
│   ├── run_unit_tests.sh       # Unit test batch job
│   └── run_pipeline.sh         # MAS pipeline batch job
├── analysis/                   # NEW — evaluation & plots
│   ├── compare_assignments.py  # Random vs unit-test-based
│   └── plots/
├── results/                    # Pipeline run results
└── run.py                      # Existing pipeline entry point
```
