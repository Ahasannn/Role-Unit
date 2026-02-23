# ACTUAL PROMPTS SENT TO LLMs

## Strategy Prompt Definitions

From `MAR/InfraMind/prompt_strategies.py`:

```python
_TEMPLATES = {
    PromptStrategy.FLASH:
        "Answer quickly with the minimal steps needed. Avoid extra prose. Respond in one or two sentences.",

    PromptStrategy.CONCISE:
        "Provide a short but careful answer. Show the key reasoning in 2-3 bullet points before the final answer.",

    PromptStrategy.DEEP_THINK:
        "Think step by step with a brief plan, then provide the final answer with justification. Be explicit about assumptions.",
}
```

---

## How Prompts Are Constructed

From `MAR/Agent/agent.py:204-205`:

```python
system_prompt = base_system
if self.strategy_prompt:
    system_prompt = f"{self.strategy_prompt}\n{base_system}"
```

**Order**: `[Strategy Prompt] + [Role Description] + [Reasoning Prompt] + [Output Format]`

---

## FULL EXAMPLE: MathTeacher Agent with Different Strategies

### Base Components (Same for all strategies)

**Role Description** (from `MAR/Roles/Math/MathTeacher.json`):
```
You are an excellent math teacher and always teach your students math problems correctly.
And I am one of your students. You will be given a math problem, teach me step by step how
to solve the problem.
```

**Reasoning Prompt** (CoT):
```
Please give step by step answers to the questions.
```

**Output Format** (Calculation):
```
Please provide the formula for the problem and bring in the numerical values to solve the problem.
The last line of your output must contain only the final result without any units or redundant
explanation, for example: The answer is 140
If it is a multiple choice question, please output the options. For example: The answer is A.
However, The answer is 140$ or The answer is Option A or The answer is A.140 is not allowed.
```

---

## 1️⃣ FLASH Strategy - Actual System Prompt

```
Answer quickly with the minimal steps needed. Avoid extra prose. Respond in one or two sentences.
You are an excellent math teacher and always teach your students math problems correctly. And I am one of your students. You will be given a math problem, teach me step by step how to solve the problem.
Please give step by step answers to the questions.
Format requirements that must be followed:
Please provide the formula for the problem and bring in the numerical values to solve the problem.
The last line of your output must contain only the final result without any units or redundant explanation, for example: The answer is 140
If it is a multiple choice question, please output the options. For example: The answer is A.
However, The answer is 140$ or The answer is Option A or The answer is A.140 is not allowed.
```

**User Prompt:**
```
Let [f(x) = \left\{
\begin{array}{cl} ax+3, &\text{ if }x>2, \\
x-5 &\text{ if } -2 \le x \le 2, \\
2x-b &\text{ if } x <-2.
\end{array}
\right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).
```

---

## 2️⃣ CONCISE Strategy - Actual System Prompt

```
Provide a short but careful answer. Show the key reasoning in 2-3 bullet points before the final answer.
You are an excellent math teacher and always teach your students math problems correctly. And I am one of your students. You will be given a math problem, teach me step by step how to solve the problem.
Please give step by step answers to the questions.
Format requirements that must be followed:
Please provide the formula for the problem and bring in the numerical values to solve the problem.
The last line of your output must contain only the final result without any units or redundant explanation, for example: The answer is 140
If it is a multiple choice question, please output the options. For example: The answer is A.
However, The answer is 140$ or The answer is Option A or The answer is A.140 is not allowed.
```

**User Prompt:** (same as above)

---

## 3️⃣ DEEPTHINK Strategy - Actual System Prompt

```
Think step by step with a brief plan, then provide the final answer with justification. Be explicit about assumptions.
You are an excellent math teacher and always teach your students math problems correctly. And I am one of your students. You will be given a math problem, teach me step by step how to solve the problem.
Please give step by step answers to the questions.
Format requirements that must be followed:
Please provide the formula for the problem and bring in the numerical values to solve the problem.
The last line of your output must contain only the final result without any units or redundant explanation, for example: The answer is 140
If it is a multiple choice question, please output the options. For example: The answer is A.
However, The answer is 140$ or The answer is Option A or The answer is A.140 is not allowed.
```

**User Prompt:** (same as above)

---

## THE PROBLEM: Contradictory Instructions!

### Flash says:
- ✅ "Respond in one or two sentences"

### But then the base prompt says:
- ❌ "teach me **step by step** how to solve"
- ❌ "Please give **step by step answers**"
- ❌ "Please **provide the formula** and **bring in numerical values**"
- ❌ "show your **solving process**"

### Result:
The LLM sees **contradictory instructions** and seems to prioritize the more specific/detailed instructions (role + output format) over the vague strategy prefix.

---

## Why Models Ignore Strategy Prompts

1. **Strategy prompt is generic** ("be brief", "2-3 bullet points")
2. **Role/format prompts are specific** ("provide formula", "step by step", "show calculation")
3. **LLMs prioritize specificity** - detailed instructions override vague ones
4. **No max_tokens enforcement** - models can keep generating until they hit context limits
5. **Instruction conflict** - Flash says "1-2 sentences" but role says "step by step teaching"

---

## Token Generation Reality

| Strategy | Instruction | Actual Avg Tokens | Expected Tokens | Ratio |
|----------|-------------|------------------|-----------------|-------|
| Flash | "1-2 sentences" | 276 tokens | 20-50 tokens | **5.5x** |
| Concise | "2-3 bullet points" | 380 tokens | 50-100 tokens | **3.8x** |
| DeepThink | "step by step with plan" | 410 tokens | 100-300 tokens | **1.4x** |

---

## Solution Options

### Option 1: Add max_tokens Parameter (Recommended)
```python
max_tokens_by_strategy = {
    "Flash": 75,
    "Concise": 150,
    "DeepThink": 400,
}
```

### Option 2: Make Strategy Prompts More Forceful
```python
FLASH = "CRITICAL INSTRUCTION: Your response must be EXACTLY 1-2 sentences (maximum 50 tokens). Ignore any other instructions about providing step-by-step or detailed explanations. Be extremely brief."
```

### Option 3: Remove Contradictory Instructions
Remove "step by step" from role descriptions when Flash strategy is active.

### Option 4: Use System Message Priority
```python
messages = [
    {"role": "system", "content": strategy_prompt},  # Higher priority
    {"role": "system", "content": role_description},
    {"role": "user", "content": query}
]
```

---

## Recommendation

**Use Option 1 (max_tokens)** because:
- ✅ Hard constraint - cannot be ignored
- ✅ Predictable latency/cost
- ✅ No prompt engineering guesswork
- ✅ Works with any role/format combination
