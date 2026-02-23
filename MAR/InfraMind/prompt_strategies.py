from enum import Enum
from typing import Dict, List, Optional


class PromptStrategy(str, Enum):
    FLASH = "Flash"
    CONCISE = "Concise"
    DEEP_THINK = "DeepThink"


_TEMPLATES: Dict[PromptStrategy, str] = {
    PromptStrategy.FLASH: "Respond in Flash mode. Go straight to the answer. Do not explain your reasoning or show intermediate steps. Output only the final answer in the required format.",
    PromptStrategy.CONCISE: "Respond in Concise mode. Briefly outline your key reasoning in 2-3 short points, then state the final answer. Do not repeat the question or add unnecessary commentary.",
    PromptStrategy.DEEP_THINK: "Respond in DeepThink mode. Reason thoroughly and step-by-step. State your assumptions, show all intermediate work, verify your result, and provide a well-justified final answer.",
}

def get_strategy_prompt(strategy: str) -> str:
    try:
        strategy_enum = PromptStrategy(strategy)
    except ValueError:
        strategy_enum = PromptStrategy.CONCISE
    return _TEMPLATES.get(strategy_enum, _TEMPLATES[PromptStrategy.CONCISE])


def build_messages(query: str, strategy: str, role: Optional[str] = None, context: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Produce OpenAI-style messages with a strategy-specific system prompt.
    """
    try:
        strategy_enum = PromptStrategy(strategy)
    except ValueError:
        strategy_enum = PromptStrategy.CONCISE

    system_prompt = _TEMPLATES.get(strategy_enum, _TEMPLATES[PromptStrategy.CONCISE])
    if role:
        system_prompt = f"{system_prompt}\n\nRole: {role}"

    user_content = query
    if context:
        user_content = f"{query}\n\nContext from previous nodes:\n{context}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
