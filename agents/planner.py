from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from utils.llm import extract_text, get_google

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner.txt"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text()


def planner(state: Any) -> dict:
    """Break a user task into actionable steps using Google Gemini."""
    llm = get_google()
    task = state["input"]
    project_context = state.get("project_context", "")
    prompt_template = _load_prompt()
    prompt = (
        prompt_template
        .replace("{task}", task)
        .replace("{project_context}", project_context)
    )

    logger.info("Planner: breaking down task...")
    res = llm.invoke(prompt)
    logger.info("Planner: plan created.")

    return {"plan": extract_text(res.content)}
