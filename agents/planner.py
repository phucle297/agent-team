from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from utils.llm import (
    extract_text,
    get_llm,
    invoke_with_retry,
    invoke_with_retry_and_fallback,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "planner.txt"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text()


def planner(state: Any) -> dict:
    """Break a user task into actionable steps."""
    llm, model_name = get_llm("google")
    task = state["input"]
    project_context = state.get("project_context", "")
    prompt_template = _load_prompt()
    prompt = (
        prompt_template
        .replace("{task}", task)
        .replace("{project_context}", project_context)
    )

    logger.info("Planner: breaking down task...")
    res = invoke_with_retry_and_fallback(
        llm,
        prompt,
        primary_model=model_name,
        invoke_fn=invoke_with_retry,
    )
    logger.info("Planner: plan created.")

    return {"plan": extract_text(res.content)}
