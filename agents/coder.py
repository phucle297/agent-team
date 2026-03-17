from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from utils.llm import extract_text, get_claude

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "coder.txt"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text()


def coder(state: Any) -> dict:
    """Write code based on the plan and optional research context using Claude."""
    llm = get_claude()
    plan = state.get("plan", "")
    research = state.get("research", "No research context available.")
    project_context = state.get("project_context", "")
    prompt_template = _load_prompt()
    prompt = (
        prompt_template
        .replace("{plan}", plan)
        .replace("{research}", research)
        .replace("{project_context}", project_context)
    )

    logger.info("Coder: writing code (TDD: tests first)...")
    res = llm.invoke(prompt)
    logger.info("Coder: code written.")

    return {"code": extract_text(res.content)}
