from __future__ import annotations

import logging
from typing import Any

from utils.agent_events import AgentStatus, tracker
from utils.llm import extract_text, get_claude

logger = logging.getLogger(__name__)


def fixer(state: Any) -> dict:
    """Fix code based on reviewer feedback using Claude."""
    llm = get_claude()
    code = state.get("code", "")
    review = state.get("review", "")
    plan = state.get("plan", "")

    prompt = (
        "You are an expert software engineer. The following code was reviewed "
        "and needs revisions. Apply the reviewer's suggestions and fix all issues.\n\n"
        "Original plan:\n{plan}\n\n"
        "Current code:\n{code}\n\n"
        "Review feedback:\n{review}\n\n"
        "Output the complete fixed code:"
    ).format(plan=plan, code=code, review=review)

    tracker.update("fixer", AgentStatus.WORKING, "Applying review feedback...")
    logger.info("Fixer: applying review feedback...")
    res = llm.invoke(prompt)
    logger.info("Fixer: code fixed.")
    tracker.update("fixer", AgentStatus.DONE, "Code fixed")

    return {"code": extract_text(res.content)}
