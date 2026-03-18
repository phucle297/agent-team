from __future__ import annotations

import logging
from typing import Any

from utils.agent_events import AgentStatus, tracker
from utils.llm import (
    extract_text,
    get_llm,
    invoke_with_retry,
    invoke_with_retry_and_fallback,
)

logger = logging.getLogger(__name__)


def fixer(state: Any) -> dict:
    """Fix code based on reviewer feedback."""
    llm, model_name = get_llm("anthropic")
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

    try:
        res = invoke_with_retry_and_fallback(
            llm,
            prompt,
            primary_model=model_name,
            invoke_fn=invoke_with_retry,
        )
        fixed_code = extract_text(res.content)
        logger.info("Fixer: code fixed.")
        tracker.update("fixer", AgentStatus.DONE, "Code fixed")
    except Exception as exc:
        fixed_code = code  # Return original code on failure
        logger.error("Fixer: failed with %s", exc)
        tracker.update("fixer", AgentStatus.ERROR, f"Error: {str(exc)[:50]}")

    return {"code": fixed_code}
