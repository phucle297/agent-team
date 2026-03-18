from __future__ import annotations

import logging
from typing import Any

from utils.llm import (
    extract_text,
    get_llm,
    invoke_with_retry,
    invoke_with_retry_and_fallback,
)

logger = logging.getLogger(__name__)


def researcher(state: Any) -> dict:
    """Research relevant information for the plan."""
    llm, model_name = get_llm("google")
    plan = state.get("plan", "")

    prompt = (
        "You are a technical researcher. Given the following plan, "
        "research and gather all relevant information, best practices, "
        "library recommendations, and potential pitfalls.\n\n"
        f"Plan:\n{plan}"
    )

    logger.info("Researcher: gathering information...")
    res = invoke_with_retry_and_fallback(
        llm,
        prompt,
        primary_model=model_name,
        invoke_fn=invoke_with_retry,
    )
    logger.info("Researcher: research complete.")

    return {"research": extract_text(res.content)}
