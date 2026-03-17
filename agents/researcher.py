from __future__ import annotations

import logging
from typing import Any

from utils.llm import extract_text, get_google

logger = logging.getLogger(__name__)


def researcher(state: Any) -> dict:
    """Research relevant information for the plan using Google Gemini."""
    llm = get_google()
    plan = state.get("plan", "")

    prompt = (
        "You are a technical researcher. Given the following plan, "
        "research and gather all relevant information, best practices, "
        "library recommendations, and potential pitfalls.\n\n"
        f"Plan:\n{plan}"
    )

    logger.info("Researcher: gathering information...")
    res = llm.invoke(prompt)
    logger.info("Researcher: research complete.")

    return {"research": extract_text(res.content)}
