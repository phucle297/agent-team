from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from utils.agent_events import AgentStatus, tracker
from utils.llm import extract_text, get_claude

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "reviewer.txt"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text()


def reviewer(state: Any) -> dict:
    """Review code and either approve or request revisions using Claude."""
    llm = get_claude()
    code = state.get("code", "")
    plan = state.get("plan", "")
    iteration = state.get("iteration", 0)
    prompt_template = _load_prompt()
    prompt = prompt_template.replace("{code}", code).replace("{plan}", plan)

    tracker.update("reviewer", AgentStatus.WORKING, f"Reviewing (iteration {iteration})...")
    logger.info("Reviewer: reviewing code (iteration %d)...", iteration)
    res = llm.invoke(prompt)
    review_content = extract_text(res.content)
    logger.info("Reviewer: review complete.")
    tracker.update("reviewer", AgentStatus.DONE, f"Review complete (iteration {iteration})")

    approved = "APPROVED" in str(review_content).upper().split("NEEDS_REVISION")[0]

    return {
        "review": review_content,
        "approved": approved,
        "iteration": iteration + 1,
    }
