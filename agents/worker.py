"""Worker agent that executes a single sub-task.

Each worker receives a sub-task from the orchestrator and produces
code/research output. Workers run in parallel via LangGraph's Send() API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from utils.agent_events import AgentStatus, tracker
from utils.llm import extract_text, get_claude

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "coder.txt"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text()


WORKER_PROMPT = """You are a specialized worker agent. Complete the following sub-task independently.

## Overall Plan:
{plan}

## Your Sub-Task:
- Title: {title}
- Description: {description}
- Type: {task_type}

## Project Context:
{project_context}

## Instructions:
- If type is "code": Write complete, working code with tests (TDD approach)
- If type is "research": Provide detailed findings, recommendations, and examples
- If type is "config": Write configuration files with comments

Produce your output now:"""


def worker(state: Any) -> dict:
    """Execute a single sub-task using Claude."""
    llm = get_claude()
    sub_task = state.get("sub_task", {})
    project_context = state.get("project_context", "")
    plan = state.get("plan", "")

    task_id = sub_task.get("id", "unknown")
    title = sub_task.get("title", "")
    description = sub_task.get("description", "")
    task_type = sub_task.get("type", "code")

    # Use custom tracker if provided (for testing), otherwise global
    t = state.get("_tracker", tracker)
    worker_name = f"worker_{task_id}"
    t.update(worker_name, AgentStatus.WORKING, f"{title[:40]}...")

    prompt = WORKER_PROMPT.format(
        plan=plan,
        title=title,
        description=description,
        task_type=task_type,
        project_context=project_context,
    )

    logger.info("Worker [%s]: executing sub-task '%s'...", task_id, title)
    res = llm.invoke(prompt)
    output = extract_text(res.content)
    logger.info("Worker [%s]: completed.", task_id)

    t.update(worker_name, AgentStatus.DONE, f"{title[:40]} - done")

    return {
        "worker_results": [{
            "task_id": task_id,
            "title": title,
            "type": task_type,
            "output": output,
        }],
    }
