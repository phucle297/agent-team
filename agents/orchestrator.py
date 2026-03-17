"""Orchestrator agent that decomposes a task into parallel sub-tasks.

The orchestrator uses an LLM to break a complex task into smaller,
independent sub-tasks that can be executed in parallel by worker agents.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from utils.agent_events import AgentStatus, tracker
from utils.llm import extract_text, get_google, invoke_with_retry

logger = logging.getLogger(__name__)

MAX_WORKERS = 6  # Maximum number of parallel sub-tasks


ORCHESTRATOR_PROMPT = """You are a task orchestrator. Break the following task into smaller, independent sub-tasks that can be worked on in parallel by different agents.

RULES:
- Each sub-task should be independent and self-contained
- Each sub-task should produce a clear deliverable (code, research, config, etc.)
- Maximum {max_workers} sub-tasks
- Follow TDD: include test requirements in each code task
- Output ONLY a JSON array, no markdown fences or other text

Each sub-task must have:
- "id": unique identifier (e.g., "task_1", "task_2")
- "title": short title
- "description": detailed description of what to implement/research
- "type": "code" | "research" | "config"

{project_context}

Task: {task}

JSON array of sub-tasks:"""


def orchestrator(state: Any) -> dict:
    """Decompose a task into parallel sub-tasks using Google Gemini."""
    llm = get_google()
    task = state.get("input", state.get("task", ""))
    project_context = state.get("project_context", "")

    tracker.update("orchestrator", AgentStatus.WORKING, f"Decomposing: {task[:50]}...")

    prompt = ORCHESTRATOR_PROMPT.format(
        task=task,
        project_context=project_context,
        max_workers=MAX_WORKERS,
    )

    logger.info("Orchestrator: decomposing task into sub-tasks...")
    res = invoke_with_retry(llm, prompt)
    raw = extract_text(res.content).strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)

    sub_tasks = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            sub_tasks = parsed[:MAX_WORKERS]
    except json.JSONDecodeError:
        logger.warning("Orchestrator: could not parse sub-tasks JSON, using single task fallback")

    # Fallback: if no valid sub-tasks, create a single task from the input
    if not sub_tasks:
        sub_tasks = [{
            "id": "task_1",
            "title": task,
            "description": task,
            "type": "code",
        }]

    # Build a readable plan from the sub-tasks
    plan_lines = [f"## Sub-tasks ({len(sub_tasks)} parallel workers):\n"]
    for st in sub_tasks:
        plan_lines.append(f"- [{st.get('id', '?')}] {st.get('title', '?')}: {st.get('description', '')}")

    plan = "\n".join(plan_lines)

    logger.info("Orchestrator: created %d sub-tasks", len(sub_tasks))
    tracker.update("orchestrator", AgentStatus.DONE, f"{len(sub_tasks)} sub-tasks created")

    return {
        "sub_tasks": sub_tasks,
        "plan": plan,
    }
