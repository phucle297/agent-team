"""Memory module for storing and retrieving past agent runs."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).parent.parent / "logs"
MEMORY_FILE = MEMORY_DIR / "memory.json"


def _ensure_memory_file():
    """Create the memory file if it doesn't exist."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("[]")


def save_run(state: dict) -> None:
    """Save a completed run to memory."""
    _ensure_memory_file()

    try:
        runs = json.loads(MEMORY_FILE.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        runs = []

    run_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": state.get("input", ""),
        "plan": state.get("plan", ""),
        "research": state.get("research", ""),
        "code_preview": state.get("code", "")[:500] if state.get("code") else "",
        "approved": state.get("approved", False),
        "iterations": state.get("iteration", 0),
        "files_changed": state.get("files_changed", []),
    }

    runs.append(run_record)

    # Keep only the last 50 runs
    if len(runs) > 50:
        runs = runs[-50:]

    MEMORY_FILE.write_text(json.dumps(runs, indent=2))
    logger.info("Memory: saved run to %s", MEMORY_FILE)


def load_runs(limit: int = 10) -> list[dict]:
    """Load recent runs from memory."""
    _ensure_memory_file()

    try:
        runs = json.loads(MEMORY_FILE.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return []

    return runs[-limit:]


def get_context_for_task(task: str, limit: int = 3) -> str:
    """Get relevant past run context for a new task."""
    runs = load_runs(limit=limit)
    if not runs:
        return "No previous runs found."

    context_parts = ["Previous runs for context:"]
    for i, run in enumerate(runs, 1):
        context_parts.append(
            f"\n--- Run {i} ({run.get('timestamp', 'unknown')}) ---\n"
            f"Task: {run.get('input', 'N/A')}\n"
            f"Plan preview: {run.get('plan', 'N/A')[:200]}\n"
            f"Approved: {run.get('approved', 'N/A')}\n"
            f"Iterations: {run.get('iterations', 'N/A')}"
        )

    return "\n".join(context_parts)
