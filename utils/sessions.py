"""Session management system.

Tracks agent team sessions with full state, timing, and file change history.
Sessions are stored in ~/.agents/sessions/ as individual JSON files.
Maintains at most 20 sessions (oldest are pruned).
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path.home() / ".agents" / "sessions"
MAX_SESSIONS = 20


def _ensure_sessions_dir() -> Path:
    """Ensure the sessions directory exists."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR


def create_session(workspace: str, task: str) -> dict:
    """Create a new session record.

    Args:
        workspace: The project directory.
        task: The task description.

    Returns:
        Session dict with id, metadata, and empty history.
    """
    _ensure_sessions_dir()

    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    session = {
        "id": session_id,
        "workspace": workspace,
        "task": task,
        "status": "running",  # running | completed | failed | rolled_back
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "steps": [],  # Agent step history
        "files_changed": [],
        "approved": False,
        "iterations": 0,
        "final_output": None,
    }

    _save_session(session)
    logger.info("Created session %s for task: %s", session_id, task[:80])
    return session


def _save_session(session: dict) -> None:
    """Save a session to disk."""
    _ensure_sessions_dir()
    session["updated_at"] = datetime.now(timezone.utc).isoformat()
    session_path = SESSIONS_DIR / f"{session['id']}.json"
    session_path.write_text(json.dumps(session, indent=2))


def load_session(session_id: str) -> dict | None:
    """Load a session by ID.

    Args:
        session_id: The session identifier.

    Returns:
        Session dict or None if not found.
    """
    _ensure_sessions_dir()
    session_path = SESSIONS_DIR / f"{session_id}.json"

    if not session_path.exists():
        return None

    try:
        return json.loads(session_path.read_text())
    except json.JSONDecodeError:
        logger.error("Corrupted session file: %s", session_path)
        return None


def add_step(session_id: str, agent: str, status: str, detail: str = "") -> None:
    """Record an agent step in the session history.

    Args:
        session_id: Session to update.
        agent: Agent name (e.g., 'planner', 'coder', 'reviewer').
        status: Step status (e.g., 'started', 'completed', 'failed').
        detail: Optional detail text (truncated to 500 chars).
    """
    session = load_session(session_id)
    if not session:
        logger.warning("Cannot add step: session %s not found", session_id)
        return

    step = {
        "agent": agent,
        "status": status,
        "detail": detail[:500] if detail else "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    session["steps"].append(step)
    _save_session(session)


def complete_session(
    session_id: str,
    final_output: str,
    approved: bool,
    iterations: int,
    files_changed: list[str],
) -> None:
    """Mark a session as completed.

    Args:
        session_id: Session to complete.
        final_output: The final output text.
        approved: Whether the review was approved.
        iterations: Number of review iterations.
        files_changed: List of changed file paths.
    """
    session = load_session(session_id)
    if not session:
        return

    session["status"] = "completed"
    session["completed_at"] = datetime.now(timezone.utc).isoformat()
    session["final_output"] = final_output[:5000] if final_output else ""
    session["approved"] = approved
    session["iterations"] = iterations
    session["files_changed"] = files_changed
    _save_session(session)

    # Prune old sessions
    _prune_sessions()


def fail_session(session_id: str, error: str) -> None:
    """Mark a session as failed.

    Args:
        session_id: Session to mark failed.
        error: Error description.
    """
    session = load_session(session_id)
    if not session:
        return

    session["status"] = "failed"
    session["completed_at"] = datetime.now(timezone.utc).isoformat()
    session["final_output"] = f"Error: {error[:2000]}"
    _save_session(session)


def mark_rolled_back(session_id: str) -> None:
    """Mark a session as rolled back.

    Args:
        session_id: Session to mark.
    """
    session = load_session(session_id)
    if not session:
        return

    session["status"] = "rolled_back"
    _save_session(session)


def list_sessions(limit: int = MAX_SESSIONS) -> list[dict]:
    """List sessions sorted by creation time (newest first).

    Args:
        limit: Maximum number of sessions to return.

    Returns:
        List of session dicts.
    """
    _ensure_sessions_dir()
    sessions = []

    for path in SESSIONS_DIR.glob("*.json"):
        try:
            session = json.loads(path.read_text())
            sessions.append(session)
        except json.JSONDecodeError:
            continue

    sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return sessions[:limit]


def _prune_sessions() -> int:
    """Remove sessions beyond MAX_SESSIONS limit.

    Returns:
        Number of sessions pruned.
    """
    _ensure_sessions_dir()
    sessions = list_sessions(limit=9999)  # Get all
    pruned = 0

    if len(sessions) <= MAX_SESSIONS:
        return 0

    for session in sessions[MAX_SESSIONS:]:
        session_path = SESSIONS_DIR / f"{session['id']}.json"
        if session_path.exists():
            try:
                os.remove(session_path)
                pruned += 1
            except OSError as e:
                logger.error("Failed to prune session %s: %s", session["id"], e)

    return pruned
