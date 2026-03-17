"""Agent status event tracking system.

Provides a thread-safe tracker that agents publish their status to,
and the TUI subscribes to for live updates.
"""

import threading
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


class AgentStatus(Enum):
    """Possible states for an agent."""

    IDLE = "idle"
    WORKING = "working"
    DONE = "done"
    ERROR = "error"


# Type alias for subscriber callbacks
StatusCallback = Callable[[str, AgentStatus, str], None]


class AgentTracker:
    """Thread-safe agent status tracker with pub/sub.

    Usage:
        tracker = AgentTracker()
        tracker.register("planner")
        tracker.subscribe(my_callback)
        tracker.update("planner", AgentStatus.WORKING, "Breaking down task...")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._agents: dict[str, dict[str, Any]] = {}
        self._subscribers: list[StatusCallback] = []

    def reset(self) -> None:
        """Clear all agents and subscribers."""
        with self._lock:
            self._agents.clear()
            self._subscribers.clear()

    def register(self, name: str) -> None:
        """Register an agent with initial IDLE status."""
        with self._lock:
            self._agents[name] = {
                "status": AgentStatus.IDLE,
                "detail": "",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    def register_workers(self, names: list[str]) -> None:
        """Register multiple worker agents at once."""
        for name in names:
            self.register(name)

    def update(self, name: str, status: AgentStatus, detail: str = "") -> None:
        """Update an agent's status and notify subscribers."""
        with self._lock:
            if name not in self._agents:
                self._agents[name] = {}
            self._agents[name]["status"] = status
            self._agents[name]["detail"] = detail
            self._agents[name]["updated_at"] = datetime.now(timezone.utc).isoformat()
            # Copy subscribers list to avoid holding lock during callbacks
            subs = list(self._subscribers)

        # Fire callbacks outside the lock to prevent deadlocks
        for callback in subs:
            try:
                callback(name, status, detail)
            except Exception:
                pass  # Don't let subscriber errors break the tracker

    def get(self, name: str) -> dict[str, Any] | None:
        """Get an agent's current status, or None if not registered."""
        with self._lock:
            if name not in self._agents:
                return None
            return deepcopy(self._agents[name])

    def get_all(self) -> dict[str, dict[str, Any]]:
        """Get a snapshot of all agent statuses."""
        with self._lock:
            return deepcopy(self._agents)

    def subscribe(self, callback: StatusCallback) -> None:
        """Register a callback for status change events."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: StatusCallback) -> None:
        """Remove a subscriber callback."""
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s is not callback]


# Module-level singleton for global access
tracker = AgentTracker()
