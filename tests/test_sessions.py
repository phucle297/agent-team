"""Tests for utils/sessions.py - Session management system."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.sessions import (
    add_step,
    complete_session,
    create_session,
    fail_session,
    list_sessions,
    load_session,
    mark_rolled_back,
    _prune_sessions,
)


@pytest.fixture
def sessions_dir(tmp_path):
    """Override SESSIONS_DIR to use a temp directory."""
    test_dir = tmp_path / "sessions"
    test_dir.mkdir()
    with patch("utils.sessions.SESSIONS_DIR", test_dir):
        yield test_dir


class TestCreateSession:
    def test_creates_session_file(self, sessions_dir):
        session = create_session("/my/project", "Build a REST API")
        session_path = sessions_dir / f"{session['id']}.json"
        assert session_path.exists()

    def test_session_has_required_fields(self, sessions_dir):
        session = create_session("/my/project", "Build something")
        assert "id" in session
        assert session["workspace"] == "/my/project"
        assert session["task"] == "Build something"
        assert session["status"] == "running"
        assert session["steps"] == []
        assert session["files_changed"] == []
        assert session["approved"] is False

    def test_generates_unique_ids(self, sessions_dir):
        s1 = create_session("/ws", "task 1")
        s2 = create_session("/ws", "task 2")
        assert s1["id"] != s2["id"]


class TestLoadSession:
    def test_loads_existing_session(self, sessions_dir):
        session = create_session("/ws", "test task")
        loaded = load_session(session["id"])
        assert loaded is not None
        assert loaded["task"] == "test task"

    def test_returns_none_for_missing(self, sessions_dir):
        result = load_session("nonexistent_id")
        assert result is None

    def test_returns_none_for_corrupted_file(self, sessions_dir):
        bad_path = sessions_dir / "bad_session.json"
        bad_path.write_text("{invalid json")
        result = load_session("bad_session")
        assert result is None


class TestAddStep:
    def test_adds_step_to_session(self, sessions_dir):
        session = create_session("/ws", "task")
        add_step(session["id"], "planner", "completed", "Created 5 steps")

        loaded = load_session(session["id"])
        assert len(loaded["steps"]) == 1
        assert loaded["steps"][0]["agent"] == "planner"
        assert loaded["steps"][0]["status"] == "completed"

    def test_adds_multiple_steps(self, sessions_dir):
        session = create_session("/ws", "task")
        add_step(session["id"], "planner", "completed")
        add_step(session["id"], "coder", "completed")
        add_step(session["id"], "reviewer", "completed")

        loaded = load_session(session["id"])
        assert len(loaded["steps"]) == 3

    def test_truncates_long_detail(self, sessions_dir):
        session = create_session("/ws", "task")
        long_detail = "x" * 1000
        add_step(session["id"], "coder", "completed", long_detail)

        loaded = load_session(session["id"])
        assert len(loaded["steps"][0]["detail"]) == 500

    def test_skips_if_session_not_found(self, sessions_dir):
        # Should not raise
        add_step("nonexistent", "planner", "completed")


class TestCompleteSession:
    def test_marks_session_completed(self, sessions_dir):
        session = create_session("/ws", "task")
        complete_session(session["id"], "final output", True, 2, ["file.py"])

        loaded = load_session(session["id"])
        assert loaded["status"] == "completed"
        assert loaded["approved"] is True
        assert loaded["iterations"] == 2
        assert loaded["files_changed"] == ["file.py"]
        assert loaded["completed_at"] is not None

    def test_truncates_long_output(self, sessions_dir):
        session = create_session("/ws", "task")
        long_output = "x" * 10000
        complete_session(session["id"], long_output, False, 1, [])

        loaded = load_session(session["id"])
        assert len(loaded["final_output"]) == 5000


class TestFailSession:
    def test_marks_session_failed(self, sessions_dir):
        session = create_session("/ws", "task")
        fail_session(session["id"], "API key invalid")

        loaded = load_session(session["id"])
        assert loaded["status"] == "failed"
        assert "API key invalid" in loaded["final_output"]


class TestMarkRolledBack:
    def test_marks_session_rolled_back(self, sessions_dir):
        session = create_session("/ws", "task")
        mark_rolled_back(session["id"])

        loaded = load_session(session["id"])
        assert loaded["status"] == "rolled_back"


class TestListSessions:
    def test_returns_empty_when_none(self, sessions_dir):
        result = list_sessions()
        assert result == []

    def test_returns_sessions_newest_first(self, sessions_dir):
        s1 = create_session("/ws", "first")
        s2 = create_session("/ws", "second")

        result = list_sessions()
        assert len(result) == 2
        assert result[0]["task"] == "second"

    def test_respects_limit(self, sessions_dir):
        for i in range(5):
            create_session("/ws", f"task {i}")

        result = list_sessions(limit=3)
        assert len(result) == 3


class TestPruneSessions:
    def test_prunes_old_sessions(self, sessions_dir):
        with patch("utils.sessions.MAX_SESSIONS", 3):
            for i in range(5):
                create_session("/ws", f"task {i}")

            pruned = _prune_sessions()
            assert pruned == 2

            remaining = list_sessions(limit=100)
            assert len(remaining) == 3

    def test_no_prune_when_under_limit(self, sessions_dir):
        create_session("/ws", "task")
        pruned = _prune_sessions()
        assert pruned == 0
