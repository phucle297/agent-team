"""Tests for utils/memory.py - Run memory persistence."""

import json
from pathlib import Path

import pytest

from utils.memory import (
    MEMORY_FILE,
    _ensure_memory_file,
    get_context_for_task,
    load_runs,
    save_run,
)


@pytest.fixture
def memory_dir(tmp_path, monkeypatch):
    """Use a temporary directory for memory storage."""
    mem_file = tmp_path / "memory.json"
    monkeypatch.setattr("utils.memory.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("utils.memory.MEMORY_FILE", mem_file)
    return mem_file


class TestEnsureMemoryFile:
    def test_creates_file_if_missing(self, memory_dir):
        assert not memory_dir.exists()
        _ensure_memory_file()
        assert memory_dir.exists()
        assert json.loads(memory_dir.read_text()) == []

    def test_does_not_overwrite_existing(self, memory_dir):
        memory_dir.write_text('[{"input": "existing"}]')
        _ensure_memory_file()
        data = json.loads(memory_dir.read_text())
        assert len(data) == 1
        assert data[0]["input"] == "existing"


class TestSaveRun:
    def test_saves_basic_run(self, memory_dir):
        state = {
            "input": "build a REST API",
            "plan": "step 1, step 2",
            "code": "print('hello')",
            "approved": True,
            "iteration": 1,
        }
        save_run(state)

        data = json.loads(memory_dir.read_text())
        assert len(data) == 1
        assert data[0]["input"] == "build a REST API"
        assert data[0]["approved"] is True
        assert data[0]["iterations"] == 1
        assert "timestamp" in data[0]

    def test_appends_multiple_runs(self, memory_dir):
        save_run({"input": "task 1"})
        save_run({"input": "task 2"})

        data = json.loads(memory_dir.read_text())
        assert len(data) == 2
        assert data[0]["input"] == "task 1"
        assert data[1]["input"] == "task 2"

    def test_truncates_code_preview_to_500_chars(self, memory_dir):
        long_code = "x" * 1000
        save_run({"code": long_code})

        data = json.loads(memory_dir.read_text())
        assert len(data[0]["code_preview"]) == 500

    def test_caps_at_50_runs(self, memory_dir):
        for i in range(55):
            save_run({"input": f"task {i}"})

        data = json.loads(memory_dir.read_text())
        assert len(data) == 50
        # Should keep the most recent 50
        assert data[0]["input"] == "task 5"
        assert data[-1]["input"] == "task 54"

    def test_handles_missing_state_keys(self, memory_dir):
        save_run({})
        data = json.loads(memory_dir.read_text())
        assert data[0]["input"] == ""
        assert data[0]["approved"] is False

    def test_handles_corrupted_memory_file(self, memory_dir):
        memory_dir.write_text("not valid json{{{")
        save_run({"input": "recovery test"})
        data = json.loads(memory_dir.read_text())
        assert len(data) == 1
        assert data[0]["input"] == "recovery test"


class TestLoadRuns:
    def test_returns_empty_list_when_no_file(self, memory_dir):
        runs = load_runs()
        assert runs == []

    def test_returns_all_runs_when_fewer_than_limit(self, memory_dir):
        save_run({"input": "a"})
        save_run({"input": "b"})
        runs = load_runs(limit=10)
        assert len(runs) == 2

    def test_respects_limit(self, memory_dir):
        for i in range(10):
            save_run({"input": f"task {i}"})
        runs = load_runs(limit=3)
        assert len(runs) == 3
        # Should return the last 3
        assert runs[0]["input"] == "task 7"

    def test_handles_corrupted_file(self, memory_dir):
        memory_dir.write_text("garbage")
        runs = load_runs()
        assert runs == []


class TestGetContextForTask:
    def test_returns_no_runs_message_when_empty(self, memory_dir):
        ctx = get_context_for_task("anything")
        assert ctx == "No previous runs found."

    def test_includes_run_info(self, memory_dir):
        save_run({"input": "build API", "plan": "design endpoints", "approved": True, "iteration": 2})
        ctx = get_context_for_task("new task")
        assert "build API" in ctx
        assert "Previous runs for context:" in ctx
        assert "Approved: True" in ctx

    def test_respects_limit(self, memory_dir):
        for i in range(5):
            save_run({"input": f"task {i}"})
        ctx = get_context_for_task("new", limit=2)
        assert "task 3" in ctx
        assert "task 4" in ctx
        assert "task 0" not in ctx
