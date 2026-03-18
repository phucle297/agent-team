"""Tests for agents/filesystem.py - File system agent."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import FakeLLMResponse


class TestFilesystemAgent:
    @patch("agents.filesystem.get_llm")
    def test_creates_file(self, mock_get_llm, tmp_path):
        ops = json.dumps([
            {"action": "create", "path": "src/main.py", "content": "print('hello')"}
        ])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(ops)
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.filesystem import filesystem_agent

        result = filesystem_agent({
            "code": "print('hello')",
            "plan": "create main.py",
            "workspace": str(tmp_path),
        })

        created_file = tmp_path / "src" / "main.py"
        assert created_file.exists()
        assert created_file.read_text() == "print('hello')"
        assert "created: src/main.py" in result["files_changed"]

    @patch("agents.filesystem.get_llm")
    def test_modifies_file(self, mock_get_llm, tmp_path):
        # Create existing file
        target = tmp_path / "app.py"
        target.write_text("old content")

        ops = json.dumps([
            {"action": "modify", "path": "app.py", "content": "new content"}
        ])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(ops)
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.filesystem import filesystem_agent

        result = filesystem_agent({
            "code": "new content",
            "plan": "update app.py",
            "workspace": str(tmp_path),
        })

        assert target.read_text() == "new content"
        assert "modified: app.py" in result["files_changed"]

    @patch("agents.filesystem.get_llm")
    def test_deletes_file(self, mock_get_llm, tmp_path):
        target = tmp_path / "old.py"
        target.write_text("delete me")

        ops = json.dumps([
            {"action": "delete", "path": "old.py"}
        ])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(ops)
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.filesystem import filesystem_agent

        result = filesystem_agent({
            "code": "",
            "plan": "remove old.py",
            "workspace": str(tmp_path),
        })

        assert not target.exists()
        assert "deleted: old.py" in result["files_changed"]

    @patch("agents.filesystem.get_llm")
    def test_delete_nonexistent_file_is_safe(self, mock_get_llm, tmp_path):
        ops = json.dumps([
            {"action": "delete", "path": "nonexistent.py"}
        ])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(ops)
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.filesystem import filesystem_agent

        result = filesystem_agent({
            "code": "",
            "plan": "",
            "workspace": str(tmp_path),
        })

        assert result["files_changed"] == []

    @patch("agents.filesystem.get_llm")
    def test_handles_invalid_json(self, mock_get_llm, tmp_path):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse("not json at all")
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.filesystem import filesystem_agent

        result = filesystem_agent({
            "code": "",
            "plan": "",
            "workspace": str(tmp_path),
        })

        assert result["file_operations"] == []
        assert result["files_changed"] == []

    @patch("agents.filesystem.get_llm")
    def test_strips_markdown_fences(self, mock_get_llm, tmp_path):
        ops_with_fences = '```json\n' + json.dumps([
            {"action": "create", "path": "test.txt", "content": "hello"}
        ]) + '\n```'
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(ops_with_fences)
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.filesystem import filesystem_agent

        result = filesystem_agent({
            "code": "",
            "plan": "",
            "workspace": str(tmp_path),
        })

        assert (tmp_path / "test.txt").exists()
        assert "created: test.txt" in result["files_changed"]

    @patch("agents.filesystem.get_llm")
    def test_skips_ops_with_empty_path(self, mock_get_llm, tmp_path):
        ops = json.dumps([
            {"action": "create", "path": "", "content": "hello"}
        ])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(ops)
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.filesystem import filesystem_agent

        result = filesystem_agent({
            "code": "",
            "plan": "",
            "workspace": str(tmp_path),
        })

        assert result["files_changed"] == []

    @patch("agents.filesystem.get_llm")
    def test_multiple_operations(self, mock_get_llm, tmp_path):
        ops = json.dumps([
            {"action": "create", "path": "a.py", "content": "a"},
            {"action": "create", "path": "b.py", "content": "b"},
        ])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(ops)
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.filesystem import filesystem_agent

        result = filesystem_agent({
            "code": "",
            "plan": "",
            "workspace": str(tmp_path),
        })

        assert (tmp_path / "a.py").exists()
        assert (tmp_path / "b.py").exists()
        assert len(result["files_changed"]) == 2

    @patch("agents.filesystem.invoke_with_retry")
    @patch("agents.filesystem.get_llm")
    def test_handles_llm_failure_gracefully(self, mock_get_llm, mock_retry, tmp_path):
        """FileSystem agent should return empty results instead of crashing on LLM failure."""
        mock_retry.side_effect = ConnectionError("Connection refused")
        mock_get_llm.return_value = (MagicMock(), "mock-model")

        from agents.filesystem import filesystem_agent

        result = filesystem_agent({
            "code": "some code",
            "plan": "some plan",
            "workspace": str(tmp_path),
        })

        assert result["file_operations"] == []
        assert result["files_changed"] == []
