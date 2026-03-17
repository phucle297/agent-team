"""Tests for agents/tools.py - Tool execution agent."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import FakeLLMResponse


class TestToolAgent:
    @patch("agents.tools.subprocess.run")
    @patch("agents.tools.get_claude")
    def test_executes_safe_command(self, mock_get_claude, mock_run, tmp_path):
        commands = json.dumps([
            {"command": "echo hello", "description": "say hello", "type": "terminal"}
        ])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(commands)
        mock_get_claude.return_value = mock_llm

        mock_run.return_value = MagicMock(
            returncode=0, stdout="hello\n", stderr=""
        )

        from agents.tools import tool_agent

        result = tool_agent({
            "code": "print('hello')",
            "plan": "echo test",
            "workspace": str(tmp_path),
        })

        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["status"] == "success"
        assert result["tool_results"][0]["output"] == "hello\n"

    @patch("agents.tools.get_claude")
    def test_blocks_dangerous_rm_rf(self, mock_get_claude, tmp_path):
        commands = json.dumps([
            {"command": "rm -rf /", "description": "nuke it", "type": "terminal"}
        ])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(commands)
        mock_get_claude.return_value = mock_llm

        from agents.tools import tool_agent

        result = tool_agent({
            "code": "",
            "plan": "",
            "workspace": str(tmp_path),
        })

        assert result["tool_results"][0]["status"] == "blocked"

    @patch("agents.tools.get_claude")
    def test_blocks_force_push(self, mock_get_claude, tmp_path):
        commands = json.dumps([
            {"command": "git push --force", "description": "force push", "type": "git"}
        ])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(commands)
        mock_get_claude.return_value = mock_llm

        from agents.tools import tool_agent

        result = tool_agent({
            "code": "",
            "plan": "",
            "workspace": str(tmp_path),
        })

        assert result["tool_results"][0]["status"] == "blocked"

    @patch("agents.tools.get_claude")
    def test_blocks_hard_reset(self, mock_get_claude, tmp_path):
        commands = json.dumps([
            {"command": "git reset --hard HEAD~1", "description": "hard reset", "type": "git"}
        ])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(commands)
        mock_get_claude.return_value = mock_llm

        from agents.tools import tool_agent

        result = tool_agent({
            "code": "",
            "plan": "",
            "workspace": str(tmp_path),
        })

        assert result["tool_results"][0]["status"] == "blocked"

    @patch("agents.tools.subprocess.run")
    @patch("agents.tools.get_claude")
    def test_handles_failed_command(self, mock_get_claude, mock_run, tmp_path):
        commands = json.dumps([
            {"command": "python -c 'exit(1)'", "description": "fail", "type": "test"}
        ])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(commands)
        mock_get_claude.return_value = mock_llm

        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="error"
        )

        from agents.tools import tool_agent

        result = tool_agent({
            "code": "",
            "plan": "",
            "workspace": str(tmp_path),
        })

        assert result["tool_results"][0]["status"] == "failed"

    @patch("agents.tools.subprocess.run")
    @patch("agents.tools.get_claude")
    def test_handles_timeout(self, mock_get_claude, mock_run, tmp_path):
        commands = json.dumps([
            {"command": "sleep 999", "description": "long command", "type": "terminal"}
        ])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(commands)
        mock_get_claude.return_value = mock_llm

        mock_run.side_effect = subprocess.TimeoutExpired("sleep", 60)

        from agents.tools import tool_agent

        result = tool_agent({
            "code": "",
            "plan": "",
            "workspace": str(tmp_path),
        })

        assert result["tool_results"][0]["status"] == "timeout"

    @patch("agents.tools.get_claude")
    def test_handles_invalid_json(self, mock_get_claude, tmp_path):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse("not json")
        mock_get_claude.return_value = mock_llm

        from agents.tools import tool_agent

        result = tool_agent({
            "code": "",
            "plan": "",
            "workspace": str(tmp_path),
        })

        assert result["tool_results"] == []

    @patch("agents.tools.get_claude")
    def test_strips_markdown_fences(self, mock_get_claude, tmp_path):
        commands_with_fences = '```json\n' + json.dumps([
            {"command": "echo hi", "description": "test", "type": "terminal"}
        ]) + '\n```'
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(commands_with_fences)
        mock_get_claude.return_value = mock_llm

        # Patch subprocess.run too since the command will try to execute
        with patch("agents.tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="hi\n", stderr="")
            from agents.tools import tool_agent

            result = tool_agent({
                "code": "",
                "plan": "",
                "workspace": str(tmp_path),
            })

        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["status"] == "success"

    @patch("agents.tools.get_claude")
    def test_skips_empty_commands(self, mock_get_claude, tmp_path):
        commands = json.dumps([
            {"command": "", "description": "empty", "type": "terminal"}
        ])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(commands)
        mock_get_claude.return_value = mock_llm

        from agents.tools import tool_agent

        result = tool_agent({
            "code": "",
            "plan": "",
            "workspace": str(tmp_path),
        })

        assert result["tool_results"] == []

    @patch("agents.tools.subprocess.run")
    @patch("agents.tools.get_claude")
    def test_truncates_long_output(self, mock_get_claude, mock_run, tmp_path):
        commands = json.dumps([
            {"command": "cat big_file", "description": "read big file", "type": "terminal"}
        ])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(commands)
        mock_get_claude.return_value = mock_llm

        mock_run.return_value = MagicMock(
            returncode=0, stdout="x" * 5000, stderr=""
        )

        from agents.tools import tool_agent

        result = tool_agent({
            "code": "",
            "plan": "",
            "workspace": str(tmp_path),
        })

        assert len(result["tool_results"][0]["output"]) == 2000
