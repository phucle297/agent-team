"""Tests for main.py - Entry point and CLI."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestRunAgentTeam:
    @patch("utils.continuation.build_continuation_note", return_value="")
    @patch("graph.workflow.build_graph")
    @patch("utils.memory.save_run")
    @patch("utils.memory.get_context_for_task", return_value="No previous runs found.")
    def test_returns_final_output(self, mock_ctx, mock_save, mock_build, mock_cont):
        """Patch at source modules since run_agent_team uses lazy imports."""
        mock_app = MagicMock()
        mock_app.invoke.return_value = {
            "final": "## Status: APPROVED\n\ngreat code",
            "approved": True,
            "iteration": 1,
        }
        mock_build.return_value = mock_app

        from main import run_agent_team

        result = run_agent_team("Build a REST API")

        assert "APPROVED" in result or "great code" in result

    @patch("utils.continuation.build_continuation_note", return_value="")
    @patch("graph.workflow.build_graph")
    @patch("utils.memory.save_run")
    @patch("utils.memory.get_context_for_task", return_value="No previous runs.")
    def test_passes_task_to_graph(self, mock_ctx, mock_save, mock_build, mock_cont):
        mock_app = MagicMock()
        mock_app.invoke.return_value = {"final": "done"}
        mock_build.return_value = mock_app

        from main import run_agent_team

        run_agent_team("My specific task")

        call_args = mock_app.invoke.call_args[0][0]
        assert call_args["input"] == "My specific task"

    @patch("utils.continuation.build_continuation_note", return_value="")
    @patch("graph.workflow.build_graph")
    @patch("utils.memory.save_run")
    @patch("utils.memory.get_context_for_task", return_value="No previous runs.")
    def test_saves_run_to_memory(self, mock_ctx, mock_save, mock_build, mock_cont):
        mock_app = MagicMock()
        mock_app.invoke.return_value = {"final": "done", "approved": True}
        mock_build.return_value = mock_app

        from main import run_agent_team

        run_agent_team("test task")

        mock_save.assert_called_once()

    @patch("utils.continuation.build_continuation_note", return_value="")
    @patch("graph.workflow.build_graph")
    @patch("utils.memory.save_run")
    @patch("utils.memory.get_context_for_task", return_value="No previous runs.")
    def test_uses_provided_workspace(self, mock_ctx, mock_save, mock_build, mock_cont):
        mock_app = MagicMock()
        mock_app.invoke.return_value = {"final": "done"}
        mock_build.return_value = mock_app

        from main import run_agent_team

        run_agent_team("task", workspace="/custom/path")

        call_args = mock_app.invoke.call_args[0][0]
        assert call_args["workspace"] == "/custom/path"

    @patch("utils.continuation.build_continuation_note", return_value="")
    @patch("graph.workflow.build_graph")
    @patch("utils.memory.save_run")
    @patch("utils.memory.get_context_for_task", return_value="No previous runs.")
    def test_returns_no_output_message_on_missing_final(self, mock_ctx, mock_save, mock_build, mock_cont):
        mock_app = MagicMock()
        mock_app.invoke.return_value = {}
        mock_build.return_value = mock_app

        from main import run_agent_team

        result = run_agent_team("task")
        assert result == "No output generated."


class TestMainCLI:
    @patch("main.run_agent_team", return_value="output")
    def test_exits_on_empty_task(self, mock_run):
        from main import main

        with patch.object(sys, "argv", ["main.py"]):
            with patch("builtins.input", return_value=""):
                with pytest.raises(SystemExit):
                    main()

    @patch("main.run_agent_team", return_value="output")
    def test_uses_argv_as_task(self, mock_run):
        from main import main

        with patch.object(sys, "argv", ["main.py", "Build", "a", "CLI"]):
            main()

        mock_run.assert_called_once_with("Build a CLI")

    @patch("main.run_agent_team", return_value="output")
    def test_uses_input_when_no_argv(self, mock_run):
        from main import main

        with patch.object(sys, "argv", ["main.py"]):
            with patch("builtins.input", return_value="Interactive task"):
                main()

        mock_run.assert_called_once_with("Interactive task")
