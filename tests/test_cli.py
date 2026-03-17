"""Tests for cli/main.py - CLI entry point."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestCLIMain:
    @patch("cli.main._launch_tui")
    def test_default_launches_tui(self, mock_tui):
        with patch.object(sys, "argv", ["agents"]):
            from cli.main import main
            main()
        mock_tui.assert_called_once()

    @patch("cli.main._show_sessions")
    def test_sessions_flag(self, mock_sessions):
        with patch.object(sys, "argv", ["agents", "--sessions"]):
            from cli.main import main
            main()
        mock_sessions.assert_called_once()

    @patch("cli.main._rollback_last")
    def test_rollback_flag(self, mock_rollback):
        with patch.object(sys, "argv", ["agents", "--rollback"]):
            from cli.main import main
            main()
        mock_rollback.assert_called_once()

    @patch("cli.main._show_help")
    def test_help_flag(self, mock_help):
        with patch.object(sys, "argv", ["agents", "--help"]):
            from cli.main import main
            main()
        mock_help.assert_called_once()


class TestShowSessions:
    @patch("utils.sessions.list_sessions", return_value=[])
    @patch("utils.sessions.SESSIONS_DIR")
    def test_handles_empty_sessions(self, mock_dir, mock_list, capsys):
        from cli.main import _show_sessions
        _show_sessions()
        captured = capsys.readouterr()
        assert "No sessions found" in captured.out

    @patch("utils.sessions.list_sessions")
    @patch("utils.sessions.SESSIONS_DIR")
    def test_displays_sessions(self, mock_dir, mock_list, capsys):
        mock_list.return_value = [
            {
                "id": "sess_001",
                "status": "completed",
                "task": "Build API",
                "created_at": "2026-03-17T07:00:00",
            }
        ]
        from cli.main import _show_sessions
        _show_sessions()
        captured = capsys.readouterr()
        assert "sess_001" in captured.out
        assert "Build API" in captured.out
