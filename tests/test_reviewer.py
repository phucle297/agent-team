"""Tests for agents/reviewer.py - Reviewer agent with APPROVED/NEEDS_REVISION logic."""

from unittest.mock import MagicMock, patch

from tests.conftest import FakeLLMResponse


class TestReviewer:
    @patch("agents.reviewer.get_llm")
    def test_approves_when_approved_in_response(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(
            "## Status: APPROVED\n\n## Summary:\nCode looks good."
        )
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.reviewer import reviewer

        result = reviewer({"code": "print('hello')", "plan": "say hello", "iteration": 0})

        assert result["approved"] is True
        assert result["iteration"] == 1
        assert "review" in result

    @patch("agents.reviewer.get_llm")
    def test_rejects_when_needs_revision(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(
            "## Status: NEEDS_REVISION\n\n## Issues Found:\n1. Missing error handling"
        )
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.reviewer import reviewer

        result = reviewer({"code": "print('hello')", "plan": "say hello", "iteration": 0})

        assert result["approved"] is False
        assert result["iteration"] == 1

    @patch("agents.reviewer.get_llm")
    def test_increments_iteration(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse("## Status: APPROVED")
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.reviewer import reviewer

        result = reviewer({"code": "x", "iteration": 2})
        assert result["iteration"] == 3

    @patch("agents.reviewer.get_llm")
    def test_defaults_iteration_to_zero(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse("## Status: APPROVED")
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.reviewer import reviewer

        result = reviewer({"code": "x"})
        assert result["iteration"] == 1

    @patch("agents.reviewer.get_llm")
    def test_includes_code_and_plan_in_prompt(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse("APPROVED")
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.reviewer import reviewer

        reviewer({"code": "MY_UNIQUE_CODE", "plan": "MY_UNIQUE_PLAN"})

        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "MY_UNIQUE_CODE" in prompt_sent
        assert "MY_UNIQUE_PLAN" in prompt_sent

    @patch("agents.reviewer.get_llm")
    def test_approved_only_when_before_needs_revision(self, mock_get_llm):
        """APPROVED should be detected only if it appears before NEEDS_REVISION."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(
            "## Status: NEEDS_REVISION\n\nPreviously APPROVED but now needs fixes."
        )
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.reviewer import reviewer

        result = reviewer({"code": "x"})
        # APPROVED appears after NEEDS_REVISION, so it should not be approved
        assert result["approved"] is False

    @patch("agents.reviewer.invoke_with_retry")
    @patch("agents.reviewer.get_llm")
    def test_handles_llm_failure_gracefully(self, mock_get_llm, mock_retry):
        """Reviewer should return an error review instead of crashing on LLM failure."""
        mock_retry.side_effect = ConnectionError("Connection refused")
        mock_get_llm.return_value = (MagicMock(), "mock-model")

        from agents.reviewer import reviewer

        result = reviewer({"code": "x", "plan": "p", "iteration": 0})

        assert "[ERROR]" in result["review"]
        assert "Connection refused" in result["review"]
        assert result["approved"] is False
        assert result["iteration"] == 1
