"""Tests for agents/fixer.py - Code fixer agent."""

from unittest.mock import MagicMock, patch

from tests.conftest import FakeLLMResponse


class TestFixer:
    @patch("agents.fixer.get_llm")
    def test_returns_fixed_code(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(
            "def hello():\n    try:\n        print('hello')\n    except Exception as e:\n        raise"
        )
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.fixer import fixer

        result = fixer({
            "code": "def hello():\n    print('hello')",
            "review": "Add error handling",
            "plan": "say hello safely",
        })

        assert "code" in result
        assert "try:" in result["code"]

    @patch("agents.fixer.get_llm")
    def test_includes_all_context_in_prompt(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse("fixed")
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.fixer import fixer

        fixer({
            "code": "ORIGINAL_CODE",
            "review": "REVIEW_FEEDBACK",
            "plan": "ORIGINAL_PLAN",
        })

        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "ORIGINAL_CODE" in prompt_sent
        assert "REVIEW_FEEDBACK" in prompt_sent
        assert "ORIGINAL_PLAN" in prompt_sent

    @patch("agents.fixer.get_llm")
    def test_handles_empty_state(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse("fixed code")
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.fixer import fixer

        result = fixer({})
        assert "code" in result

    @patch("agents.fixer.invoke_with_retry")
    @patch("agents.fixer.get_llm")
    def test_handles_llm_failure_gracefully(self, mock_get_llm, mock_retry):
        """Fixer should return original code instead of crashing on LLM failure."""
        mock_retry.side_effect = ConnectionError("Connection refused")
        mock_get_llm.return_value = (MagicMock(), "mock-model")

        from agents.fixer import fixer

        result = fixer({
            "code": "original code here",
            "review": "fix something",
            "plan": "the plan",
        })

        # Should return the original code, not crash
        assert result["code"] == "original code here"
