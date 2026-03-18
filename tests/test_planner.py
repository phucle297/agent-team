"""Tests for agents/planner.py - Planner agent."""

from unittest.mock import MagicMock, patch

from tests.conftest import FakeLLMResponse


class TestPlanner:
    @patch("agents.planner.get_llm")
    def test_returns_plan_from_llm(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(
            "1. Design API (type: code)\n2. Research JWT (type: research)"
        )
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.planner import planner

        result = planner({"input": "Build a REST API"})

        assert "plan" in result
        assert "Design API" in result["plan"]

    @patch("agents.planner.get_llm")
    def test_passes_task_to_prompt(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse("plan")
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.planner import planner

        planner({"input": "Build a CLI tool"})

        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "Build a CLI tool" in prompt_sent

    @patch("agents.planner.get_llm")
    def test_loads_prompt_template(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse("plan")
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.planner import planner

        planner({"input": "test"})

        prompt_sent = mock_llm.invoke.call_args[0][0]
        # Should contain template text from prompts/planner.txt
        assert "senior software architect" in prompt_sent.lower() or "actionable steps" in prompt_sent.lower()

    @patch("agents.planner.get_llm")
    def test_raises_on_missing_input(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.planner import planner
        import pytest

        with pytest.raises(KeyError):
            planner({})
