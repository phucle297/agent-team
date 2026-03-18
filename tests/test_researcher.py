"""Tests for agents/researcher.py - Researcher agent."""

from unittest.mock import MagicMock, patch

from tests.conftest import FakeLLMResponse


class TestResearcher:
    @patch("agents.researcher.get_llm")
    def test_returns_research_from_llm(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse(
            "JWT best practices: use RS256, rotate keys..."
        )
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.researcher import researcher

        result = researcher({"plan": "Research JWT auth"})

        assert "research" in result
        assert "JWT best practices" in result["research"]

    @patch("agents.researcher.get_llm")
    def test_includes_plan_in_prompt(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse("research")
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.researcher import researcher

        researcher({"plan": "Design database schema"})

        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "Design database schema" in prompt_sent

    @patch("agents.researcher.get_llm")
    def test_handles_empty_plan(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse("general research")
        mock_get_llm.return_value = (mock_llm, "mock-model")

        from agents.researcher import researcher

        result = researcher({})
        assert "research" in result
