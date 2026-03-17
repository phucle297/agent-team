"""Tests for agents/coder.py - Coder agent."""

from unittest.mock import MagicMock, patch

from tests.conftest import FakeLLMResponse


class TestCoder:
    @patch("agents.coder.get_claude")
    def test_returns_code_from_llm(self, mock_get_claude):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse("def hello():\n    print('hello')")
        mock_get_claude.return_value = mock_llm

        from agents.coder import coder

        result = coder({"plan": "Write a hello function"})

        assert "code" in result
        assert "def hello()" in result["code"]

    @patch("agents.coder.get_claude")
    def test_includes_plan_in_prompt(self, mock_get_claude):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse("code")
        mock_get_claude.return_value = mock_llm

        from agents.coder import coder

        coder({"plan": "Create a REST API with JWT"})

        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "Create a REST API with JWT" in prompt_sent

    @patch("agents.coder.get_claude")
    def test_includes_research_in_prompt(self, mock_get_claude):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse("code")
        mock_get_claude.return_value = mock_llm

        from agents.coder import coder

        coder({"plan": "build API", "research": "Use Express.js framework"})

        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "Use Express.js framework" in prompt_sent

    @patch("agents.coder.get_claude")
    def test_defaults_research_when_missing(self, mock_get_claude):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse("code")
        mock_get_claude.return_value = mock_llm

        from agents.coder import coder

        coder({"plan": "build API"})

        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "No research context available." in prompt_sent

    @patch("agents.coder.get_claude")
    def test_handles_empty_plan(self, mock_get_claude):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = FakeLLMResponse("some code")
        mock_get_claude.return_value = mock_llm

        from agents.coder import coder

        result = coder({})
        assert "code" in result
