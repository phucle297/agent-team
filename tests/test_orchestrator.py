"""Tests for agents/orchestrator.py - Task decomposition into parallel sub-tasks."""

from unittest.mock import MagicMock, patch

from tests.conftest import FakeLLMResponse


class TestOrchestrator:
    """Test the orchestrator agent that splits tasks into sub-tasks."""

    @patch("agents.orchestrator.get_google")
    def test_returns_sub_tasks_list(self, mock_llm_factory):
        from agents.orchestrator import orchestrator

        llm = MagicMock()
        llm.invoke.return_value = FakeLLMResponse(
            '[\n'
            '  {"id": "task_1", "title": "Setup project", "description": "Init React app", "type": "code"},\n'
            '  {"id": "task_2", "title": "Create board", "description": "Build board component", "type": "code"}\n'
            ']'
        )
        mock_llm_factory.return_value = llm

        state = {"input": "Create a Jira-like kanban board", "project_context": ""}
        result = orchestrator(state)

        assert "sub_tasks" in result
        assert len(result["sub_tasks"]) == 2
        assert result["sub_tasks"][0]["id"] == "task_1"
        assert result["sub_tasks"][1]["id"] == "task_2"

    @patch("agents.orchestrator.get_google")
    def test_sets_plan_from_sub_tasks(self, mock_llm_factory):
        from agents.orchestrator import orchestrator

        llm = MagicMock()
        llm.invoke.return_value = FakeLLMResponse(
            '[{"id": "t1", "title": "Do X", "description": "desc", "type": "code"}]'
        )
        mock_llm_factory.return_value = llm

        state = {"input": "Build feature", "project_context": ""}
        result = orchestrator(state)

        assert "plan" in result
        assert "Do X" in result["plan"]

    @patch("agents.orchestrator.get_google")
    def test_handles_invalid_json_gracefully(self, mock_llm_factory):
        from agents.orchestrator import orchestrator

        llm = MagicMock()
        llm.invoke.return_value = FakeLLMResponse("Not valid JSON at all")
        mock_llm_factory.return_value = llm

        state = {"input": "Build something", "project_context": ""}
        result = orchestrator(state)

        # Should fall back to a single task
        assert "sub_tasks" in result
        assert len(result["sub_tasks"]) == 1
        assert result["sub_tasks"][0]["title"] == "Build something"

    @patch("agents.orchestrator.get_google")
    def test_handles_markdown_fenced_json(self, mock_llm_factory):
        from agents.orchestrator import orchestrator

        llm = MagicMock()
        llm.invoke.return_value = FakeLLMResponse(
            '```json\n'
            '[{"id": "t1", "title": "Task 1", "description": "desc", "type": "code"}]\n'
            '```'
        )
        mock_llm_factory.return_value = llm

        state = {"input": "Build it", "project_context": ""}
        result = orchestrator(state)

        assert len(result["sub_tasks"]) == 1
        assert result["sub_tasks"][0]["id"] == "t1"

    @patch("agents.orchestrator.get_google")
    def test_includes_project_context_in_prompt(self, mock_llm_factory):
        from agents.orchestrator import orchestrator

        llm = MagicMock()
        llm.invoke.return_value = FakeLLMResponse(
            '[{"id": "t1", "title": "T", "description": "d", "type": "code"}]'
        )
        mock_llm_factory.return_value = llm

        state = {"input": "Build feature", "project_context": "Python/FastAPI project"}
        orchestrator(state)

        prompt = llm.invoke.call_args[0][0]
        assert "Python/FastAPI project" in prompt

    @patch("agents.orchestrator.get_google")
    def test_caps_sub_tasks_at_max(self, mock_llm_factory):
        """Orchestrator should not create more than MAX_WORKERS sub-tasks."""
        from agents.orchestrator import MAX_WORKERS, orchestrator

        tasks = [
            {"id": f"t{i}", "title": f"Task {i}", "description": "d", "type": "code"}
            for i in range(20)
        ]
        import json
        llm = MagicMock()
        llm.invoke.return_value = FakeLLMResponse(json.dumps(tasks))
        mock_llm_factory.return_value = llm

        state = {"input": "Huge task", "project_context": ""}
        result = orchestrator(state)

        assert len(result["sub_tasks"]) <= MAX_WORKERS
