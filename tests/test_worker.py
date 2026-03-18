"""Tests for agents/worker.py - Parallel worker agent that executes sub-tasks."""

from unittest.mock import MagicMock, patch

from tests.conftest import FakeLLMResponse


class TestWorker:
    """Test the worker agent that executes a single sub-task."""

    @patch("agents.worker.get_llm")
    def test_returns_worker_result(self, mock_get_llm):
        from agents.worker import worker

        llm = MagicMock()
        llm.invoke.return_value = FakeLLMResponse("function Board() { return <div>Board</div> }")
        mock_get_llm.return_value = (llm, "mock-model")

        state = {
            "sub_task": {
                "id": "task_1",
                "title": "Create Board component",
                "description": "Build the kanban board UI",
                "type": "code",
            },
            "project_context": "React project",
            "plan": "Build a kanban board",
        }
        result = worker(state)

        assert "worker_results" in result
        assert len(result["worker_results"]) == 1
        assert result["worker_results"][0]["task_id"] == "task_1"
        assert "Board" in result["worker_results"][0]["output"]

    @patch("agents.worker.get_llm")
    def test_includes_task_info_in_prompt(self, mock_get_llm):
        from agents.worker import worker

        llm = MagicMock()
        llm.invoke.return_value = FakeLLMResponse("code output")
        mock_get_llm.return_value = (llm, "mock-model")

        state = {
            "sub_task": {
                "id": "t1",
                "title": "Setup database",
                "description": "Create PostgreSQL schema",
                "type": "code",
            },
            "project_context": "Python/FastAPI",
            "plan": "Overall plan",
        }
        worker(state)

        prompt = llm.invoke.call_args[0][0]
        assert "Setup database" in prompt
        assert "Create PostgreSQL schema" in prompt
        assert "Python/FastAPI" in prompt

    @patch("agents.worker.get_llm")
    def test_handles_research_type(self, mock_get_llm):
        from agents.worker import worker

        llm = MagicMock()
        llm.invoke.return_value = FakeLLMResponse("Research findings here")
        mock_get_llm.return_value = (llm, "mock-model")

        state = {
            "sub_task": {
                "id": "r1",
                "title": "Research drag-drop libs",
                "description": "Find best React DnD library",
                "type": "research",
            },
            "project_context": "",
            "plan": "Plan",
        }
        result = worker(state)

        assert result["worker_results"][0]["type"] == "research"

    @patch("agents.worker.get_llm")
    def test_handles_empty_sub_task(self, mock_get_llm):
        from agents.worker import worker

        llm = MagicMock()
        llm.invoke.return_value = FakeLLMResponse("output")
        mock_get_llm.return_value = (llm, "mock-model")

        state = {
            "sub_task": {"id": "t1", "title": "", "description": "", "type": "code"},
            "project_context": "",
            "plan": "",
        }
        result = worker(state)

        assert len(result["worker_results"]) == 1

    @patch("agents.worker.get_llm")
    def test_updates_agent_tracker(self, mock_get_llm):
        from agents.worker import worker
        from utils.agent_events import AgentStatus, AgentTracker

        llm = MagicMock()
        llm.invoke.return_value = FakeLLMResponse("code")
        mock_get_llm.return_value = (llm, "mock-model")

        tracker = AgentTracker()
        tracker.reset()
        tracker.register("worker_task_1")

        events = []
        tracker.subscribe(lambda name, status, detail: events.append((name, status)))

        state = {
            "sub_task": {"id": "task_1", "title": "Build X", "description": "desc", "type": "code"},
            "project_context": "",
            "plan": "",
            "_tracker": tracker,
        }
        worker(state)

        # Should have WORKING and DONE events
        statuses = [e[1] for e in events if e[0] == "worker_task_1"]
        assert AgentStatus.WORKING in statuses
        assert AgentStatus.DONE in statuses

    @patch("agents.worker.invoke_with_retry")
    @patch("agents.worker.get_llm")
    def test_handles_llm_failure_gracefully(self, mock_get_llm, mock_retry):
        """Worker should return an error result instead of crashing on LLM failure."""
        from agents.worker import worker
        from utils.agent_events import AgentStatus, AgentTracker

        mock_retry.side_effect = ConnectionError("Connection refused")
        mock_get_llm.return_value = (MagicMock(), "mock-model")

        tracker = AgentTracker()
        tracker.reset()
        tracker.register("worker_task_1")

        events = []
        tracker.subscribe(lambda name, status, detail: events.append((name, status)))

        state = {
            "sub_task": {"id": "task_1", "title": "Build X", "description": "desc", "type": "code"},
            "project_context": "",
            "plan": "",
            "_tracker": tracker,
        }
        result = worker(state)

        # Should still return a result, not crash
        assert "worker_results" in result
        assert len(result["worker_results"]) == 1
        assert "[ERROR]" in result["worker_results"][0]["output"]
        assert "Connection refused" in result["worker_results"][0]["output"]

        # Should have WORKING then ERROR status
        statuses = [e[1] for e in events if e[0] == "worker_task_1"]
        assert AgentStatus.WORKING in statuses
        assert AgentStatus.ERROR in statuses
