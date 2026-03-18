"""Tests for graph/workflow.py - LangGraph workflow structure, routing, finalize."""

from unittest.mock import MagicMock, patch

from graph.workflow import (
    MAX_ITERATIONS,
    AgentState,
    aggregate,
    build_graph,
    fan_out_to_workers,
    finalize,
    should_continue_review,
)


class TestShouldContinueReview:
    def test_returns_approved_when_approved(self):
        state = {"approved": True, "iteration": 1}
        assert should_continue_review(state) == "approved"

    def test_returns_needs_revision_when_not_approved_and_under_max(self):
        state = {"approved": False, "iteration": 1}
        assert should_continue_review(state) == "needs_revision"

    def test_returns_max_iterations_when_at_max(self):
        state = {"approved": False, "iteration": MAX_ITERATIONS}
        assert should_continue_review(state) == "max_iterations"

    def test_returns_max_iterations_when_over_max(self):
        state = {"approved": False, "iteration": MAX_ITERATIONS + 1}
        assert should_continue_review(state) == "max_iterations"

    def test_defaults_to_needs_revision_with_empty_state(self):
        assert should_continue_review({}) == "needs_revision"

    def test_approved_takes_precedence_over_max_iterations(self):
        state = {"approved": True, "iteration": MAX_ITERATIONS + 10}
        assert should_continue_review(state) == "approved"


class TestFanOutToWorkers:
    def test_creates_send_per_sub_task(self):
        state = {
            "sub_tasks": [
                {"id": "t1", "title": "A", "description": "d1", "type": "code"},
                {"id": "t2", "title": "B", "description": "d2", "type": "research"},
            ],
            "plan": "the plan",
            "project_context": "ctx",
        }
        sends = fan_out_to_workers(state)
        assert len(sends) == 2
        assert sends[0].node == "worker"
        assert sends[1].node == "worker"
        assert sends[0].arg["sub_task"]["id"] == "t1"
        assert sends[1].arg["sub_task"]["id"] == "t2"

    def test_passes_plan_and_context_to_each_worker(self):
        state = {
            "sub_tasks": [{"id": "t1", "title": "A", "description": "d", "type": "code"}],
            "plan": "master plan",
            "project_context": "react project",
        }
        sends = fan_out_to_workers(state)
        assert sends[0].arg["plan"] == "master plan"
        assert sends[0].arg["project_context"] == "react project"

    def test_returns_empty_list_when_no_sub_tasks(self):
        sends = fan_out_to_workers({"sub_tasks": []})
        assert sends == []


class TestAggregate:
    def test_merges_code_results(self):
        state = {
            "worker_results": [
                {"task_id": "t1", "title": "Board", "type": "code", "output": "function Board() {}"},
                {"task_id": "t2", "title": "Card", "type": "code", "output": "function Card() {}"},
            ]
        }
        result = aggregate(state)
        assert "Board" in result["code"]
        assert "Card" in result["code"]

    def test_separates_research_from_code(self):
        state = {
            "worker_results": [
                {"task_id": "t1", "title": "Code task", "type": "code", "output": "code here"},
                {"task_id": "r1", "title": "Research task", "type": "research", "output": "findings"},
            ]
        }
        result = aggregate(state)
        assert "code here" in result["code"]
        assert "findings" in result["research"]

    def test_handles_empty_results(self):
        result = aggregate({"worker_results": []})
        assert result["code"] == ""
        assert result["research"] == "No research context."


class TestFinalize:
    def test_approved_output(self):
        state = {
            "code": "print('hello')",
            "review": "Looks good",
            "approved": True,
            "iteration": 1,
        }
        result = finalize(state)
        assert "APPROVED" in result["final"]
        assert "print('hello')" in result["final"]
        assert "Looks good" in result["final"]

    def test_best_effort_output(self):
        state = {
            "code": "print('hello')",
            "review": "Still has issues",
            "approved": False,
            "iteration": 3,
        }
        result = finalize(state)
        assert "BEST EFFORT" in result["final"]
        assert "after 3 iterations" in result["final"]

    def test_handles_empty_state(self):
        result = finalize({})
        assert "final" in result
        assert "BEST EFFORT" in result["final"]

    def test_includes_worker_count(self):
        state = {
            "code": "code",
            "review": "review",
            "approved": True,
            "iteration": 1,
            "worker_results": [
                {"task_id": "t1", "output": "out1"},
                {"task_id": "t2", "output": "out2"},
            ],
        }
        result = finalize(state)
        assert "2 parallel tasks" in result["final"]


class TestBuildGraph:
    def test_graph_compiles(self):
        """Graph should compile without errors."""
        app = build_graph()
        assert app is not None

    def test_graph_has_all_nodes(self):
        app = build_graph()
        node_names = set(app.get_graph().nodes.keys())
        expected = {
            "__start__", "__end__",
            "orchestrator", "worker", "aggregate",
            "reviewer", "fixer",
            "filesystem", "tools", "finalize",
        }
        assert expected == node_names

    def test_graph_has_orchestrator_entry(self):
        app = build_graph()
        edges = [(e.source, e.target) for e in app.get_graph().edges]
        assert ("__start__", "orchestrator") in edges

    def test_graph_has_worker_to_aggregate(self):
        app = build_graph()
        edges = [(e.source, e.target) for e in app.get_graph().edges]
        assert ("worker", "aggregate") in edges

    def test_graph_has_aggregate_to_reviewer(self):
        app = build_graph()
        edges = [(e.source, e.target) for e in app.get_graph().edges]
        assert ("aggregate", "reviewer") in edges

    def test_graph_has_fixer_loop(self):
        app = build_graph()
        edges = [(e.source, e.target) for e in app.get_graph().edges]
        assert ("fixer", "reviewer") in edges

    def test_graph_has_conditional_edges_from_reviewer(self):
        app = build_graph()
        edges = [(e.source, e.target) for e in app.get_graph().edges]
        assert ("reviewer", "filesystem") in edges
        assert ("reviewer", "fixer") in edges

    def test_graph_ends_at_finalize(self):
        app = build_graph()
        edges = [(e.source, e.target) for e in app.get_graph().edges]
        assert ("finalize", "__end__") in edges

    def test_graph_has_orchestrator_to_worker_conditional(self):
        app = build_graph()
        edges = [(e.source, e.target) for e in app.get_graph().edges]
        assert ("orchestrator", "worker") in edges


class TestGraphIntegration:
    """Integration test with mocked LLMs to verify the full flow."""

    @patch("agents.tools.get_llm")
    @patch("agents.filesystem.get_llm")
    @patch("agents.reviewer.get_llm")
    @patch("agents.worker.get_llm")
    @patch("agents.orchestrator.get_llm")
    def test_full_flow_approved_first_pass(
        self, mock_orchestrator_llm, mock_worker_llm, mock_reviewer_llm,
        mock_filesystem_llm, mock_tools_llm
    ):
        from tests.conftest import FakeLLMResponse

        # Orchestrator returns 2 sub-tasks
        orch_llm = MagicMock()
        orch_llm.invoke.return_value = FakeLLMResponse(
            '[{"id": "t1", "title": "Part A", "description": "do A", "type": "code"},'
            ' {"id": "t2", "title": "Part B", "description": "do B", "type": "research"}]'
        )
        mock_orchestrator_llm.return_value = (orch_llm, "mock-model")

        # Worker
        worker_llm = MagicMock()
        worker_llm.invoke.return_value = FakeLLMResponse("def greet(): print('hi')")
        mock_worker_llm.return_value = (worker_llm, "mock-model")

        # Reviewer - APPROVED
        reviewer_llm = MagicMock()
        reviewer_llm.invoke.return_value = FakeLLMResponse("## Status: APPROVED\n\nLooks great!")
        mock_reviewer_llm.return_value = (reviewer_llm, "mock-model")

        # Filesystem agent
        filesystem_llm = MagicMock()
        filesystem_llm.invoke.return_value = FakeLLMResponse("[]")
        mock_filesystem_llm.return_value = (filesystem_llm, "mock-model")

        # Tool agent
        tools_llm = MagicMock()
        tools_llm.invoke.return_value = FakeLLMResponse("[]")
        mock_tools_llm.return_value = (tools_llm, "mock-model")

        app = build_graph()
        result = app.invoke({
            "input": "Say hello",
            "plan": "",
            "research": "",
            "code": "",
            "review": "",
            "final": "",
            "approved": False,
            "iteration": 0,
            "sub_tasks": [],
            "worker_results": [],
        })

        assert result["approved"] is True
        assert result["iteration"] == 1
        assert "APPROVED" in result["final"]
        # Both workers should have run
        assert len(result["worker_results"]) == 2

    @patch("agents.tools.get_llm")
    @patch("agents.filesystem.get_llm")
    @patch("agents.fixer.get_llm")
    @patch("agents.reviewer.get_llm")
    @patch("agents.worker.get_llm")
    @patch("agents.orchestrator.get_llm")
    def test_iteration_loop_fix_then_approve(
        self, mock_orchestrator_llm, mock_worker_llm, mock_reviewer_llm,
        mock_fixer_llm, mock_filesystem_llm, mock_tools_llm
    ):
        from tests.conftest import FakeLLMResponse

        # Orchestrator - 1 sub-task
        orch_llm = MagicMock()
        orch_llm.invoke.return_value = FakeLLMResponse(
            '[{"id": "t1", "title": "Build it", "description": "do it", "type": "code"}]'
        )
        mock_orchestrator_llm.return_value = (orch_llm, "mock-model")

        # Worker
        worker_llm = MagicMock()
        worker_llm.invoke.return_value = FakeLLMResponse("bad code v1")
        mock_worker_llm.return_value = (worker_llm, "mock-model")

        # Reviewer - first rejects, then approves
        reviewer_llm = MagicMock()
        reviewer_llm.invoke.side_effect = [
            FakeLLMResponse("## Status: NEEDS_REVISION\n\nFix error handling"),
            FakeLLMResponse("## Status: APPROVED\n\nNow it's good!"),
        ]
        mock_reviewer_llm.return_value = (reviewer_llm, "mock-model")

        # Fixer
        fixer_llm = MagicMock()
        fixer_llm.invoke.return_value = FakeLLMResponse("fixed code v2")
        mock_fixer_llm.return_value = (fixer_llm, "mock-model")

        # Filesystem agent
        filesystem_llm = MagicMock()
        filesystem_llm.invoke.return_value = FakeLLMResponse("[]")
        mock_filesystem_llm.return_value = (filesystem_llm, "mock-model")

        # Tool agent
        tools_llm = MagicMock()
        tools_llm.invoke.return_value = FakeLLMResponse("[]")
        mock_tools_llm.return_value = (tools_llm, "mock-model")

        app = build_graph()
        result = app.invoke({
            "input": "Build feature",
            "plan": "",
            "research": "",
            "code": "",
            "review": "",
            "final": "",
            "approved": False,
            "iteration": 0,
            "sub_tasks": [],
            "worker_results": [],
        })

        assert result["approved"] is True
        assert result["iteration"] == 2
        assert "APPROVED" in result["final"]

    @patch("agents.tools.get_llm")
    @patch("agents.filesystem.get_llm")
    @patch("agents.fixer.get_llm")
    @patch("agents.reviewer.get_llm")
    @patch("agents.worker.get_llm")
    @patch("agents.orchestrator.get_llm")
    def test_max_iterations_stops_loop(
        self, mock_orchestrator_llm, mock_worker_llm, mock_reviewer_llm,
        mock_fixer_llm, mock_filesystem_llm, mock_tools_llm
    ):
        from tests.conftest import FakeLLMResponse

        orch_llm = MagicMock()
        orch_llm.invoke.return_value = FakeLLMResponse(
            '[{"id": "t1", "title": "Task", "description": "d", "type": "code"}]'
        )
        mock_orchestrator_llm.return_value = (orch_llm, "mock-model")

        worker_llm = MagicMock()
        worker_llm.invoke.return_value = FakeLLMResponse("code")
        mock_worker_llm.return_value = (worker_llm, "mock-model")

        # Reviewer always rejects
        reviewer_llm = MagicMock()
        reviewer_llm.invoke.return_value = FakeLLMResponse("## Status: NEEDS_REVISION\n\nStill bad")
        mock_reviewer_llm.return_value = (reviewer_llm, "mock-model")

        fixer_llm = MagicMock()
        fixer_llm.invoke.return_value = FakeLLMResponse("attempted fix")
        mock_fixer_llm.return_value = (fixer_llm, "mock-model")

        filesystem_llm = MagicMock()
        filesystem_llm.invoke.return_value = FakeLLMResponse("[]")
        mock_filesystem_llm.return_value = (filesystem_llm, "mock-model")

        tools_llm = MagicMock()
        tools_llm.invoke.return_value = FakeLLMResponse("[]")
        mock_tools_llm.return_value = (tools_llm, "mock-model")

        app = build_graph()
        result = app.invoke({
            "input": "Impossible task",
            "plan": "",
            "research": "",
            "code": "",
            "review": "",
            "final": "",
            "approved": False,
            "iteration": 0,
            "sub_tasks": [],
            "worker_results": [],
        })

        assert result["approved"] is False
        assert result["iteration"] == MAX_ITERATIONS
        assert "BEST EFFORT" in result["final"]
