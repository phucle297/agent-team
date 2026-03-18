"""LangGraph workflow orchestrating the AI agent team.

Features:
- Orchestrator decomposes tasks into parallel sub-tasks
- Parallel worker execution via LangGraph Send() API
- Aggregator merges all worker outputs into unified code
- Iteration loop: review -> fix -> review -> finalize (max 3 iterations)
- File system agent: creates/modifies files from generated code
- Tool agent: runs git, tests, terminal commands
- Memory: stores past runs for context
- TDD: agents follow red-green-refactor workflow
- Agent status tracking via event bus
"""

import logging
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from agents.filesystem import filesystem_agent
from agents.fixer import fixer
from agents.orchestrator import orchestrator
from agents.reviewer import reviewer
from agents.tools import tool_agent
from agents.worker import worker
from utils.agent_events import AgentStatus, tracker

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3


class AgentState(TypedDict, total=False):
    """State schema for the agent team workflow."""

    # Core fields
    input: str
    plan: str
    research: str
    code: str
    review: str
    final: str

    # Orchestrator / parallel workers
    sub_tasks: list
    worker_results: Annotated[list, add]

    # Iteration loop
    approved: bool
    iteration: int

    # File system
    workspace: str
    file_operations: list
    files_changed: list

    # Tools
    tool_results: list

    # Memory
    memory_context: str

    # Project context (detected from workspace)
    project_context: str


def fan_out_to_workers(state: Any) -> list[Send]:
    """Fan out sub-tasks to parallel worker nodes via Send()."""
    sub_tasks = state.get("sub_tasks", [])
    plan = state.get("plan", "")
    project_context = state.get("project_context", "")

    tracker.update(
        "orchestrator",
        AgentStatus.WORKING,
        f"Dispatching {len(sub_tasks)} worker(s)...",
    )

    sends = []
    for st in sub_tasks:
        sends.append(Send("worker", {
            "sub_task": st,
            "plan": plan,
            "project_context": project_context,
        }))

    logger.info("Fan-out: dispatching %d workers in parallel", len(sends))
    return sends


def aggregate(state: Any) -> dict:
    """Merge all worker results into a unified code block and research."""
    worker_results = state.get("worker_results", [])

    code_parts = []
    research_parts = []

    for wr in worker_results:
        task_id = wr.get("task_id", "?")
        title = wr.get("title", "?")
        output = wr.get("output", "")
        task_type = wr.get("type", "code")

        header = f"// === [{task_id}] {title} ===\n"

        if task_type == "research":
            research_parts.append(f"## {title}\n\n{output}\n")
        else:
            code_parts.append(f"{header}{output}\n")

    code = "\n".join(code_parts) if code_parts else ""
    research = "\n".join(research_parts) if research_parts else "No research context."

    logger.info("Aggregate: merged %d worker results (%d code, %d research)",
                len(worker_results), len(code_parts), len(research_parts))

    return {
        "code": code,
        "research": research,
    }


def should_continue_review(state: Any) -> str:
    """Decide whether to continue the review-fix loop or finalize."""
    if state.get("approved", False):
        logger.info("Review loop: code APPROVED at iteration %d", state.get("iteration", 0))
        return "approved"

    iteration = state.get("iteration", 0)
    if iteration >= MAX_ITERATIONS:
        logger.info("Review loop: max iterations (%d) reached, finalizing", MAX_ITERATIONS)
        return "max_iterations"

    logger.info("Review loop: NEEDS_REVISION, continuing to iteration %d", iteration + 1)
    return "needs_revision"


def finalize(state: Any) -> dict:
    """Produce the final output from the last code and review."""
    code = state.get("code", "")
    review = state.get("review", "")
    approved = state.get("approved", False)
    iteration = state.get("iteration", 0)
    files_changed = state.get("files_changed", [])
    tool_results = state.get("tool_results", [])
    worker_results = state.get("worker_results", [])

    status = "APPROVED" if approved else f"BEST EFFORT (after {iteration} iterations)"

    final_output = f"## Status: {status}\n\n"

    if worker_results:
        final_output += f"## Workers: {len(worker_results)} parallel tasks completed\n\n"

    final_output += f"## Final Code:\n\n{code}\n\n"
    final_output += f"## Last Review:\n\n{review}\n\n"

    if files_changed:
        final_output += "## Files Changed:\n\n"
        for f in files_changed:
            final_output += f"- {f}\n"
        final_output += "\n"

    if tool_results:
        final_output += "## Tool Results:\n\n"
        for r in tool_results:
            status_str = r.get("status", "unknown")
            cmd = r.get("command", "?")
            final_output += f"- [{status_str}] `{cmd}`\n"

    return {"final": final_output}


def build_graph():
    """Build the LangGraph workflow with parallel worker execution.

    Flow:
        entry -> orchestrator -> [worker x N] (parallel via Send)
        -> aggregate -> reviewer
        reviewer -> (approved?) -> filesystem -> tools -> finalize -> END
        reviewer -> (needs revision?) -> fixer -> reviewer (loop)
        reviewer -> (max iterations?) -> filesystem -> tools -> finalize -> END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("worker", worker)
    graph.add_node("aggregate", aggregate)
    graph.add_node("reviewer", reviewer)
    graph.add_node("fixer", fixer)
    graph.add_node("filesystem", filesystem_agent)
    graph.add_node("tools", tool_agent)
    graph.add_node("finalize", finalize)

    # Entry: start with orchestrator
    graph.set_entry_point("orchestrator")

    # Orchestrator fans out to parallel workers via Send()
    graph.add_conditional_edges(
        "orchestrator",
        fan_out_to_workers,
        ["worker"],
    )

    # All workers feed into aggregate
    graph.add_edge("worker", "aggregate")

    # Aggregate feeds into reviewer
    graph.add_edge("aggregate", "reviewer")

    # Review loop with conditional edges
    graph.add_conditional_edges(
        "reviewer",
        should_continue_review,
        {
            "approved": "filesystem",
            "needs_revision": "fixer",
            "max_iterations": "filesystem",
        },
    )

    # Fixer feeds back into reviewer (iteration loop)
    graph.add_edge("fixer", "reviewer")

    # After filesystem, run tools (e.g., tests), then finalize
    graph.add_edge("filesystem", "tools")
    graph.add_edge("tools", "finalize")

    # Finalize goes to END
    graph.add_edge("finalize", END)

    return graph.compile()
