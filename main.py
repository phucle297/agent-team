"""AI Agent Team - Main entry point.

A multi-agent system orchestrated by LangGraph:
- Orchestrator (Google Gemini) decomposes tasks into parallel sub-tasks
- Workers (Claude) execute sub-tasks in parallel via Send() API
- Reviewer (Claude) validates and improves output
- Fixer (Claude) applies review feedback in iteration loops

Features:
- Parallel worker execution via LangGraph Send() API (max 6 workers)
- Review -> fix -> review iteration loop (max 3 rounds)
- File system agent for repo modifications
- Memory of past runs (JSON)
- Tool execution (git, terminal, tests)
- Agent status tracking via event bus
"""

import logging
import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "agent_team.log")),
    ],
)
logger = logging.getLogger(__name__)


def run_agent_team(task: str, workspace: str | None = None) -> str:
    """Run the agent team on a given task.

    Args:
        task: The task description for the agents to work on.
        workspace: Optional workspace directory for file operations.

    Returns:
        The final output from the agent team.
    """
    from graph.workflow import build_graph
    from utils.context import get_context_prompt
    from utils.memory import get_context_for_task, save_run

    logger.info("=" * 60)
    logger.info("Starting AI Agent Team")
    logger.info("Task: %s", task)
    logger.info("=" * 60)

    # Get memory context from past runs
    memory_context = get_context_for_task(task)

    # Scan project context from workspace
    ws = workspace or os.getcwd()
    project_context = get_context_prompt(ws)

    # Build and run the workflow
    app = build_graph()

    initial_state = {
        "input": task,
        "plan": "",
        "research": "",
        "code": "",
        "review": "",
        "final": "",
        "approved": False,
        "iteration": 0,
        "workspace": ws,
        "file_operations": [],
        "files_changed": [],
        "tool_results": [],
        "memory_context": memory_context,
        "project_context": project_context,
        "sub_tasks": [],
        "worker_results": [],
    }

    logger.info("Running workflow...")
    result = app.invoke(initial_state)

    # Save run to memory
    save_run(result)

    final_output = result.get("final", "No output generated.")
    logger.info("=" * 60)
    logger.info("Agent Team Complete")
    logger.info("Approved: %s", result.get("approved", False))
    logger.info("Iterations: %s", result.get("iteration", 0))
    logger.info("=" * 60)

    return final_output


def main():
    """Interactive CLI entry point."""
    print("\n" + "=" * 60)
    print("  AI Agent Team - Local Multi-Agent System")
    print("  Orchestrator + Parallel Workers + Reviewer")
    print("=" * 60 + "\n")

    # Check for API keys
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY not set in .env")
    if not os.environ.get("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY not set in .env")

    if len(sys.argv) > 1:
        # Task provided as command-line argument
        task = " ".join(sys.argv[1:])
    else:
        # Interactive mode
        task = input("Enter your task: ").strip()

    if not task:
        print("No task provided. Exiting.")
        sys.exit(1)

    print(f"\nTask: {task}\n")
    print("Running agent team...\n")

    result = run_agent_team(task)

    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
