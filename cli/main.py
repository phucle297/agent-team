"""CLI entry point for the `agents` command.

Usage:
    agents              Launch the interactive TUI in the current directory
    agents --sessions   List past sessions
    agents --rollback   Rollback the last session
"""

import logging
import os
import sys


def _configure_logging(workspace: str) -> None:
    """Configure logging to write to <workspace>/logs/agent_team.log.

    Creates the logs directory if it doesn't exist. All agent activity,
    errors, and tracebacks are captured here for debugging.
    """
    log_dir = os.path.join(workspace, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "agent_team.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
        ],
        force=True,  # Override any prior basicConfig
    )


def main():
    """Entry point for the `agents` command."""
    # Ensure the agent-team project root is in sys.path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Load .env from the agent-team project (API keys)
    from dotenv import load_dotenv

    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)

    # Parse simple args
    args = sys.argv[1:]

    if "--sessions" in args:
        _show_sessions()
        return

    if "--rollback" in args:
        _rollback_last()
        return

    if "--help" in args or "-h" in args:
        _show_help()
        return

    # Default: launch the TUI in the current working directory
    _launch_tui()


def _launch_tui():
    """Launch the Textual TUI app."""
    from cli.app import AgentTeamApp

    workspace = os.getcwd()
    _configure_logging(workspace)
    app = AgentTeamApp(workspace=workspace)
    app.run()


def _show_sessions():
    """Show session history in a simple text format."""
    from utils.sessions import list_sessions

    sessions = list_sessions()
    if not sessions:
        print("No sessions found.")
        return

    print(f"\n{'ID':<30} {'Status':<15} {'Task':<40} {'Created':<20}")
    print("-" * 105)
    for s in sessions:
        sid = s.get("id", "?")[:28]
        status = s.get("status", "?")
        task = s.get("task", "")[:38]
        created = s.get("created_at", "")[:19]
        print(f"{sid:<30} {status:<15} {task:<40} {created:<20}")
    print()


def _rollback_last():
    """Rollback the most recent session."""
    from utils.sessions import list_sessions, mark_rolled_back
    from utils.snapshots import rollback_session

    sessions = list_sessions(limit=1)
    if not sessions:
        print("No sessions to rollback.")
        return

    last = sessions[0]
    session_id = last["id"]
    print(f"Rolling back session: {session_id}")
    print(f"  Task: {last.get('task', '?')}")

    result = rollback_session(session_id)
    mark_rolled_back(session_id)

    for f in result.get("restored", []):
        print(f"  Restored: {f}")
    for f in result.get("deleted", []):
        print(f"  Deleted: {f}")
    for e in result.get("errors", []):
        print(f"  Error: {e}")

    print("Rollback complete.")


def _show_help():
    """Show help text."""
    print("""
Agent Team - AI-Powered Development with TDD

Usage:
    agents              Launch interactive TUI in the current directory
    agents --sessions   List past sessions
    agents --rollback   Rollback the most recent session
    agents --help       Show this help

The TUI will detect your project's language, framework, and test setup
automatically, then use an AI agent team to implement your tasks using
a strict TDD (red-green-refactor) approach.

Keyboard shortcuts in TUI:
    Ctrl+J    Submit task
    F1        Help
    F2        Session history & rollback
    F3        Quick rollback last session
    Ctrl+C    Quit
""")


if __name__ == "__main__":
    main()
