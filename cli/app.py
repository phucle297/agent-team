"""Textual TUI application for the AI Agent Team.

A Claude Code-inspired interactive terminal UI with:
- Task input area
- Live agent activity feed (selectable, copyable)
- Agent status panel showing which agents are active/idle/done
- File changes panel
- Session history and rollback
- Keyboard shortcuts
"""

import logging
import os
import re
from datetime import datetime

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    Static,
    TextArea,
)

from utils.agent_events import AgentStatus, AgentTracker, tracker
from utils.context import get_context_prompt, scan_project
from utils.sessions import (
    add_step,
    complete_session,
    create_session,
    fail_session,
    list_sessions,
    mark_rolled_back,
)
from utils.snapshots import (
    backup_file,
    cleanup_old_snapshots,
    create_snapshot,
    rollback_session,
)

logger = logging.getLogger(__name__)

# Status display symbols
STATUS_SYMBOLS = {
    AgentStatus.IDLE: "[dim]  IDLE[/dim]",
    AgentStatus.WORKING: "[bold yellow]  WORK[/bold yellow]",
    AgentStatus.DONE: "[bold green]  DONE[/bold green]",
    AgentStatus.ERROR: "[bold red]  ERR![/bold red]",
}


class TaskInput(TextArea):
    """TextArea subclass with Ctrl+Backspace bound to delete word left."""

    BINDINGS = [
        Binding("ctrl+backspace", "delete_word_left", "Delete word left", show=False),
    ]


class ActivityLog(TextArea):
    """Read-only TextArea for agent activity with copy support.

    Supports Ctrl+Shift+C and y (yank) to copy selected text.
    """

    BINDINGS = [
        Binding("ctrl+shift+c", "copy", "Copy", show=False, priority=True),
        Binding("y", "yank", "Yank", show=False),
    ]

    def action_yank(self) -> None:
        """Copy selected text to clipboard (vim-style yank)."""
        self.action_copy()


class SessionsScreen(ModalScreen):
    """Modal screen showing session history with rollback option."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("r", "rollback", "Rollback selected"),
    ]

    CSS = """
    SessionsScreen {
        align: center middle;
    }

    #sessions-container {
        width: 90%;
        height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #sessions-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        color: $text;
    }

    #sessions-table {
        height: 1fr;
    }

    #sessions-help {
        text-align: center;
        padding: 1;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="sessions-container"):
            yield Label("Session History (last 20)", id="sessions-title")
            yield DataTable(id="sessions-table")
            yield Label("[R] Rollback  [Esc] Close", id="sessions-help")

    def on_mount(self) -> None:
        table = self.query_one("#sessions-table", DataTable)
        table.add_columns("ID", "Task", "Status", "Time", "Files")
        table.cursor_type = "row"

        sessions = list_sessions()
        for s in sessions:
            task_preview = s.get("task", "")[:40]
            status = s.get("status", "?")
            created = s.get("created_at", "")[:19]
            files_count = len(s.get("files_changed", []))
            table.add_row(
                s.get("id", "?"),
                task_preview,
                status,
                created,
                str(files_count),
                key=s.get("id", ""),
            )

    def action_rollback(self) -> None:
        table = self.query_one("#sessions-table", DataTable)
        if table.row_count == 0:
            return

        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        session_id = str(row_key)

        result = rollback_session(session_id)
        mark_rolled_back(session_id)

        restored = len(result.get("restored", []))
        deleted = len(result.get("deleted", []))
        errors = len(result.get("errors", []))

        self.app.notify(
            f"Rollback: {restored} restored, {deleted} deleted, {errors} errors",
            title="Session Rolled Back",
        )
        self.dismiss()


class HelpScreen(ModalScreen):
    """Help screen with keyboard shortcuts."""

    BINDINGS = [Binding("escape", "dismiss", "Close")]

    CSS = """
    HelpScreen {
        align: center middle;
    }

    #help-container {
        width: 60%;
        height: 60%;
        background: $surface;
        border: thick $primary;
        padding: 2 3;
    }

    #help-title {
        text-align: center;
        text-style: bold;
        padding: 1;
    }

    #help-content {
        padding: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="help-container"):
            yield Label("Keyboard Shortcuts", id="help-title")
            yield Static(
                "[b]Ctrl+J[/b]  Submit task\n"
                "[b]F1[/b]  Show this help\n"
                "[b]F2[/b]  Session history & rollback\n"
                "[b]F3[/b]  Rollback last session\n"
                "[b]Ctrl+C[/b]  Quit\n\n"
                "[b]Text Editing:[/b]\n"
                "  Ctrl+Backspace  Delete word left\n"
                "  Ctrl+Shift+K    Delete line\n"
                "  Ctrl+A          Select all\n"
                "  Ctrl+C/V/X      Copy/Paste/Cut\n\n"
                "[b]Agents Flow:[/b]\n"
                "  Orchestrator -> Workers x N (parallel)\n"
                "  -> Aggregate -> Reviewer -> Fixer (loop)\n"
                "  -> Filesystem -> Tools -> Finalize\n\n"
                "[b]TDD Approach:[/b]\n"
                "  RED: Write failing tests first\n"
                "  GREEN: Write minimum code to pass\n"
                "  REFACTOR: Clean up, keep tests green",
                id="help-content",
            )

    def on_key(self, event) -> None:
        self.dismiss()


class AgentTeamApp(App):
    """The main Textual TUI application."""

    TITLE = "Agent Team"
    SUB_TITLE = "AI-Powered Development with TDD"

    CSS = """
    #main-container {
        height: 1fr;
    }

    #left-panel {
        width: 3fr;
        height: 1fr;
    }

    #right-panel {
        width: 1fr;
        height: 1fr;
        border-left: thick $primary;
    }

    #task-input-area {
        height: auto;
        max-height: 8;
        padding: 1 2;
        border-bottom: solid $primary;
    }

    #task-label {
        padding: 0 0 1 0;
        text-style: bold;
        color: $accent;
    }

    #task-input {
        width: 100%;
        height: 3;
    }

    #agent-log {
        height: 1fr;
        border-bottom: solid $primary;
    }

    #agent-log-title {
        padding: 0 1;
        text-style: bold;
        background: $primary;
        color: $text;
    }

    #activity-log {
        height: 1fr;
    }

    #files-panel {
        height: auto;
        max-height: 30%;
        padding: 0 1;
    }

    #files-title {
        text-style: bold;
        padding: 0 0 1 0;
        color: $accent;
    }

    #files-list {
        height: auto;
    }

    #agent-status-panel {
        padding: 1;
        height: auto;
        border-bottom: solid $primary;
    }

    #agent-status-title {
        text-style: bold;
        padding: 0 0 1 0;
        color: $accent;
    }

    #agent-status-content {
        height: auto;
    }

    #context-panel {
        padding: 1;
        height: auto;
    }

    #context-title {
        text-style: bold;
        padding: 0 0 1 0;
        color: $accent;
    }

    #context-info {
        height: auto;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: $primary;
        color: $text;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("f1", "help", "Help"),
        Binding("f2", "sessions", "Sessions"),
        Binding("f3", "rollback_last", "Rollback"),
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+j", "submit_task", "Submit", priority=True),
    ]

    def __init__(self, workspace: str | None = None):
        super().__init__()
        self.workspace = workspace or os.getcwd()
        self.current_session_id: str | None = None
        self._project_context: dict | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="left-panel"):
                with Container(id="task-input-area"):
                    yield Label("> Enter your task (Ctrl+J to submit):", id="task-label")
                    yield TaskInput(
                        id="task-input",
                    )
                with Container(id="agent-log"):
                    yield Label(" Agent Activity ", id="agent-log-title")
                    yield ActivityLog(
                        id="activity-log",
                        read_only=True,
                    )
                with Container(id="files-panel"):
                    yield Label("Files Changed", id="files-title")
                    yield Static("No changes yet.", id="files-list")
            with Vertical(id="right-panel"):
                with Container(id="agent-status-panel"):
                    yield Label("Agent Status", id="agent-status-title")
                    yield Static("No agents running.", id="agent-status-content")
                with Container(id="context-panel"):
                    yield Label("Project Context", id="context-title")
                    yield Static("Scanning...", id="context-info")
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app on mount."""
        self._update_status("Ready. Enter a task and press Ctrl+J to submit.")
        self.query_one("#task-input", TaskInput).focus()
        # Scan project context in background so TUI opens instantly
        self._scan_project_context_async()

    @work(thread=True)
    def _scan_project_context_async(self) -> None:
        """Scan project context in a background thread."""
        ctx = scan_project(self.workspace)
        self._project_context = ctx
        self.call_from_thread(self._display_context, ctx)

    def _display_context(self, ctx: dict) -> None:
        """Update the context panel with scan results (called from main thread)."""
        info_parts = []
        info_parts.append(f"[b]Dir:[/b] {ctx['workspace']}")

        if ctx.get("languages"):
            info_parts.append(f"[b]Lang:[/b] {', '.join(ctx['languages'])}")
        if ctx.get("frameworks"):
            info_parts.append(f"[b]Frameworks:[/b] {', '.join(ctx['frameworks'])}")
        if ctx.get("test_frameworks"):
            info_parts.append(f"[b]Testing:[/b] {', '.join(ctx['test_frameworks'])}")
        if ctx.get("has_git"):
            info_parts.append("[b]Git:[/b] yes")
        if ctx.get("key_files"):
            info_parts.append(f"[b]Key files:[/b]\n  " + "\n  ".join(ctx["key_files"][:10]))

        info_text = "\n".join(info_parts)
        self.query_one("#context-info", Static).update(info_text)

    def _update_status(self, text: str) -> None:
        """Update the status bar."""
        self.query_one("#status-bar", Static).update(f" {text}")

    def _log_agent(self, agent: str, message: str, style: str = "") -> None:
        """Log an agent activity message to the activity TextArea."""
        log = self.query_one("#activity-log", ActivityLog)
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Strip Rich markup tags for plain text display in TextArea
        plain_message = re.sub(r"\[/?[a-z_ ]+\]", "", message)

        line = f"{timestamp} [{agent}] {plain_message}\n"
        log.insert(line, log.document.end)

    def _update_files(self, files: list[str]) -> None:
        """Update the files changed panel."""
        if not files:
            return
        files_widget = self.query_one("#files-list", Static)
        display = "\n".join(f"  {f}" for f in files[-15:])  # Show last 15
        if len(files) > 15:
            display = f"  ... ({len(files) - 15} more)\n" + display
        files_widget.update(display)

    def _update_agent_status_display(self) -> None:
        """Refresh the agent status panel from the tracker."""
        statuses = tracker.get_all()
        if not statuses:
            self.query_one("#agent-status-content", Static).update("No agents running.")
            return

        lines = []
        for name, info in statuses.items():
            status = info.get("status", AgentStatus.IDLE)
            detail = info.get("detail", "")
            symbol = STATUS_SYMBOLS.get(status, "  ?")
            detail_str = f"  {detail}" if detail else ""
            lines.append(f"{symbol}  [b]{name}[/b]{detail_str}")

        self.query_one("#agent-status-content", Static).update("\n".join(lines))

    def _on_agent_event(self, agent_name: str, status: AgentStatus, detail: str) -> None:
        """Handle agent status change events (called from worker threads)."""
        self.call_from_thread(self._update_agent_status_display)
        self.call_from_thread(self._log_agent, agent_name, f"{status.value}: {detail}")

    def action_submit_task(self) -> None:
        """Handle task submission via Ctrl+J."""
        text_area = self.query_one("#task-input", TaskInput)
        task = text_area.text.strip()
        if not task:
            self.notify("Please enter a task.", severity="warning")
            return

        # Disable input while running
        text_area.disabled = True

        self._log_agent("System", f"Task: [b]{task}[/b]")
        self._update_status(f"Running: {task[:60]}...")

        # Run the agent workflow in a background worker
        self._run_agent_team(task)

    @work(thread=True)
    def _run_agent_team(self, task: str) -> None:
        """Run the agent team in a background thread."""
        from graph.workflow import build_graph
        from utils.context import get_context_prompt
        from utils.memory import get_context_for_task, save_run

        # Create session
        session = create_session(self.workspace, task)
        self.current_session_id = session["id"]

        # Create snapshot for rollback
        create_snapshot(session["id"], self.workspace)
        cleanup_old_snapshots(keep=20)

        self.call_from_thread(
            self._log_agent, "System", f"Session: {session['id']}"
        )

        try:
            # Reset tracker and register all known agents
            tracker.reset()
            tracker.register("orchestrator")
            tracker.register("reviewer")
            tracker.register("fixer")
            tracker.register("filesystem")
            tracker.register("tools")

            # Subscribe TUI to agent events
            tracker.subscribe(self._on_agent_event)
            self.call_from_thread(self._update_agent_status_display)

            # Get context
            memory_context = get_context_for_task(task)
            project_context = get_context_prompt(self.workspace)

            # Build workflow
            self.call_from_thread(self._log_agent, "System", "Building workflow...")

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
                "workspace": self.workspace,
                "file_operations": [],
                "files_changed": [],
                "tool_results": [],
                "memory_context": memory_context,
                "project_context": project_context,
                "sub_tasks": [],
                "worker_results": [],
            }

            self.call_from_thread(
                self._log_agent, "System", "Starting agent pipeline..."
            )

            # Run the graph
            result = app.invoke(initial_state)

            # Log results
            approved = result.get("approved", False)
            iterations = result.get("iteration", 0)
            files_changed = result.get("files_changed", [])
            worker_results = result.get("worker_results", [])

            # Back up any changed files for rollback
            for f_entry in files_changed:
                # Extract filepath from "created: path" or "modified: path"
                parts = f_entry.split(": ", 1)
                if len(parts) == 2:
                    filepath = os.path.join(self.workspace, parts[1])
                    backup_file(session["id"], filepath)

            status = "APPROVED" if approved else f"BEST EFFORT ({iterations} iterations)"
            self.call_from_thread(
                self._log_agent,
                "System",
                f"[b]Result: {status}[/b] | {len(worker_results)} workers completed",
            )

            if files_changed:
                self.call_from_thread(self._update_files, files_changed)
                for f in files_changed:
                    self.call_from_thread(self._log_agent, "Filesystem", f)

            tool_results = result.get("tool_results", [])
            for tr in tool_results:
                status_str = tr.get("status", "?")
                cmd = tr.get("command", "?")
                self.call_from_thread(
                    self._log_agent, "Tools", f"[{status_str}] {cmd}"
                )

            # Save to memory and session
            save_run(result)
            complete_session(
                session["id"],
                result.get("final", ""),
                approved,
                iterations,
                files_changed,
            )

            # Log final output
            final = result.get("final", "No output generated.")
            self.call_from_thread(
                self._log_agent, "System", "--- Final Output ---"
            )
            # Show a truncated preview in the log
            preview = final[:500] + ("..." if len(final) > 500 else "")
            self.call_from_thread(self._log_agent, "System", preview)

            self.call_from_thread(
                self._update_status,
                f"Completed: {status} | Session: {session['id']}",
            )

        except Exception as e:
            error_msg = str(e)
            logger.exception("Agent pipeline failed: %s", error_msg)
            self.call_from_thread(
                self._log_agent, "System", f"[red]Error: {error_msg}[/red]"
            )
            fail_session(session["id"], error_msg)
            self.call_from_thread(
                self._update_status, f"Failed: {error_msg[:60]}"
            )
        finally:
            # Unsubscribe and re-enable input
            tracker.unsubscribe(self._on_agent_event)
            self.call_from_thread(self._enable_input)

    def _enable_input(self) -> None:
        """Re-enable the task input."""
        text_area = self.query_one("#task-input", TaskInput)
        text_area.disabled = False
        text_area.clear()
        text_area.focus()

    def action_help(self) -> None:
        """Show the help screen."""
        self.push_screen(HelpScreen())

    def action_sessions(self) -> None:
        """Show session history."""
        self.push_screen(SessionsScreen())

    def action_rollback_last(self) -> None:
        """Rollback the most recent session."""
        sessions = list_sessions(limit=1)
        if not sessions:
            self.notify("No sessions to rollback.", severity="warning")
            return

        last = sessions[0]
        session_id = last["id"]
        result = rollback_session(session_id)
        mark_rolled_back(session_id)

        restored = len(result.get("restored", []))
        deleted = len(result.get("deleted", []))
        errors = len(result.get("errors", []))

        self._log_agent(
            "System",
            f"Rolled back session {session_id}: "
            f"{restored} restored, {deleted} deleted, {errors} errors",
        )
        self.notify(
            f"Rolled back: {restored} restored, {deleted} deleted",
            title="Rollback Complete",
        )
