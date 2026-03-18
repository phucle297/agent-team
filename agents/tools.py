import json
import logging
import os
import subprocess
from typing import Any

from utils.agent_events import AgentStatus, tracker
from utils.llm import (
    extract_text,
    get_llm,
    invoke_with_retry,
    invoke_with_retry_and_fallback,
)

logger = logging.getLogger(__name__)


def tool_agent(state: Any) -> dict:
    """Execute tools: git commands, terminal commands, and test runner."""
    code = state.get("code", "")
    plan = state.get("plan", "")
    workspace = state.get("workspace", os.getcwd())

    llm, model_name = get_llm("anthropic")

    prompt = (
        "You are a tool execution agent. Given the following code and plan, "
        "determine what shell commands need to be run (e.g., install dependencies, "
        "run tests, git operations).\n\n"
        "Output a JSON array of commands to execute. Each command should have:\n"
        '- "command": the shell command to run\n'
        '- "description": what this command does\n'
        '- "type": "git" | "test" | "terminal"\n\n'
        "IMPORTANT: Output ONLY the JSON array, no markdown fences or other text.\n"
        "Only suggest safe, non-destructive commands.\n"
        "Do NOT suggest commands that delete files or force-push.\n\n"
        f"Plan:\n{plan}\n\n"
        f"Code:\n{code}\n\n"
        "JSON array of commands:"
    )

    tracker.update("tools", AgentStatus.WORKING, "Determining commands to run...")
    logger.info("Tool Agent: determining commands to run...")

    try:
        res = invoke_with_retry_and_fallback(
            llm,
            prompt,
            primary_model=model_name,
            invoke_fn=invoke_with_retry,
        )
        raw = extract_text(res.content).strip()
    except Exception as exc:
        logger.error("Tool Agent: LLM call failed with %s", exc)
        tracker.update("tools", AgentStatus.ERROR, f"Error: {str(exc)[:50]}")
        return {"tool_results": []}

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)

    commands = []
    try:
        commands = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Tool Agent: could not parse commands JSON")
        return {"tool_results": []}

    results = []
    for cmd in commands:
        command = cmd.get("command", "")
        description = cmd.get("description", "")
        cmd_type = cmd.get("type", "terminal")

        if not command:
            continue

        # Safety check: block destructive commands
        dangerous = ["rm -rf", "force", "--hard", "drop", "format"]
        if any(d in command.lower() for d in dangerous):
            results.append({
                "command": command,
                "description": description,
                "status": "blocked",
                "output": "Blocked: potentially destructive command",
            })
            logger.warning("Tool Agent: blocked dangerous command: %s", command)
            continue

        try:
            logger.info("Tool Agent: running [%s] %s", cmd_type, command)
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=workspace,
            )
            results.append({
                "command": command,
                "description": description,
                "status": "success" if result.returncode == 0 else "failed",
                "output": result.stdout[:2000] if result.stdout else "",
                "error": result.stderr[:2000] if result.stderr else "",
                "returncode": result.returncode,
            })
        except subprocess.TimeoutExpired:
            results.append({
                "command": command,
                "description": description,
                "status": "timeout",
                "output": "Command timed out after 60s",
            })
        except Exception as e:
            results.append({
                "command": command,
                "description": description,
                "status": "error",
                "output": str(e),
            })

    tracker.update("tools", AgentStatus.DONE, f"{len(results)} commands executed")
    return {"tool_results": results}
