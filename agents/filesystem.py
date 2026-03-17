import json
import logging
import os
from pathlib import Path

from utils.agent_events import AgentStatus, tracker
from utils.llm import extract_text, get_claude, invoke_with_retry

logger = logging.getLogger(__name__)


def filesystem_agent(state: dict) -> dict:
    """Read/write files and modify the repository based on generated code."""
    code = state.get("code", "")
    plan = state.get("plan", "")
    workspace = state.get("workspace", os.getcwd())

    llm = get_claude()

    prompt = (
        "You are a file system agent. Given the following code and plan, "
        "determine what files need to be created or modified.\n\n"
        "Output a JSON array of file operations. Each operation should have:\n"
        '- "action": "create" | "modify" | "delete"\n'
        '- "path": relative file path\n'
        '- "content": file content (for create/modify)\n\n'
        "IMPORTANT: Output ONLY the JSON array, no markdown fences or other text.\n\n"
        f"Plan:\n{plan}\n\n"
        f"Code:\n{code}\n\n"
        "JSON array of file operations:"
    )

    tracker.update("filesystem", AgentStatus.WORKING, "Analyzing file operations...")
    logger.info("FileSystem Agent: analyzing code for file operations...")
    res = invoke_with_retry(llm, prompt)
    raw = extract_text(res.content).strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)

    operations = []
    try:
        operations = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("FileSystem Agent: could not parse file operations JSON")
        return {"file_operations": [], "files_changed": []}

    files_changed = []
    for op in operations:
        action = op.get("action", "")
        filepath = op.get("path", "")
        content = op.get("content", "")

        if not filepath:
            continue

        full_path = Path(workspace) / filepath

        if action == "create":
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            files_changed.append(f"created: {filepath}")
            logger.info("FileSystem Agent: created %s", filepath)

        elif action == "modify":
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            files_changed.append(f"modified: {filepath}")
            logger.info("FileSystem Agent: modified %s", filepath)

        elif action == "delete":
            if full_path.exists():
                full_path.unlink()
                files_changed.append(f"deleted: {filepath}")
                logger.info("FileSystem Agent: deleted %s", filepath)

    tracker.update("filesystem", AgentStatus.DONE, f"{len(files_changed)} files changed")
    return {"file_operations": operations, "files_changed": files_changed}
