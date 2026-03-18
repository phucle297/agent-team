import json
import logging
from typing import Any

from utils.llm import extract_text, get_llm, invoke_with_retry

logger = logging.getLogger(__name__)


SUMMARY_PROMPT = """You are creating a continuation note so the next session can continue the work.

Write a compact, high-signal summary with ONLY the essentials needed to continue.

Include:
- Goal
- Current state / what was done
- Key decisions + rationale
- Files changed (paths)
- Open TODOs (ordered)
- Commands run + outcomes (if any)
- Any important constraints / gotchas

Keep it under 250-400 lines.

Input:
Task: {task}

Plan:
{plan}

Review:
{review}

Files changed:
{files_changed}

Tool results:
{tool_results}

Now write the continuation note:"""


def build_continuation_note(state: dict[str, Any]) -> str:
    llm, _ = get_llm("google")
    prompt = SUMMARY_PROMPT.format(
        task=state.get("input", ""),
        plan=state.get("plan", ""),
        review=state.get("review", ""),
        files_changed="\n".join(state.get("files_changed", []) or []),
        tool_results=json.dumps(state.get("tool_results", []) or [], indent=2)[:6000],
    )

    try:
        res = invoke_with_retry(llm, prompt)
        return extract_text(getattr(res, "content", res)).strip()
    except Exception as exc:
        logger.warning("Failed to build continuation note: %s", exc)
        return ""
