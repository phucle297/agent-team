import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
DEFAULT_GOOGLE_MODEL = "gemini-3-flash-preview"


def extract_text(content: Any) -> str:
    """Safely extract text from an LLM response's .content field.

    LangChain model responses can return .content as:
    - str: plain text
    - list[dict]: content blocks like [{"type": "text", "text": "..."}]
    - list[str]: plain string list
    - None or other: fallback to empty string / str()
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
            # Skip non-text blocks (images, etc.)
        return "".join(parts)
    return str(content)


def get_claude():
    model = os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
    return ChatAnthropic(model=model)


def get_google():
    model = os.environ.get("GOOGLE_MODEL", DEFAULT_GOOGLE_MODEL)
    return ChatGoogleGenerativeAI(model=model)
