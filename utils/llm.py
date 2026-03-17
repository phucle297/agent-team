import logging
import os
import time
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
DEFAULT_GOOGLE_MODEL = "gemini-3-flash-preview"

MAX_RETRIES = 3
BASE_DELAY = 2.0  # seconds


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


def invoke_with_retry(
    llm: Any,
    prompt: str,
    *,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
) -> Any:
    """Invoke an LLM with retry and exponential backoff.

    Retries on connection errors, timeouts, and rate-limit (429) errors.
    Returns the LLM response on success.
    Raises the last exception after all retries are exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()

            # Determine if this error is retryable
            retryable = isinstance(exc, (ConnectionError, TimeoutError, OSError)) or (
                "connection" in exc_str
                or "timeout" in exc_str
                or "timed out" in exc_str
                or "rate" in exc_str
                or "429" in exc_str
                or "ssl" in exc_str
                or "eof" in exc_str
                or "reset" in exc_str
                or "temporarily" in exc_str
                or "overloaded" in exc_str
                or "unavailable" in exc_str
                or "502" in exc_str
                or "503" in exc_str
            )

            if not retryable or attempt == max_retries:
                logger.error(
                    "LLM invoke failed (attempt %d/%d, non-retryable or exhausted): %s",
                    attempt, max_retries, exc,
                )
                raise

            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "LLM invoke failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt, max_retries, delay, exc,
            )
            time.sleep(delay)

    # Should never reach here, but satisfy type checker
    raise last_exc  # type: ignore[misc]


def get_claude():
    model = os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
    return ChatAnthropic(model=model)


def get_google():
    model = os.environ.get("GOOGLE_MODEL", DEFAULT_GOOGLE_MODEL)
    return ChatGoogleGenerativeAI(model=model)
