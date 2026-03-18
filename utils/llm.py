import logging
import os
import time
from collections.abc import Callable
from typing import Any, cast

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
DEFAULT_GOOGLE_MODEL = "gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Unified model array
# ---------------------------------------------------------------------------
# Single env var: AVAILABLE_MODELS (comma-separated).
# Example: AVAILABLE_MODELS="claude-sonnet-4-20250514,gemini-3-flash-preview"
#
# Provider is auto-detected from the model name:
#   - Starts with "claude"  -> Anthropic
#   - Starts with "gemini" or "gemma" -> Google
#
# The array order matters for fallback: on rate-limit the system tries the
# next model in the list that belongs to a *different* provider.
# ---------------------------------------------------------------------------
AVAILABLE_MODELS_ENV = "AVAILABLE_MODELS"


def _parse_model_list(value: str | None) -> list[str]:
    """Parse a comma-separated model list string into a list."""
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def detect_provider(model_name: str) -> str:
    """Auto-detect the provider from a model name.

    Returns ``"anthropic"`` or ``"google"``.  Raises ``ValueError`` for
    unknown prefixes.
    """
    lower = model_name.lower()
    if lower.startswith("claude"):
        return "anthropic"
    if lower.startswith(("gemini", "gemma")):
        return "google"
    raise ValueError(f"Cannot detect provider for model: {model_name!r}")


def get_available_models() -> list[str]:
    """Return the configured model array (from ``AVAILABLE_MODELS`` env).

    Falls back to ``[DEFAULT_CLAUDE_MODEL, DEFAULT_GOOGLE_MODEL]`` when the
    env var is not set.
    """
    parsed = _parse_model_list(os.environ.get(AVAILABLE_MODELS_ENV))
    if parsed:
        return parsed
    return [DEFAULT_CLAUDE_MODEL, DEFAULT_GOOGLE_MODEL]


def _pick_model_for_provider(provider: str) -> str:
    """Pick the first model in the array that matches *provider*.

    Falls back to the built-in default for that provider when no match is
    found in the array.
    """
    defaults = {"anthropic": DEFAULT_CLAUDE_MODEL, "google": DEFAULT_GOOGLE_MODEL}
    for m in get_available_models():
        try:
            if detect_provider(m) == provider:
                return m
        except ValueError:
            continue
    return defaults.get(provider, DEFAULT_CLAUDE_MODEL)


def get_model_name(provider: str) -> str:
    """Return the model name that will be used for *provider* (``"anthropic"`` or ``"google"``)."""
    env_overrides = {"anthropic": "CLAUDE_MODEL", "google": "GOOGLE_MODEL"}
    env_var = env_overrides.get(provider)
    if env_var:
        explicit = os.environ.get(env_var)
        if explicit:
            return explicit
    return _pick_model_for_provider(provider)


def get_fallback_models(primary_model: str) -> list[str]:
    """Return models to try after *primary_model* hits a rate limit.

    Walks the ``AVAILABLE_MODELS`` array and returns every model whose
    provider differs from the primary's provider, preserving order.
    """
    try:
        primary_provider = detect_provider(primary_model)
    except ValueError:
        return []

    fallbacks: list[str] = []
    for m in get_available_models():
        try:
            if detect_provider(m) != primary_provider:
                fallbacks.append(m)
        except ValueError:
            continue
    return fallbacks

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
    fail_fast_on_rate_limit: bool = False,
) -> Any:
    """Invoke an LLM with retry and exponential backoff.

    Retries on connection errors, timeouts, and rate-limit (429) errors.
    Returns the LLM response on success.
    Raises the last exception after all retries are exhausted.

    When *fail_fast_on_rate_limit* is True, rate-limit / quota errors
    are raised immediately on the first attempt so that a higher-level
    fallback mechanism (e.g. ``invoke_with_retry_and_fallback``) can
    switch to a different provider without wasting time retrying the
    exhausted one.
    """
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()

            is_rate_limit = _is_rate_limit_error(exc)

            # When called from the fallback wrapper, surface rate-limit
            # errors immediately so the next provider is tried without
            # wasting minutes on hopeless retries.
            if fail_fast_on_rate_limit and is_rate_limit:
                logger.warning(
                    "LLM invoke rate-limited (attempt %d, fail-fast enabled): %s",
                    attempt, exc,
                )
                raise

            # Determine if this error is retryable
            retryable = is_rate_limit or isinstance(exc, (ConnectionError, TimeoutError, OSError)) or (
                "connection" in exc_str
                or "timeout" in exc_str
                or "timed out" in exc_str
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
            if is_rate_limit:
                # Output-token rate limits often require waiting for the next minute window.
                delay = max(delay, float(os.environ.get("LLM_RATE_LIMIT_DELAY", "20")))
            logger.warning(
                "LLM invoke failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt, max_retries, delay, exc,
            )
            time.sleep(delay)

    # Should never reach here, but satisfy type checker
    raise last_exc  # type: ignore[misc]


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a 429 / rate-limit / quota error."""
    exc_str = str(exc).lower()
    return (
        "429" in exc_str
        or "rate_limit" in exc_str
        or "rate limit" in exc_str
        or "resource_exhausted" in exc_str
        or "resource exhausted" in exc_str
        or "quota" in exc_str
    )


def _create_llm_for_model(model_name: str) -> Any:
    """Instantiate the right LangChain LLM for *model_name*."""
    provider = detect_provider(model_name)
    if provider == "anthropic":
        return _build_anthropic_llm(model_name)
    return ChatGoogleGenerativeAI(model=model_name)


def invoke_with_retry_and_fallback(
    primary_llm: Any,
    prompt: str,
    *,
    primary_model: str | None = None,
    invoke_fn: Callable[..., Any] | None = None,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
) -> Any:
    """Invoke *primary_llm* with retry; on rate-limit walk the model array.

    When the primary model is rate-limited (429) and ``AVAILABLE_MODELS``
    contains models from a different provider, each fallback model is tried
    in array order.  Non-rate-limit errors are re-raised immediately.
    """
    if invoke_fn is None:
        invoke_fn = invoke_with_retry

    try:
        return invoke_fn(
            primary_llm, prompt,
            max_retries=max_retries,
            base_delay=base_delay,
            fail_fast_on_rate_limit=bool(primary_model and get_fallback_models(primary_model)),
        )
    except Exception as exc:
        if not _is_rate_limit_error(exc):
            raise

        if primary_model is None:
            raise

        fallbacks = get_fallback_models(primary_model)
        if not fallbacks:
            raise

        for fb_model in fallbacks:
            logger.warning(
                "Primary model %s rate-limited; trying fallback %s",
                primary_model, fb_model,
            )
            try:
                fb_llm = _create_llm_for_model(fb_model)
                return invoke_with_retry(fb_llm, prompt, max_retries=max_retries, base_delay=base_delay)
            except Exception as fb_exc:
                logger.warning("Fallback model %s also failed: %s", fb_model, fb_exc)
                continue

        # All fallbacks exhausted – re-raise the original error
        raise


def _get_ca_cert_path() -> tuple[str, str] | None:
    # Priority: explicit TLS bundle vars first, then Nix fallback.
    for var in (
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NIX_SSL_CERT_FILE",
    ):
        path = os.environ.get(var)
        if path:
            expanded = os.path.expanduser(path)
            return var, expanded
    return None


def _ensure_ssl_cert_env() -> None:
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return

    detected = _get_ca_cert_path()
    if not detected:
        return

    var, ca_path = detected
    if var == "SSL_CERT_FILE":
        return

    if os.path.isfile(ca_path):
        os.environ["SSL_CERT_FILE"] = ca_path
        logger.info("Propagated SSL_CERT_FILE from %s=%s", var, ca_path)
    else:
        logger.warning("Ignoring %s=%s (not a file)", var, ca_path)


def _build_anthropic_llm(model: str) -> Any:
    """Build a ``ChatAnthropic`` instance with optional SSL / max-tokens.

    SSL certificates are handled via the ``SSL_CERT_FILE`` environment
    variable which ``httpx`` (used internally by ``langchain-anthropic``)
    picks up automatically.  We no longer pass a custom ``http_client``
    because ``langchain-anthropic >= 1.x`` creates its own and rejects
    the extra kwarg.
    """
    _ensure_ssl_cert_env()

    anthropic_cls = cast(Any, ChatAnthropic)

    kwargs: dict[str, Any] = {"model_name": model, "timeout": None, "stop": None}

    max_tokens_env = os.environ.get("CLAUDE_MAX_TOKENS")
    if max_tokens_env:
        kwargs["max_tokens_to_sample"] = int(max_tokens_env)

    return anthropic_cls(**kwargs)


def get_claude():
    """Return a Claude LLM instance using the first Anthropic model in the array."""
    model = os.environ.get("CLAUDE_MODEL") or _pick_model_for_provider("anthropic")
    return _build_anthropic_llm(model)


def get_google():
    """Return a Google Gemini LLM instance using the first Google model in the array."""
    model = os.environ.get("GOOGLE_MODEL") or _pick_model_for_provider("google")
    _ensure_ssl_cert_env()
    return ChatGoogleGenerativeAI(model=model)


def get_llm(preferred_provider: str = "anthropic") -> tuple[Any, str]:
    """Return an LLM instance and the resolved model name.

    Tries *preferred_provider* first.  If no model for that provider
    exists in ``AVAILABLE_MODELS``, falls back to the first model of
    any other provider that is available.

    Returns ``(llm_instance, model_name)`` so callers can pass the
    model name to ``invoke_with_retry_and_fallback``.
    """
    models = get_available_models()

    # 1) Try preferred provider
    for m in models:
        try:
            if detect_provider(m) == preferred_provider:
                return _create_llm_for_model(m), m
        except ValueError:
            continue

    # 2) Fall back to first available model regardless of provider
    for m in models:
        try:
            return _create_llm_for_model(m), m
        except ValueError:
            continue

    # 3) Last resort: built-in defaults
    default = DEFAULT_CLAUDE_MODEL if preferred_provider == "anthropic" else DEFAULT_GOOGLE_MODEL
    return _create_llm_for_model(default), default
