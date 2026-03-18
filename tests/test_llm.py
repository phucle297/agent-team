"""Tests for utils/llm.py - LLM wrapper functions."""

import os
from unittest.mock import MagicMock, patch

import pytest

from utils.llm import (
    _is_rate_limit_error,
    extract_text,
    get_claude,
    get_google,
    invoke_with_retry,
    invoke_with_retry_and_fallback,
)


class TestGetClaude:
    """Tests for the get_claude factory function."""

    @patch("utils.llm.ChatAnthropic")
    def test_returns_chat_anthropic_instance(self, mock_cls):
        result = get_claude()
        mock_cls.assert_called_once()
        assert result is mock_cls.return_value

    @patch("utils.llm.ChatAnthropic")
    def test_uses_correct_model(self, mock_cls):
        get_claude()
        _, kwargs = mock_cls.call_args
        assert kwargs["model_name"] == "claude-sonnet-4-20250514"

    @patch("utils.llm.ChatAnthropic")
    def test_no_http_client_when_no_cert_env(self, mock_cls):
        """No http_client kwarg is ever passed (SSL handled via env vars)."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")}
        with patch.dict(os.environ, env, clear=True):
            get_claude()
        _, kwargs = mock_cls.call_args
        assert "http_client" not in kwargs

    @patch("utils.llm.ChatAnthropic")
    def test_propagates_requests_ca_bundle_to_ssl_cert_file(self, mock_cls, tmp_path):
        """When REQUESTS_CA_BUNDLE is set, _ensure_ssl_cert_env propagates it to SSL_CERT_FILE."""
        ca_path = tmp_path / "corp-ca.pem"
        ca_path.write_text("test")
        env = {k: v for k, v in os.environ.items()
               if k not in ("SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE")}
        env["REQUESTS_CA_BUNDLE"] = str(ca_path)
        with patch.dict(os.environ, env, clear=True):
            get_claude()
            assert os.environ.get("SSL_CERT_FILE") == str(ca_path)
        _, kwargs = mock_cls.call_args
        assert "http_client" not in kwargs

    @patch("utils.llm.ChatAnthropic")
    def test_propagates_curl_ca_bundle_to_ssl_cert_file(self, mock_cls, tmp_path):
        """When CURL_CA_BUNDLE is set, _ensure_ssl_cert_env propagates it to SSL_CERT_FILE."""
        ca_path = tmp_path / "curl-ca.pem"
        ca_path.write_text("test")
        env = {k: v for k, v in os.environ.items()
               if k not in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE")}
        env["CURL_CA_BUNDLE"] = str(ca_path)
        with patch.dict(os.environ, env, clear=True):
            get_claude()
            assert os.environ.get("SSL_CERT_FILE") == str(ca_path)
        _, kwargs = mock_cls.call_args
        assert "http_client" not in kwargs

    @patch("utils.llm.ChatAnthropic")
    def test_ssl_cert_file_already_set_not_overwritten(self, mock_cls, tmp_path):
        """When SSL_CERT_FILE is already set, _ensure_ssl_cert_env does not overwrite it."""
        priority = tmp_path / "priority.crt"
        fallback = tmp_path / "fallback.pem"
        priority.write_text("test")
        fallback.write_text("test")
        with patch.dict(os.environ, {
            "SSL_CERT_FILE": str(priority),
            "REQUESTS_CA_BUNDLE": str(fallback),
        }, clear=False):
            get_claude()
            assert os.environ.get("SSL_CERT_FILE") == str(priority)
        _, kwargs = mock_cls.call_args
        assert "http_client" not in kwargs

    @patch("utils.llm.ChatAnthropic")
    def test_custom_model_via_env(self, mock_cls):
        """CLAUDE_MODEL env var overrides the default model."""
        with patch.dict(os.environ, {"CLAUDE_MODEL": "claude-haiku-3"}, clear=False):
            get_claude()
        _, kwargs = mock_cls.call_args
        assert kwargs["model_name"] == "claude-haiku-3"


class TestGetGoogle:
    """Tests for the get_google factory function."""

    @patch("utils.llm.ChatGoogleGenerativeAI")
    def test_returns_chat_google_instance(self, mock_cls):
        result = get_google()
        mock_cls.assert_called_once_with(model="gemini-3-flash-preview")
        assert result is mock_cls.return_value

    @patch("utils.llm.ChatGoogleGenerativeAI")
    def test_uses_correct_model(self, mock_cls):
        get_google()
        _, kwargs = mock_cls.call_args
        assert kwargs["model"] == "gemini-3-flash-preview"


class TestExtractText:
    """Tests for the extract_text helper function."""

    def test_string_content_returned_as_is(self):
        assert extract_text("hello world") == "hello world"

    def test_list_of_text_blocks(self):
        content = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ]
        assert extract_text(content) == "Hello world"

    def test_list_with_non_text_blocks_ignored(self):
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "image", "data": "..."},
            {"type": "text", "text": " world"},
        ]
        assert extract_text(content) == "Hello world"

    def test_empty_string(self):
        assert extract_text("") == ""

    def test_empty_list(self):
        assert extract_text([]) == ""

    def test_none_returns_empty_string(self):
        assert extract_text(None) == ""

    def test_list_of_strings(self):
        """Some providers return a plain list of strings."""
        assert extract_text(["hello", " ", "world"]) == "hello world"

    def test_integer_converted_to_string(self):
        assert extract_text(42) == "42"


class TestInvokeWithRetry:
    """Tests for the invoke_with_retry helper function."""

    def test_succeeds_on_first_attempt(self):
        llm = MagicMock()
        llm.invoke.return_value = "success"
        result = invoke_with_retry(llm, "prompt", max_retries=3, base_delay=0)
        assert result == "success"
        assert llm.invoke.call_count == 1

    def test_retries_on_connection_error(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            ConnectionError("Connection refused"),
            "success",
        ]
        result = invoke_with_retry(llm, "prompt", max_retries=3, base_delay=0)
        assert result == "success"
        assert llm.invoke.call_count == 2

    def test_retries_on_timeout_error(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            TimeoutError("Request timed out"),
            "success",
        ]
        result = invoke_with_retry(llm, "prompt", max_retries=3, base_delay=0)
        assert result == "success"
        assert llm.invoke.call_count == 2

    def test_retries_on_ssl_error(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            Exception("SSL: CERTIFICATE_VERIFY_FAILED"),
            "success",
        ]
        result = invoke_with_retry(llm, "prompt", max_retries=3, base_delay=0)
        assert result == "success"
        assert llm.invoke.call_count == 2

    def test_retries_on_rate_limit_429(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            Exception("Error 429: Rate limit exceeded"),
            "success",
        ]
        result = invoke_with_retry(llm, "prompt", max_retries=3, base_delay=0)
        assert result == "success"
        assert llm.invoke.call_count == 2

    def test_raises_after_max_retries_exhausted(self):
        llm = MagicMock()
        llm.invoke.side_effect = ConnectionError("Connection refused")
        import pytest
        with pytest.raises(ConnectionError, match="Connection refused"):
            invoke_with_retry(llm, "prompt", max_retries=2, base_delay=0)
        assert llm.invoke.call_count == 2

    def test_raises_immediately_on_non_retryable_error(self):
        llm = MagicMock()
        llm.invoke.side_effect = ValueError("Invalid prompt format")
        import pytest
        with pytest.raises(ValueError, match="Invalid prompt format"):
            invoke_with_retry(llm, "prompt", max_retries=3, base_delay=0)
        assert llm.invoke.call_count == 1

    def test_retries_on_503_unavailable(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            Exception("503 Service Unavailable"),
            "success",
        ]
        result = invoke_with_retry(llm, "prompt", max_retries=3, base_delay=0)
        assert result == "success"
        assert llm.invoke.call_count == 2

    def test_retries_on_overloaded(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            Exception("API is temporarily overloaded"),
            "success",
        ]
        result = invoke_with_retry(llm, "prompt", max_retries=3, base_delay=0)
        assert result == "success"

    def test_passes_prompt_to_llm(self):
        llm = MagicMock()
        llm.invoke.return_value = "ok"
        invoke_with_retry(llm, "my specific prompt", max_retries=1, base_delay=0)
        llm.invoke.assert_called_once_with("my specific prompt")

    def test_fail_fast_on_rate_limit_raises_immediately(self):
        """When fail_fast_on_rate_limit=True, rate-limit errors raise on first attempt."""
        llm = MagicMock()
        llm.invoke.side_effect = Exception("429 RESOURCE_EXHAUSTED")
        with pytest.raises(Exception, match="429 RESOURCE_EXHAUSTED"):
            invoke_with_retry(
                llm, "prompt", max_retries=3, base_delay=0, fail_fast_on_rate_limit=True,
            )
        # Should NOT retry — only 1 attempt
        assert llm.invoke.call_count == 1

    def test_fail_fast_on_rate_limit_does_not_affect_non_rate_limit(self):
        """fail_fast_on_rate_limit does not affect non-rate-limit retryable errors."""
        llm = MagicMock()
        llm.invoke.side_effect = [
            ConnectionError("Connection refused"),
            "success",
        ]
        result = invoke_with_retry(
            llm, "prompt", max_retries=3, base_delay=0, fail_fast_on_rate_limit=True,
        )
        assert result == "success"
        assert llm.invoke.call_count == 2

    def test_retries_on_resource_exhausted(self):
        """RESOURCE_EXHAUSTED is retryable when fail_fast is off."""
        llm = MagicMock()
        llm.invoke.side_effect = [
            Exception("RESOURCE_EXHAUSTED quota exceeded"),
            "success",
        ]
        result = invoke_with_retry(llm, "prompt", max_retries=3, base_delay=0)
        assert result == "success"
        assert llm.invoke.call_count == 2


class TestIsRateLimitError:
    """Tests for the _is_rate_limit_error helper."""

    def test_matches_429(self):
        assert _is_rate_limit_error(Exception("Error 429: Rate limit exceeded"))

    def test_matches_rate_limit(self):
        assert _is_rate_limit_error(Exception("rate limit exceeded"))

    def test_matches_rate_limit_underscore(self):
        assert _is_rate_limit_error(Exception("rate_limit_error"))

    def test_matches_resource_exhausted(self):
        assert _is_rate_limit_error(Exception("RESOURCE_EXHAUSTED"))

    def test_matches_resource_exhausted_with_space(self):
        assert _is_rate_limit_error(Exception("resource exhausted"))

    def test_matches_quota(self):
        assert _is_rate_limit_error(Exception("You exceeded your current quota"))

    def test_does_not_match_generic_error(self):
        assert not _is_rate_limit_error(Exception("Connection refused"))

    def test_matches_google_error_message(self):
        """Matches the real Google RESOURCE_EXHAUSTED error format."""
        msg = (
            "Error calling model 'gemini-3-flash-preview' (RESOURCE_EXHAUSTED): "
            "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
            "'You exceeded your current quota'}}"
        )
        assert _is_rate_limit_error(Exception(msg))


class TestInvokeWithRetryAndFallback:
    """Tests for the invoke_with_retry_and_fallback wrapper."""

    @patch.dict(os.environ, {"AVAILABLE_MODELS": "gemini-3-flash-preview,claude-sonnet-4-20250514"})
    @patch("utils.llm._create_llm_for_model")
    def test_falls_back_to_claude_on_google_rate_limit(self, mock_create):
        """When Google returns 429, should fall back to Claude."""
        primary_llm = MagicMock()
        primary_llm.invoke.side_effect = Exception("429 RESOURCE_EXHAUSTED")

        fallback_llm = MagicMock()
        fallback_llm.invoke.return_value = "fallback success"
        mock_create.return_value = fallback_llm

        result = invoke_with_retry_and_fallback(
            primary_llm,
            "test prompt",
            primary_model="gemini-3-flash-preview",
            max_retries=1,
            base_delay=0,
        )
        assert result == "fallback success"
        mock_create.assert_called_once_with("claude-sonnet-4-20250514")

    @patch.dict(os.environ, {"AVAILABLE_MODELS": "claude-sonnet-4-20250514,gemini-3-flash-preview"})
    @patch("utils.llm._create_llm_for_model")
    def test_falls_back_to_google_on_claude_rate_limit(self, mock_create):
        """When Claude returns 429, should fall back to Google."""
        primary_llm = MagicMock()
        primary_llm.invoke.side_effect = Exception("429 rate limit exceeded")

        fallback_llm = MagicMock()
        fallback_llm.invoke.return_value = "google fallback"
        mock_create.return_value = fallback_llm

        result = invoke_with_retry_and_fallback(
            primary_llm,
            "test prompt",
            primary_model="claude-sonnet-4-20250514",
            max_retries=1,
            base_delay=0,
        )
        assert result == "google fallback"
        mock_create.assert_called_once_with("gemini-3-flash-preview")

    def test_raises_non_rate_limit_error_immediately(self):
        """Non-rate-limit errors should be raised without fallback."""
        primary_llm = MagicMock()
        primary_llm.invoke.side_effect = ValueError("Invalid prompt")

        with pytest.raises(ValueError, match="Invalid prompt"):
            invoke_with_retry_and_fallback(
                primary_llm,
                "test prompt",
                primary_model="gemini-3-flash-preview",
                max_retries=1,
                base_delay=0,
            )

    def test_returns_primary_result_on_success(self):
        """When primary succeeds, should return its result directly."""
        primary_llm = MagicMock()
        primary_llm.invoke.return_value = "primary success"

        result = invoke_with_retry_and_fallback(
            primary_llm,
            "test prompt",
            primary_model="gemini-3-flash-preview",
            max_retries=1,
            base_delay=0,
        )
        assert result == "primary success"

    @patch.dict(os.environ, {"AVAILABLE_MODELS": "gemini-3-flash-preview"})
    def test_raises_when_no_fallback_models_available(self):
        """When no cross-provider models exist, should raise the original error."""
        primary_llm = MagicMock()
        primary_llm.invoke.side_effect = Exception("429 RESOURCE_EXHAUSTED")

        with pytest.raises(Exception, match="429 RESOURCE_EXHAUSTED"):
            invoke_with_retry_and_fallback(
                primary_llm,
                "test prompt",
                primary_model="gemini-3-flash-preview",
                max_retries=1,
                base_delay=0,
            )
