"""Tests for utils/llm.py - LLM wrapper functions."""

from unittest.mock import MagicMock, patch

from utils.llm import extract_text, get_claude, get_google, invoke_with_retry


class TestGetClaude:
    """Tests for the get_claude factory function."""

    @patch("utils.llm.ChatAnthropic")
    def test_returns_chat_anthropic_instance(self, mock_cls):
        result = get_claude()
        mock_cls.assert_called_once_with(model="claude-sonnet-4-20250514")
        assert result is mock_cls.return_value

    @patch("utils.llm.ChatAnthropic")
    def test_uses_correct_model(self, mock_cls):
        get_claude()
        _, kwargs = mock_cls.call_args
        assert kwargs["model"] == "claude-sonnet-4-20250514"


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
