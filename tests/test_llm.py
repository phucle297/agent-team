"""Tests for utils/llm.py - LLM wrapper functions."""

from unittest.mock import patch

from utils.llm import extract_text, get_claude, get_google


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
