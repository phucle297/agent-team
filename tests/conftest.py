"""Shared test fixtures and helpers."""

from unittest.mock import MagicMock

import pytest


class FakeLLMResponse:
    """Mock LLM response with a .content attribute."""

    def __init__(self, content: str):
        self.content = content


def make_fake_llm(content: str) -> MagicMock:
    """Create a mock LLM that returns a fixed response."""
    llm = MagicMock()
    llm.invoke.return_value = FakeLLMResponse(content)
    return llm


@pytest.fixture
def fake_claude(monkeypatch):
    """Patch get_claude to return a mock LLM."""
    llm = make_fake_llm("mock claude response")

    def _get(content="mock claude response"):
        nonlocal llm
        llm = make_fake_llm(content)
        monkeypatch.setattr("utils.llm.get_claude", lambda: llm)
        return llm

    # Apply default immediately
    monkeypatch.setattr("utils.llm.get_claude", lambda: llm)
    _get.llm = llm
    _get.set = _get
    return _get


@pytest.fixture
def fake_google(monkeypatch):
    """Patch get_google to return a mock LLM."""
    llm = make_fake_llm("mock google response")

    def _get(content="mock google response"):
        nonlocal llm
        llm = make_fake_llm(content)
        monkeypatch.setattr("utils.llm.get_google", lambda: llm)
        return llm

    monkeypatch.setattr("utils.llm.get_google", lambda: llm)
    _get.llm = llm
    _get.set = _get
    return _get
