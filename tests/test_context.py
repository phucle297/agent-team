"""Tests for utils/context.py - Project context scanner."""

import json
import os
from pathlib import Path

import pytest

from utils.context import (
    _build_file_tree,
    _build_summary,
    _detect_frameworks,
    _detect_languages,
    _detect_test_frameworks,
    _find_key_files,
    get_context_prompt,
    scan_project,
)


class TestDetectLanguages:
    def test_detects_python(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')")
        result = _detect_languages(tmp_path)
        assert "python" in result

    def test_detects_javascript(self, tmp_path):
        (tmp_path / "package.json").write_text("{}")
        result = _detect_languages(tmp_path)
        assert "javascript" in result

    def test_detects_typescript(self, tmp_path):
        (tmp_path / "tsconfig.json").write_text("{}")
        result = _detect_languages(tmp_path)
        assert "typescript" in result

    def test_detects_rust(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]")
        result = _detect_languages(tmp_path)
        assert "rust" in result

    def test_detects_go(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example")
        result = _detect_languages(tmp_path)
        assert "go" in result

    def test_returns_empty_for_empty_dir(self, tmp_path):
        result = _detect_languages(tmp_path)
        assert result == []


class TestDetectFrameworks:
    def test_detects_react_from_package_json(self, tmp_path):
        pkg = {"dependencies": {"react": "^18.0.0"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        result = _detect_frameworks(tmp_path)
        assert "react" in result

    def test_detects_fastapi_from_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\ndependencies = ["fastapi"]\n'
        )
        result = _detect_frameworks(tmp_path)
        assert "fastapi" in result

    def test_detects_django_from_requirements(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("django==4.0\n")
        result = _detect_frameworks(tmp_path)
        assert "django" in result

    def test_returns_empty_for_empty_dir(self, tmp_path):
        result = _detect_frameworks(tmp_path)
        assert result == []


class TestDetectTestFrameworks:
    def test_detects_pytest(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
        result = _detect_test_frameworks(tmp_path)
        assert "pytest" in result

    def test_detects_jest(self, tmp_path):
        (tmp_path / "jest.config.js").write_text("module.exports = {};")
        result = _detect_test_frameworks(tmp_path)
        assert "jest" in result

    def test_returns_empty_for_empty_dir(self, tmp_path):
        result = _detect_test_frameworks(tmp_path)
        assert result == []


class TestBuildFileTree:
    def test_builds_tree_with_files(self, tmp_path):
        (tmp_path / "main.py").write_text("")
        (tmp_path / "utils").mkdir()
        (tmp_path / "utils" / "helper.py").write_text("")

        tree = _build_file_tree(tmp_path, max_depth=2, max_files=10)
        assert "main.py" in tree
        assert "utils/" in tree
        assert "helper.py" in tree

    def test_excludes_git_and_pycache(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "main.py").write_text("")

        tree = _build_file_tree(tmp_path)
        assert ".git" not in tree
        assert "__pycache__" not in tree
        assert "main.py" in tree

    def test_respects_max_files(self, tmp_path):
        for i in range(20):
            (tmp_path / f"file_{i}.py").write_text("")

        tree = _build_file_tree(tmp_path, max_files=5)
        assert "truncated" in tree


class TestFindKeyFiles:
    def test_finds_readme(self, tmp_path):
        (tmp_path / "README.md").write_text("# Hello")
        result = _find_key_files(tmp_path)
        assert "README.md" in result

    def test_finds_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]")
        result = _find_key_files(tmp_path)
        assert "pyproject.toml" in result

    def test_returns_empty_for_no_key_files(self, tmp_path):
        (tmp_path / "random.txt").write_text("nothing")
        result = _find_key_files(tmp_path)
        assert result == []


class TestScanProject:
    def test_scans_python_project(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\ntestpaths = ['tests']"
        )
        (tmp_path / ".git").mkdir()

        ctx = scan_project(str(tmp_path))
        assert ctx["exists"] is True
        assert "python" in ctx["languages"]
        assert "pytest" in ctx["test_frameworks"]
        assert ctx["has_git"] is True

    def test_handles_nonexistent_dir(self):
        ctx = scan_project("/nonexistent/path/xyz")
        assert ctx["exists"] is False
        assert ctx["languages"] == []

    def test_includes_summary(self, tmp_path):
        (tmp_path / "main.py").write_text("")
        ctx = scan_project(str(tmp_path))
        assert isinstance(ctx["summary"], str)


class TestGetContextPrompt:
    def test_returns_formatted_string(self, tmp_path):
        (tmp_path / "main.py").write_text("")
        result = get_context_prompt(str(tmp_path))
        assert "## Project Context" in result
        assert str(tmp_path) in result

    def test_includes_tdd_note_when_tests_detected(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
        result = get_context_prompt(str(tmp_path))
        assert "TDD" in result
        assert "red-green-refactor" in result
