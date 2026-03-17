"""Project context scanner.

Detects language, framework, file structure, and test setup of the
current working directory to give agents rich context about the project.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# File patterns that indicate languages
LANGUAGE_INDICATORS = {
    "python": ["*.py", "pyproject.toml", "setup.py", "setup.cfg", "Pipfile", "requirements.txt"],
    "javascript": ["*.js", "*.mjs", "package.json"],
    "typescript": ["*.ts", "*.tsx", "tsconfig.json"],
    "rust": ["Cargo.toml", "*.rs"],
    "go": ["go.mod", "go.sum", "*.go"],
    "java": ["pom.xml", "build.gradle", "*.java"],
    "ruby": ["Gemfile", "*.rb", "Rakefile"],
    "php": ["composer.json", "*.php"],
    "c": ["Makefile", "CMakeLists.txt", "*.c", "*.h"],
    "cpp": ["CMakeLists.txt", "*.cpp", "*.hpp", "*.cc"],
    "shell": ["*.sh", "*.bash"],
}

# Config files that indicate frameworks
FRAMEWORK_INDICATORS = {
    "django": ["manage.py", "django"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "react": ["react", "next.config"],
    "nextjs": ["next.config.js", "next.config.ts", "next.config.mjs"],
    "vue": ["vue.config.js", "nuxt.config"],
    "express": ["express"],
    "rails": ["config/routes.rb", "Gemfile"],
    "spring": ["spring"],
    "actix": ["actix"],
    "gin": ["gin"],
}

# Test framework indicators
TEST_INDICATORS = {
    "pytest": ["pytest.ini", "conftest.py", "pyproject.toml"],
    "jest": ["jest.config.js", "jest.config.ts"],
    "mocha": [".mocharc.yml", ".mocharc.json"],
    "vitest": ["vitest.config.ts", "vitest.config.js"],
    "rspec": [".rspec", "spec/"],
    "go_test": ["*_test.go"],
    "cargo_test": ["Cargo.toml"],
}


def scan_project(workspace: str) -> dict:
    """Scan a project directory to detect its characteristics.

    Args:
        workspace: Path to the project root directory.

    Returns:
        Dict with detected languages, frameworks, test setup,
        file structure, and other project metadata.
    """
    workspace_path = Path(workspace).resolve()

    if not workspace_path.exists():
        return {
            "workspace": str(workspace_path),
            "exists": False,
            "languages": [],
            "frameworks": [],
            "test_frameworks": [],
            "has_git": False,
            "file_tree": "",
            "key_files": [],
            "summary": f"Directory does not exist: {workspace_path}",
        }

    context = {
        "workspace": str(workspace_path),
        "exists": True,
        "languages": _detect_languages(workspace_path),
        "frameworks": _detect_frameworks(workspace_path),
        "test_frameworks": _detect_test_frameworks(workspace_path),
        "has_git": (workspace_path / ".git").is_dir(),
        "file_tree": _build_file_tree(workspace_path, max_depth=2, max_files=30),
        "key_files": _find_key_files(workspace_path),
    }

    context["summary"] = _build_summary(context)
    return context


def _detect_languages(root: Path) -> list[str]:
    """Detect programming languages used in the project.

    Only checks root and one level deep to avoid slow recursive scans.
    """
    detected = []
    for lang, patterns in LANGUAGE_INDICATORS.items():
        for pattern in patterns:
            if pattern.startswith("*"):
                # Check root level first
                matches = list(root.glob(pattern))[:1]
                # Then one level deep only (NOT recursive **)
                if not matches:
                    matches = list(root.glob(f"*/{pattern}"))[:1]
                if matches:
                    detected.append(lang)
                    break
            else:
                # Exact filename
                if (root / pattern).exists():
                    detected.append(lang)
                    break
    return detected


def _detect_frameworks(root: Path) -> list[str]:
    """Detect frameworks by checking config files and dependency files."""
    detected = []

    # Check package.json for JS/TS frameworks
    pkg_json = root / "package.json"
    if pkg_json.exists():
        try:
            import json
            pkg = json.loads(pkg_json.read_text())
            all_deps = {}
            all_deps.update(pkg.get("dependencies", {}))
            all_deps.update(pkg.get("devDependencies", {}))

            for framework, indicators in FRAMEWORK_INDICATORS.items():
                for indicator in indicators:
                    if indicator in all_deps:
                        detected.append(framework)
                        break
        except (json.JSONDecodeError, OSError):
            pass

    # Check pyproject.toml for Python frameworks
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text().lower()
            for framework in ["django", "flask", "fastapi"]:
                if framework in content:
                    detected.append(framework)
        except OSError:
            pass

    # Check requirements.txt
    reqs = root / "requirements.txt"
    if reqs.exists():
        try:
            content = reqs.read_text().lower()
            for framework in ["django", "flask", "fastapi"]:
                if framework in content:
                    detected.append(framework)
        except OSError:
            pass

    # Check for specific config files
    for framework, indicators in FRAMEWORK_INDICATORS.items():
        for indicator in indicators:
            if (root / indicator).exists():
                if framework not in detected:
                    detected.append(framework)
                break

    return list(set(detected))


def _detect_test_frameworks(root: Path) -> list[str]:
    """Detect test frameworks used in the project."""
    detected = []

    for framework, indicators in TEST_INDICATORS.items():
        for indicator in indicators:
            if indicator.endswith("/"):
                if (root / indicator.rstrip("/")).is_dir():
                    detected.append(framework)
                    break
            elif indicator.startswith("*"):
                # Only check root and one level deep
                if list(root.glob(indicator))[:1] or list(root.glob(f"*/{indicator}"))[:1]:
                    detected.append(framework)
                    break
            else:
                if (root / indicator).exists():
                    # Special case: pyproject.toml only counts for pytest
                    # if it contains [tool.pytest]
                    if indicator == "pyproject.toml" and framework == "pytest":
                        try:
                            content = (root / indicator).read_text()
                            if "[tool.pytest" in content:
                                detected.append(framework)
                                break
                        except OSError:
                            pass
                    else:
                        detected.append(framework)
                        break

    return list(set(detected))


def _build_file_tree(root: Path, max_depth: int = 3, max_files: int = 50) -> str:
    """Build a file tree string representation of the project.

    Excludes common noise directories like node_modules, .git, __pycache__, etc.
    """
    EXCLUDE_DIRS = {
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "dist", "build", ".next", ".nuxt", "target", ".direnv",
        "egg-info", ".eggs", "htmlcov",
    }

    lines = []
    file_count = 0

    def _walk(path: Path, prefix: str, depth: int):
        nonlocal file_count
        if depth > max_depth or file_count >= max_files:
            return

        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except PermissionError:
            return

        # Filter out excluded directories and hidden files at depth > 0
        entries = [
            e for e in entries
            if e.name not in EXCLUDE_DIRS
            and not any(e.name.endswith(suffix) for suffix in [".egg-info"])
        ]

        for i, entry in enumerate(entries):
            if file_count >= max_files:
                lines.append(f"{prefix}... (truncated)")
                return

            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "

            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                _walk(entry, prefix + extension, depth + 1)
            else:
                lines.append(f"{prefix}{connector}{entry.name}")
                file_count += 1

    lines.append(f"{root.name}/")
    _walk(root, "", 0)
    return "\n".join(lines)


def _find_key_files(root: Path) -> list[str]:
    """Find key project files (configs, READMEs, entry points)."""
    key_patterns = [
        "README.md", "README.rst", "README.txt",
        "pyproject.toml", "setup.py", "setup.cfg",
        "package.json", "tsconfig.json",
        "Cargo.toml", "go.mod",
        "Makefile", "Dockerfile", "docker-compose.yml",
        ".env.example", ".gitignore",
        "main.py", "app.py", "index.ts", "index.js",
        "manage.py",
    ]

    found = []
    for pattern in key_patterns:
        if (root / pattern).exists():
            found.append(pattern)

    return found


def _build_summary(context: dict) -> str:
    """Build a human-readable summary of the project context."""
    parts = []

    if context["languages"]:
        parts.append(f"Languages: {', '.join(context['languages'])}")

    if context["frameworks"]:
        parts.append(f"Frameworks: {', '.join(context['frameworks'])}")

    if context["test_frameworks"]:
        parts.append(f"Test frameworks: {', '.join(context['test_frameworks'])}")

    if context["has_git"]:
        parts.append("Git repository: yes")

    if context["key_files"]:
        parts.append(f"Key files: {', '.join(context['key_files'])}")

    return "\n".join(parts) if parts else "No project characteristics detected."


def get_context_prompt(workspace: str) -> str:
    """Generate a context prompt string for agents.

    Args:
        workspace: Path to the project root.

    Returns:
        Formatted string describing the project context.
    """
    ctx = scan_project(workspace)

    prompt = f"""## Project Context

Working directory: {ctx['workspace']}

{ctx['summary']}

### File Structure
```
{ctx['file_tree']}
```
"""

    if ctx["test_frameworks"]:
        prompt += f"\n### Testing\nDetected test framework(s): {', '.join(ctx['test_frameworks'])}\n"
        prompt += "IMPORTANT: Follow TDD (red-green-refactor). Write failing tests FIRST, then implement code to pass them.\n"

    # Cap total prompt size to avoid RESOURCE_EXHAUSTED errors with LLM APIs
    max_len = 4000
    if len(prompt) > max_len:
        prompt = prompt[:max_len] + "\n... (context truncated)\n"

    return prompt
