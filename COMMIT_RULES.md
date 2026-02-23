# Commit Rules

These rules apply to all commits in this repository.

## 1. Message Format

Use Conventional Commit style, single summary line:

- `feat: ...` for new user-facing features
- `fix: ...` for bug fixes
- `docs: ...` for documentation-only changes
- `chore: ...` for maintenance (configs, tooling, format-only)
- `refactor: ...` for internal changes with no behavior change
- `test: ...` for adding or updating tests
- `build: ...` for build system / dependency changes
- `ci: ...` for CI-related changes
- `perf: ...` for performance improvements

Examples:

- `feat: add orchestrator skeleton for LLM loop`
- `feat(orchestrator): support multiple tasks per run`
- `fix: handle test failures without losing state`
- `docs: describe spec-driven requirements format`

Write messages about the **why** more than the raw list of files.

## 2. Scope (Optional)

When helpful, you may add a scope in parentheses after the type:

- `feat(orchestrator): initial loop skeleton`
- `docs(spec): clarify web-only focus`

Scopes should be short paths or concepts (e.g. `spec`, `agents`, `infra`).

## 3. Content Requirements

- Each commit must be logically cohesive.
- Do not mix unrelated concerns (e.g. spec rewrite + formatting across whole repo).
- Prefer smaller commits over large, multi-purpose ones.
- Never commit secrets or real credentials.

## 4. Tooling Expectations

Before committing:

- Run tests relevant to your change when they exist.
- Ensure `AI_DEV_KIT_SETUP.md` remains accurate if you modify behavior of the LLM loop or agents.

## 5. Amending and History

- It is acceptable to amend the most recent commit when:
  - it has not been pushed, and
  - you are only fixing message typos or adding closely-related files.
- Do not amend or rewrite history of commits that have already been pushed to shared branches.

These rules are intentionally minimal; evolve them as the project grows.
