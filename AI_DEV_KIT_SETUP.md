# AI DEV KIT – SPEC‑DRIVEN LLM AUTONOMOUS DEVELOPMENT SYSTEM

This document defines a fully LLM‑driven, spec‑first development automation system.

The end‑to‑end workflow is controlled by LLM agents orchestrated in Python. There are **no shell (`.sh`) scripts** in the control path. Shell commands (tests, git, builds) are only invoked as tools from Python when needed.

LangChain (or similar libraries) can be used to implement the agent and tool abstractions, but it is an implementation detail. The design below is framework‑agnostic and focuses on behavior and data contracts.

---

# SYSTEM GOALS

- Automate web‑oriented software development from high‑level specs.
- Use an LLM to plan, implement, test, and review changes in a loop.
- Be **spec‑driven**: requirements live in files, not in ad‑hoc prompts.
- Run on free/cheap LLM backends (local models, free tiers, etc.).
- Keep orchestration in Python so the system is portable and debuggable.

High‑level loop:

Requirement Spec  
↓  
Planner Agent (plan steps)  
↓  
Coder Agent (apply web code changes)  
↓  
Tester Agent (run project tests)  
↓  
Reviewer Agent (enforce spec + quality)  
↓  
Git Commit + Push (optional)  
↓  
Repeat Automatically Until Spec Is Satisfied

---

# SECTION 1 – PROJECT STRUCTURE

Canonical layout for a single project using AI DEV KIT:

```text
ai-dev-kit/
│
├── config/
│   └── settings.env
│
├── tasks/
│   └── requirements.txt
│
├── state/
│   └── state.json
│
├── logs/
│   └── run.log
│
├── agents/
│   ├── planner.py
│   ├── coder.py
│   ├── tester.py
│   ├── reviewer.py
│   └── orchestrator.py
│
└── main.py
```

- `config/settings.env`
  - Basic configuration: model name, API/base URL, temperature, max tokens, test command, git settings, etc.

- `tasks/requirements.txt`
  - Current **task spec** in plain text / lightweight markup.
  - Describes desired changes, especially for web code (UI, APIs, full‑stack flows).

- `state/state.json`
  - Machine‑readable run state: active task, current plan, completed steps, last test results, last reviewer verdict.

- `logs/run.log`
  - Append‑only human‑oriented log of runs: timestamps, agent calls, errors.

- `agents/*.py`
  - Python modules implementing each logical agent.

- `main.py`
  - Bootstrap entrypoint: loads config, initializes the LLM client, creates the orchestrator, starts/resumes the loop.

There are **no `.sh` files** in this layout. All orchestration and control flow live in Python modules that call the LLM and, when necessary, call out to the system via subprocess tools.

---

# SECTION 2 – LLM AND TOOLING LAYER

The system is model‑ and framework‑agnostic.

- LLM client
  - Any chat‑style LLM with a JSON API (OpenAI‑compatible, Ollama, LM Studio, local vLLM, etc.).
  - Should support: system + user messages, temperature/top‑p control, reasonable context window.

- Optional orchestration frameworks
  - **LangChain** is a good fit but not required.
  - Alternatives: direct HTTP calls, LlamaIndex, custom wrappers.
  - The spec assumes generic concepts: "call LLM with prompt", "invoke tool", "maintain memory".

- Tools (implemented in Python, exposed to the LLM)
  - `read_file(path, offset, limit)` – safe read of project files.
  - `write_file(path, content)` / `apply_patch(path, patch)` – apply edits with validation.
  - `list_files(glob_pattern)` – discover relevant code.
  - `run_tests()` – execute configured test command, capture exit code and output.
  - `git_status()`, `git_diff()`, `git_commit(message)`, `git_push()` – optional git operations.

The agents never talk directly to the shell. They **request** actions through these tools; Python implements the actual operations.

---

# SECTION 3 – AGENT ROLES

Each agent has a narrow responsibility and communicates via structured messages and the shared `state/state.json`.

## Planner Agent (`planner.py`)

Purpose: turn a human‑written spec into an executable plan.

- Inputs
  - Raw spec from `tasks/requirements.txt`.
  - High‑level context: project type (e.g. Next.js app, Django API), tech stack hints from config.

- Output
  - Plan object (JSON) stored in `state/state.json`, for example:

    ```json
    {
      "task_id": "2026-02-23-homepage-redesign",
      "summary": "Redesign marketing homepage hero section",
      "steps": [
        {
          "id": "analyze",
          "title": "Analyze current homepage implementation",
          "kind": "analysis",
          "target_files": ["web/app/page.tsx"],
          "acceptance": "Clear understanding of current layout and dependencies"
        },
        {
          "id": "implement-ui",
          "title": "Implement new hero layout",
          "kind": "implementation",
          "target_files": ["web/app/page.tsx", "web/app/components/Hero.tsx"],
          "acceptance": "Matches spec text and passes existing tests"
        }
      ]
    }
    ```

- Behavior
  - Never edits code directly.
  - Produces small, testable steps oriented around web features (routes, components, API handlers).

## Coder Agent (`coder.py`)

Purpose: implement plan steps as concrete code changes, especially web code.

- Inputs
  - One plan step from `state/state.json`.
  - Relevant source files (read via tools).

- Output
  - A set of patches to apply (per file) plus a short summary.

- Behavior
  - Focused on web stacks (e.g. React/Next.js/Vue/Svelte, Node/Express, Django/FastAPI).
  - Uses only the file/tool API to inspect and modify code.
  - Writes changes incrementally so tests can run frequently.

## Tester Agent (`tester.py`)

Purpose: run project tests and interpret results.

- Inputs
  - Summary of recent changes (from `state/state.json`).

- Output
  - Structured test result object: command used, exit code, truncated stdout/stderr, pass/fail.

- Behavior
  - Calls a single configurable test command (e.g. `npm test`, `pnpm test`, `pytest`, `cargo test`).
  - Suggests targeted re‑runs if failure logs are noisy.

## Reviewer Agent (`reviewer.py`)

Purpose: enforce the spec and quality bar.

- Inputs
  - Original spec.
  - Plan and completed steps.
  - Latest diffs and test results.

- Output
  - Verdict: `approved` or `changes_required`.
  - If `changes_required`, a list of concrete follow‑up actions mapped back to plan steps or new steps.

- Behavior
  - Checks spec compliance, web UX constraints, code style conventions.
  - Avoids over‑editing; requests only necessary changes.

## Orchestrator (`orchestrator.py`)

Purpose: run the loop and coordinate agents.

- Responsibilities
  - Load and validate config.
  - Read current spec and state.
  - Call Planner when there is no active plan.
  - Iterate: Coder → Tester → Reviewer until task is done or a hard failure occurs.
  - Persist updated `state/state.json` and write human logs.
  - Optionally call git tools to commit and push when Reviewer approves.

---

# SECTION 4 – SPEC FORMAT (SPEC‑DRIVEN INPUT)

`tasks/requirements.txt` is the **single source of truth** for what the system should do.

Recommended sections (plain text, easy for humans and LLMs):

```text
TASK: Short name of the feature or fix

AREA: frontend | backend | fullstack | infra

STACK: e.g. Next.js + TypeScript + Tailwind

USER STORY:
As a visitor, I want to see a modern hero section on the homepage so that I immediately understand the product.

ACCEPTANCE CRITERIA:
1. Hero shows a headline, subheadline, primary CTA button.
2. Layout is responsive on mobile and desktop.
3. Existing tests pass. Add/update tests when necessary.

CONSTRAINTS:
- Reuse existing design tokens.
- Do not introduce new dependencies without approval.

NOTES:
- Any extra details, mock copy, etc.
```

The Planner Agent is responsible for parsing this text and turning it into a structured plan in `state/state.json`.

---

# SECTION 5 – ORCHESTRATION LOOP

Conceptual loop implemented in `main.py` + `orchestrator.py`:

1. **Bootstrap**
   - Load `config/settings.env`.
   - Initialize LLM client.
   - Load `state/state.json` if it exists; otherwise create an empty state.

2. **Ensure Plan Exists**
   - If there is no active plan for the current spec, call Planner.
   - Save the plan to `state/state.json`.

3. **Execute Next Step**
   - Select the next pending step.
   - Call Coder with that step.
   - Apply produced patches.
   - Mark step as `in_progress` or `completed` depending on the Coder result.

4. **Test**
   - When a meaningful unit of work is done (one or more steps), call Tester.
   - Save test results in `state/state.json`.

5. **Review**
   - Call Reviewer with spec, plan, diffs, and test results.
   - If `approved`:
     - Optionally run git commit/push using configured commands.
     - Mark task as `done`.
   - If `changes_required`:
     - Update the plan or add follow‑up steps.
     - Go back to Coder.

6. **Repeat**
   - Continue until the spec is satisfied or a fatal error occurs (e.g. repeated test failures).

All of this logic resides in Python. The only shell interaction is through tools invoked by the orchestrator (tests, git), not hard‑coded `.sh` workflows.

---

# SECTION 6 – OPTIONAL LANGCHAIN INTEGRATION

While the spec does not depend on LangChain, it can simplify implementation:

- Represent each agent as a LangChain `Runnable` or Agent with a system prompt and restricted tools.
- Wrap file, test, and git utilities as LangChain `Tool` objects.
- Use simple memory (JSON + optional LangChain memory components) so that agents can reference past steps without re‑reading everything.

Any other orchestration library can be used as long as it respects the roles, inputs/outputs, and loop described above.

---

# SECTION 7 – NON‑GOALS AND LIMITATIONS

- Not a general chat assistant; it is focused on **spec‑driven web development tasks**.
- Not a CI/CD system; it may call tests and git, but deployment is out of scope.
- Not tied to a specific vendor; it must run with free or low‑cost LLM options when configured.

This document is the authoritative specification for how the autonomous development loop should behave. Python code in `agents/` and `main.py` must follow this design.

---

# SECTION 2 – LANGCHAIN STACK

This system uses LangChain to manage LLM calls, tools, and multi-agent workflows.

- Core LLM interface: `ChatOpenAI`-compatible client (can be backed by local models via Ollama/LM Studio or any free-tier API).
- Orchestration: a central LangChain "controller" chain in `agents/orchestrator.py` that routes tasks to specialist agents.
- Memory: simple JSON-backed memory in `state/state.json` and optional LangChain `ConversationSummaryMemory` for long runs.
- Tools: LangChain `Tool` objects that wrap
  - file system access (read/write/search project files),
  - test runners (e.g. `pytest`, `npm test`),
  - git operations (status/diff/commit) invoked via Python.

All communication between agents and tools goes through LangChain abstractions; there are no direct shell pipelines.

---

# SECTION 3 – AGENT DEFINITIONS

Each agent is a LangChain Runnable (or Agent) with a focused system prompt and limited tool set.

- Planner Agent (`planner.py`)
  - Input: natural language requirement from `tasks/requirements.txt` + current repo snapshot.
  - Output: structured JSON plan (steps with goals, files to touch, acceptance criteria).
  - Tools: repo introspection (list files, read file excerpts).

- Coder Agent (`coder.py`)
  - Input: a single plan step + relevant files.
  - Output: concrete diffs/patches to apply to the codebase.
  - Tools: structured file edit tool (read/patch/write), code search.
  - Focus: web code (frontend/backend) generation and refactoring according to the plan/spec.

- Tester Agent (`tester.py`)
  - Input: summary of recent changes.
  - Output: test command selection and interpretation of results.
  - Tools: test runner tool (wraps Python `subprocess`), log reader.

- Reviewer Agent (`reviewer.py`)
  - Input: plan, applied diffs, test results.
  - Output: approval/rejection + list of follow-up changes.
  - Tools: read diffs, inspect files, query tests.

- Orchestrator (`orchestrator.py`)
  - Calls each agent in sequence using LangChain chains.
  - Persists and reloads `state/state.json` so runs can resume.
  - Decides when a task is done or when another iteration is needed.

---

# SECTION 4 – SPEC-DRIVEN WEB CODING FLOW

Tasks are driven by specs, not ad-hoc prompts.

1. You write a task spec in `tasks/requirements.txt`, for example:

   - feature name
   - affected area (e.g. `frontend`, `backend`, `fullstack`)
   - user story and acceptance criteria
   - tech constraints (frameworks, style guides, test expectations)

2. `main.py` starts the orchestrator, which:
   - loads the spec,
   - calls the Planner Agent via LangChain to create a plan,
   - saves the plan into `state/state.json`.

3. For each plan step, the Coder Agent:
   - reads existing web code (React/Vue/Next/etc.),
   - proposes concrete edits,
   - applies them through a safe patch tool.

4. After meaningful changes, the Tester Agent runs the project's tests and reports back via LangChain.

5. The Reviewer Agent checks:
   - plan vs. actual code,
   - test results,
   - spec compliance.

6. If the Reviewer Agent approves, the orchestrator can optionally run a git commit and push (via tools).

Throughout this flow, agents talk only through LangChain (messages, tools, and memory). There are no shell scripts; everything is LLM-driven inside Python.
