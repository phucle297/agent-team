# 🧠 AI Agent Team (Claude Code–like, Custom & No Platform Fee)

This project recreates a lightweight version of Claude Code Agent Teams using your own APIs.

## Stack

- Claude API → coding tasks
- Google GenAI API → reasoning / research
- LangGraph → orchestration
- Direnv + Nix → reproducible environment

---

# ⚙️ 1. Concept

Agent Teams = multiple specialized agents coordinated by a planner.

User Input  
↓  
Planner Agent  
↓  
Coder (Claude) + Researcher (Google)  
↓  
Reviewer (Claude)  
↓  
Final Output

---

# 📁 2. Project Structure

ai-agent-team/  
│  
├── README.md  
├── .env  
├── main.py  
│  
├── agents/  
│ ├── planner.py  
│ ├── coder.py  
│ ├── researcher.py  
│ ├── reviewer.py  
│  
├── graph/  
│ └── workflow.py  
│  
├── prompts/  
│ ├── planner.txt  
│ ├── coder.txt  
│ ├── reviewer.txt  
│  
└── utils/  
  └── llm.py

---

# 📄 3. README Content

## Overview

Local multi-agent system similar to Claude Code.

## Agents

- Planner → breaks tasks into steps
- Coder → writes code (Claude only)
- Researcher → gathers info (Google)
- Reviewer → validates/improves output

## Setup

Run: direnv allow

Add .env:

ANTHROPIC_API_KEY=your_key  
GOOGLE_API_KEY=your_key

## Run

Run: python main.py

---

# 🧱 4. LLM Wrapper (utils/llm.py)

    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI

    def get_claude():
        return ChatAnthropic(model="claude-3-5-sonnet-latest")

    def get_google():
        return ChatGoogleGenerativeAI(model="gemini-pro")

---

# 🤖 5. Agents

## Planner (Google)

    from utils.llm import get_google

    llm = get_google()

    def planner(state):
        task = state["input"]
        res = llm.invoke(f"Break this into steps:\n{task}")
        return {"plan": res.content}

---

## Coder (Claude)

    from utils.llm import get_claude

    llm = get_claude()

    def coder(state):
        plan = state["plan"]
        res = llm.invoke(f"Write code for:\n{plan}")
        return {"code": res.content}

---

## Researcher (Google)

    from utils.llm import get_google

    llm = get_google()

    def researcher(state):
        plan = state["plan"]
        res = llm.invoke(f"Research needed info:\n{plan}")
        return {"research": res.content}

---

## Reviewer (Claude)

    from utils.llm import get_claude

    llm = get_claude()

    def reviewer(state):
        code = state["code"]
        res = llm.invoke(f"Review and improve:\n{code}")
        return {"final": res.content}

---

# 🔁 6. Workflow (LangGraph)

    from langgraph.graph import StateGraph
    from agents.planner import planner
    from agents.coder import coder
    from agents.researcher import researcher
    from agents.reviewer import reviewer

    def build_graph():
        graph = StateGraph(dict)

        graph.add_node("planner", planner)
        graph.add_node("coder", coder)
        graph.add_node("researcher", researcher)
        graph.add_node("reviewer", reviewer)

        graph.set_entry_point("planner")

        graph.add_edge("planner", "coder")
        graph.add_edge("planner", "researcher")
        graph.add_edge("coder", "reviewer")

        return graph.compile()

---

# ▶️ 7. Main

    from graph.workflow import build_graph

    app = build_graph()

    result = app.invoke({
        "input": "Build a Node.js REST API with JWT auth"
    })

    print(result["final"])

---

# 🚀 8. Next Upgrades (Important)

### 1. Parallel execution

Run coder + researcher simultaneously

### 2. Iteration loop (VERY important)

review → fix → review → finalize

### 3. File system agent

- read/write files
- modify repo automatically

### 4. Memory

- store past runs (FAISS or simple JSON)

### 5. Tool usage

- git
- terminal
- test runner

---

# 💡 Key Insight

Claude Code is basically:

Planner + Tools + Iteration Loop

You already have ~70%. Missing pieces:

- feedback loop
- tool execution
- repo awareness

---

# 🎯 Final Goal

Build your own:

Local AI Dev Team with full control and no platform fee

---

Next step ideas:

- CLI like Claude Code
- Auto-edit files (real dev agent)
- Multi-agent debugging system 🚀

# How to use

- Install Nix package manager and direnv
- Allow direnv
- source .venv/bin/activate
- pip install -e .
