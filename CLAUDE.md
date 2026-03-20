# Local Autonomous Research Agent — Project Instructions

## Project Summary

This is a local-first autonomous research agent built on Ollama/MLX with self-improving rules. It runs entirely on Apple M4 Max (36GB unified RAM, 410 GB/s bandwidth).

The agent researches topics autonomously, enforces Constitutional AI rules on its outputs, learns from failures to improve its rules, and maintains a hierarchical memory system across sessions.

## Architecture At A Glance

```
Orchestrator (LangGraph)
    ↓
LLM Layer (MLX + Ollama)
    ├─ qwen3:8b (routing, 80+ tok/s)
    ├─ qwen3:32b (reasoning, 25-35 tok/s via MLX)
    ├─ qwen3:30b-a3b (MoE, 70-92 tok/s)
    └─ qwen2.5-coder:32b (code, 25-35 tok/s)
    ↓
Memory Layer (LanceDB + SQLite)
    ├─ Tier 1: Working memory (in-context deque)
    ├─ Tier 2: Vector store (LanceDB, hybrid search)
    └─ Tier 3: Metadata + checkpoints (SQLite)
    ↓
Tools (async, parallelized)
    ├─ Web search (Tavily, LLM-optimized)
    ├─ Content extraction (Jina Reader, clean markdown)
    └─ File operations
    ↓
Rules Engine (Constitutional AI)
    ├─ Hard rules (must follow)
    ├─ Soft rules (should follow, can improve)
    └─ Self-improvement via A/B testing
```

## Hardware Constraints (Load-Bearing Facts)

- **36GB RAM** → max concurrent model size ~32B at Q4_K_M (~19GB), +17GB for KV cache + OS
- **Never load two 32B models simultaneously** (38GB > 36GB) — Ollama auto-unloads, but plan for it
- **qwen3:8b always resident** (~5GB) for fast orchestration
- **MLX is 2x faster than Ollama** on Apple Silicon — prefer MLX when available
- **Memory bandwidth: 410 GB/s** → theoretical ceiling ~21.6 passes/sec for 32B model

## Code Standards

**Python 3.11+ only**. Enforce these:

1. **Type hints everywhere**: `def foo(x: str, y: list[int]) -> dict[str, Any]:`
2. **Async-first**: All I/O (network, disk) must be async (httpx, aiohttp, asyncio)
3. **Pydantic v2 data models**: Every function parameter list is a Pydantic model (except simple scalars)
4. **Never block the event loop**: Use `asyncio.to_thread()` for sync calls
5. **No raw dicts in function signatures**: Use dataclasses/Pydantic instead
6. **Structured logging only**: `structlog` logger, never `print()` or bare `logging`
7. **LLM calls only via LLMClient**: Never call Ollama/MLX directly, always route through `src/llm/router.py`
8. **No ChromaDB**: Use LanceDB only (hybrid search, no server needed)
9. **No LangChain AgentExecutor**: Use LangGraph only (supports cycles, checkpointing)
10. **Docstrings**: Every public function/class has a docstring with types and example

## File Layout

```
local_llm/
├── CLAUDE.md                        ← You are here
├── README.md                        ← Getting started, quick test
├── requirements.txt                 ← Python deps, all pinned
├── .env.example                     ← Template (TAVILY_API_KEY, etc.)
├── .gitignore                       ← Ignore .env, __pycache__, data/, logs/
│
├── config/
│   ├── rules.yaml                   ← Constitutional AI rules (YAML → XML)
│   ├── model_config.yaml            ← Role assignments (router config)
│   └── prompts/                     ← Jinja2 templates
│       ├── system.j2
│       ├── research.j2
│       ├── reflect.j2
│       └── rule_critique.j2
│
├── src/
│   ├── __init__.py
│   │
│   ├── agent/                       ← LangGraph orchestration
│   │   ├── __init__.py
│   │   ├── state.py                 ← AgentState TypedDict
│   │   ├── core.py                  ← Main state machine (nodes: plan, research, synthesize)
│   │   ├── objectives.py            ← Goal decomposition
│   │   └── nodes/                   ← Reusable graph nodes
│   │       ├── plan.py
│   │       ├── research.py
│   │       ├── reflect.py
│   │       └── synthesize.py
│   │
│   ├── llm/                         ← LLM backends
│   │   ├── __init__.py
│   │   ├── base.py                  ← Abstract LLMClient
│   │   ├── ollama_client.py         ← Async Ollama wrapper
│   │   ├── mlx_client.py            ← MLX wrapper (2x faster)
│   │   └── router.py                ← Model selection logic
│   │
│   ├── memory/                      ← Hierarchical memory
│   │   ├── __init__.py
│   │   ├── working.py               ← In-context deque (4K tokens)
│   │   ├── lancedb_store.py         ← Vector DB layer
│   │   ├── sqlite_store.py          ← Metadata + checkpoints
│   │   ├── manager.py               ← Unified interface
│   │   └── reflection.py            ← Compression + summarization
│   │
│   ├── retrieval/                   ← Advanced retrieval (HyDE, hybrid, rerank)
│   │   ├── __init__.py
│   │   ├── hyde.py                  ← Hypothetical doc expansion
│   │   ├── hybrid.py                ← BM25 + vector + RRF fusion
│   │   ├── reranker.py              ← Cross-encoder reranking
│   │   └── chunker.py               ← Hierarchical chunking
│   │
│   ├── tools/                       ← Tool implementations
│   │   ├── __init__.py              ← Tool registry + schema
│   │   ├── base.py                  ← Abstract Tool class
│   │   ├── web.py                   ← Tavily search + Jina Reader
│   │   ├── files.py                 ← File I/O (read, write)
│   │   └── compute.py               ← Safe Python execution
│   │
│   └── rules/                       ← Constitutional AI + learning
│       ├── __init__.py
│       ├── engine.py                ← Rule enforcement
│       ├── loader.py                ← YAML → internal format
│       ├── learner.py               ← Self-improvement + A/B testing
│       └── optimizer.py             ← DSPy prompt optimization
│
├── scripts/
│   ├── __init__.py
│   ├── agent.py                     ← CLI entry point
│   └── setup.py                     ← One-shot: pull models, setup DB
│
├── tests/
│   ├── test_memory.py
│   ├── test_retrieval.py
│   ├── test_rules.py
│   └── test_agent.py
│
├── data/                            ← .gitignored
│   ├── research_memory.lancedb/     ← LanceDB storage
│   └── agent_state.db               ← SQLite checkpoints
│
└── logs/                            ← .gitignored
    └── agent.log
```

## Performance Requirements

**These are non-negotiable constraints:**

- **Orchestration calls** (plan, route, decide): Always `qwen3:8b` or `qwen3:30b-a3b` (fast models)
- **Research/reasoning calls**: `qwen3:32b` with `thinking=ON`
- **Synthesis calls**: `qwen3:32b` with `thinking=OFF` (faster, no extended reasoning needed)
- **Code calls**: `qwen2.5-coder:32b` only
- **All tool calls must be parallelized**: Use `asyncio.gather(*tasks)` — never sequential
- **Context window enforcement**: 16K max, budget 1024+2048+4096+4096+3072+2048 tokens
- **KV cache**: Set `num_ctx=16384` to prevent OOM when swapping between 32B models

## Memory Budget (Hard Limits)

For a 16K context window, allocate as follows:

```
System prompt:         1,024 tokens
Tool definitions:      2,048 tokens
Retrieved memory:      4,096 tokens (from LanceDB HyDE+hybrid search)
Conversation history:  4,096 tokens (compressed by ReflectionEngine if >80%)
Workspace/scratch:     3,072 tokens (current task, intermediate findings)
Response buffer:       2,048 tokens (leave room for generation)
────────────────────────────────
Total:                16,384 tokens
```

Enforce this in `src/memory/manager.py` via `ContextBudget` dataclass. Never exceed 80% utilization — when approaching, trigger `ReflectionEngine.compress()`.

## What NOT To Do (Anti-Patterns)

- ❌ **Never load two 32B models together** — OOM at 38GB > 36GB
- ❌ **Never use ChromaDB** — LanceDB is faster, has hybrid search
- ❌ **Never use LangChain AgentExecutor** — LangGraph is more flexible
- ❌ **Never skip context budget enforcement** — model degrades past 80% fill
- ❌ **Never commit rule changes without A/B evaluation** — could regress quality
- ❌ **Never make synchronous calls in async context** — use `asyncio.to_thread()`
- ❌ **Never call Ollama directly** — always route through `OllamaClient`
- ❌ **Never use print() for logging** — use `structlog`
- ❌ **Never have tool calls in sequence** — always `asyncio.gather()`
- ❌ **Never store secrets in code** — use `.env` only

## Rule System (Constitutional AI)

Rules are stored in `config/rules.yaml`, compiled to XML, and enforced before every response:

**Hard Rules** (agent cannot modify):
```yaml
hard:
  - id: H1
    rule: "Verify claims against minimum 2 independent sources"
  - id: H2
    rule: "Flag all sources with potential conflicts of interest"
```

**Soft Rules** (agent can propose improvements):
```yaml
soft:
  - id: S1
    confidence: 0.8
    rule: "Prefer primary sources over secondary summaries"
  - id: S2
    confidence: 0.7
    rule: "For technical topics, prefer papers < 18 months old"
```

**Enforcement cycle** (run after every response):
```python
response = await research_agent.think(query)

for rule in all_rules:
    critique = await llm.critique(rule_id, response)
    if critique.violates:
        response = await llm.revise(response, critique)

store(response, rule_violations=[])
```

**Self-improvement cycle** (after task completion):
```
1. Identify failures (score < 0.6)
2. Agent proposes rule changes
3. Test on holdout test set
4. A/B eval: new_score - old_score
5. If improvement > 5%, commit; else reject
```

## Getting Started

1. **Setup environment**: `python scripts/setup.py`
   - Pulls Qwen3 models
   - Creates LanceDB + SQLite schemas
   - Installs Python deps

2. **Run a simple research task**:
   ```bash
   python -m scripts.agent run \
     --objective "Find 3 facts about local LLM inference" \
     --max-steps 10
   ```

3. **Check rule proposals**: `python -m scripts.agent review-rules`

4. **Query your memory**: `python -m scripts.agent query "What did you find about quantization?"`

## Debugging Tips

- **Structured logs** go to `logs/agent.log` — search for `step=N` to trace execution
- **Use `/tmp/claude_debug` for temporary files**
- **Run tests first**: `pytest tests/` before running full agent
- **Check model memory**: `ollama list` to see loaded models
- **Profile bottlenecks**: Add `@timer` decorator from `src/util/profiling.py`

## Key Dependencies

- **langraph**: State machine with cycles + checkpointing
- **ollama**: API client for Ollama
- **mlx-lm**: Apple Silicon native inference (2x faster)
- **lancedb**: Vector store with hybrid search
- **pydantic**: Data validation
- **httpx**: Async HTTP client
- **structlog**: Structured logging
- **tavily-python**: Web search API
- **rank-bm25**: BM25 scoring for hybrid search
- **sentence-transformers**: Cross-encoder reranker
- **dspy-ai**: Prompt optimization

All pinned in `requirements.txt` — no floating versions.

## Contributing to This Project

If you're working with Claude Code on this project:

1. **Read the ARCHITECTURE section in README.md** before making changes
2. **Follow the code standards** above (types, async, Pydantic, structlog)
3. **Run tests** before committing: `pytest tests/ -v`
4. **Update CLAUDE.md** if you change project structure or add constraints
5. **Commit messages** should reference the component: `[memory] improve hybrid search quality` or `[agent] fix loop exit condition`

This is an aggressive, first-principles implementation. Favor clarity + correctness over cleverness.
