# Local Autonomous Research Agent

A high-performance autonomous research agent running entirely on local hardware (Apple M4 Max) using Ollama/MLX, with self-improving Constitutional AI rules, hierarchical memory system, and sophisticated retrieval pipelines.

## Features

- **Local-first inference**: No external LLM APIs, full privacy
- **Hardware-optimized**: Apple M4 Max with 36GB RAM, MLX backend for 2x speed
- **Self-improving rules**: Constitutional AI with A/B testing feedback loops
- **Hierarchical memory**: Working (FIFO), episodic (SQLite), semantic (LanceDB vectors)
- **Sophisticated retrieval**: HyDE expansion + hybrid BM25+vector search + cross-encoder reranking
- **Web research**: Tavily API + Jina Reader for content extraction
- **Checkpointing**: Resume long-running research tasks from checkpoints
- **Async-first**: Full asyncio stack for parallelism

## Quick Start

### Prerequisites
- macOS (M-series Apple Silicon)
- Python 3.11+
- Ollama installed (`brew install ollama`)
- 36GB+ RAM

### Setup

1. **Clone and install**:
```bash
cd local_llm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your Tavily API key (TAVILY_API_KEY)
```

3. **Pull models**:
```bash
ollama pull qwen3:32b      # 19GB, main reasoning model
ollama pull qwen3:30b-a3b  # 18GB, fast MoE orchestrator
ollama pull qwen3:8b       # 5GB, fast routing
ollama pull qwen2.5-coder:32b  # Code generation
ollama pull nomic-embed-text    # 8K context embeddings
```

4. **Run a research task**:
```bash
python -m scripts.agent run \
  --objective "Research state of local LLM inference in 2025" \
  --max-steps 20 \
  --rules config/rules.yaml
```

## Architecture

### Three-Layer System

```
┌──────────────────────────────────────────┐
│ LAYER 1: ORCHESTRATION (LangGraph)       │
│ qwen3:8b — routes, plans, dispatches     │
│ qwen3:30b-a3b — triage, classification   │
├──────────────────────────────────────────┤
│ LAYER 2: INTELLIGENCE                    │
│ qwen3:32b w/thinking → deep research     │
│ qwen2.5-coder:32b → code generation      │
├──────────────────────────────────────────┤
│ LAYER 3: KNOWLEDGE                       │
│ LanceDB — hybrid vector+BM25 search      │
│ SQLite — metadata, checkpoints, rules    │
│ Rules Engine — Constitutional AI         │
└──────────────────────────────────────────┘
```

### Execution Flow

```
Objective → Plan (qwen3:8b)
    ↓
Loop (max 15 steps):
  Think (qwen3:32b + thinking=ON)
    ↓
  Act (parallel tools via asyncio.gather)
    → Tavily web search
    → Jina Reader for content
    → Store findings in LanceDB
    ↓
  Observe (update findings, compress memory at 80%)
    ↓
  Reflect (decide continue vs synthesize)
    ↓
  Enforce Rules (Constitutional AI critique)
    ↓
Synthesize (qwen3:32b + thinking=OFF)
    ↓
Output with citations
```

## Project Structure

```
local_llm/
├── CLAUDE.md                 # Project instructions for AI assistants
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Package metadata & build config
├── pytest.ini               # Test configuration
├── .env.example             # Environment template
│
├── config/
│   ├── rules.yaml           # Constitutional AI rules (hard/soft/learning)
│   ├── model_config.yaml    # Model role assignments
│   └── prompts/             # Jinja2 prompt templates
│       ├── system.j2
│       ├── research.j2
│       ├── reflect.j2
│       └── rule_critique.j2
│
├── src/
│   ├── llm/                 # LLM inference layer
│   │   ├── base.py          # Abstract LLMClient
│   │   ├── ollama_client.py # Async Ollama wrapper
│   │   ├── mlx_client.py    # MLX backend (2x faster)
│   │   └── router.py        # Smart model routing + constraints
│   │
│   ├── memory/              # Hierarchical memory system
│   │   ├── models.py        # Pydantic data models
│   │   ├── working.py       # FIFO in-context buffer (4K tokens)
│   │   ├── lancedb_store.py # Vector store (hybrid search)
│   │   ├── sqlite_store.py  # Metadata + checkpoints
│   │   └── manager.py       # Unified memory coordinator
│   │
│   ├── retrieval/           # Sophisticated search
│   │   ├── models.py        # Data structures
│   │   ├── hyde.py          # Hypothetical Document Expansion
│   │   ├── hybrid.py        # BM25 + vector + RRF fusion
│   │   ├── reranker.py      # Cross-encoder reranking
│   │   └── chunker.py       # Hierarchical chunking
│   │
│   ├── rules/               # Constitutional AI + self-improvement
│   │   ├── models.py        # Rule data structures
│   │   ├── loader.py        # YAML → XML compilation
│   │   ├── engine.py        # Critique-revise loop
│   │   ├── learner.py       # A/B testing + rule proposals
│   │   └── optimizer.py     # DSPy prompt optimization
│   │
│   ├── agent/               # LangGraph orchestrator
│   │   ├── state.py         # Agent state schema
│   │   ├── core.py          # Main state machine
│   │   └── nodes/           # Graph nodes
│   │       ├── plan.py      # Decompose objective
│   │       ├── think.py     # Reasoning with thinking=ON
│   │       ├── act.py       # Tool execution
│   │       ├── observe.py   # Store findings
│   │       ├── reflect.py   # Compress memory + decide
│   │       └── synthesize.py # Combine findings
│   │
│   └── tools/               # Tool implementations
│       ├── web.py           # Tavily + Jina
│       ├── memory.py        # Memory operations
│       └── __init__.py      # Tool registry
│
├── scripts/
│   ├── agent.py             # CLI: run, resume, query, review-rules, export
│   └── setup.py             # One-shot environment setup
│
└── tests/
    ├── conftest.py          # Pytest fixtures
    ├── test_memory.py       # Memory layer tests
    ├── test_retrieval.py    # Retrieval tests
    ├── test_rules.py        # Rules engine tests
    ├── test_agent.py        # Agent orchestrator tests
    ├── test_integration.py  # Full-system tests
    ├── fixtures/            # Test data
    └── data/               # Static test files
```

## Commands

### Run Research

```bash
python -m scripts.agent run \
  --objective "Your research question" \
  --max-steps 20 \
  --model qwen3:32b \
  --rules config/rules.yaml
```

### Resume Session

```bash
python -m scripts.agent resume --session <session_id>
```

### Query Memory

```bash
python -m scripts.agent query "What did you find about X?"
```

### Review Rules

```bash
python -m scripts.agent review-rules
```

### Optimize Rules (DSPy)

```bash
python -m scripts.agent optimize-rules
```

### Export Results

```bash
python -m scripts.agent export --session <id> --format markdown
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

# Only integration tests
pytest tests/test_integration.py -v

# Only memory tests
pytest tests/test_memory.py -v -m memory
```

## Performance Notes

- **Hardware constraints enforced**: Never load 2x 32B models (would be 38GB > 36GB)
- **Context budget**: 16K tokens max
  - System: 1KB
  - Tools: 2KB
  - Retrieved memory: 4KB
  - Conversation: 4KB
  - Workspace: 3KB
  - Buffer: 2KB
- **Token targets**:
  - Orchestration: 80+ tok/s (qwen3:8b)
  - MoE routing: 90+ tok/s (qwen3:30b-a3b)
  - Reasoning: 15-22 tok/s (qwen3:32b)
  - With MLX backend: 2x faster

## Configuration

### Rules Format (`config/rules.yaml`)

```yaml
version: 1
meta_rules:
  - id: M1
    priority: critical
    rule: "When rules conflict, prefer accuracy"

research_rules:
  hard:
    - id: R1
      rule: "Verify claims against 2+ sources"
  soft:
    - id: S1
      confidence: 0.8
      rule: "Prefer primary sources"

learning_rules: []  # Auto-generated from failures
```

### Model Assignment (`config/model_config.yaml`)

```yaml
roles:
  orchestrate:
    primary: qwen3:8b
    fallback: [qwen3:32b]
  reason:
    primary: qwen3:32b
    thinking_enabled: true
  code:
    primary: qwen2.5-coder:32b
    fallback: [qwen3:32b]
```

## Environment Variables

Create `.env` from `.env.example`:

```bash
TAVILY_API_KEY=your_key_here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=300
DEBUG=false
```

## Contributing

This project uses:
- **Code formatting**: black (line-length: 100)
- **Linting**: ruff
- **Type checking**: mypy
- **Testing**: pytest + pytest-asyncio
- **Async**: full asyncio stack, no blocking

## License

MIT

## References

- **Hardware**: Apple M4 Max (32-core GPU, 410 GB/s bandwidth, 36GB RAM)
- **Models**: Qwen3 family (32B/8B/30B-a3b)
- **Infrastructure**: Ollama + MLX-LM
- **Memory**: LanceDB (hybrid search) + SQLite
- **Orchestration**: LangGraph
- **Rules**: Constitutional AI patterns
- **Retrieval**: HyDE + RRF fusion

## Status

- ✅ Phase 1: Foundation (CLAUDE.md, config, LLM layer)
- ✅ Phase 2: LLM Backend (base, ollama, router)
- 🔄 Phase 3-6: Implementation in progress (memory, retrieval, rules, agent, tests)
- ⏳ Phase 7: Integration & validation
- ⏳ Phase 8: Optimization & deployment
