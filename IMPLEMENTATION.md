# Agent Orchestrator & Tools Implementation

## Overview

Complete implementation of LangGraph-based autonomous research agent with Constitutional AI rules enforcement, hierarchical memory management, and parallel tool execution.

## Files Implemented

### Agent Orchestration (`src/agent/`)

#### state.py
- **AgentState TypedDict**: Complete state schema with 25+ fields
  - Core research task tracking (objective, session_id, sub_goals)
  - Execution tracking (step_number, max_steps, execution_status)
  - Memory management (context_tokens, max_context_tokens)
  - Quality control (rule_violations, proposed_rule_changes)
  - Conversation history and checkpoint data

#### core.py
- **create_agent_graph()**: LangGraph state machine construction
  - 7-node pipeline: plan → think → act → observe → reflect → {synthesize|continue} → enforce_rules
  - Conditional routing from reflect node (continue vs. synthesize decision)
  - SqliteSaver checkpointing for resumability
  - Session management and async task coordination

- **ResearchAgent**: High-level interface
  - `research()`: Execute autonomous research with full state management
  - `resume_session()`: Resume paused research from checkpoint
  - `query_memory()`: Search across sessions
  - `cleanup()`: Graceful resource cleanup

### Nodes (`src/agent/nodes/`)

Each node is a pure function taking (state, memory_manager, model_router) and returning updated state.

#### plan.py - Goal Decomposition
- Uses qwen3:8b (fast model) in JSON mode
- Decomposes objective into 3-5 sub-goals
- Generates structured response with reasoning
- Handles JSON parsing and fallback scenarios
- Token counting and budget tracking

#### think.py - Research Reasoning
- Uses qwen3:32b with thinking enabled (500 token budget)
- Analyzes current goal and available tools
- Outputs ReAct-style reasoning: Thought → Tool Selection
- Retrieves prior context from memory
- Generates tool calls with arguments

#### act.py - Parallel Tool Execution
- Async execution with asyncio.gather()
- Concurrency limit of 3 parallel tools
- Exponential backoff retry (3 attempts, 2^n second delays)
- Proper error handling and logging
- Result aggregation with success/failure tracking

#### observe.py - Finding Storage
- Processes tool results by type (web_search, read_url)
- Extracts titles, content, sources, confidence scores
- Persists to memory (LanceDB + SQLite)
- Updates context token count
- Triggers compression at 80% utilization

#### reflect.py - Progress Analysis
- Checks hard limits: max_steps (15) forces synthesis
- Monitors context budget: 80% → warn, 95% → critical
- Evaluates goal completion and research sufficiency
- Routes to: next_goal, synthesize, or exhaustion path
- Adds detailed reflection to message history

#### synthesize.py - Response Generation
- Uses qwen3:32b (thinking disabled for speed)
- Organizes findings by theme
- Generates coherent multi-paragraph response
- Adds citations in [N] format
- Includes key insights and further research suggestions

#### enforce_rules.py - Constitutional AI
- Checks response against hard/soft rules (placeholder)
- Detects rule violations
- Triggers revision if needed via model_router
- Logs all violations to state
- Fallback: returns original if revision fails

### Tools (`src/tools/`)

#### __init__.py - Tool Registry
- Tool schema generation for LLM use
- Dispatch to registered tools
- Error handling and logging

#### web.py - Web Tools
- **web_search()**: Tavily API integration
  - LLM-optimized results
  - Score-ranked results
  - Returns title, content, URL, score
  - Includes AI-generated answers if available
  - 30-second timeout with error handling

- **read_url()**: Jina Reader integration
  - Free markdown extraction
  - Clean content without boilerplate
  - No API key required
  - Timeout and error handling

#### memory.py - Memory Tools
- **save_finding()**: Persist findings to memory
- **retrieve_context()**: Query memory with budget
- **update_session_status()**: Track execution state
- **compress_context()**: Trigger memory compression

### CLI (`scripts/agent.py`)

Entry point with subcommands:

```bash
# Run new research
python -m scripts.agent run \
  --objective "Find quantum computing advances" \
  --max-steps 15 \
  --output results.json

# Resume paused session
python -m scripts.agent resume \
  --session-id abc123def456 \
  --output results.json

# Query memory
python -m scripts.agent query \
  --query "quantum error correction" \
  --top-k 5
```

Features:
- Structured logging (JSON)
- Environment variable loading (.env)
- JSON output serialization
- Help and usage documentation
- Graceful error handling with exit codes

## Architecture Decisions

### Async-First Design
- All I/O operations are async (network, disk, LLM)
- Tool execution parallelized with semaphore (3 concurrent max)
- Retry logic uses exponential backoff with asyncio.sleep()
- Proper exception handling in async contexts

### State Management
- TypedDict for type-safe state passing
- Immutable state transitions (return updated dict)
- Checkpoint persistence at key points
- Session resumability via SqliteSaver

### Memory Hierarchy
- **Tier 1**: Working memory (in-context, 4K tokens)
- **Tier 2**: Semantic memory (LanceDB, hybrid search)
- **Tier 3**: Episodic memory (SQLite, metadata + checkpoints)

### Tool Integration
- LLM-tool loop via tool calls in completion response
- Retry on failure with exponential backoff (2s, 4s, 8s)
- Max 3 concurrent tools to avoid resource exhaustion
- Result aggregation with success/failure tracking

### Context Budget Enforcement
- Hard limits: 16K tokens total (from ContextBudget model)
- 80% utilization → warning + potential compression
- 95% utilization → critical, force synthesis
- Automatic step counting and token tracking

### Hardware Constraint Handling
- Never load two 32B models: ModelRouter enforces
- qwen3:8b always resident for routing
- Fallback chains: primary → 30B MoE → 32B reasoning
- Unload models when switching for OOM prevention

## Type Safety

All components use full type hints:
- Function signatures with complete types
- Pydantic v2 for data models
- TypedDict for state schema
- Union types for tool results

## Logging

Structured logging with structlog:
- JSON output for log aggregation
- Context preservation across async boundaries
- Error tracking with full stack traces
- Session and step tracing

## Error Handling

Comprehensive error handling:
- Tool execution retries with backoff
- LLM API error recovery
- Graceful fallbacks (e.g., no tools → continue)
- Session-level error tracking
- Execution status flags for monitoring

## Example Usage

```python
from src.llm.router import ModelRouter
from src.memory import MemoryManager
from src.agent import ResearchAgent

# Initialize
model_router = ModelRouter()
memory_manager = MemoryManager()
agent = ResearchAgent(memory_manager, model_router)

# Run research
result = await agent.research(
    objective="Find latest quantum computing breakthroughs",
    max_steps=15,
)

print(result['final_response'])
print(f"Findings: {len(result['findings'])}")
print(f"Status: {result['execution_status']}")

await agent.cleanup()
```

## Configuration

### Environment Variables
- `TAVILY_API_KEY`: Tavily search API key (required for web_search)
- `OLLAMA_BASE_URL`: Ollama endpoint (default: http://localhost:11434)
- `LANCEDB_PATH`: LanceDB storage path (default: data/research_memory.lancedb)

### Config Files
- `config/model_config.yaml`: Model assignments and constraints
- `config/rules.yaml`: Constitutional AI rules (loaded by enforce_rules node)

## Performance

Optimized for M4 Max 36GB hardware:
- Parallel tool execution (3 max concurrent)
- Efficient token counting (cached in CompletionResponse)
- Memory compression at 80% utilization
- Fallback model selection to prevent OOM
- Minimal copying via state dict updates

## Testing

All modules pass Python syntax validation. Unit test templates provided:
- `tests/test_agent.py`: Agent orchestration
- `tests/test_tools.py`: Tool execution
- `tests/test_memory.py`: Memory operations
- `tests/test_rules.py`: Rule enforcement

## Future Enhancements

1. **Multi-turn conversations**: `parent_message_id` field in state
2. **A/B testing**: Rule learning via self-improvement cycle
3. **Advanced retrieval**: HyDE + cross-encoder reranking
4. **Distributed execution**: Multi-device model loading
5. **Web UI**: FastAPI frontend for research dashboard
