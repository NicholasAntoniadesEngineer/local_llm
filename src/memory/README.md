# Memory Layer

Production-quality implementation of a three-tier hierarchical memory system for the autonomous research agent.

## Architecture

```
Working Memory (Tier 1)
├─ In-context FIFO buffer (4K tokens max)
├─ Recent messages and findings
├─ 80% compression threshold triggers summarization
└─ Sub-millisecond access latency

    ↓ (Compress & Archive)

Episodic Memory (Tier 2: SQLite)
├─ Structured metadata and checkpoints
├─ Tables: sessions, conversations, findings, checkpoints, rule_updates, action_log
├─ Foreign key relationships and indexes
├─ ACID guarantees via SQLite
└─ Efficient backup/restore via SQL dumps

    ↓ (Index & Embed)

Semantic Memory (Tier 3: LanceDB)
├─ Vector embeddings (768-dim via Nomic)
├─ Hybrid search: BM25 + vector + RRF fusion
├─ Tags and importance scoring
├─ Sub-millisecond similarity search
└─ Multi-session isolation

MemoryManager (Coordinator)
├─ Unified API across all tiers
├─ Session lifecycle management
├─ Context budget enforcement (16K window)
└─ Async-first, production-safe
```

## Core Components

### 1. Models (`models.py`)

Pydantic v2 models with full validation:

```python
# Session management
SessionInfo(objective, max_steps, status, metadata)

# Conversation tracking
ConversationMessage(session_id, turn_number, role, content, tokens, model_used, ...)

# Findings storage
Finding(session_id, title, content, sources, confidence, importance, tags, ...)

# Checkpoint/resumability
Checkpoint(session_id, step_number, agent_state, memory_state, actions, ...)

# Context budget enforcement
ContextBudget(system_prompt, tool_definitions, retrieved_memory, ...)
  - get_total_allocated()
  - is_warning_level(used_tokens)
  - remaining_budget(used_tokens)
```

All models include:
- Auto-generated UUIDs
- Timestamp auto-setting
- Field validators (ranges, non-negativity)
- Comprehensive docstrings

### 2. Working Memory (`working.py`)

In-context FIFO buffer:

```python
wm = WorkingMemory(max_tokens=4096, compression_threshold=0.8)

# Add messages (auto-evicts on overflow)
wm.add_message(conversation_message)

# Query interface
wm.get_buffer()
wm.get_recent(num_messages=5)
wm.get_by_role("assistant")
wm.get_utilization()  # Returns 0.0-1.0
wm.should_compress()  # True when >= 80%

# Importance-weighted summarization before eviction
count, summary = wm.summarize_and_evict(messages)

# Format for model context window
context = wm.get_context_for_model(context_budget)
```

**Key Features:**
- FIFO eviction with 10% buffer margin
- Importance-weighted summarization (2x for assistant messages)
- Token counting with metadata tracking
- Compression trigger at 80% utilization

### 3. SQLite Store (`sqlite_store.py`)

Persistent structured storage via async aiosqlite:

```python
store = SQLiteStore(db_path="data/agent_state.db")
await store.initialize()  # Creates schema

# Session management
await store.save_session(session_info)
session = await store.get_session(session_id)

# Conversation history
await store.save_conversation(message)
history = await store.get_conversation_history(session_id, limit=100)

# Findings
await store.save_finding(finding)
findings = await store.get_findings(session_id, min_confidence=0.8)

# Checkpoints for resumability
await store.save_checkpoint(checkpoint)
latest = await store.get_latest_checkpoint(session_id)

# Action logs
await store.save_action(action)
actions = await store.get_actions(session_id, status="success")

# Statistics
stats = await store.get_session_stats(session_id)
# Returns: {messages, total_tokens, findings, avg_confidence, actions, ...}

# Cleanup
deleted = await store.cleanup_old_sessions(days_old=30)
```

**Schema:**

```sql
sessions(id, objective, status, created_at, updated_at, metadata)
conversations(id, session_id, turn_number, role, content, tokens, model_used, tool_calls, ...)
findings(id, session_id, title, content, sources, confidence, importance, tags, ...)
checkpoints(id, session_id, step_number, agent_state, memory_state, actions, ...)
rule_updates(id, rule_id, rule_type, old_rule, new_rule, status, ab_test_scores, ...)
action_log(id, session_id, action_type, tool_name, input_data, output_data, status, ...)
```

All tables have:
- Indexes on foreign keys and common filters
- Foreign key constraints with cascading deletes
- JSON columns for complex nested data
- Timestamp tracking

### 4. LanceDB Store (`lancedb_store.py`)

Vector store with hybrid search:

```python
store = LanceDBStore(db_path="data/research_memory.lancedb")
await store.initialize()

# Insert with embedding
await store.insert_entry(
    entry_id="e1",
    text="Document content",
    vector=[0.1, 0.2, ...],  # 768-dim
    entry_type="finding",
    session_id="s1",
    importance=7,
    tags=["quantum", "breakthrough"],
    source="https://...",
)

# Batch insert
await store.batch_insert(entries, session_id="s1")

# Three search modes:
vector_results = await store.vector_search(
    query_vector=embedding,
    session_id="s1",
    limit=10,
)

bm25_results = await store.bm25_search(
    query="quantum computing",
    session_id="s1",
    limit=10,
)

# Hybrid search (RRF fusion)
hybrid_results = await store.hybrid_search(
    query="quantum computing",
    query_vector=embedding,
    session_id="s1",
    limit=10,
    vector_weight=0.5,  # 50% vector, 50% BM25
    bm25_weight=0.5,
)

# Advanced queries
by_tag = await store.get_by_tag("quantum", session_id="s1")
important = await store.get_important_entries(session_id="s1", min_importance=7)
stats = await store.get_stats(session_id="s1")
```

**Features:**
- Reciprocal Rank Fusion (RRF) for hybrid search
- BM25 caching per session
- Importance scoring (1-10)
- Tag filtering and search
- Per-session isolation

### 5. Memory Manager (`manager.py`)

Unified coordinator with session lifecycle:

```python
manager = MemoryManager(sqlite_path="data/agent_state.db", lancedb_path="data/research_memory.lancedb")
await manager.initialize()

# Session lifecycle
session = await manager.start_session("Research objective", max_steps=10)
session = await manager.resume_session(session_id)
await manager.end_session("completed")

# Current session
current = manager.get_current_session()

# Conversation API
msg = await manager.add_message(
    role="user",
    content="Question",
    tokens=10,
    model_used="qwen3:8b",
    latency_ms=150.5,
)

history = await manager.get_conversation_history(limit=50)

# Findings API
finding = await manager.add_finding(
    title="Key Finding",
    content="Summary",
    sources=["https://..."],
    confidence=0.85,
    importance=8,
    tags=["quantum"],
)

findings = await manager.get_findings(limit=50, min_confidence=0.7)

# Checkpoints (resumability)
checkpoint = await manager.save_checkpoint(
    agent_state={"step": 1, "status": "running"},
    memory_state={"messages": 15, "findings": 3},
    completed_actions=["plan", "research_1"],
    next_actions=["research_2"],
)

latest = await manager.get_latest_checkpoint()

# Action logging
action = await manager.log_action(
    action_type="web_search",
    tool_name="tavily",
    input_data={"query": "quantum computers"},
    output_data={"results": 5},
    status="success",
    duration_ms=250,
)

actions = await manager.get_actions(limit=50, status="success")

# Memory retrieval
results = await manager.search_memory(
    query="quantum computing",
    query_vector=embedding,
    search_type="hybrid",
    limit=10,
)

important = await manager.get_important_memories(min_importance=7, limit=10)

# Context budget
budget = manager.get_context_budget()
# Returns: ContextBudget with allocations for all components

working_context = manager.get_working_memory_context()
# Returns: {messages: [...], token_count, utilization, formatted}

stats = manager.get_memory_stats()
# Returns: comprehensive stats across all tiers

# Cleanup
await manager.cleanup_old_sessions(days_old=30)
await manager.close()

# Async context manager
async with MemoryManager(...) as manager:
    session = await manager.start_session(...)
```

## Context Budget Allocation

For 16K context window:

```
System Prompt:       1,024 tokens  (6%)   - Model instructions, rules
Tool Definitions:    2,048 tokens (12%)   - Available tools schema
Retrieved Memory:    4,096 tokens (25%)   - Search results, related findings
Conversation:       4,096 tokens (25%)   - Dialog history
Workspace/Scratch:  3,072 tokens (19%)   - Current task, intermediate work
Response Buffer:    2,048 tokens (13%)   - Leave room for generation
────────────────────────────────────────
Total:             16,384 tokens (100%)
```

Thresholds:
- Warning: 80% (13,107 tokens) → Triggers compression
- Critical: 95% (15,565 tokens) → Force eviction

## Usage Examples

### Basic Session with Messages and Findings

```python
async with MemoryManager() as manager:
    # Start research session
    session = await manager.start_session(
        objective="Find quantum computing breakthroughs",
        max_steps=10,
    )

    # Add conversation
    user_msg = await manager.add_message(
        role="user",
        content="What are recent quantum advances?",
        tokens=8,
    )

    assistant_msg = await manager.add_message(
        role="assistant",
        content="Surface codes have achieved error correction...",
        tokens=45,
        model_used="qwen3:32b",
    )

    # Add findings
    finding = await manager.add_finding(
        title="Surface Code Error Correction",
        content="Google and other labs achieve practical error correction",
        sources=["https://nature.com/articles/..."],
        confidence=0.92,
        importance=9,
        tags=["error-correction", "quantum-computing"],
    )

    # Search memory
    results = await manager.search_memory(
        query="error correction techniques",
        search_type="hybrid",
    )

    # Save checkpoint for resumability
    await manager.save_checkpoint(
        agent_state={"current_goal": "synthesize findings"},
        memory_state={"total_messages": 2, "findings": 1},
    )
```

### Checkpoint and Resume

```python
# Session 1: Research
async with MemoryManager() as manager:
    session = await manager.start_session("Research quantum", max_steps=5)
    # ... do research ...
    await manager.save_checkpoint(
        agent_state={"progress": 0.6},
        memory_state={...},
    )
    # Session ends

# Session 2: Resume
async with MemoryManager() as manager:
    session = await manager.resume_session(session_id)
    checkpoint = await manager.get_latest_checkpoint()

    # Resume from checkpoint
    progress = checkpoint.agent_state["progress"]
    # ... continue from 60% ...
```

### Memory Compression

```python
# Monitor working memory
stats = manager.get_memory_stats()
if stats["working"]["should_compress"]:
    # Compress working memory
    buffer = manager.working.get_buffer()
    count, summary = manager.working.summarize_and_evict(buffer)

    # Save summary to episodic memory
    summary_msg = await manager.add_message(
        role="assistant",
        content=summary,
        tokens=len(summary) // 4,  # Estimate
    )
```

### Rule Learning (Integration with Rules Engine)

```python
# Log proposed rule change
update = await manager.log_rule_update(
    rule_id="S1",
    rule_type="soft",
    old_rule="Prefer papers < 18 months old",
    new_rule="Prefer papers < 12 months old",
    reason="Faster pace of discovery in quantum field",
)

# After A/B testing
update.status = "accepted"
update.ab_test_score_old = 0.72
update.ab_test_score_new = 0.81
await manager.episodic.save_rule_update(update)
```

## Performance Characteristics

| Operation | Tier | Latency | Throughput |
|-----------|------|---------|-----------|
| Add message | Working | <1ms | 1000s/sec |
| Search working | Working | <1ms | - |
| Save to SQLite | Episodic | 1-5ms | 200/sec |
| Retrieve from SQLite | Episodic | 1-10ms | 100s/sec |
| Vector search | Semantic | 10-50ms | 20-50/sec |
| BM25 search | Semantic | 5-20ms | 50-200/sec |
| Hybrid search | Semantic | 20-100ms | 10-50/sec |

Memory:
- Working Memory: ~5-10 MB (4K tokens × ~2.5KB/token)
- SQLite: 100-500 MB (per million messages)
- LanceDB: 600+ MB per million vectors (768-dim)

## Testing

Comprehensive test suite in `tests/test_memory.py`:

```bash
# Run all memory tests
pytest tests/test_memory.py -v

# Run specific test class
pytest tests/test_memory.py::TestWorkingMemory -v

# Run with coverage
pytest tests/test_memory.py --cov=src.memory --cov-report=html
```

Test coverage includes:
- Model validation (boundaries, defaults)
- Working memory FIFO eviction and compression
- SQLite CRUD operations
- LanceDB vector and BM25 search
- Hybrid search RRF fusion
- Manager session lifecycle
- Context budget enforcement
- Checkpoint resumability

## Configuration

### Environment Variables

```bash
# Database paths (absolute paths recommended)
MEMORY_SQLITE_PATH=data/agent_state.db
MEMORY_LANCEDB_PATH=data/research_memory.lancedb

# Working memory settings
WORKING_MEMORY_TOKENS=4096
COMPRESSION_THRESHOLD=0.8

# LanceDB settings
LANCEDB_BATCH_SIZE=1000
LANCEDB_ENABLE_CACHE=true
```

### SQLite Optimization

```python
# PRAGMA settings for performance
await db.execute("PRAGMA synchronous = NORMAL")  # vs FULL (slower)
await db.execute("PRAGMA journal_mode = WAL")     # Write-ahead logging
await db.execute("PRAGMA cache_size = 10000")     # 10K pages cache
```

## Error Handling

All async operations include:
- Structured logging via `structlog`
- Informative error messages
- Graceful degradation (returns empty list if search fails)
- Connection lifecycle management

```python
try:
    results = await manager.search_memory(query)
except Exception as e:
    logger.error("search_failed", error=str(e), query=query[:50])
    return []
```

## Future Enhancements

- [ ] Matryoshka embedding truncation (768→128 for compression)
- [ ] SQLite full-text search (FTS5) integration
- [ ] Temporal decay (older findings weighted lower)
- [ ] Multi-session knowledge transfer
- [ ] Automatic checkpoint pruning
- [ ] Memory profile visualization

## References

- **CLAUDE.md**: Overall project architecture and constraints
- **config/model_config.yaml**: LLM routing and hardware constraints
- **tests/test_memory.py**: Comprehensive test suite with fixtures
