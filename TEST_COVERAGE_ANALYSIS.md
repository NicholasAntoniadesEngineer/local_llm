# Test Coverage Analysis

## Overview

This document analyzes the current test coverage of the `local_llm` codebase and identifies areas where testing should be improved. The project contains **36 source modules** across `src/`, `src/runtime/`, and `skills/`, but only **14 test files** in `tests/`.

---

## Current Test Coverage Map

| Source Module | Test File | Coverage Level |
|---|---|---|
| `src/runtime/controller.py` | `test_controller_contracts.py` | Partial |
| `src/runtime/improve_runner.py` | `test_improve_runner.py` | Partial |
| `src/runtime/mlx_adapter.py` | `test_mlx_adapter.py` | Good |
| `src/runtime/prompt_builder.py` | `test_prompt_builder.py` | Good |
| `src/runtime/repo_bootstrap.py` | `test_repo_bootstrap.py` | Minimal |
| `src/runtime/runtime_support.py` | `test_runtime_support.py` | Good |
| `src/runtime/self_improve_runtime.py` | `test_self_improve_runtime.py` | Minimal |
| `src/runtime/task_state.py` | `test_runtime_task_state.py` | Good |
| `src/runtime/tool_call_parser.py` | `test_tool_call_parser.py` | Partial |
| `src/runtime/tools.py` | `test_tools_editing.py` | Partial |
| `src/runtime/turboquant_mlx_setup.py` | `test_turboquant_mlx_setup.py` | Minimal |
| `src/runtime/verifier.py` | `test_verifier_skill_gates.py` | Good |
| `src/context_manager.py` | `test_context_budget.py` | Partial |
| `src/skill_tree.py` | `test_improve_loop_recovery.py` | Minimal |

### Modules With Zero Test Coverage (22 modules)

**Core (`src/`)**:
- `src/agent.py` — Main agent loop, model loading, context compression
- `src/config.py` — Agent and model configuration dataclasses
- `src/memory.py` — Session memory, similarity retrieval, persistence
- `src/paths.py` — Run directory generation and sanitization
- `src/write_guard.py` — Atomic file writes with rollback

**Runtime (`src/runtime/`)**:
- `src/runtime/policy.py` — Policy engine, step decisions, tool batching
- `src/runtime/patcher.py` — Mutation coordinator with rollback
- `src/runtime/llm_text.py` — Thinking tag stripping, code extraction
- `src/runtime/benchmark_suite.py` — Benchmark case definitions
- `src/runtime/state_store.py` — Persistent state (partially covered via task_state tests)

**Skills (`skills/`)** — All 12 skill modules have zero formal test coverage:
- `skills/code_validator.py`
- `skills/confidence_scorer.py`
- `skills/error_recovery.py`
- `skills/loop_detector.py`
- `skills/memory_compressor.py`
- `skills/metrics.py`
- `skills/orchestrator.py`
- `skills/result_evaluator.py`
- `skills/search_cache.py`
- `skills/self_evaluator.py`
- `skills/smart_router.py`
- `skills/strategy_learner.py`
- `skills/task_planner.py`

> **Note**: Many skill modules contain inline `if __name__ == "__main__"` smoke tests, but these are not integrated into the unittest framework and are not run as part of any test suite.

---

## Priority 1: High-Risk Gaps (Recommend Immediate Action)

### 1. `src/agent.py` — No tests at all

This is the central orchestration module (~450 lines) containing `MLXAgent`, the main ReAct loop, context compression, and model loading. A failure here breaks everything.

**What to test**:
- `_compress_context()`: Verify tiered compression triggers at 40%/60%/80% fill thresholds
- `_format_prompt()`: Ensure Qwen3 chat template is applied correctly
- `_count_tokens()`: Validate token counting with fallback behavior
- `_load_skill_instance()`: Test skill module caching (cache hit vs. miss)
- `reset_for_new_task()`: Verify session state is properly cleared without reloading the model
- Error paths: Model loading failures, malformed prompts, missing skill files

### 2. `src/skill_tree.py` — Minimal coverage for a complex module

This module (~500 lines) manages a SQLite-backed skill DAG with UCB1 bandit scoring, impact propagation, and cycle detection. Only `peek_next_skill()` and `record_pull()` are tested.

**What to test**:
- `add_skill()`: Cycle detection in DAG, duplicate handling, prerequisite validation
- `get_weakest_skill()`: Quality score calculation and ranking
- `_select_best_unlocked_skill_id()`: UCB1 scoring formula correctness
- `mark_completed()` / `mark_failed()`: Status transitions and impact propagation through the DAG
- `get_critical_path()`: Dependency chain analysis
- Database migration paths (`_migrate_v3()`)
- Edge cases: Empty tree, single-node tree, deeply nested dependencies

### 3. `src/runtime/policy.py` — No tests at all

The policy engine (~265 lines) drives every decision the agent makes: which tool to use, when to change phase, how to batch operations.

**What to test**:
- `build_step_policy()`: Policy selection given different task states and phases
- `select_tool_batch()`: Tool batching logic with `READ_BATCH_SAFE_TOOLS`
- `fallback_tool_call()`: Phase-based fallback selection
- `reward_from_outcome()`: Reward calculation including loop penalty
- Config loading with defaults and malformed files

### 4. `src/memory.py` — No tests at all

Session memory (~156 lines) handles persistence and similarity-based retrieval, critical for multi-step tasks.

**What to test**:
- `SessionMemory.retrieve_relevant()`: Jaccard similarity scoring and result ranking
- `MemoryManager.record_attempt()` / `record_discovery()` / `record_failure()` / `record_success()`: State tracking
- Auto-compression when iteration count exceeds threshold
- JSON serialization/deserialization round-trips
- File I/O error handling during save/load

---

## Priority 2: Moderate-Risk Gaps

### 5. `src/write_guard.py` — Atomic writes need validation

**What to test**:
- `_validate_python()`: AST syntax validation catches bad code
- `_minimum_size()`: Size ratio floor prevents accidental file truncation
- `write_text()`: Full write-backup-rollback pipeline
- Rollback behavior on write failure

### 6. `src/context_manager.py` — Partially tested, key paths missing

Currently only hard-shrink and episodic buffer creation are tested.

**What to test**:
- `EpisodicBuffer.compress_messages()`: Full compression logic with different `recent_pairs` values
- `KVCacheManager.ensure_prefix()`: Cache consistency and invalidation
- `ContextBudgetGuard._shrink_messages_to_hard_limit()`: Binary search truncation boundary conditions
- Protected message preservation across multiple compression rounds

### 7. `src/config.py` — No tests at all

**What to test**:
- Model profile validation (all 11 profiles load correctly)
- Invalid model name handling
- Context window vs. `max_tokens` constraint consistency

### 8. `src/runtime/patcher.py` — No tests at all

**What to test**:
- `apply_mutation()` / `commit_mutation()` / `rollback_mutation()` lifecycle
- Rollback restores original content; rollback of new file deletes it
- Double-commit or double-rollback handling

---

## Priority 3: Skill Modules (12 files, all untested)

All skill modules in `skills/` lack formal test coverage. Many contain inline smoke tests that should be migrated to the `tests/` directory. The highest-value targets:

### 9. `skills/loop_detector.py`
- `is_stuck()`: 3-action loop detection with 80% similarity threshold
- `similarity()`: Difflib-based comparison accuracy
- Window sliding behavior

### 10. `skills/smart_router.py`
- `pick_tool()`: Score-based tool selection
- `should_change_phase()`: Phase transition logic

### 11. `skills/error_recovery.py`
- `classify_error()`: Error type classification accuracy
- `should_retry()` / `backoff_seconds()`: Retry logic and exponential backoff

### 12. `skills/search_cache.py`
- `get()` / `set()`: Cache TTL behavior
- `cleanup()`: Expired entry eviction
- `stats()`: Hit/miss tracking accuracy

### 13. `skills/strategy_learner.py`
- `best_strategy()`: Strategy ranking by win rate
- `record_outcome()`: Input validation
- `avoid_strategy()`: Blacklist behavior

---

## Priority 4: Existing Test Quality Improvements

### 14. `test_tool_call_parser.py` — Missing edge cases
- Malformed JSON in tool calls
- Multiple tool calls in a single response
- Incomplete or unterminated code blocks

### 15. `test_turboquant_mlx_setup.py` — Only tests the disabled path
- Add tests for the enabled case (`MLX_USE_TURBO_KV=1`)
- Test behavior when model lacks `layers` attribute

### 16. `test_tools_editing.py` — Missing boundary cases
- Line ranges beyond file end
- Binary file handling
- Empty file operations

### 17. Cross-cutting gaps across all existing tests
- **No integration tests**: No tests verify multi-component interactions (e.g., policy → controller → tools)
- **No error recovery tests**: Almost no tests verify behavior when dependencies fail (disk errors, corrupted state)
- **Fake objects over mocks**: Tests use hand-rolled `FakeX` classes instead of `unittest.mock`, making it harder to verify call sequences

---

## Recommended Action Plan

| Phase | Action | Files to Create | Impact |
|---|---|---|---|
| 1 | Test `agent.py` core paths | `tests/test_agent.py` | High — validates main loop |
| 2 | Test `skill_tree.py` DAG logic | `tests/test_skill_tree.py` | High — validates skill management |
| 3 | Test `policy.py` decisions | `tests/test_policy.py` | High — validates tool selection |
| 4 | Test `memory.py` persistence | `tests/test_memory.py` | Medium — validates state tracking |
| 5 | Test `write_guard.py` atomicity | `tests/test_write_guard.py` | Medium — prevents data loss |
| 6 | Migrate skill inline tests | `tests/test_skills_*.py` | Medium — formalizes existing coverage |
| 7 | Expand existing test edge cases | Update existing test files | Low — hardens current coverage |
| 8 | Add integration tests | `tests/test_integration.py` | Medium — validates component interactions |
