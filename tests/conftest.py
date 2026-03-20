"""Pytest configuration and shared fixtures for all tests."""

import asyncio
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
import yaml

from src.llm.base import CompletionRequest, CompletionResponse, LLMClient
from src.memory.models import (
    ConversationMessage,
    Finding,
    Checkpoint,
    SessionInfo,
    ContextBudget,
    RuleUpdate,
    ActionLog,
)
from src.retrieval.models import (
    Document,
    Chunk,
    SearchResult,
    HyDeResult,
    ChunkerConfig,
    HybridSearchConfig,
)
from src.rules.models import HardRule, SoftRule, MetaRule, RuleViolation
from src.agent.state import AgentState


# ============================================================================
# Async Test Configuration
# ============================================================================


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for async tests."""
    if asyncio._get_running_loop() is None:
        policy = asyncio.get_event_loop_policy()
        policy.set_event_loop(policy.new_event_loop())
    return asyncio.get_event_loop_policy()


@pytest.fixture(scope="session")
def event_loop(event_loop_policy):
    """Create session-scoped event loop."""
    loop = event_loop_policy.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Mock LLM Client
# ============================================================================


class MockLLMClient(LLMClient):
    """Mock LLM client with deterministic responses."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str | None = None,
        response_text: str = "Mock response",
    ):
        super().__init__(model, base_url)
        self.response_text = response_text
        self.call_count = 0
        self.last_request: CompletionRequest | None = None

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Return mocked completion response."""
        self.call_count += 1
        self.last_request = request

        # Check cache first
        cached = self._cache_get(request)
        if cached:
            return cached

        response = CompletionResponse(
            text=self.response_text,
            stop_reason="end_turn",
            thinking=None,
            tool_calls=[],
            usage={"input_tokens": 50, "output_tokens": 100},
            latency_ms=10.0,
            model=request.model,
            cached=False,
        )
        self._cache_set(request, response)
        return response

    async def count_tokens(self, text: str) -> int:
        """Return estimated token count."""
        return len(text.split()) // 1.3  # Rough approximation

    async def is_available(self) -> bool:
        """Always available in tests."""
        return True


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Provide a mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def mock_llm_8b() -> MockLLMClient:
    """Mock qwen3:8b for orchestration."""
    return MockLLMClient(model="qwen3:8b", response_text="Planning response")


@pytest.fixture
def mock_llm_32b() -> MockLLMClient:
    """Mock qwen3:32b for reasoning."""
    return MockLLMClient(model="qwen3:32b", response_text="Detailed reasoning response")


@pytest.fixture
def mock_llm_coder() -> MockLLMClient:
    """Mock qwen2.5-coder:32b for code generation."""
    return MockLLMClient(model="qwen2.5-coder:32b", response_text="```python\nprint('hello')\n```")


# ============================================================================
# Test Data: Documents and Chunks
# ============================================================================


@pytest.fixture
def sample_documents() -> list[Document]:
    """Provide sample documents for retrieval tests."""
    return [
        Document(
            document_id="doc_001",
            title="Quantum Computing Advances 2024",
            content="Recent breakthroughs in quantum error correction enable larger quantum computers. "
            * 20,
            metadata={"source": "arxiv.org", "published": "2024-01-15"},
            source="https://arxiv.org/abs/2401.xxxxx",
            chunk_count=5,
            embedding=[0.1] * 768,
        ),
        Document(
            document_id="doc_002",
            title="Neural Networks and Deep Learning",
            content="Neural networks with billions of parameters can solve complex tasks. "
            * 20,
            metadata={"source": "neurips.org", "year": 2023},
            source="https://neurips.org/paper/xxxxx",
            chunk_count=3,
            embedding=[0.2] * 768,
        ),
        Document(
            document_id="doc_003",
            title="Edge Computing in IoT",
            content="Edge computing reduces latency and bandwidth requirements for IoT devices. " * 20,
            metadata={"source": "ieee.org", "year": 2024},
            source="https://ieee.org/articles/xxxxx",
            chunk_count=4,
            embedding=[0.3] * 768,
        ),
    ]


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Provide sample text chunks for retrieval tests."""
    return [
        Chunk(
            chunk_id="doc_001::0",
            document_id="doc_001",
            text="Quantum error correction codes protect quantum information from decoherence.",
            token_count=512,
            position=0,
            is_parent=True,
            embedding=[0.1] * 768,
        ),
        Chunk(
            chunk_id="doc_001::0::0",
            document_id="doc_001",
            text="Surface codes are the most practical approach to quantum error correction.",
            token_count=128,
            position=0,
            is_parent=False,
            parent_id="doc_001::0",
            embedding=[0.11] * 768,
        ),
        Chunk(
            chunk_id="doc_001::0::1",
            document_id="doc_001",
            text="Topological codes provide protection against certain error patterns.",
            token_count=128,
            position=1,
            is_parent=False,
            parent_id="doc_001::0",
            embedding=[0.12] * 768,
        ),
        Chunk(
            chunk_id="doc_002::0",
            document_id="doc_002",
            text="Deep neural networks learn hierarchical representations of data.",
            token_count=512,
            position=0,
            is_parent=True,
            embedding=[0.2] * 768,
        ),
        Chunk(
            chunk_id="doc_003::0",
            document_id="doc_003",
            text="Edge computing processes data closer to the source for lower latency.",
            token_count=512,
            position=0,
            is_parent=True,
            embedding=[0.3] * 768,
        ),
    ]


@pytest.fixture
def sample_search_results(sample_chunks) -> list[SearchResult]:
    """Provide sample search results."""
    return [
        SearchResult(
            chunk=sample_chunks[0],
            score=0.95,
            rank=1,
            bm25_rank=2,
            vector_rank=1,
            reranker_score=0.92,
            retrieval_method="hybrid",
        ),
        SearchResult(
            chunk=sample_chunks[1],
            score=0.87,
            rank=2,
            bm25_rank=1,
            vector_rank=3,
            reranker_score=0.85,
            retrieval_method="hybrid",
        ),
        SearchResult(
            chunk=sample_chunks[2],
            score=0.72,
            rank=3,
            bm25_rank=5,
            vector_rank=2,
            reranker_score=0.70,
            retrieval_method="hybrid",
        ),
    ]


# ============================================================================
# Test Data: Memory Objects
# ============================================================================


@pytest.fixture
def sample_session() -> SessionInfo:
    """Provide a sample session info object."""
    return SessionInfo(
        objective="Find latest quantum computing breakthroughs",
        max_steps=10,
        current_step=0,
        status="active",
    )


@pytest.fixture
def sample_message() -> ConversationMessage:
    """Provide a sample conversation message."""
    return ConversationMessage(
        session_id="session_001",
        turn_number=0,
        role="user",
        content="What are the latest advances in quantum computing?",
        tokens=12,
        model_used="qwen3:8b",
        latency_ms=150.0,
    )


@pytest.fixture
def sample_finding() -> Finding:
    """Provide a sample research finding."""
    return Finding(
        session_id="session_001",
        title="Surface Code Error Correction Breakthrough",
        content="Researchers achieved sustained logical qubits using surface codes with error rates below threshold.",
        sources=[
            "https://arxiv.org/abs/2401.xxxxx",
            "https://nature.com/articles/xxxxx",
        ],
        confidence=0.92,
        tags=["quantum", "error-correction", "breakthrough"],
        importance=9,
        category="technical",
    )


@pytest.fixture
def sample_checkpoint() -> Checkpoint:
    """Provide a sample checkpoint."""
    return Checkpoint(
        session_id="session_001",
        step_number=3,
        agent_state={
            "objective": "Find quantum computing breakthroughs",
            "sub_goals": ["Find recent papers", "Identify key techniques"],
            "current_goal_index": 1,
        },
        memory_state={
            "working_memory": ["Finding 1", "Finding 2"],
            "vector_store_size": 150,
        },
        completed_actions=["plan", "research_1", "research_2"],
        next_actions=["research_3", "synthesize"],
    )


@pytest.fixture
def sample_context_budget() -> ContextBudget:
    """Provide a context budget object."""
    return ContextBudget(
        system_prompt=1024,
        tool_definitions=2048,
        retrieved_memory=4096,
        conversation_history=4096,
        workspace_scratch=3072,
        response_buffer=2048,
    )


# ============================================================================
# Test Data: Rules
# ============================================================================


@pytest.fixture
def sample_hard_rules() -> list[HardRule]:
    """Provide sample hard rules."""
    return [
        HardRule(
            id="H1",
            rule="Verify claims against minimum 2 independent sources before concluding",
            enforcement="block_if_violated",
            rationale="Single-source claims have high failure rate",
        ),
        HardRule(
            id="H2",
            rule="Explicitly cite all sources used; include URL or publication details",
            enforcement="block_if_violated",
            rationale="Enables verification and tracks provenance",
        ),
    ]


@pytest.fixture
def sample_soft_rules() -> list[SoftRule]:
    """Provide sample soft rules."""
    return [
        SoftRule(
            id="S1",
            priority="high",
            confidence=0.85,
            rule="Prefer primary sources (papers, official reports) over secondary summaries",
            rationale="Reduces distortion from interpretive layers",
            effectiveness_score=0.82,
        ),
        SoftRule(
            id="S2",
            priority="high",
            confidence=0.78,
            rule="For technical topics, prefer sources published in last 18 months",
            rationale="Rapidly evolving fields; old papers may be superseded",
            effectiveness_score=0.76,
        ),
    ]


@pytest.fixture
def sample_rule_violation(sample_finding) -> RuleViolation:
    """Provide a sample rule violation."""
    return RuleViolation(
        rule_id="H1",
        rule_text="Verify claims against minimum 2 independent sources",
        violation_type="insufficient_sources",
        violation_description="Finding only cites 1 source, needs 2 minimum",
        severity="critical",
        affected_content=sample_finding.content,
        suggested_fix="Search for additional sources to verify claim",
    )


# ============================================================================
# Test Data: Agent State
# ============================================================================


@pytest.fixture
def sample_agent_state(sample_session) -> AgentState:
    """Provide a sample agent state."""
    return AgentState(
        objective=sample_session.objective,
        session_id=sample_session.id,
        sub_goals=[
            "Identify recent quantum breakthroughs",
            "Analyze technical innovations",
            "Synthesize findings",
        ],
        current_goal_index=0,
        step_number=0,
        max_steps=10,
        findings=[],
        messages=[],
        context_tokens=1000,
        max_context_tokens=16384,
        rule_violations=[],
        proposed_rule_changes=[],
        last_tool_results={},
        tool_error=None,
        checkpoint_data={},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        should_continue=True,
        synthesis_complete=False,
        final_response="",
        error_message=None,
        execution_status="planning",
        parent_message_id=None,
        metadata={},
    )


@pytest.fixture
def sample_hyde_result() -> HyDeResult:
    """Provide a sample HyDE expansion result."""
    return HyDeResult(
        query="quantum error correction breakthrough",
        hypotheticals=[
            "Surface code error correction achieves sub-threshold error rates enabling logical qubits",
            "Topological codes protect quantum information from decoherence in large systems",
            "Bosonic codes enable quantum error correction with fewer physical qubits",
        ],
        query_embedding=[0.1] * 768,
        hypothetical_embeddings=[
            [0.11] * 768,
            [0.12] * 768,
            [0.13] * 768,
        ],
        averaged_embedding=[0.115] * 768,
        generation_latency_ms=250.0,
    )


# ============================================================================
# Test Data: YAML Files
# ============================================================================


@pytest.fixture
def sample_rules_yaml() -> str:
    """Provide sample rules YAML content."""
    return """version: 1
updated: 2024-03-20

hard_rules:
  - id: H1
    rule: "Verify claims against minimum 2 independent sources"
    enforcement: "block_if_violated"
    rationale: "Single-source claims have high failure rate"

soft_rules:
  - id: S1
    priority: high
    confidence: 0.85
    rule: "Prefer primary sources over secondary summaries"
    rationale: "Reduces distortion from interpretive layers"
    effectiveness_score: 0.82

meta_rules:
  - id: M1
    priority: critical
    rule: "When rules conflict, prefer accuracy over completeness"
    rationale: "False information is worse than incomplete information"
"""


@pytest.fixture
def sample_model_config_yaml() -> str:
    """Provide sample model config YAML content."""
    return """roles:
  orchestration:
    model: qwen3:8b
    description: "Fast routing and planning decisions"
    context_window: 8192
    max_batch_size: 4

  reasoning:
    model: qwen3:32b
    description: "Deep research and analysis"
    context_window: 16384
    max_batch_size: 1

  code_generation:
    model: qwen2.5-coder:32b
    description: "Code analysis and generation"
    context_window: 16384
    max_batch_size: 1

model_defaults:
  temperature: 0.3
  top_p: 0.9
  top_k: 40
  max_tokens: 2000
"""


@pytest.fixture
def sample_rules_yaml_file(tmp_path, sample_rules_yaml) -> Path:
    """Create a temporary rules.yaml file."""
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(sample_rules_yaml)
    return rules_file


@pytest.fixture
def sample_model_config_yaml_file(tmp_path, sample_model_config_yaml) -> Path:
    """Create a temporary model_config.yaml file."""
    config_file = tmp_path / "model_config.yaml"
    config_file.write_text(sample_model_config_yaml)
    return config_file


# ============================================================================
# Temporary Databases
# ============================================================================


@pytest.fixture
def sqlite_db() -> Generator[sqlite3.Connection, None, None]:
    """Provide an in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")

    # Create schema
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            objective TEXT NOT NULL,
            created_at DATETIME,
            status TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            turn_number INTEGER,
            role TEXT,
            content TEXT,
            tokens INTEGER,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE findings (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            title TEXT,
            content TEXT,
            confidence REAL,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
        """
    )
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def temp_lancedb_path() -> Generator[Path, None, None]:
    """Provide a temporary LanceDB directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Async Utilities
# ============================================================================


@pytest.fixture
def async_mock() -> AsyncMock:
    """Provide an async mock."""
    return AsyncMock()


@pytest.fixture
def async_mock_tavily_search() -> AsyncMock:
    """Provide mock for Tavily search API."""
    mock = AsyncMock()
    mock.return_value = {
        "results": [
            {
                "title": "Result 1",
                "url": "https://example.com/1",
                "content": "Result 1 content",
            },
            {
                "title": "Result 2",
                "url": "https://example.com/2",
                "content": "Result 2 content",
            },
        ]
    }
    return mock


@pytest.fixture
def async_mock_jina_reader() -> AsyncMock:
    """Provide mock for Jina Reader API."""
    mock = AsyncMock()
    mock.return_value = "# Document Title\n\nCleaned markdown content."
    return mock


# ============================================================================
# Parametrized Test Data
# ============================================================================


@pytest.fixture(
    params=[
        ("query1", 0.8),
        ("query2", 0.9),
        ("query3", 0.7),
    ]
)
def parametrized_queries(request) -> tuple[str, float]:
    """Parametrized queries with expected relevance scores."""
    return request.param


# ============================================================================
# Cleanup and Utilities
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_temp_files() -> Generator[None, None, None]:
    """Auto-cleanup temp files after each test."""
    yield
    # Cleanup happens automatically with context managers


def pytest_configure(config):
    """Configure pytest plugins and markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async (deselect with '-m \"not asyncio\"')"
    )
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow (deselect with '-m \"not slow\"')")


@pytest.fixture
def mock_config_files(tmp_path) -> dict[str, Path]:
    """Provide mock configuration files."""
    rules_file = tmp_path / "rules.yaml"
    model_config_file = tmp_path / "model_config.yaml"

    rules_file.write_text(
        """
version: 1
hard_rules:
  - id: H1
    rule: "Test hard rule"
soft_rules:
  - id: S1
    rule: "Test soft rule"
"""
    )

    model_config_file.write_text(
        """
roles:
  orchestration:
    model: qwen3:8b
"""
    )

    return {"rules": rules_file, "model_config": model_config_file}
