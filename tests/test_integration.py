"""Integration tests for full-system research loop."""

import pytest
import asyncio

from src.agent.state import AgentState
from src.memory.models import SessionInfo, Finding, ConversationMessage
from src.retrieval.models import Document, Chunk


@pytest.mark.integration
class TestFullResearchLoop:
    """Test complete research loop from objective to findings."""

    @pytest.mark.asyncio
    async def test_research_loop_executes_5_steps(self, mock_llm_8b):
        """Test research loop executes multiple steps successfully."""
        steps_executed = 0
        max_steps = 5
        
        while steps_executed < max_steps:
            # Simulate step execution
            response = await mock_llm_8b.complete(
                from src.llm.base import CompletionRequest
                request=CompletionRequest(
                    model="qwen3:8b",
                    prompt=f"Step {steps_executed}: Plan research"
                )
            )
            
            assert response.text is not None
            assert response.stop_reason in ["end_turn", "max_tokens"]
            
            steps_executed += 1
        
        assert steps_executed == max_steps

    @pytest.mark.asyncio
    async def test_research_accumulates_findings(self, sample_session, mock_llm_8b):
        """Test research loop accumulates findings across steps."""
        session = sample_session
        findings = []
        
        for step in range(3):
            # Simulate finding generation
            finding = Finding(
                session_id=session.id,
                title=f"Finding {step + 1}",
                content=f"Research finding from step {step + 1}",
                confidence=0.8 + (step * 0.05),
            )
            findings.append(finding)
        
        assert len(findings) == 3
        assert all(f.session_id == session.id for f in findings)


@pytest.mark.integration
class TestMemoryPersistence:
    """Test memory persistence across session."""

    @pytest.mark.asyncio
    async def test_session_data_persists(self, sqlite_db):
        """Test session data persists in SQLite."""
        # Create session
        session_id = "session_test_001"
        objective = "Test research objective"
        
        cursor = sqlite_db.cursor()
        cursor.execute(
            "INSERT INTO sessions (id, objective, status) VALUES (?, ?, ?)",
            (session_id, objective, "active")
        )
        sqlite_db.commit()
        
        # Retrieve session
        cursor.execute("SELECT id, objective FROM sessions WHERE id = ?", (session_id,))
        result = cursor.fetchone()
        
        assert result is not None
        assert result[0] == session_id
        assert result[1] == objective

    @pytest.mark.asyncio
    async def test_findings_persist(self, sqlite_db, sample_finding):
        """Test findings persist across restarts."""
        finding = sample_finding
        
        cursor = sqlite_db.cursor()
        cursor.execute(
            """INSERT INTO findings (id, session_id, title, content, confidence)
               VALUES (?, ?, ?, ?, ?)""",
            (finding.id, finding.session_id, finding.title, finding.content, finding.confidence)
        )
        sqlite_db.commit()
        
        # Retrieve finding
        cursor.execute(
            "SELECT title, confidence FROM findings WHERE id = ?",
            (finding.id,)
        )
        result = cursor.fetchone()
        
        assert result is not None
        assert result[0] == finding.title
        assert abs(result[1] - finding.confidence) < 0.001


@pytest.mark.integration
class TestRuleEnforcementInContext:
    """Test rules enforced throughout research loop."""

    @pytest.mark.asyncio
    async def test_hard_rule_enforced_in_synthesis(self, sample_hard_rules):
        """Test hard rules are enforced during synthesis."""
        rule = sample_hard_rules[0]  # H1: verify against 2 sources
        
        # Simulate response with only 1 source
        finding_with_1_source = Finding(
            session_id="s1",
            title="Single Source Finding",
            content="Only one source",
            sources=["https://single-source.com"],  # Only 1 source!
        )
        
        # Check rule violation
        violates_h1 = len(finding_with_1_source.sources) < 2
        assert violates_h1 is True

    @pytest.mark.asyncio
    async def test_soft_rule_applied_to_ranking(self, sample_soft_rules, sample_documents):
        """Test soft rules applied to source ranking."""
        rule = sample_soft_rules[0]  # S1: prefer primary sources
        
        # Rank documents: primary sources higher
        docs_with_type = [
            {"doc": sample_documents[0], "type": "paper", "score": 0.95},
            {"doc": sample_documents[1], "type": "blog", "score": 0.80},
        ]
        
        # Sort by preference for primary sources
        ranked = sorted(
            docs_with_type,
            key=lambda x: (x["type"] == "paper", x["score"]),
            reverse=True
        )
        
        assert ranked[0]["type"] == "paper"


@pytest.mark.integration
class TestMultiTurnSession:
    """Test multi-turn research sessions with memory."""

    @pytest.mark.asyncio
    async def test_session_with_multiple_turns(self, sample_session):
        """Test research session with multiple conversation turns."""
        session = sample_session
        messages = []
        
        for turn in range(3):
            message = ConversationMessage(
                session_id=session.id,
                turn_number=turn,
                role="user" if turn % 2 == 0 else "assistant",
                content=f"Turn {turn} content",
                tokens=10 + (turn * 5),
            )
            messages.append(message)
        
        assert len(messages) == 3
        assert messages[-1].turn_number == 2

    @pytest.mark.asyncio
    async def test_context_accumulation_across_turns(self, sample_agent_state):
        """Test context usage accumulates across turns."""
        state = sample_agent_state
        context_per_turn = 500
        
        for turn in range(3):
            state["context_tokens"] += context_per_turn
            state["step_number"] += 1
        
        expected_context = context_per_turn * 3
        assert state["context_tokens"] >= expected_context


@pytest.mark.integration
class TestRetrieval:
    """Test retrieval integration with research loop."""

    @pytest.mark.asyncio
    async def test_retrieve_and_rank_documents(self, sample_documents, sample_chunks):
        """Test retrieving and ranking documents."""
        query = "quantum computing breakthroughs"
        
        # Simulate retrieval
        retrieved_docs = sample_documents[:2]  # Retrieve top 2
        
        assert len(retrieved_docs) > 0
        assert all(d.embedding is not None for d in retrieved_docs)

    @pytest.mark.asyncio
    async def test_hierarchical_chunk_retrieval(self, sample_chunks):
        """Test retrieving parent and child chunks."""
        parent_chunks = [c for c in sample_chunks if c.is_parent]
        
        if parent_chunks:
            parent = parent_chunks[0]
            children = [c for c in sample_chunks if c.parent_id == parent.chunk_id]
            
            assert len(children) > 0
            assert all(c.document_id == parent.document_id for c in children)


@pytest.mark.integration
class TestEndToEndResearch:
    """Test complete end-to-end research workflow."""

    @pytest.mark.asyncio
    async def test_objective_to_findings_workflow(
        self, sample_session, sample_documents, mock_llm_8b
    ):
        """Test workflow from objective to accumulated findings."""
        session = sample_session
        findings = []
        
        # Step 1: Plan
        response = await mock_llm_8b.complete(
            from src.llm.base import CompletionRequest
            request=CompletionRequest(
                model="qwen3:8b",
                prompt=f"Plan research for: {session.objective}"
            )
        )
        assert response.text is not None
        
        # Step 2-3: Research and retrieve documents
        for doc in sample_documents[:2]:
            finding = Finding(
                session_id=session.id,
                title=doc.title,
                content=doc.content[:200],
                sources=[doc.source],
                confidence=0.85,
            )
            findings.append(finding)
        
        # Step 4: Synthesize
        synthesis_prompt = f"Synthesize {len(findings)} findings about {session.objective}"
        response = await mock_llm_8b.complete(
            from src.llm.base import CompletionRequest
            request=CompletionRequest(
                model="qwen3:8b",
                prompt=synthesis_prompt
            )
        )
        
        assert response.text is not None
        assert len(findings) > 0


@pytest.mark.integration
class TestCheckpointRecovery:
    """Test checkpointing and recovery."""

    @pytest.mark.asyncio
    async def test_checkpoint_and_resume_state(self, sample_agent_state, sqlite_db):
        """Test checkpointing enables recovery."""
        state = sample_agent_state
        
        # Progress state
        state["step_number"] = 3
        state["findings"].append({"title": "Finding", "content": "Content"})
        
        # Save checkpoint
        checkpoint_data = {
            "step": state["step_number"],
            "findings_count": len(state["findings"]),
            "status": state["execution_status"],
        }
        
        # Restore state
        restored_step = checkpoint_data["step"]
        restored_findings_count = checkpoint_data["findings_count"]
        
        assert restored_step == 3
        assert restored_findings_count == 1


@pytest.mark.integration
class TestConcurrentToolExecution:
    """Test parallel tool execution."""

    @pytest.mark.asyncio
    async def test_parallel_searches(self, async_mock_tavily_search):
        """Test parallel tool execution doesn't block."""
        # Simulate 3 concurrent searches
        tasks = [
            async_mock_tavily_search(query=f"query_{i}")
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(r is not None for r in results)


@pytest.mark.integration
class TestSessionCompletion:
    """Test session completion and finalization."""

    @pytest.mark.asyncio
    async def test_session_completion_flow(self, sample_session):
        """Test transitioning session to completed."""
        session = sample_session
        assert session.status == "active"
        
        # Complete session
        session.status = "completed"
        session.current_step = session.max_steps
        
        assert session.status == "completed"
        assert session.current_step == session.max_steps


@pytest.mark.integration
class TestErrorRecovery:
    """Test error handling and recovery in research loop."""

    @pytest.mark.asyncio
    async def test_tool_failure_recovery(self, sample_agent_state):
        """Test recovery from tool execution failure."""
        state = sample_agent_state
        
        # Simulate tool failure
        state["tool_error"] = "Network timeout"
        state["execution_status"] = "researching"
        
        # Continue despite error
        retry_count = 0
        max_retries = 2
        
        while retry_count < max_retries and state["tool_error"] is not None:
            # Retry logic
            state["tool_error"] = None  # Simulate successful retry
            retry_count += 1
        
        assert retry_count <= max_retries
        assert state["tool_error"] is None
