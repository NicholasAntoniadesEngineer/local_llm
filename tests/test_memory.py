"""Tests for memory layer (working, vector, persistent)."""

import pytest
from datetime import datetime, timedelta
from collections import deque
from typing import Any

from src.memory.models import (
    ConversationMessage,
    Finding,
    Checkpoint,
    SessionInfo,
    ContextBudget,
    RuleUpdate,
    ActionLog,
    MemoryEntryType,
)


@pytest.mark.unit
class TestContextBudget:
    """Test context budget allocation and enforcement."""

    def test_context_budget_initialization(self, sample_context_budget: ContextBudget):
        """Test ContextBudget initializes with correct allocations."""
        budget = sample_context_budget
        assert budget.system_prompt == 1024
        assert budget.tool_definitions == 2048
        assert budget.retrieved_memory == 4096
        assert budget.conversation_history == 4096
        assert budget.workspace_scratch == 3072
        assert budget.response_buffer == 2048
        assert budget.total_budget == 16384

    def test_total_allocated_calculation(self, sample_context_budget: ContextBudget):
        """Test total allocated tokens calculation."""
        budget = sample_context_budget
        total = budget.get_total_allocated()
        expected = 1024 + 2048 + 4096 + 4096 + 3072 + 2048
        assert total == expected == 16384

    def test_utilization_at_zero(self, sample_context_budget: ContextBudget):
        """Test utilization is 0% at 0 tokens used."""
        budget = sample_context_budget
        utilization = budget.get_utilization(0)
        assert utilization == 0.0

    def test_utilization_at_50_percent(self, sample_context_budget: ContextBudget):
        """Test utilization is 0.5 at 50% fill."""
        budget = sample_context_budget
        utilization = budget.get_utilization(8192)
        assert abs(utilization - 0.5) < 0.01

    def test_utilization_at_80_percent(self, sample_context_budget: ContextBudget):
        """Test utilization at warning threshold."""
        budget = sample_context_budget
        utilization = budget.get_utilization(13107)  # 80% of 16384
        assert abs(utilization - 0.8) < 0.01

    def test_utilization_exceeds_budget(self, sample_context_budget: ContextBudget):
        """Test utilization caps at 1.0 when exceeding budget."""
        budget = sample_context_budget
        utilization = budget.get_utilization(20000)
        assert utilization == 1.0

    def test_warning_level_below_threshold(self, sample_context_budget: ContextBudget):
        """Test warning level not triggered below 80%."""
        budget = sample_context_budget
        is_warning = budget.is_warning_level(8000)
        assert is_warning is False

    def test_warning_level_at_threshold(self, sample_context_budget: ContextBudget):
        """Test warning level triggered at 80%."""
        budget = sample_context_budget
        is_warning = budget.is_warning_level(13107)  # 80% of 16384
        assert is_warning is True

    def test_critical_level_below_threshold(self, sample_context_budget: ContextBudget):
        """Test critical level not triggered below 95%."""
        budget = sample_context_budget
        is_critical = budget.is_critical_level(14000)
        assert is_critical is False

    def test_critical_level_at_threshold(self, sample_context_budget: ContextBudget):
        """Test critical level triggered at 95%."""
        budget = sample_context_budget
        is_critical = budget.is_critical_level(15564)  # 95% of 16384
        assert is_critical is True

    def test_remaining_budget_at_zero_usage(self, sample_context_budget: ContextBudget):
        """Test remaining budget is full at zero usage."""
        budget = sample_context_budget
        remaining = budget.remaining_budget(0)
        assert remaining == 16384

    def test_remaining_budget_at_half_usage(self, sample_context_budget: ContextBudget):
        """Test remaining budget at 50% usage."""
        budget = sample_context_budget
        remaining = budget.remaining_budget(8192)
        assert remaining == 8192

    def test_remaining_budget_at_full_usage(self, sample_context_budget: ContextBudget):
        """Test remaining budget is 0 at full usage."""
        budget = sample_context_budget
        remaining = budget.remaining_budget(16384)
        assert remaining == 0

    def test_remaining_budget_over_limit(self, sample_context_budget: ContextBudget):
        """Test remaining budget never goes negative."""
        budget = sample_context_budget
        remaining = budget.remaining_budget(20000)
        assert remaining == 0


@pytest.mark.unit
class TestConversationMessage:
    """Test conversation message model."""

    def test_message_creation(self, sample_message: ConversationMessage):
        """Test creating a conversation message."""
        msg = sample_message
        assert msg.session_id == "session_001"
        assert msg.turn_number == 0
        assert msg.role == "user"
        assert msg.tokens == 12
        assert msg.model_used == "qwen3:8b"

    def test_message_has_unique_id(self, sample_message: ConversationMessage):
        """Test each message gets a unique ID."""
        msg1 = ConversationMessage(
            session_id="sess1",
            turn_number=0,
            role="user",
            content="Hello",
            tokens=1,
        )
        msg2 = ConversationMessage(
            session_id="sess1",
            turn_number=1,
            role="assistant",
            content="Hi",
            tokens=1,
        )
        assert msg1.id != msg2.id

    def test_message_turn_number_validation(self):
        """Test turn number validation rejects negative values."""
        with pytest.raises(ValueError):
            ConversationMessage(
                session_id="sess1",
                turn_number=-1,
                role="user",
                content="Test",
                tokens=1,
            )

    def test_message_tokens_validation(self):
        """Test tokens validation rejects negative values."""
        with pytest.raises(ValueError):
            ConversationMessage(
                session_id="sess1",
                turn_number=0,
                role="user",
                content="Test",
                tokens=-5,
            )

    def test_message_latency_validation(self):
        """Test latency validation rejects negative values."""
        with pytest.raises(ValueError):
            ConversationMessage(
                session_id="sess1",
                turn_number=0,
                role="user",
                content="Test",
                tokens=1,
                latency_ms=-10.0,
            )

    def test_message_timestamp_auto_set(self, sample_message: ConversationMessage):
        """Test timestamp is automatically set."""
        before = datetime.utcnow()
        msg = ConversationMessage(
            session_id="sess1",
            turn_number=0,
            role="user",
            content="Test",
            tokens=1,
        )
        after = datetime.utcnow()
        assert before <= msg.timestamp <= after


@pytest.mark.unit
class TestFinding:
    """Test research finding model."""

    def test_finding_creation(self, sample_finding: Finding):
        """Test creating a finding."""
        finding = sample_finding
        assert finding.session_id == "session_001"
        assert finding.title == "Surface Code Error Correction Breakthrough"
        assert finding.confidence == 0.92
        assert len(finding.sources) == 2
        assert finding.importance == 9

    def test_finding_confidence_validation_too_low(self):
        """Test confidence validation rejects values below 0."""
        with pytest.raises(ValueError):
            Finding(
                session_id="sess1",
                title="Test",
                content="Test content",
                confidence=-0.1,
            )

    def test_finding_confidence_validation_too_high(self):
        """Test confidence validation rejects values above 1."""
        with pytest.raises(ValueError):
            Finding(
                session_id="sess1",
                title="Test",
                content="Test content",
                confidence=1.5,
            )

    def test_finding_importance_validation_too_low(self):
        """Test importance validation rejects values below 1."""
        with pytest.raises(ValueError):
            Finding(
                session_id="sess1",
                title="Test",
                content="Test content",
                importance=0,
            )

    def test_finding_importance_validation_too_high(self):
        """Test importance validation rejects values above 10."""
        with pytest.raises(ValueError):
            Finding(
                session_id="sess1",
                title="Test",
                content="Test content",
                importance=11,
            )

    def test_finding_default_confidence(self):
        """Test finding defaults to 0.5 confidence."""
        finding = Finding(
            session_id="sess1",
            title="Test",
            content="Test content",
        )
        assert finding.confidence == 0.5

    def test_finding_default_importance(self):
        """Test finding defaults to 5 importance."""
        finding = Finding(
            session_id="sess1",
            title="Test",
            content="Test content",
        )
        assert finding.importance == 5

    def test_finding_unique_id(self):
        """Test each finding gets a unique ID."""
        f1 = Finding(
            session_id="sess1",
            title="Finding 1",
            content="Content 1",
        )
        f2 = Finding(
            session_id="sess1",
            title="Finding 2",
            content="Content 2",
        )
        assert f1.id != f2.id


@pytest.mark.unit
class TestCheckpoint:
    """Test session checkpoint model."""

    def test_checkpoint_creation(self, sample_checkpoint: Checkpoint):
        """Test creating a checkpoint."""
        checkpoint = sample_checkpoint
        assert checkpoint.session_id == "session_001"
        assert checkpoint.step_number == 3
        assert checkpoint.agent_state["objective"] == "Find quantum computing breakthroughs"
        assert len(checkpoint.completed_actions) == 3

    def test_checkpoint_step_number_validation(self):
        """Test step number validation rejects negative values."""
        with pytest.raises(ValueError):
            Checkpoint(
                session_id="sess1",
                step_number=-1,
                agent_state={},
                memory_state={},
            )

    def test_checkpoint_unique_id(self):
        """Test each checkpoint gets a unique ID."""
        cp1 = Checkpoint(
            session_id="sess1",
            step_number=0,
            agent_state={},
            memory_state={},
        )
        cp2 = Checkpoint(
            session_id="sess1",
            step_number=1,
            agent_state={},
            memory_state={},
        )
        assert cp1.id != cp2.id

    def test_checkpoint_timestamp_auto_set(self):
        """Test checkpoint timestamp is automatically set."""
        before = datetime.utcnow()
        checkpoint = Checkpoint(
            session_id="sess1",
            step_number=0,
            agent_state={},
            memory_state={},
        )
        after = datetime.utcnow()
        assert before <= checkpoint.timestamp <= after

    def test_checkpoint_completed_actions_list(self, sample_checkpoint: Checkpoint):
        """Test checkpoint tracks completed actions."""
        assert sample_checkpoint.completed_actions == ["plan", "research_1", "research_2"]

    def test_checkpoint_next_actions_list(self, sample_checkpoint: Checkpoint):
        """Test checkpoint tracks next actions."""
        assert sample_checkpoint.next_actions == ["research_3", "synthesize"]


@pytest.mark.unit
class TestSessionInfo:
    """Test session information model."""

    def test_session_creation(self, sample_session: SessionInfo):
        """Test creating a session."""
        session = sample_session
        assert session.objective == "Find latest quantum computing breakthroughs"
        assert session.max_steps == 10
        assert session.current_step == 0
        assert session.status == "active"

    def test_session_counter_validation_negative_max_steps(self):
        """Test max_steps validation rejects negative values."""
        with pytest.raises(ValueError):
            SessionInfo(
                objective="Test",
                max_steps=-1,
            )

    def test_session_counter_validation_negative_current_step(self):
        """Test current_step validation rejects negative values."""
        with pytest.raises(ValueError):
            SessionInfo(
                objective="Test",
                current_step=-1,
            )

    def test_session_default_values(self):
        """Test session has sensible defaults."""
        session = SessionInfo(objective="Test objective")
        assert session.max_steps == 10
        assert session.current_step == 0
        assert session.status == "active"
        assert session.findings_count == 0
        assert session.messages_count == 0
        assert session.total_tokens == 0

    def test_session_unique_id(self):
        """Test each session gets a unique ID."""
        s1 = SessionInfo(objective="Objective 1")
        s2 = SessionInfo(objective="Objective 2")
        assert s1.id != s2.id


@pytest.mark.unit
class TestRuleUpdate:
    """Test rule update model."""

    def test_rule_update_creation(self):
        """Test creating a rule update."""
        update = RuleUpdate(
            rule_id="S1",
            rule_type="soft",
            old_rule="Old rule text",
            new_rule="New rule text",
            reason="Improved based on failures",
            status="proposed",
        )
        assert update.rule_id == "S1"
        assert update.rule_type == "soft"
        assert update.status == "proposed"

    def test_rule_update_ab_test_score_validation_too_low(self):
        """Test A/B test score validation rejects values below 0."""
        with pytest.raises(ValueError):
            RuleUpdate(
                rule_id="S1",
                rule_type="soft",
                new_rule="New rule",
                reason="Test",
                ab_test_score_old=-0.1,
            )

    def test_rule_update_ab_test_score_validation_too_high(self):
        """Test A/B test score validation rejects values above 1."""
        with pytest.raises(ValueError):
            RuleUpdate(
                rule_id="S1",
                rule_type="soft",
                new_rule="New rule",
                reason="Test",
                ab_test_score_new=1.5,
            )

    def test_rule_update_valid_ab_test_scores(self):
        """Test rule update accepts valid A/B test scores."""
        update = RuleUpdate(
            rule_id="S1",
            rule_type="soft",
            new_rule="New rule",
            reason="Test",
            ab_test_score_old=0.65,
            ab_test_score_new=0.75,
        )
        assert update.ab_test_score_old == 0.65
        assert update.ab_test_score_new == 0.75

    def test_rule_update_status_transitions(self):
        """Test rule update status transitions."""
        update = RuleUpdate(
            rule_id="S1",
            rule_type="soft",
            new_rule="New rule",
            reason="Test",
            status="proposed",
        )
        assert update.status == "proposed"

        # Update status
        update.status = "accepted"
        assert update.status == "accepted"


@pytest.mark.unit
class TestActionLog:
    """Test action log model."""

    def test_action_log_creation(self):
        """Test creating an action log."""
        log = ActionLog(
            session_id="sess1",
            action_type="web_search",
            tool_name="tavily",
            status="success",
        )
        assert log.session_id == "sess1"
        assert log.action_type == "web_search"
        assert log.tool_name == "tavily"
        assert log.status == "success"

    def test_action_log_duration_validation(self):
        """Test duration validation rejects negative values."""
        with pytest.raises(ValueError):
            ActionLog(
                session_id="sess1",
                action_type="test",
                duration_ms=-50.0,
            )

    def test_action_log_with_error(self):
        """Test action log with error status and message."""
        log = ActionLog(
            session_id="sess1",
            action_type="web_search",
            status="failed",
            error_message="Network timeout after 30s",
        )
        assert log.status == "failed"
        assert log.error_message == "Network timeout after 30s"

    def test_action_log_input_output_data(self):
        """Test action log tracks input and output data."""
        log = ActionLog(
            session_id="sess1",
            action_type="web_search",
            input_data={"query": "quantum computing"},
            output_data={"results": 5, "total_found": 1000},
        )
        assert log.input_data["query"] == "quantum computing"
        assert log.output_data["results"] == 5


@pytest.mark.unit
class TestMemoryFIFOEviction:
    """Test FIFO eviction behavior for working memory."""

    def test_deque_fifo_with_max_length(self):
        """Test deque FIFO eviction with maxlen."""
        # Simulates working memory with max 5 entries
        memory: deque[str] = deque(maxlen=5)

        # Add 3 items
        memory.append("msg_1")
        memory.append("msg_2")
        memory.append("msg_3")
        assert len(memory) == 3

        # Add 4 more (should evict 2 oldest)
        memory.append("msg_4")
        memory.append("msg_5")
        memory.append("msg_6")
        memory.append("msg_7")

        assert len(memory) == 5
        assert list(memory) == ["msg_3", "msg_4", "msg_5", "msg_6", "msg_7"]
        assert "msg_1" not in memory
        assert "msg_2" not in memory

    def test_fifo_eviction_maintains_order(self):
        """Test FIFO eviction preserves order of remaining items."""
        memory: deque[str] = deque(maxlen=3)
        memory.append("a")
        memory.append("b")
        memory.append("c")
        memory.append("d")  # 'a' should be evicted

        assert list(memory) == ["b", "c", "d"]
        assert memory[0] == "b"  # Oldest remaining


@pytest.mark.unit
class TestMemoryDeduplication:
    """Test deduplication logic for memory entries."""

    def test_deduplicate_identical_messages(self):
        """Test deduplication removes identical messages."""
        messages = [
            ConversationMessage(
                session_id="s1",
                turn_number=0,
                role="user",
                content="What is quantum computing?",
                tokens=5,
            ),
            ConversationMessage(
                session_id="s1",
                turn_number=1,
                role="user",
                content="What is quantum computing?",  # Duplicate
                tokens=5,
            ),
            ConversationMessage(
                session_id="s1",
                turn_number=2,
                role="user",
                content="Different question?",
                tokens=3,
            ),
        ]

        # Simple deduplication by content hash
        seen_content = set()
        deduplicated = []
        for msg in messages:
            if msg.content not in seen_content:
                deduplicated.append(msg)
                seen_content.add(msg.content)

        assert len(deduplicated) == 2
        assert deduplicated[0].content == "What is quantum computing?"
        assert deduplicated[1].content == "Different question?"

    def test_deduplicate_findings(self):
        """Test deduplication for findings with similar content."""
        findings = [
            Finding(
                session_id="s1",
                title="Breakthrough A",
                content="Surface codes achieve error correction",
            ),
            Finding(
                session_id="s1",
                title="Breakthrough B",
                content="Surface codes achieve error correction",  # Duplicate
            ),
        ]

        # Deduplicate by content hash
        seen_content = set()
        deduplicated = []
        for finding in findings:
            if finding.content not in seen_content:
                deduplicated.append(finding)
                seen_content.add(finding.content)

        assert len(deduplicated) == 1


@pytest.mark.unit
class TestMemoryCompression:
    """Test memory compression at 80% threshold."""

    def test_compression_triggered_at_80_percent(self):
        """Test compression should trigger at 80% budget fill."""
        budget = ContextBudget()
        used_tokens = int(budget.total_budget * 0.8)

        assert budget.is_warning_level(used_tokens) is True

    def test_compression_not_triggered_below_80(self):
        """Test compression should not trigger below 80%."""
        budget = ContextBudget()
        used_tokens = int(budget.total_budget * 0.79)

        assert budget.is_warning_level(used_tokens) is False

    def test_summarization_reduces_tokens(self):
        """Test that summarization reduces message count."""
        messages = [
            ConversationMessage(
                session_id="s1",
                turn_number=i,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}" * 10,  # Roughly 20 tokens each
                tokens=20,
            )
            for i in range(10)
        ]

        # Simulate compression: keep only last N and summarize rest
        compression_ratio = 0.3
        keep_count = max(2, int(len(messages) * compression_ratio))
        compressed = messages[-keep_count:]

        assert len(compressed) < len(messages)
        assert len(compressed) == keep_count


@pytest.mark.unit
class TestSessionLifecycle:
    """Test session creation, update, and completion."""

    def test_session_created_in_active_state(self):
        """Test new session starts in active state."""
        session = SessionInfo(objective="Test objective")
        assert session.status == "active"

    def test_session_update_timestamp(self):
        """Test session updated_at changes on update."""
        session = SessionInfo(objective="Test objective")
        original_updated = session.updated_at

        # Simulate update
        import time
        time.sleep(0.01)  # Small delay
        session.updated_at = datetime.utcnow()

        assert session.updated_at > original_updated

    def test_session_progress_tracking(self):
        """Test session tracks progress (current_step/max_steps)."""
        session = SessionInfo(objective="Test", max_steps=5)
        assert session.current_step == 0
        assert session.max_steps == 5
        assert (session.current_step / session.max_steps) == 0.0

        # Simulate progress
        session.current_step = 3
        assert (session.current_step / session.max_steps) == 0.6

    def test_session_completion_transition(self):
        """Test transitioning session to completed."""
        session = SessionInfo(objective="Test", max_steps=5)
        assert session.status == "active"

        session.status = "completed"
        assert session.status == "completed"


@pytest.mark.unit
class TestCheckpointResumeability:
    """Test checkpoint creation and resumption."""

    def test_checkpoint_captures_agent_state(self, sample_agent_state):
        """Test checkpoint captures full agent state."""
        checkpoint = Checkpoint(
            session_id=sample_agent_state["session_id"],
            step_number=sample_agent_state["step_number"],
            agent_state={
                "objective": sample_agent_state["objective"],
                "sub_goals": sample_agent_state["sub_goals"],
                "current_goal_index": sample_agent_state["current_goal_index"],
            },
            memory_state={},
            completed_actions=["plan", "research_1"],
            next_actions=["research_2"],
        )

        assert checkpoint.agent_state["objective"] == sample_agent_state["objective"]
        assert checkpoint.agent_state["sub_goals"] == sample_agent_state["sub_goals"]

    def test_checkpoint_enables_resume(self, sample_agent_state):
        """Test checkpoint can be used to resume execution."""
        # Create checkpoint at step 3
        checkpoint = Checkpoint(
            session_id=sample_agent_state["session_id"],
            step_number=3,
            agent_state=sample_agent_state,
            memory_state={},
            completed_actions=["plan", "research_1", "research_2"],
            next_actions=["research_3", "synthesize"],
        )

        # Resume from checkpoint
        assert checkpoint.step_number == 3
        assert checkpoint.completed_actions[-1] == "research_2"
        assert checkpoint.next_actions[0] == "research_3"

    def test_checkpoint_preserves_findings(self):
        """Test checkpoint preserves accumulated findings."""
        findings = [
            Finding(session_id="s1", title="Finding 1", content="Content 1"),
            Finding(session_id="s1", title="Finding 2", content="Content 2"),
        ]

        checkpoint = Checkpoint(
            session_id="s1",
            step_number=2,
            agent_state={"findings": [f.dict() for f in findings]},
            memory_state={},
        )

        assert len(checkpoint.agent_state["findings"]) == 2
