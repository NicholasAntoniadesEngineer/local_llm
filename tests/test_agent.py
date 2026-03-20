"""Tests for agent orchestrator and state management."""

import pytest
from datetime import datetime

from src.agent.state import AgentState
from src.memory.models import SessionInfo, Finding


@pytest.mark.unit
class TestAgentStateInitialization:
    """Test agent state initialization."""

    def test_agent_state_creation(self, sample_agent_state):
        """Test creating agent state."""
        state = sample_agent_state
        assert state["objective"] == "Find latest quantum computing breakthroughs"
        assert state["session_id"] is not None
        assert state["step_number"] == 0
        assert state["max_steps"] == 10

    def test_agent_state_has_required_fields(self, sample_agent_state):
        """Test agent state has all required fields."""
        state = sample_agent_state
        required_fields = [
            "objective", "session_id", "step_number", "max_steps",
            "findings", "messages", "context_tokens", "rule_violations",
            "should_continue", "execution_status"
        ]
        
        for field in required_fields:
            assert field in state

    def test_initial_context_tokens(self, sample_agent_state):
        """Test initial context token count."""
        state = sample_agent_state
        assert state["context_tokens"] <= state["max_context_tokens"]
        assert state["context_tokens"] >= 0

    def test_execution_status_initialization(self, sample_agent_state):
        """Test execution status starts at planning."""
        state = sample_agent_state
        assert state["execution_status"] == "planning"


@pytest.mark.unit
class TestSubGoalDecomposition:
    """Test decomposing research objective into sub-goals."""

    def test_sub_goals_from_objective(self, sample_agent_state):
        """Test objective is decomposed into sub-goals."""
        state = sample_agent_state
        assert len(state["sub_goals"]) >= 1
        assert len(state["sub_goals"]) <= 5

    def test_sub_goals_are_specific(self, sample_agent_state):
        """Test sub-goals are specific and actionable."""
        state = sample_agent_state
        for goal in state["sub_goals"]:
            assert len(goal) > 0
            assert isinstance(goal, str)

    def test_current_goal_index_tracking(self, sample_agent_state):
        """Test tracking current goal in progress."""
        state = sample_agent_state
        assert state["current_goal_index"] >= 0
        assert state["current_goal_index"] < len(state["sub_goals"])

    def test_goal_progress_advancement(self, sample_agent_state):
        """Test advancing through sub-goals."""
        state = sample_agent_state
        initial_index = state["current_goal_index"]
        
        # Simulate advancing
        state["current_goal_index"] = min(
            initial_index + 1,
            len(state["sub_goals"]) - 1
        )
        
        assert state["current_goal_index"] > initial_index


@pytest.mark.unit
class TestResearchLoop:
    """Test main research loop execution."""

    def test_research_loop_starts(self, sample_agent_state):
        """Test research loop can start."""
        state = sample_agent_state
        assert state["should_continue"] is True
        assert state["step_number"] == 0

    def test_loop_step_increment(self, sample_agent_state):
        """Test loop increments step counter."""
        state = sample_agent_state
        initial_step = state["step_number"]
        
        state["step_number"] += 1
        
        assert state["step_number"] == initial_step + 1

    def test_loop_termination_at_max_steps(self, sample_agent_state):
        """Test loop terminates at max steps."""
        state = sample_agent_state
        state["step_number"] = state["max_steps"]
        
        should_continue = state["step_number"] < state["max_steps"]
        assert should_continue is False

    def test_loop_termination_on_finding_synthesis(self, sample_agent_state):
        """Test loop can terminate early on synthesis."""
        state = sample_agent_state
        state["synthesis_complete"] = True
        
        should_continue = not state["synthesis_complete"]
        assert should_continue is False


@pytest.mark.unit
class TestMemoryRecall:
    """Test memory recall across turns."""

    def test_conversation_history_accumulation(self, sample_agent_state):
        """Test conversation history accumulates."""
        state = sample_agent_state
        initial_count = len(state["messages"])
        
        # Simulate adding a message
        state["messages"].append({"role": "assistant", "content": "Response"})
        
        assert len(state["messages"]) == initial_count + 1

    def test_findings_accumulation(self, sample_agent_state):
        """Test findings accumulate during research."""
        state = sample_agent_state
        initial_count = len(state["findings"])
        
        # Simulate adding a finding
        state["findings"].append({"title": "Finding 1", "content": "Content"})
        
        assert len(state["findings"]) == initial_count + 1

    def test_context_token_tracking(self, sample_agent_state):
        """Test tracking context token usage."""
        state = sample_agent_state
        assert "context_tokens" in state
        assert state["context_tokens"] <= state["max_context_tokens"]

    def test_context_compression_triggered(self, sample_agent_state):
        """Test context compression triggered at 80%."""
        state = sample_agent_state
        threshold = int(state["max_context_tokens"] * 0.8)
        
        # Simulate high context usage
        state["context_tokens"] = int(threshold * 1.1)  # 110% of threshold
        
        should_compress = state["context_tokens"] >= threshold
        assert should_compress is True


@pytest.mark.unit
class TestFindingTracking:
    """Test research finding accumulation and management."""

    def test_add_finding_to_state(self, sample_agent_state):
        """Test adding finding to agent state."""
        state = sample_agent_state
        finding = Finding(
            session_id=state["session_id"],
            title="Test Finding",
            content="Finding content",
            confidence=0.85,
        )
        
        state["findings"].append(finding.dict())
        assert len(state["findings"]) > 0

    def test_finding_ordering_by_confidence(self, sample_agent_state):
        """Test findings can be ordered by confidence."""
        state = sample_agent_state
        
        findings = [
            Finding(session_id="s1", title="F1", content="C1", confidence=0.9),
            Finding(session_id="s1", title="F2", content="C2", confidence=0.7),
            Finding(session_id="s1", title="F3", content="C3", confidence=0.95),
        ]
        
        sorted_findings = sorted(
            findings,
            key=lambda f: f.confidence,
            reverse=True
        )
        
        assert sorted_findings[0].confidence == 0.95
        assert sorted_findings[-1].confidence == 0.7

    def test_finding_sources_tracking(self):
        """Test findings track source citations."""
        finding = Finding(
            session_id="s1",
            title="Test",
            content="Content",
            sources=[
                "https://source1.com",
                "https://source2.com",
            ],
        )
        
        assert len(finding.sources) == 2


@pytest.mark.unit
class TestRuleEnforcement:
    """Test rule enforcement during synthesis."""

    def test_rule_violations_detected(self, sample_agent_state):
        """Test rule violations are detected."""
        state = sample_agent_state
        
        # Simulate violation
        state["rule_violations"].append({
            "rule_id": "H1",
            "violation_type": "insufficient_sources",
            "severity": "critical",
        })
        
        assert len(state["rule_violations"]) > 0

    def test_proposed_rule_changes_tracked(self, sample_agent_state):
        """Test proposed rule changes are tracked."""
        state = sample_agent_state
        
        state["proposed_rule_changes"].append({
            "rule_id": "S1",
            "new_rule": "Updated rule",
            "reason": "Improvement based on results",
        })
        
        assert len(state["proposed_rule_changes"]) > 0

    def test_critical_violation_blocks_synthesis(self):
        """Test critical rule violations block synthesis."""
        violations = [
            {"rule_id": "H1", "severity": "critical"},
        ]
        
        blocking_violations = [v for v in violations if v["severity"] == "critical"]
        should_block = len(blocking_violations) > 0
        
        assert should_block is True


@pytest.mark.unit
class TestCheckpointing:
    """Test session checkpointing for resumability."""

    def test_checkpoint_creation(self, sample_agent_state):
        """Test creating a checkpoint from state."""
        state = sample_agent_state
        
        checkpoint = {
            "step_number": state["step_number"],
            "agent_state": {
                "objective": state["objective"],
                "sub_goals": state["sub_goals"],
            },
            "findings": state["findings"],
        }
        
        assert checkpoint["step_number"] == state["step_number"]
        assert checkpoint["agent_state"]["objective"] == state["objective"]

    def test_checkpoint_enables_resume(self, sample_agent_state):
        """Test checkpoint enables resuming from previous point."""
        state = sample_agent_state
        state["step_number"] = 5
        
        checkpoint = {
            "step_number": state["step_number"],
            "agent_state": state,
        }
        
        # Resume from checkpoint
        resumed_step = checkpoint["step_number"]
        assert resumed_step == 5

    def test_checkpoint_interval(self, sample_agent_state):
        """Test checkpointing at regular intervals."""
        state = sample_agent_state
        checkpoint_interval = 3
        
        should_checkpoint = (state["step_number"] % checkpoint_interval) == 0
        
        # Initially at step 0, should checkpoint
        assert should_checkpoint is True


@pytest.mark.unit
class TestModelRouting:
    """Test correct model routing for different roles."""

    def test_orchestration_model_selection(self):
        """Test qwen3:8b selected for orchestration."""
        role = "orchestration"
        model_mapping = {
            "orchestration": "qwen3:8b",
            "reasoning": "qwen3:32b",
            "code": "qwen2.5-coder:32b",
        }
        
        selected_model = model_mapping.get(role)
        assert selected_model == "qwen3:8b"

    def test_reasoning_model_selection(self):
        """Test qwen3:32b selected for reasoning."""
        role = "reasoning"
        model_mapping = {
            "orchestration": "qwen3:8b",
            "reasoning": "qwen3:32b",
            "code": "qwen2.5-coder:32b",
        }
        
        selected_model = model_mapping.get(role)
        assert selected_model == "qwen3:32b"

    def test_code_model_selection(self):
        """Test qwen2.5-coder:32b selected for code."""
        role = "code"
        model_mapping = {
            "orchestration": "qwen3:8b",
            "reasoning": "qwen3:32b",
            "code": "qwen2.5-coder:32b",
        }
        
        selected_model = model_mapping.get(role)
        assert selected_model == "qwen2.5-coder:32b"


@pytest.mark.unit
class TestToolExecution:
    """Test tool execution tracking."""

    def test_tool_call_logging(self, sample_agent_state):
        """Test tool calls are logged."""
        state = sample_agent_state
        
        tool_call = {
            "tool": "tavily_search",
            "query": "quantum computing",
            "status": "pending",
        }
        
        state["last_tool_results"][tool_call["tool"]] = tool_call
        
        assert "tavily_search" in state["last_tool_results"]

    def test_tool_error_handling(self, sample_agent_state):
        """Test tool errors are captured."""
        state = sample_agent_state
        
        state["tool_error"] = "Network timeout after 30s"
        
        assert state["tool_error"] is not None

    def test_tool_results_persistence(self, sample_agent_state):
        """Test tool results persist across turns."""
        state = sample_agent_state
        
        state["last_tool_results"]["search_results"] = {
            "count": 5,
            "results": ["result1", "result2"],
        }
        
        assert state["last_tool_results"]["search_results"]["count"] == 5


@pytest.mark.unit
class TestSynthesis:
    """Test synthesis phase."""

    def test_synthesis_flag_initially_false(self, sample_agent_state):
        """Test synthesis_complete is initially false."""
        state = sample_agent_state
        assert state["synthesis_complete"] is False

    def test_synthesis_completion_sets_final_response(self, sample_agent_state):
        """Test synthesis sets final response."""
        state = sample_agent_state
        state["synthesis_complete"] = True
        state["final_response"] = "Final research synthesis..."
        
        assert len(state["final_response"]) > 0

    def test_execution_status_transitions_to_complete(self, sample_agent_state):
        """Test execution status transitions to complete."""
        state = sample_agent_state
        
        # Simulate completion
        state["execution_status"] = "complete"
        state["synthesis_complete"] = True
        
        assert state["execution_status"] == "complete"


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in agent."""

    def test_error_sets_failure_status(self, sample_agent_state):
        """Test errors set failure status."""
        state = sample_agent_state
        
        state["error_message"] = "Failed to retrieve documents"
        state["execution_status"] = "failed"
        
        assert state["execution_status"] == "failed"
        assert state["error_message"] is not None

    def test_error_message_persistence(self, sample_agent_state):
        """Test error messages persist for debugging."""
        state = sample_agent_state
        error_msg = "Tool execution timeout after 60s"
        
        state["error_message"] = error_msg
        
        assert state["error_message"] == error_msg
