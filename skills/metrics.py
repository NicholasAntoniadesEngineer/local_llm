from typing import Dict, Any

class AgentMetrics:
    """Metrics tracker for agent operations. Tracks tool usage, success rates, and performance metrics."""

    def __init__(self):
        """Initialize metrics tracking with default values."""
        self.tool_call_count: Dict[str, int] = {}
        self.success_count: int = 0
        self.total_steps: int = 0
        self.total_time: float = 0.0
        self.total_tokens_used: int = 0
        self.phase_transitions: list = []

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a float between 0 and 1."""
        return self.success_count / self.total_steps if self.total_steps > 0 else 0.0

    def record_step(self, tool: str, success: bool, duration: float, tokens_used: int = 0) -> None:
        """Record a step with tool usage, success status, duration, and token usage."""
        if not isinstance(tool, str) or not tool:
            raise ValueError("Tool name must be a non-empty string")
        if not isinstance(success, bool):
            raise ValueError("Success status must be a boolean")
        if not isinstance(duration, (int, float)) or duration < 0:
            raise ValueError("Duration must be a non-negative number")
        if not isinstance(tokens_used, int) or tokens_used < 0:
            raise ValueError("Token usage must be a non-negative integer")

        self.total_steps += 1
        self.total_time += duration
        self.total_tokens_used += tokens_used
        if success:
            self.success_count += 1
        self.tool_call_count[tool] = self.tool_call_count.get(tool, 0) + 1

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of all tracked metrics."""
        return {
            'tool_call_count': self.tool_call_count,
            'success_rate': self.success_rate,
            'avg_time_per_step': self.total_time / self.total_steps if self.total_steps > 0 else 0,
            'total_tokens_used': self.total_tokens_used,
            'phase_transitions': self.phase_transitions
        }

    def report(self) -> str:
        """Generate a human-readable report of the metrics."""
        return (
            f"Tool Call Count: {self.tool_call_count}\n"
            f"Success Rate: {self.success_rate:.2%}\n"
            f"Avg Time/Step: {self.total_time / max(1, self.total_steps):.2f}s\n"
            f"Total Tokens: {self.total_tokens_used}\n"
        )

if __name__ == "__main__":
    # Test 1: Basic functionality
    m = AgentMetrics()
    for i in range(10):
        m.record_step("tool_a" if i % 2 == 0 else "tool_b", i < 7, 1.0)
    assert m.success_rate == 0.7, f"Expected 0.7, got {m.success_rate}"
    assert m.total_steps == 10
    assert m.tool_call_count["tool_a"] == 5
    assert m.tool_call_count["tool_b"] == 5
    s = m.summary()
    assert s["success_rate"] == 0.7
    assert s["avg_time_per_step"] == 1.0

    # Test 2: Edge case with zero steps
    m2 = AgentMetrics()
    assert m2.success_rate == 0.0
    assert m2.total_steps == 0
    assert m2.tool_call_count == {}
    assert m2.total_time == 0.0
    assert m2.total_tokens_used == 0

    # Test 3: Invalid tool name
    try:
        m.record_step(123, True, 1.0)
    except ValueError as e:
        assert str(e) == "Tool name must be a non-empty string"

    # Test 4: Invalid success status
    try:
        m.record_step("tool_a", "True", 1.0)
    except ValueError as e:
        assert str(e) == "Success status must be a boolean"

    # Test 5: Negative duration
    try:
        m.record_step("tool_a", True, -1.0)
    except ValueError as e:
        assert str(e) == "Duration must be a non-negative number"

    # Test 6: Negative token usage
    try:
        m.record_step("tool_a", True, 1.0, -5)
    except ValueError as e:
        assert str(e) == "Token usage must be a non-negative integer"

    # Test 7: Empty tool name
    try:
        m.record_step("", True, 1.0)
    except ValueError as e:
        assert str(e) == "Tool name must be a non-empty string"

    # Test 8: Phase transitions
    m.phase_transitions.append("phase_1_to_2")
    m.phase_transitions.append("phase_2_to_3")
    assert len(m.phase_transitions) == 2

    print(m.report())
    print('ALL TESTS PASSED')