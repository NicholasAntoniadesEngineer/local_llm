class AgentMetrics:
    def __init__(self):
        self.tool_call_count = {}
        self.success_count = 0
        self.total_steps = 0
        self.total_time = 0
        self.total_tokens_used = 0
        self.phase_transitions = []

    @property
    def success_rate(self):
        return self.success_count / self.total_steps if self.total_steps > 0 else 0.0

    def record_step(self, tool, success, duration, tokens_used=0):
        self.total_steps += 1
        self.total_time += duration
        self.total_tokens_used += tokens_used
        if success:
            self.success_count += 1
        self.tool_call_count[tool] = self.tool_call_count.get(tool, 0) + 1

    def summary(self):
        return {
            'tool_call_count': self.tool_call_count,
            'success_rate': self.success_rate,
            'avg_time_per_step': self.total_time / self.total_steps if self.total_steps > 0 else 0,
            'total_tokens_used': self.total_tokens_used,
            'phase_transitions': self.phase_transitions
        }

    def report(self):
        return (
            f"Tool Call Count: {self.tool_call_count}\n"
            f"Success Rate: {self.success_rate:.2%}\n"
            f"Avg Time/Step: {self.total_time / max(1, self.total_steps):.2f}s\n"
            f"Total Tokens: {self.total_tokens_used}\n"
        )

if __name__ == "__main__":
    m = AgentMetrics()
    # Deterministic test - 7 successes out of 10
    for i in range(10):
        m.record_step("tool_a" if i % 2 == 0 else "tool_b", i < 7, 1.0)
    assert m.success_rate == 0.7, f"Expected 0.7, got {m.success_rate}"
    assert m.total_steps == 10
    assert m.tool_call_count["tool_a"] == 5
    assert m.tool_call_count["tool_b"] == 5
    s = m.summary()
    assert s["success_rate"] == 0.7
    assert s["avg_time_per_step"] == 1.0
    print(m.report())
    print('ALL TESTS PASSED')
