class AgentMetrics:
    def __init__(self):
        self.tool_call_count = {}
        self.success_rate = 0
        self.total_steps = 0
        self.total_time = 0
        self.total_tokens_used = 0
        self.phase_transitions = []

    def record_step(self, tool, success, duration, tokens_used=0):
        self.total_steps += 1
        self.total_time += duration
        self.total_tokens_used += tokens_used

        if success:
            self.success_rate = (self.success_rate * (self.total_steps - 1) + 1) / self.total_steps
        else:
            self.success_rate = (self.success_rate * (self.total_steps - 1)) / self.total_steps

        if tool in self.tool_call_count:
            self.tool_call_count[tool] += 1
        else:
            self.tool_call_count[tool] = 1

    def summary(self):
        return {
            'tool_call_count': self.tool_call_count,
            'success_rate': self.success_rate,
            'avg_time_per_step': self.total_time / self.total_steps if self.total_steps > 0 else 0,
            'total_tokens_used': self.total_tokens_used,
            'phase_transitions': self.phase_transitions
        }

    def report(self):
        report = """
        Agent Metrics Report:
        """
        report += f"Tool Call Count: {self.tool_call_count}\n"
        report += f"Success Rate: {self.success_rate:.2%}\n"
        report += f"Average Time per Step: {self.total_time / self.total_steps if self.total_steps > 0 else 0:.2f} seconds\n"
        report += f"Total Tokens Used: {self.total_tokens_used}\n"
        report += f"Phase Transitions: {self.phase_transitions}\n"
        return report

# Test block
if __name__ == "__main__":
    metrics = AgentMetrics()
    import random
    
    for _ in range(20):
        tool = random.choice(['tool1', 'tool2', 'tool3'])
        success = random.choice([True, False])
        duration = random.uniform(0.1, 2.0)
        tokens_used = random.randint(10, 100)
        metrics.record_step(tool, success, duration, tokens_used)

    print(metrics.report())
    print('ALL TESTS PASSED')