"""Reflection engine for adaptive decision-making based on accumulated memory."""

from memory import SessionMemory
from typing import Optional


class ReflectionEngine:
    """Analyzes patterns in agent memory and recommends actions."""

    def __init__(self, memory: SessionMemory):
        """Initialize reflection engine with agent memory."""
        self.memory = memory

    def detect_loop(self, loop_threshold: int = 3) -> bool:
        """Detect if agent is stuck in a loop (repeating same action)."""
        if len(self.memory.iterations) < loop_threshold:
            return False

        # Check last N iterations
        recent = self.memory.iterations[-loop_threshold:]
        tools = [it.tool_used for it in recent]
        args = [str(it.input_args) for it in recent]

        # If all same tool + same args = loop
        if len(set(tools)) == 1 and len(set(args)) == 1:
            return True
        return False

    def analyze_failure_patterns(self) -> dict:
        """Identify patterns in failures."""
        if not self.memory.failures:
            return {}

        patterns = {}
        for failure in self.memory.failures:
            reason = failure.get("reason", "").lower()
            # Extract error type
            if "connection" in reason or "network" in reason:
                error_type = "network_error"
            elif "timeout" in reason:
                error_type = "timeout"
            elif "import" in reason or "module" in reason:
                error_type = "missing_library"
            elif "404" in reason or "not found" in reason:
                error_type = "endpoint_error"
            else:
                error_type = "other"

            patterns[error_type] = patterns.get(error_type, 0) + 1

        return patterns

    def get_reflection_prompt(self) -> str:
        """Generate reflection prompt forcing tool usage based on memory analysis."""

        # Build analysis but embed action recommendation
        action = "web_search"  # Default
        action_arg = "prediction markets API"

        # 1. Loop detection - change strategy
        if self.detect_loop():
            last_tool = self.memory.iterations[-1].tool_used if self.memory.iterations else "web_search"
            if last_tool == "run_python":
                action = "web_search"
                action_arg = "alternative prediction market API"
            elif last_tool == "web_search":
                action = "run_python"
                action_arg = "try different code approach"
            elif last_tool == "bash":
                action = "run_python"
                action_arg = "test with Python instead"

        # 2. Failure analysis - suggest different approach
        patterns = self.analyze_failure_patterns()
        if patterns:
            if patterns.get("endpoint_error"):
                action = "web_search"
                action_arg = "alternative API or different endpoint"
            elif patterns.get("network_error"):
                action = "run_python"
                action_arg = "use httpx or different request method"
            elif patterns.get("missing_library"):
                action = "run_python"
                action_arg = "use only built-in libraries"

        # 3. Success tracking - continue what works
        if self.memory.successes:
            last_success = self.memory.successes[-1]
            approach = last_success.get('approach', '').lower()
            if 'web_search' in approach:
                action = "run_python"
                action_arg = "implement what you learned from search"
            elif 'run_python' in approach:
                action = "write_file"
                action_arg = "save working code"

        # AGGRESSIVE: Return ACTUAL TOOL CALL not narrative
        # Build proper tool call the model will echo
        if action == "web_search":
            tool_call = f'<tool>web_search</tool>\n<args>{{"query": "{action_arg}"}}</args>'
        elif action == "run_python":
            tool_call = f'<tool>run_python</tool>\n<args>{{"code": "{action_arg}"}}</args>'
        elif action == "write_file":
            tool_call = f'<tool>write_file</tool>\n<args>{{"path": "{action_arg}.txt", "content": "Results"}}</args>'
        else:
            tool_call = f'<tool>{action}</tool>\n<args>{{"query": "{action_arg}"}}</args>'

        return tool_call

    def _suggest_strategy_for_failures(self, patterns: dict) -> str:
        """Suggest strategy based on failure patterns."""
        suggestions = "\n**Recommendations**:\n"

        if patterns.get("endpoint_error"):
            suggestions += "  → Try different API endpoint or library\n"
        if patterns.get("network_error"):
            suggestions += "  → Use different network request method (httpx vs requests)\n"
        if patterns.get("timeout"):
            suggestions += "  → Increase timeout or optimize code\n"
        if patterns.get("missing_library"):
            suggestions += "  → Use only built-in libraries or install missing ones\n"

        return suggestions

    def _get_next_steps(self) -> str:
        """Get recommended next steps."""
        if self.detect_loop():
            return "  1. Identify what you were trying\n  2. Pick a DIFFERENT library/API/approach\n  3. Try again\n"

        if self.memory.iterations:
            last_tool = self.memory.iterations[-1].tool_used
            if last_tool == "web_search":
                return "  1. You searched. Now WRITE and TEST code based on findings\n"
            elif last_tool == "run_python":
                return "  1. You ran code. ANALYZE result, search for solutions, or try different approach\n"
            elif last_tool == "bash":
                return "  1. You ran bash. Next: test with Python or search for help\n"

        return "  1. Use web_search to gather information\n  2. Use run_python to test code\n  3. Use write_file to save progress\n  4. Iterate and refine\n"

    def should_reflect(self) -> bool:
        """Should agent take time to reflect?"""
        # Reflect every N iterations
        return len(self.memory.iterations) % 5 == 0 and len(self.memory.iterations) > 0

    def get_memory_summary_for_llm(self, max_lines: int = 20) -> str:
        """Generate concise memory summary for LLM context."""
        summary = f"📚 **Session Memory Summary**:\n"
        summary += f"   Iterations: {len(self.memory.iterations)}\n"
        summary += f"   Success rate: {len(self.memory.successes) / max(1, len(self.memory.iterations)):.1%}\n"

        if self.memory.discoveries:
            summary += f"\n   **Recent Discoveries**:\n"
            for d in self.memory.discoveries[-3:]:
                summary += f"   • {d[:60]}\n"

        if self.memory.strategies_tried:
            summary += f"\n   **Strategies Tried**:\n"
            for s in self.memory.strategies_tried[-3:]:
                summary += f"   • {s}\n"

        patterns = self.analyze_failure_patterns()
        if patterns:
            summary += f"\n   **Common Errors**: {', '.join(patterns.keys())}\n"

        return summary
