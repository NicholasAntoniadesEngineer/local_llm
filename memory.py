"""Agent working memory and learning history."""

import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any


@dataclass
class Iteration:
    """Single iteration/attempt in the agent's work."""

    iteration_num: int
    step: int
    tool_used: str
    input_args: dict
    result: str
    success: bool
    learning: str = ""  # What was learned from this attempt


@dataclass
class SessionMemory:
    """Complete session working memory."""

    session_id: str
    goal: str
    start_time: str
    iterations: list[Iteration] = field(default_factory=list)
    discoveries: list[str] = field(default_factory=list)  # What agent found
    strategies_tried: list[str] = field(default_factory=list)  # Approaches attempted
    libraries_evaluated: dict[str, str] = field(default_factory=dict)  # lib → evaluation
    failures: list[dict] = field(default_factory=list)  # Failed attempts + why
    successes: list[dict] = field(default_factory=list)  # Successful approaches
    current_strategy: str = ""  # Current approach being pursued
    next_strategy: str = ""  # Next strategy to try if current fails
    progress_score: float = 0.0  # 0-1, how close to goal

    def add_iteration(
        self,
        step: int,
        tool_used: str,
        input_args: dict,
        result: str,
        success: bool,
        learning: str = "",
    ) -> None:
        """Record an iteration."""
        iteration = Iteration(
            iteration_num=len(self.iterations),
            step=step,
            tool_used=tool_used,
            input_args=input_args,
            result=result,
            success=success,
            learning=learning,
        )
        self.iterations.append(iteration)

    def get_iteration_summary(self) -> str:
        """Get concise summary of recent iterations for agent context."""
        if not self.iterations:
            return "No iterations yet."

        recent = self.iterations[-10:]  # Last 10 iterations
        summary = f"**Session Progress (Iterations {len(self.iterations)} total):**\n\n"

        for it in recent:
            status = "✓" if it.success else "✗"
            summary += f"{status} Step {it.step}: {it.tool_used}\n"
            if it.learning:
                summary += f"   → {it.learning}\n"

        summary += f"\n**Current Strategy:** {self.current_strategy}\n"
        summary += f"**Progress:** {self.progress_score:.1%}\n"

        if self.discoveries:
            summary += f"\n**Discoveries:**\n"
            for d in self.discoveries[-5:]:  # Last 5
                summary += f"- {d}\n"

        return summary

    def get_failure_analysis(self) -> str:
        """Analyze patterns in failures."""
        if not self.failures:
            return ""

        analysis = "**Recent Failures & Patterns:**\n"
        for failure in self.failures[-3:]:
            analysis += f"- {failure.get('attempt')}: {failure.get('reason')}\n"

        return analysis

    def save(self, path: Path) -> None:
        """Save to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @staticmethod
    def load(path: Path) -> "SessionMemory":
        """Load from disk."""
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return SessionMemory(**data)


class MemoryManager:
    """Manages agent working memory across iterations."""

    def __init__(self, goal: str, session_dir: Path = None):
        """Initialize memory manager."""
        # Use absolute path: ~/.claude/sessions
        if session_dir is None:
            session_dir = Path.home() / ".claude" / "sessions"
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.memory = SessionMemory(
            session_id=session_id,
            goal=goal,
            start_time=datetime.now().isoformat(),
        )
        self.memory_file = self.session_dir / f"session_{session_id}.json"

    def record_attempt(
        self,
        step: int,
        tool: str,
        args: dict,
        result: str,
        success: bool,
        learning: str = "",
    ) -> None:
        """Record tool attempt and learning."""
        self.memory.add_iteration(step, tool, args, result, success, learning)
        self.memory.save(self.memory_file)

    def record_discovery(self, discovery: str) -> None:
        """Record important discovery."""
        if discovery not in self.memory.discoveries:
            self.memory.discoveries.append(discovery)
            self.memory.save(self.memory_file)

    def record_strategy(self, strategy: str, current: bool = False) -> None:
        """Record strategy attempted."""
        if strategy not in self.memory.strategies_tried:
            self.memory.strategies_tried.append(strategy)
        if current:
            self.memory.current_strategy = strategy
        self.memory.save(self.memory_file)

    def record_failure(self, attempt: str, reason: str) -> None:
        """Record failed attempt."""
        self.memory.failures.append(
            {
                "attempt": attempt,
                "reason": reason,
                "iteration": len(self.memory.iterations),
            }
        )
        self.memory.save(self.memory_file)

    def record_success(self, approach: str, result: str) -> None:
        """Record successful approach."""
        self.memory.successes.append(
            {
                "approach": approach,
                "result": result,
                "iteration": len(self.memory.iterations),
            }
        )
        self.memory.save(self.memory_file)

    def get_context_for_agent(self) -> str:
        """Get memory summary to include in agent context."""
        return self.memory.get_iteration_summary()

    def update_progress(self, score: float) -> None:
        """Update progress score (0-1)."""
        self.memory.progress_score = score
        self.memory.save(self.memory_file)

    def get_next_strategy(self) -> str:
        """Recommend next strategy based on failures."""
        if not self.memory.failures:
            return "Continue with current approach"

        last_failure = self.memory.failures[-1]
        reason = last_failure.get("reason", "").lower()

        # Strategy suggestions based on failure type
        if "api" in reason or "endpoint" in reason:
            return "Try different API/library. Search for alternatives."
        elif "timeout" in reason:
            return "Increase timeout or optimize code for speed."
        elif "import" in reason or "module" in reason:
            return "Install missing library or use alternative."
        elif "syntax" in reason or "error" in reason:
            return "Review and fix code logic, try simpler approach."
        else:
            return "Reflect on approach, try different strategy."
