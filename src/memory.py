"""Agent session memory - tracks iterations, discoveries, and failures."""

import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class Iteration:
    """Single tool execution record."""
    step: int
    tool_used: str
    input_args: dict
    result: str
    success: bool
    learning: str = ""


@dataclass
class SessionMemory:
    """Complete session state."""
    session_id: str
    goal: str
    start_time: str
    iterations: list[Iteration] = field(default_factory=list)
    discoveries: list[str] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    successes: list[dict] = field(default_factory=list)

    def add_iteration(self, step: int, tool_used: str, input_args: dict,
                      result: str, success: bool, learning: str = "") -> None:
        self.iterations.append(Iteration(
            step=step, tool_used=tool_used, input_args=input_args,
            result=result, success=success, learning=learning,
        ))
        # Auto-compress: keep only last 30 discoveries
        if len(self.discoveries) > 30:
            self.discoveries = self.discoveries[-20:]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @staticmethod
    def load(path: Path) -> "SessionMemory":
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        # Reconstruct Iteration objects
        data["iterations"] = [Iteration(**it) for it in data.get("iterations", [])]
        return SessionMemory(**data)


    @staticmethod
    def retrieve_relevant(goal: str, session_dir: Path = None, n: int = 3) -> list[dict]:
        """Find past sessions with similar goals and extract their learnings.

        Returns list of dicts with: goal, successes, failures, key_learning
        """
        if session_dir is None:
            session_dir = Path.home() / ".claude" / "sessions"
        if not session_dir.exists():
            return []

        goal_words = set(goal.lower().split())
        scored = []

        for f in sorted(session_dir.glob("session_*.json"), reverse=True)[:50]:
            try:
                data = json.loads(f.read_text())
                past_goal = data.get("goal", "")
                past_words = set(past_goal.lower().split())
                # Jaccard similarity
                overlap = len(goal_words & past_words)
                union = len(goal_words | past_words)
                sim = overlap / max(1, union)
                if sim > 0.15:
                    scored.append((sim, {
                        "goal": past_goal,
                        "successes": [s.get("approach", "")[:80] for s in data.get("successes", [])[:3]],
                        "failures": [f.get("reason", "")[:80] for f in data.get("failures", [])[:3]],
                        "tools_used": len(data.get("iterations", [])),
                    }))
            except Exception:
                continue

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:n]]


class MemoryManager:
    """Manages session memory with auto-save."""

    def __init__(self, goal: str, session_dir: Path = None):
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

    def record_attempt(self, step: int, tool: str, args: dict,
                       result: str, success: bool, learning: str = "") -> None:
        self.memory.add_iteration(step, tool, args, result, success, learning)
        self.memory.save(self.memory_file)

    def record_discovery(self, discovery: str) -> None:
        if discovery not in self.memory.discoveries:
            self.memory.discoveries.append(discovery)
            self.memory.save(self.memory_file)

    def record_failure(self, attempt: str, reason: str) -> None:
        self.memory.failures.append({
            "attempt": attempt, "reason": reason,
            "iteration": len(self.memory.iterations),
        })
        self.memory.save(self.memory_file)

    def record_success(self, approach: str, result: str) -> None:
        self.memory.successes.append({
            "approach": approach, "result": result,
            "iteration": len(self.memory.iterations),
        })
        self.memory.save(self.memory_file)
