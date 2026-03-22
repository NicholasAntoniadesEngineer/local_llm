import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class CompressedSessionMemory:
    session_id: str
    goal: str
    start_time: str
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    discoveries: List[str] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    successes: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    @classmethod
    def compress(cls, session_memory: 'SessionMemory', max_iterations: int = 10) -> 'CompressedSessionMemory':
        # Keep only the last max_iterations iterations
        recent_iterations = session_memory.iterations[-max_iterations:] if len(session_memory.iterations) > max_iterations else session_memory.iterations

        # Summarize older iterations
        older_iterations = session_memory.iterations[:-max_iterations] if len(session_memory.iterations) > max_iterations else []
        summary = "\n".join([f"Step {it.step}: {it.tool_used} - {it.result[:50]}..." for it in older_iterations])

        return cls(
            session_id=session_memory.session_id,
            goal=session_memory.goal,
            start_time=session_memory.start_time,
            iterations=recent_iterations,
            discoveries=session_memory.discoveries,
            failures=session_memory.failures,
            successes=session_memory.successes,
            summary=summary
        )

# Test block
def test_compress():
    # Create a fake SessionMemory with 50 iterations
    from memory import SessionMemory, Iteration
    fake_iterations = [Iteration(step=i, tool_used=f"tool_{i%5}", input_args={f"arg_{i%5}": i}, result=f"result_{i}", success=True) for i in range(50)]
    fake_session_memory = SessionMemory(
        session_id="test_session",
        goal="test_goal",
        start_time="2023-01-01T00:00:00",
        iterations=fake_iterations
    )

    # Compress the session memory
    compressed_memory = CompressedSessionMemory.compress(fake_session_memory)

    # Verify the result has <= 10 iterations
    assert len(compressed_memory.iterations) <= 10, f"Expected <=10 iterations, got {len(compressed_memory.iterations)}"
    assert compressed_memory.summary != "", "Summary should not be empty"

    print('ALL TESTS PASSED')

if __name__ == "__main__":
    test_compress()