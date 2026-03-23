import difflib
from typing import List, Tuple, Optional, Dict, Any
import json
from metrics import AgentMetrics

class LoopDetector:
    def __init__(self):
        self.actions = []  # Stores tuples of (tool, args_str, result_str)
        self.max_actions = 10  # Sliding window of last 10 actions
        self.metrics = AgentMetrics()

    def record(self, tool: str, args_str: str, result_str: str):
        """Add a new action to the sliding window"""
        if not isinstance(tool, str) or not isinstance(args_str, str) or not isinstance(result_str, str):
            raise ValueError("All parameters must be strings")
        
        self.actions.append((tool, args_str, result_str))
        # Keep only the last max_actions items
        self.actions = self.actions[-self.max_actions:]
        
        # Record step in metrics
        self.metrics.record_step(tool, True, 0.0, 0)

    def similarity(self, a: str, b: str) -> float:
        """Calculate similarity between two strings using difflib"""
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValueError("Inputs must be strings")
        
        return difflib.SequenceMatcher(None, a, b).ratio()

    def is_stuck(self) -> bool:
        """Check if the agent is stuck in a loop"""
        if len(self.actions) < 3:
            return False

        # Check if last 3 actions are from the same tool
        tool, _, _ = self.actions[-1]
        if any(a[0] != tool for a in self.actions[-3:-1]):
            return False

        # Check if results are >80% similar
        results = [a[2] for a in self.actions[-3:]]
        
        # Compare all pairs of results
        for i in range(3):
            for j in range(i+1, 3):
                if self.similarity(results[i], results[j]) <= 0.8:
                    return False
        
        return True

    def suggest_escape(self, current_tool: str) -> Optional[str]:
        """Suggest a different tool/approach when stuck"""
        if not isinstance(current_tool, str):
            raise ValueError("Current tool must be a string")
        
        if len(self.actions) < 3:
            return None

        # Check if last 3 actions are from the same tool
        tool, _, _ = self.actions[-1]
        if any(a[0] != tool for a in self.actions[-3:-1]):
            return None

        # Find a different tool from the ones we've used recently
        used_tools = set(a[0] for a in self.actions[-3:])
        
        # Suggest a tool that's different from the current one
        for tool in used_tools:
            if tool != current_tool:
                return f"Consider trying a different approach using tool: {tool}"
        
        # If all used tools are the same as current_tool, suggest a general escape
        return "Try a different approach or tool to break the loop"

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about the agent's actions"""
        return self.metrics.summary()

    def reset(self) -> None:
        """Reset the loop detector"""
        self.actions = []
        self.metrics = AgentMetrics()

if __name__ == "__main__":
    # Test 1: 3 identical calls should be stuck
    detector = LoopDetector()
    detector.record("toolA", "args1", "result1")
    detector.record("toolA", "args1", "result1")
    detector.record("toolA", "args1", "result1")
    assert detector.is_stuck(), "Test 1 failed: 3 identical calls should be stuck"

    # Test 2: 3 different calls should not be stuck
    detector = LoopDetector()
    detector.record("toolA", "args1", "result1")
    detector.record("toolB", "args2", "result2")
    detector.record("toolC", "args3", "result3")
    assert not detector.is_stuck(), "Test 2 failed: 3 different calls should not be stuck"

    # Test 3: Similar but not identical results
    detector = LoopDetector()
    detector.record("toolA", "args1", "result1")
    detector.record("toolA", "args1", "result1 similar")
    detector.record("toolA", "args1", "result1 slightly different")
    assert not detector.is_stuck(), "Test 3 failed: Similar but not identical results should not be considered stuck"

    # Test 4: Empty actions
    detector = LoopDetector()
    assert not detector.is_stuck(), "Test 4 failed: Empty actions should not be stuck"

    # Test 5: Single action
    detector = LoopDetector()
    detector.record("toolA", "args1", "result1")
    assert not detector.is_stuck(), "Test 5 failed: Single action should not be stuck"

    # Test 6: Two actions
    detector = LoopDetector()
    detector.record("toolA", "args1", "result1")
    detector.record("toolA", "args1", "result1")
    assert not detector.is_stuck(), "Test 6 failed: Two actions should not be stuck"

    # Test 7: Three actions with different tools
    detector = LoopDetector()
    detector.record("toolA", "args1", "result1")
    detector.record("toolB", "args2", "result2")
    detector.record("toolC", "args3", "result3")
    assert not detector.is_stuck(), "Test 7 failed: Three actions with different tools should not be stuck"

    # Test 8: Three actions with same tool but different results
    detector = LoopDetector()
    detector.record("toolA", "args1", "result1")
    detector.record("toolA", "args1", "result2")
    detector.record("toolA", "args1", "result3")
    assert not detector.is_stuck(), "Test 8 failed: Three actions with same tool but different results should not be stuck"

    # Test 9: Three actions with same tool and similar results
    detector = LoopDetector()
    detector.record("toolA", "args1", "result1")
    detector.record("toolA", "args1", "result1 similar")
    detector.record("toolA", "args1", "result1 slightly different")
    assert not detector.is_stuck(), "Test 9 failed: Three actions with same tool and similar results should not be stuck"

    # Test 10: Three actions with same tool and identical results
    detector = LoopDetector()
    detector.record("toolA", "args1", "result1")
    detector.record("toolA", "args1", "result1")
    detector.record("toolA", "args1", "result1")
    assert detector.is_stuck(), "Test 10 failed: Three actions with same tool and identical results should be stuck"

    print("ALL TESTS PASSED")