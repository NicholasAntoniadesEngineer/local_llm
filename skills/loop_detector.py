from typing import Dict, List, Tuple, Any
from metrics import AgentMetrics
import difflib


class LoopDetector:
    """Detects repeated actions with similar results to identify potential loops in agent behavior."""

    def __init__(self, metrics: AgentMetrics = None):
        """Initialize loop detector with optional metrics tracking.

        Args:
            metrics: Optional AgentMetrics instance to track loop detection metrics
        """
        self.metrics = metrics or AgentMetrics()
        self.action_history: List[Tuple[str, str]] = []  # (action, result)
        self.similarity_threshold = 0.8  # Minimum similarity for considering actions as similar
        self.loop_count = 0
        self.last_escape_suggestion = None

    def record(self, action: str, result: str) -> None:
        """Record an action and its result for loop detection.

        Args:
            action: The action taken
            result: The result of the action
        """
        if not isinstance(action, str) or not action:
            raise ValueError("Action must be a non-empty string")
        if not isinstance(result, str) or not result:
            raise ValueError("Result must be a non-empty string")

        self.action_history.append((action, result))
        self.metrics.record_step("loop_detector", True, 0.0)

    def is_stuck(self, window_size: int = 5) -> bool:
        """Detect if the agent is stuck in a loop.

        Args:
            window_size: Number of recent actions to check for repetition

        Returns:
            True if the agent is likely stuck in a loop, False otherwise
        """
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError("Window size must be a positive integer")
        if len(self.action_history) < window_size:
            return False

        # Get the most recent actions
        recent_actions = self.action_history[-window_size:]
        
        # Check if all actions in the window are similar
        for i in range(len(recent_actions)):
            for j in range(i + 1, len(recent_actions)):
                if not self._are_similar(recent_actions[i], recent_actions[j]):
                    return False
        # Add a check for minimum similarity threshold
        if len(recent_actions) >= 3:
            return True
        return False
        
        self.loop_count += 1
        return True

    def suggest_escape(self, max_attempts: int = 3) -> str:
        """Suggest an escape strategy if the agent is stuck.

        Args:
            max_attempts: Maximum number of attempts to find an escape strategy

        Returns:
            Suggested escape strategy or 'No escape strategy found' if none can be determined
        """
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("Max attempts must be a positive integer")

        # Check if we have enough data to suggest an escape
        if len(self.action_history) < 2:
            return "Insufficient data to suggest an escape strategy"

        # Try to find a different action that hasn't been tried recently
        tried_actions = set(action for action, _ in self.action_history[-max_attempts:])
        
        # For simplicity, we'll just suggest a random action not in the tried set
        # In a real implementation, this would involve more sophisticated strategy learning
        all_actions = ["search", "analyze", "plan", "execute", "review"]
        for action in all_actions:
            if action not in tried_actions:
                self.last_escape_suggestion = action
                return action
        
        return "No escape strategy found"

    def similarity(self, action1: str, action2: str) -> float:
        """Calculate the similarity between two actions.

        Args:
            action1: First action
            action2: Second action

        Returns:
            Similarity score between 0 and 1
        """
        if not isinstance(action1, str) or not action1:
            raise ValueError("Action1 must be a non-empty string")
        if not isinstance(action2, str) or not action2:
            raise ValueError("Action2 must be a non-empty string")

        # Use difflib to calculate similarity
        return difflib.SequenceMatcher(None, action1, action2).ratio()

    def _are_similar(self, action1: Tuple[str, str], action2: Tuple[str, str]) -> bool:
        """Check if two actions are similar based on both action and result.

        Args:
            action1: Tuple of (action, result)
            action2: Tuple of (action, result)

        Returns:
            True if both action and result are similar, False otherwise
        """
        action_similarity = self.similarity(action1[0], action2[0])
        result_similarity = self.similarity(action1[1], action2[1])
        
        return (action_similarity >= self.similarity_threshold and 
                result_similarity >= self.similarity_threshold)

    
if __name__ == "__main__":
    # Test 1: Basic functionality
    detector = LoopDetector()
    detector.record("search", "results found")
    detector.record("search", "results found")
    detector.record("search", "results found")
    assert detector.is_stuck() == True, "Test 1 failed: is_stuck should return True for repeated actions"
    assert detector.suggest_escape() == "analyze", "Test 1 failed: suggest_escape should return 'analyze'"

    # Test 2: Different actions
    detector = LoopDetector()
    detector.record("search", "results found")
    detector.record("analyze", "insights generated")
    assert detector.is_stuck() == False, "Test 2 failed: is_stuck should return False for different actions"
    assert detector.suggest_escape() == "plan", "Test 2 failed: suggest_escape should return 'plan'"

    # Test 3: Edge case with empty input
    detector = LoopDetector()
    assert detector.is_stuck() == False, "Test 3 failed: is_stuck should return False for empty input"
    assert detector.suggest_escape() == "search", "Test 3 failed: suggest_escape should return 'search'"

    # Test 4: Boundary values
    detector = LoopDetector()
    for i in range(5):
        detector.record("search", "results found")
    assert detector.is_stuck() == True, "Test 4 failed: is_stuck should return True for 5 repeated actions"
    assert detector.suggest_escape() == "analyze", "Test 4 failed: suggest_escape should return 'analyze'"

    # Test 5: Mixed actions
    detector = LoopDetector()
    detector.record("search", "results found")
    detector.record("search", "results found")
    detector.record("analyze", "insights generated")
    detector.record("search", "results found")
    assert detector.is_stuck() == False, "Test 5 failed: is_stuck should return False for mixed actions"
    assert detector.suggest_escape() == "plan", "Test 5 failed: suggest_escape should return 'plan'"

    print("ALL TESTS PASSED")