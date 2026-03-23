from typing import Optional, Union

class ConfidenceScorer:
    """A class to score confidence levels based on knowledge, capability, and progress."

    def score_knowledge(self, discoveries_count: int, relevant_keywords: Optional[list] = None) -> float:
        """Score based on number of discoveries and relevant keywords.

        Args:
            discoveries_count (int): Number of discoveries made.
            relevant_keywords (list, optional): List of relevant keywords. Defaults to empty list.

        Returns:
            float: Confidence score between 0.0 and 1.0.
        """
        if discoveries_count < 0:
            raise ValueError("Discoveries count cannot be negative.")
        if relevant_keywords is None:
            relevant_keywords = []
        if not isinstance(relevant_keywords, list):
            raise TypeError("Relevant keywords must be a list.")
        
        keyword_score = min(1.0, len(relevant_keywords) / 10)
        return (discoveries_count / 10) * keyword_score

    def score_capability(self, tool_success_rate: float, errors_count: int) -> float:
        """Score based on tool success rate and number of errors.

        Args:
            tool_success_rate (float): Success rate of the tool (0.0 to 100.0).
            errors_count (int): Number of errors encountered.

        Returns:
            float: Confidence score between 0.0 and 1.0.
        """
        if tool_success_rate < 0 or tool_success_rate > 100:
            raise ValueError("Tool success rate must be between 0.0 and 100.0.")
        if errors_count < 0:
            raise ValueError("Error count cannot be negative.")
        
        error_score = max(0.0, 1.0 - (errors_count / 10))
        return (tool_success_rate / 100) * error_score

    def score_progress(self, steps_taken: int, max_steps: int, files_written: int) -> float:
        """Score based on progress through steps and files written.

        Args:
            steps_taken (int): Number of steps taken.
            max_steps (int): Total number of steps.
            files_written (int): Number of files written.

        Returns:
            float: Confidence score between 0.0 and 1.0.
        """
        if steps_taken < 0 or max_steps < 0 or files_written < 0:
            raise ValueError("Steps taken, max steps, and files written cannot be negative.")
        if max_steps == 0:
            raise ValueError("Max steps cannot be zero.")
        
        step_score = steps_taken / max_steps
        file_score = min(1.0, files_written / 5)
        return (step_score + file_score) / 2

    def should_act(self, knowledge: float, capability: float, progress: float) -> str:
        """Decide what action to take based on the scores.

        Args:
            knowledge (float): Confidence score for knowledge.
            capability (float): Confidence score for capability.
            progress (float): Confidence score for progress.

        Returns:
            str: Action to take ('research', 'abort', 'save', or 'code').
        """
        if not (0.0 <= knowledge <= 1.0 and 0.0 <= capability <= 1.0 and 0.0 <= progress <= 1.0):
            raise ValueError("Confidence scores must be between 0.0 and 1.0.")
        
        if knowledge < 0.3:
            return 'research'
        elif capability < 0.5:
            return 'abort'
        elif progress > 0.7:
            return 'save'
        else:
            return 'code'

    def overall(self, knowledge: float, capability: float, progress: float) -> float:
        """Combined score of knowledge, capability, and progress.

        Args:
            knowledge (float): Confidence score for knowledge.
            capability (float): Confidence score for capability.
            progress (float): Confidence score for progress.

        Returns:
            float: Combined confidence score between 0.0 and 1.0.
        """
        if not (0.0 <= knowledge <= 1.0 and 0.0 <= capability <= 1.0 and 0.0 <= progress <= 1.0):
            raise ValueError("Confidence scores must be between 0.0 and 1.0.")
        
        return (knowledge + capability + progress) / 3

if __name__ == "__main__":
    scorer = ConfidenceScorer()

    # Test 1: Low knowledge should return 'research'
    assert scorer.should_act(0.2, 0.8, 0.5) == 'research', "Test 1 Failed"

    # Test 2: High everything should return 'save'
    assert scorer.should_act(0.9, 0.9, 0.9) == 'save', "Test 2 Failed"

    # Test 3: Many errors should return 'abort'
    assert scorer.should_act(0.6, 0.2, 0.5) == 'abort', "Test 3 Failed"

    # Test 4: Balanced should return 'code'
    assert scorer.should_act(0.5, 0.5, 0.5) == 'code', "Test 4 Failed"

    # Test 5: Edge case - knowledge exactly 0.3
    assert scorer.should_act(0.3, 0.8, 0.5) == 'code', "Test 5 Failed"

    # Test 6: Edge case - capability exactly 0.5
    assert scorer.should_act(0.5, 0.5, 0.5) == 'code', "Test 6 Failed"

    # Test 7: Edge case - progress exactly 0.7
    assert scorer.should_act(0.5, 0.5, 0.7) == 'save', "Test 7 Failed"

    # Test 8: Invalid input - negative discoveries_count
    try:
        scorer.score_knowledge(-1, [])
    except ValueError:
        pass
    else:
        assert False, "Test 8 Failed"

    # Test 9: Invalid input - non-list relevant_keywords
    try:
        scorer.score_knowledge(5, 'not a list')
    except TypeError:
        pass
    else:
        assert False, "Test 9 Failed"

    # Test 10: Invalid input - tool success rate out of range
    try:
        scorer.score_capability(101, 0)
    except ValueError:
        pass
    else:
        assert False, "Test 10 Failed"

    print('ALL TESTS PASSED')