from typing import Dict, Any
from metrics import AgentMetrics
from error_recovery import ErrorRecovery

class ConfidenceScorer:
    def score_knowledge(self, discoveries_count: int, relevant_keywords: list) -> float:
        """Score based on number of discoveries and relevant keywords"""
        if discoveries_count is None or relevant_keywords is None:
            return 0.0
        if discoveries_count == 0:
            return 0.0
        keyword_score = min(1.0, len(relevant_keywords) / 10)
        return (discoveries_count / 10) * keyword_score

    def score_capability(self, tool_success_rate: float, errors_count: int) -> float:
        """Score based on tool success rate and number of errors"""
        if tool_success_rate is None or errors_count is None:
            return 0.0
        if tool_success_rate == 0:
            return 0.0
        error_score = max(0.0, 1.0 - (errors_count / 10))
        return (tool_success_rate / 100) * error_score

    def score_progress(self, steps_taken: int, max_steps: int, files_written: int) -> float:
        """Score based on progress through steps and files written"""
        if steps_taken is None or max_steps is None or files_written is None:
            return 0.0
        if max_steps == 0:
            return 0.0
        step_score = steps_taken / max_steps
        file_score = min(1.0, files_written / 5)
        return (step_score + file_score) / 2

    def should_act(self, knowledge: float, capability: float, progress: float) -> str:
        """Decide what action to take based on the scores"""
        if knowledge is None or capability is None or progress is None:
            return 'abort'
        if knowledge < 0.3:
            return 'research'
        elif capability < 0.5:
            return 'abort'
        elif progress > 0.7:
            return 'save'
        else:
            return 'code'

    def overall(self, knowledge: float, capability: float, progress: float) -> float:
        """Combined score of knowledge, capability, and progress"""
        if knowledge is None or capability is None or progress is None:
            return 0.0
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
    
    # Test 5: None values should return 'abort'
    assert scorer.should_act(None, 0.8, 0.5) == 'abort', "Test 5 Failed"
    
    # Test 6: Zero discoveries should return 0.0
    assert scorer.score_knowledge(0, ['test']) == 0.0, "Test 6 Failed"
    
    # Test 7: Zero success rate should return 0.0
    assert scorer.score_capability(0, 5) == 0.0, "Test 7 Failed"
    
    # Test 8: Zero max steps should return 0.0
    assert scorer.score_progress(5, 0, 3) == 0.0, "Test 8 Failed"
    
    print('ALL TESTS PASSED')