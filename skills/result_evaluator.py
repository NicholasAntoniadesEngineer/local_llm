import difflib
import math
from typing import List, Dict, Tuple, Optional
from search_cache import SearchCache

class ResultEvaluator:
    def __init__(self):
        self.search_cache = SearchCache()

    def score_search_result(self, query: str, title: str, snippet: str) -> float:
        # Check if this result is in the search cache
        cached = self.search_cache.get(f"{query}_{title}")
        if cached:
            try:
                return float(cached)
            except (TypeError, ValueError):
                return 0.5

        # Calculate relevance based on query and title
        query_words = query.lower().split()
        title_words = title.lower().split()
        
        # Calculate title relevance
        title_relevance = len(set(query_words) & set(title_words)) / max(len(query_words), 1)
        
        # Calculate snippet relevance
        snippet_relevance = sum(1 for word in query_words if word in snippet.lower()) / len(query_words) if query_words else 0
        
        # Combine scores
        score = (title_relevance * 0.6) + (snippet_relevance * 0.4)
        
        # Cache the result
        self.search_cache.set(f"{query}_{title}", str(score))
        
        return score

    def _text_similarity(self, a: str, b: str) -> float:
        # Use difflib to calculate text similarity
        return difflib.SequenceMatcher(None, a, b).ratio()

    def score_code_output(self, code: str, output: str, has_error: bool) -> float:
        if has_error:
            return 0.0  # Code with errors is not useful
        
        # Check if output matches expected pattern
        # For simplicity, we'll assume code that produces output is good
        # In a real system, we would check against expected results
        return 1.0  # Assume code is good if it runs without errors

    def is_duplicate(self, new_result: str, previous_results: List[str]) -> bool:
        # Check if new_result is a duplicate of any previous result
        for prev in previous_results:
            if self._text_similarity(new_result, prev) > 0.9:
                return True
        return False

    def summarize_quality(self, scores_list: List[float]) -> Dict[str, float]:
        if not scores_list:
            return {'avg': 0.0, 'min': 0.0, 'max': 0.0, 'assessment': 'No results to evaluate'}
        
        avg = sum(scores_list) / len(scores_list)
        min_score = min(scores_list)
        max_score = max(scores_list)
        
        if avg > 0.8:
            assessment = 'High quality'
        elif avg > 0.5:
            assessment = 'Moderate quality'
        else:
            assessment = 'Low quality'
        
        return {'avg': round(avg, 2), 'min': round(min_score, 2), 'max': round(max_score, 2), 'assessment': assessment}

if __name__ == "__main__":
    evaluator = ResultEvaluator()

    # Test 1: Relevant result scores higher than irrelevant
    score1 = evaluator.score_search_result("Python programming", "Python Basics Tutorial", "Learn the fundamentals of Python programming...")
    score2 = evaluator.score_search_result("Python programming", "How to cook spaghetti", "This is a recipe for spaghetti...")
    assert score1 > score2, f"Relevant ({score1}) should beat irrelevant ({score2})"
    assert 0 <= score1 <= 1.0, f"Score must be 0-1, got {score1}"

    # Test 2: Code with error scores 0
    assert evaluator.score_code_output("x", "error", True) == 0.0

    # Test 3: Working code scores > 0
    assert evaluator.score_code_output("print(1)", "1", False) > 0

    # Test 4: Duplicate detection
    assert evaluator.is_duplicate("test", ["test", "other"]) is True
    assert evaluator.is_duplicate("unique", ["different", "also different"]) is False

    # Test 5: Quality summary
    summary = evaluator.summarize_quality([0.9, 0.8, 0.7])
    assert summary["avg"] > 0.7
    assert summary["min"] == 0.7
    assert summary["max"] == 0.9
    assert summary["assessment"] in ("High quality", "Moderate quality")

    # Test 6: Empty quality summary
    empty_summary = evaluator.summarize_quality([])
    assert empty_summary["avg"] == 0.0

    print("ALL TESTS PASSED")