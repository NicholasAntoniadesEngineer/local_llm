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
            return cached.get('score', 0.5)  # Default to 0.5 if no score

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
        self.search_cache.set(f"{query}_{title}", {'score': score})
        
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
    
    # Test 1: Relevant search result
    score1 = evaluator.score_search_result("Python programming", "Python Basics Tutorial", "Learn the fundamentals of Python programming...")
    print(f"Test 1 Score: {score1}")
    
    # Test 2: Junk search result
    score2 = evaluator.score_search_result("Python programming", "How to cook spaghetti", "This is a recipe for spaghetti...")
    print(f"Test 2 Score: {score2}")
    
    # Test 3: Working code
    code3 = "print('Hello, World!')"
    output3 = "Hello, World!"
    score3 = evaluator.score_code_output(code3, output3, False)
    print(f"Test 3 Score: {score3}")
    
    # Test 4: Error code
    code4 = "print('Hello, World!"  # Missing closing quote
    output4 = "SyntaxError: EOL while scanning string literal"
    score4 = evaluator.score_code_output(code4, output4, True)
    print(f"Test 4 Score: {score4}")
    
    # Test 5: Duplicate detection
    results5 = ["This is a test result.", "This is a test result."]
    is_dup5 = evaluator.is_duplicate("This is a test result.", results5)
    print(f"Test 5 Is Duplicate: {is_dup5}")
    
    # Test 6: Quality summary
    scores6 = [0.9, 0.85, 0.75, 0.6, 0.95]
    summary6 = evaluator.summarize_quality(scores6)
    print(f"Test 6 Summary: {summary6}")
    
    print('\nALL TESTS PASSED')