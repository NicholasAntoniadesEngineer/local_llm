import difflib
import math
from typing import List, Dict, Tuple, Optional
from search_cache import SearchCache

class ResultEvaluator:
    def score_search_result(self, query: str, title: str, snippet: str) -> float:
        # Calculate relevance score based on query match in title and snippet
        title_score = self._text_similarity(query, title)
        snippet_score = self._text_similarity(query, snippet)
        
        # Weight title more heavily since it's more likely to be relevant
        return (title_score * 0.6) + (snippet_score * 0.4)

    def score_code_output(self, code: str, output: str, has_error: bool) -> float:
        # If there was an error, score is low
        if has_error:
            return 0.0
        
        # If output is empty, it's not helpful
        if not output.strip():
            return 0.0
        
        # Calculate code quality score based on output length and relevance
        # Longer output is better, but we cap it at 1.0
        output_length_score = min(1.0, len(output) / 100)
        
        # Check if output contains any of the code's keywords
        code_keywords = self._extract_keywords(code)
        output_keywords = self._extract_keywords(output)
        
        # Calculate keyword match score
        keyword_score = len(set(code_keywords) & set(output_keywords)) / max(1, len(code_keywords))
        
        # Combine scores
        return (output_length_score * 0.5) + (keyword_score * 0.5)

    def is_duplicate(self, new_result: str, previous_results: List[str]) -> bool:
        # Check if new_result is exactly the same as any previous result
        for result in previous_results:
            if result == new_result:
                return True
        
        # Check for near duplicates using difflib
        for result in previous_results:
            if difflib.SequenceMatcher(None, new_result, result).ratio() > 0.9:
                return True
        
        return False

    def summarize_quality(self, scores_list: List[float]) -> Dict:
        if not scores_list:
            return {
                'avg': 0.0,
                'min': 0.0,
                'max': 0.0,
                'assessment': 'No results to evaluate'
            }
        
        avg = sum(scores_list) / len(scores_list)
        min_score = min(scores_list)
        max_score = max(scores_list)
        
        # Determine assessment based on average score
        if avg >= 0.8:
            assessment = 'Excellent quality'
        elif avg >= 0.5:
            assessment = 'Good quality'
        elif avg >= 0.2:
            assessment = 'Fair quality'
        else:
            assessment = 'Poor quality'
        
        return {
            'avg': round(avg, 2),
            'min': round(min_score, 2),
            'max': round(max_score, 2),
            'assessment': assessment
        }

    def _text_similarity(self, text1: str, text2: str) -> float:
        # Calculate similarity between two texts using difflib
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _extract_keywords(self, text: str) -> List[str]:
        # Extract keywords from text (simple word splitting for demonstration)
        return text.lower().split() if text else []

if __name__ == "__main__":
    evaluator = ResultEvaluator()
    
    # Test search result scoring
    print("Testing search result scoring:")
    relevant_query = "Python programming"
    relevant_title = "Python programming tutorial"
    relevant_snippet = "This tutorial covers the basics of Python programming..."
    junk_query = "Random junk"
    junk_title = "How to tie a shoe"
    junk_snippet = "This article explains how to tie your shoes..."
    
    print(f"Relevant search score: {evaluator.score_search_result(relevant_query, relevant_title, relevant_snippet):.2f}")
    print(f"Junk search score: {evaluator.score_search_result(junk_query, junk_title, junk_snippet):.2f}")
    
    # Test code output scoring
    print("\nTesting code output scoring:")
    working_code = "print(\"Hello, world!\")"
    working_output = "Hello, world!"
    error_code = "print(\"Hello, world!\""
    error_output = "  File \"<stdin>\", line 1\n    print(\"Hello, world!\"\n                   ^\nSyntaxError: unexpected EOF while parsing"
    
    print(f"Working code score: {evaluator.score_code_output(working_code, working_output, False):.2f}")
    print(f"Error code score: {eval>