from typing import Optional, Dict, List
from search_cache import SearchCache
import time
import difflib


class ResultEvaluator:
    """Evaluates the quality of tool results and detects duplicates."""

    def __init__(self):
        """Initialize the result evaluator with a search cache."""
        self.cache = SearchCache()

    def score_search(self, results: List[Dict[str, str]]) -> Dict[str, float]:
        """Score search results based on relevance and quality.

        Args:
            results: List of search results with 'title', 'url', and 'snippet' fields

        Returns:
            Dictionary of scores for each result
        """
        if not results:
            return {}

        scores = {}
        for i, result in enumerate(results):
            if not all(key in result for key in ['title', 'url', 'snippet']):
                raise ValueError(f"Invalid result format at index {i}")

            # Calculate relevance score based on title and snippet
            title_score = self._calculate_relevance(result['title'])
            snippet_score = self._calculate_relevance(result['snippet'])
            
            # Calculate quality score based on URL domain
            domain_score = self._calculate_domain_quality(result['url'])
            
            # Combine scores with weights
            total_score = (title_score * 0.4) + (snippet_score * 0.4) + (domain_score * 0.2)
            scores[result['url']] = total_score

        return scores

    def _calculate_relevance(self, text: str) -> float:
        """Calculate relevance score for a text snippet.

        Args:
            text: Text to analyze

        Returns:
            Relevance score between 0 and 1
        """
        if not text:
            return 0.0

        # Simple keyword matching for relevance
        keywords = ['python', 'api', 'tutorial', 'guide', 'example']
        match_count = sum(1 for keyword in keywords if keyword in text.lower())
        return min(match_count / len(keywords), 1.0)

    def _calculate_domain_quality(self, url: str) -> float:
        """Calculate quality score based on URL domain.

        Args:
            url: URL to analyze

        Returns:
            Quality score between 0 and 1
        """
        if not url:
            return 0.0

        # Simple domain quality assessment
        domain = url.split('//')[-1].split('/')[0]
        
        # Common high-quality domains
        high_quality_domains = ['python.org', 'github.com', 'stackoverflow.com', 'w3schools.com']
        
        # Common low-quality domains
        low_quality_domains = ['clickhere.com', 'freebies.net', 'badsite.org']
        
        if any(domain in high_quality_domains for domain in high_quality_domains):
            return 1.0
        elif any(domain in low_quality_domains for domain in low_quality_domains):
            return 0.3
        else:
            return 0.7

    def score_code(self, code: str) -> float:
        """Score the quality of code based on syntax and structure.

        Args:
            code: Python code to score

        Returns:
            Quality score between 0 and 1
        """
        if not code:
            return 0.0

        # Simple syntax check
        try:
            compile(code, '<string>', 'exec')
            return 0.8
        except SyntaxError:
            return 0.3

    def is_duplicate(self, text: str, threshold: float = 0.8) -> bool:
        """Check if text is a duplicate of previously seen content.

        Args:
            text: Text to check
            threshold: Similarity threshold for duplication

        Returns:
            True if text is a duplicate, False otherwise
        """
        if not text:
            return False

        # Check cache first
        cached = self.cache.get(text)
        if cached:
            return True

        # Check for duplicates in cache
        for key in self.cache.stats()['keys']:
            if difflib.SequenceMatcher(None, text, key).ratio() > threshold:
                self.cache.set(text, text)
                return True

        self.cache.set(text, text)
        return False

    def summarize(self, text: str, max_length: int = 200) -> str:
        """Create a summary of the text.

        Args:
            text: Text to summarize
            max_length: Maximum length of the summary

        Returns:
            Summary of the text
        """
        if not text:
            return ""

        # Simple summary by taking first max_length characters
        return text[:max_length] + ('...' if len(text) > max_length else '')


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    # Test cases
    evaluator = ResultEvaluator()

    # Test score_search
    results = [
        {'title': 'Python Tutorial', 'url': 'https://python.org', 'snippet': 'Learn Python programming language'},
        {'title': 'Advanced Python', 'url': 'https://advancedpython.com', 'snippet': 'Deep dive into Python concepts'},
        {'title': 'Python for Beginners', 'url': 'https://beginnerpython.com', 'snippet': 'Start learning Python with our beginner course'}
    ]
    scores = evaluator.score_search(results)
    assert all(0 <= score <= 1 for score in scores.values()), "Search score validation failed"

    # Test score_code
    valid_code = "print('Hello, world!')"
    assert evaluator.score_code(valid_code) > 0.5, "Code score validation failed"

    # Test is_duplicate
    assert not evaluator.is_duplicate("This is a test"), "Duplicate check failed"
    assert evaluator.is_duplicate("This is a test"), "Duplicate check failed"

    # Test summarize
    long_text = """This is a long text that needs to be summarized. """ * 10
    summary = evaluator.summarize(long_text)
    assert len(summary) <= 200, "Summarization validation failed"

    print("ALL TESTS PASSED")