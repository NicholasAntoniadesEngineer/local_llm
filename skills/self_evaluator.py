from code_validator import CodeValidator
from result_evaluator import ResultEvaluator
from metrics import AgentMetrics

class SelfEvaluator:
    def __init__(self):
        self.code_validator = CodeValidator()
        self.result_evaluator = ResultEvaluator()
        self.metrics = AgentMetrics()

    def evaluate_file(self, file_path: str) -> dict:
        # Validate code syntax and imports
        code_validation = self.code_validator.validate_all(file_path)

        # Evaluate result quality
        result_quality = self.result_evaluator.score_code_output(file_path)

        # Get performance metrics
        metrics = self.metrics.report()

        # Calculate overall score
        syntax_score = 1.0 if code_validation['syntax_ok'] else 0.0
        imports_score = 1.0 if code_validation['imports_ok'] else 0.0
        tests_score = 1.0 if code_validation['tests_ok'] else 0.0
        result_score = result_quality['score']
        
        overall_score = (syntax_score + imports_score + tests_score + result_score) / 4
        
        recommendation = ''
        if overall_score >= 0.8:
            recommendation = 'keep'
        elif overall_score >= 0.5:
            recommendation = 'fix'
        else:
            recommendation = 'discard'
        
        return {
            'syntax_ok': code_validation['syntax_ok'],
            'imports_ok': code_validation['imports_ok'],
            'tests_ok': code_validation['tests_ok'],
            'result_quality': result_quality,
            'overall_score': overall_score,
            'recommendation': recommendation,
            'metrics': metrics
        }

if __name__ == "__main__":
    # Test cases
    evaluator = SelfEvaluator()
    
    # Test 1: Valid file
    with open('test_valid.py', 'w') as f:
        f.write("""
def test_add():
    assert 1 + 1 == 2
""")
    
    result1 = evaluator.evaluate_file('test_valid.py')
    assert result1['recommendation'] == 'keep', f"Test 1 Failed: {result1['recommendation']}",
    
    # Test 2: File with syntax error
    with open('test_error.py', 'w') as f:
        f.write("""
def test_add(
    assert 1 + 1 == 2
""")
    
    result2 = evaluator.evaluate_file('test_error.py')
    assert result2['recommendation'] == 'discard', f"Test 2 Failed: {result2['recommendation']}",
    
    # Test 3: File with failing test
    with open('test_failing.py', 'w') as f:
        f.write("""
def test_add():
    assert 1 + 1 == 3
""")
    
    result3 = evaluator.evaluate to 'fix', f"Test 3 Failed: {result3['recommendation']}",
    
    print('ALL TESTS PASSED')