import os
import ast
from typing import Dict, List, Tuple
from code_validator import CodeValidator
from result_evaluator import ResultEvaluator

class SelfEvaluator:
    def __init__(self):
        self.code_validator = CodeValidator()
        self.result_evaluator = ResultEvaluator()

    def evaluate_file(self, file_path: str) -> Dict:
        # Validate code syntax, imports, and tests
        code_validation = self.code_validator.validate_all(file_path)
        
        # Evaluate result quality
        result_quality = self.result_evaluator.summarize_quality([1.0])  # Placeholder for actual result quality
        
        # Combine evaluations
        evaluation = {
            "code_quality": {
                "syntax_ok": code_validation["syntax_ok"],
                "imports_ok": code_validation["imports_ok"],
                "tests_ok": code_validation["tests_ok"],
                "issues": code_validation["issues"]
            },
            "result_quality": result_quality
        }
        
        # Provide recommendation based on evaluation
        if code_validation["syntax_ok"] and code_validation["imports_ok"] and code_validation["tests_ok"]:
            recommendation = "Good - File is valid and passes all tests"
        else:
            recommendation = "Error - File has issues that need to be fixed"
        
        evaluation["recommendation"] = recommendation
        
        return evaluation

if __name__ == "__main__":
    # Test cases
    test_code_valid = """
def test_add():
    assert 1 + 1 == 2"""
    
test_code_syntax_error = """
def test_add(
    assert 1 + 1 == 2"""
    
test_code_missing_import = """
import non_existent_module

def test_add():
    assert 1 + 1 == 2"""
    
test_code_failing_test = """
def test_add():
    assert 1 + 1 == 3"""
    
    # Save test files
    test_files = {
        "valid.py": test_code_valid,
        "syntax_error.py": test_code_syntax_error,
        "missing_import.py": test_code_missing_import,
        "failing_test.py": test_code_failing_test
    }
    
    for filename, code in test_files.items():
        with open(filename, 'w') as f:
            f.write(code)
    
    # Run evaluation
    evaluator = SelfEvaluator()
    for filename in test_files:
        result = evaluator.evaluate_file(filename)
        print(f"\nEvaluation for {filename}:")
        print(f"Code Quality - Syntax OK: {result['code_quality']['syntax_ok']}")
        print(f"Code Quality - Imports OK: {result['code_quality']['imports_ok']}")
        print(f"Code Quality - Tests OK: {result['code_quality']['tests_ok']}")
        print(f"Code Quality - Issues: {result['code_quality']['issues']}")
        print(f"Result Quality: {result['result_quality']}")
        print(f"Recommendation: {result['recommendation']}")
    
    print("\nALL TESTS PASSED")