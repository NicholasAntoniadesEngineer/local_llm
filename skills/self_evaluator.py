import os
import ast
from typing import Dict, List, Tuple

class SelfEvaluator:
    @staticmethod
    def evaluate_file(file_path: str) -> Dict[str, any]:
        """Evaluate the quality of a file's code."""
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "message": f"File not found: {file_path}",
                "score": 0.0,
                "recommendation": "Create the file first"
            }

        # Read file content
        try:
            with open(file_path, 'r') as f:
                code_str = f.read()
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error reading file: {str(e)}",
                "score": 0.0,
                "recommendation": "Fix file permissions"
            }

        # Check syntax
        syntax_ok, syntax_msg = SelfEvaluator._check_syntax(code_str)
        
        # Check imports
        missing_modules = SelfEvaluator._check_imports(code_str)
        imports_ok = len(missing_modules) == 0
        
        # Run tests
        tests_ok, tests_msg = SelfEvaluator._run_tests(file_path)
        
        # Calculate score
        score = 1.0
        if not syntax_ok:
            score = 0.0
        elif not imports_ok:
            score = 0.5
        elif not tests_ok:
            score = 0.75
        
        # Generate recommendation
        if syntax_ok and imports_ok and tests_ok:
            recommendation = "File is valid and passes all tests"
        elif not syntax_ok:
            recommendation = f"Fix syntax error: {syntax_msg}"
        elif not imports_ok:
            recommendation = f"Install missing modules: {', '.join(missing_modules)}"
        elif not tests_ok:
            recommendation = f"Fix test failures: {tests_msg}"
        else:
            recommendation = "Unknown issue"

        return {
            "status": "success" if (syntax_ok and imports_ok and tests_ok) else "error",
            "message": "File is valid and passes all tests" if (syntax_ok and imports_ok and tests_ok) else f"{syntax_msg if not syntax_ok else ''}{', '.join(missing_modules) if not imports_ok else ''}{tests_msg if not tests_ok else ''}",
            "score": score,
            "recommendation": recommendation
        }

    @staticmethod
    def _check_syntax(code_str: str) -> Tuple[bool, str]:
        try:
            ast.parse(code_str)
            return True, "No syntax errors"
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    @staticmethod
    def _check_imports(code_str: str) -> List[str]:
        try:
            tree = ast.parse(code_str)
            imports = [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
            
            # Check for from ... import ... syntax
            from_imports = []
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    from_imports.append(node.module)
            
            all_imports = imports + from_imports
            
            # Check if modules are importable
            missing_modules = []
            for module in all_imports:
                try:
                    importlib.util.find_spec(module)
                except ValueError:
                    missing_modules.append(module)
            return missing_modules
        except Exception as e:
            return [f"Error analyzing imports: {str(e)}"]

    @staticmethod
    def _run_tests(file_path: str) -> Tuple[bool, str]:
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}"
            
            # Run tests using subprocess
            result = subprocess.run([
                "python3",
                "-m",
                "unittest",
                file_path
            ],
            capture_output=True,
            text=True,
            timeout=10)
            
            if result.returncode == 0:
                return True, "All tests passed"
            else:
                return False, f"Test failures:\n{result.stderr}"
        except Exception as e:
            return False, f"Error running tests: {str(e)}"

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
    
    # Run validation
    for filename in test_files:
        result = SelfEvaluator.evaluate_file(filename)
        print(f"\nValidation for {filename}:")
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        print(f"Score: {result['score']}")
        print(f"Recommendation: {result['recommendation']}")
    
    print("\nALL TESTS PASSED")