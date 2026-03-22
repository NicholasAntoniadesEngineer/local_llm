import ast
import importlib.util
import sys
import subprocess
import os
from typing import Tuple, List, Dict

class CodeValidator:
    @staticmethod
    def check_syntax(code_str: str) -> Tuple[bool, str]:
        try:
            ast.parse(code_str)
            return True, "No syntax errors"
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    @staticmethod
    def check_imports(code_str: str) -> List[str]:
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
    def run_tests(file_path: str) -> Tuple[bool, str]:
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

    @staticmethod
    def validate_all(file_path: str) -> Dict:
        try:
            with open(file_path, 'r') as f:
                code_str = f.read()
        except Exception as e:
            return {
                "syntax_ok": False,
                "imports_ok": False,
                "tests_ok": False,
                "issues": [f"Error reading file: {str(e)}"]
            }
            
        syntax_ok, syntax_msg = CodeValidator.check_syntax(code_str)
        imports_ok = len(CodeValidator.check_imports(code_str)) == 0
        
        tests_ok, tests_msg = CodeValidator.run_tests(file_path)
        
        issues = []
        if not syntax_ok:
            issues.append(f"Syntax error: {syntax_msg}")
        if not imports_ok:
            missing_modules = CodeValidator.check_imports(code_str)
            issues.extend([f"Missing import: {m}" for m in missing_modules])
        if not tests_ok:
            issues.append(f"Test failure: {tests_msg}")
        
        return {
            "syntax_ok": syntax_ok,
            "imports_ok": imports_ok,
            "tests_ok": tests_ok,
            "issues": issues
        }

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
        result = CodeValidator.validate_all(filename)
        print(f"\nValidation for {filename}:")
        print(f"Syntax OK: {result['syntax_ok']}")
        print(f"Imports OK: {result['imports_ok']}")
        print(f"Tests OK: {result['tests_ok']}")
        print(f"Issues: {result['issues']}")
        
    print("\nALL TESTS PASSED")