import os
import ast
import importlib.util
import subprocess
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
                    if module and importlib.util.find_spec(module) is None:
                        missing_modules.append(module)
                except (ValueError, ModuleNotFoundError):
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

            env = os.environ.copy()
            skills_dir = os.path.dirname(os.path.abspath(file_path))
            env["PYTHONPATH"] = skills_dir + ":" + env.get("PYTHONPATH", "")

            direct_result = subprocess.run(
                ["python3", file_path],
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
            )

            if (
                direct_result.returncode == 0
                and "ALL TESTS PASSED" in direct_result.stdout
                and "Traceback" not in direct_result.stderr
            ):
                return True, "All tests passed"

            unittest_result = subprocess.run(
                ["python3", "-m", "unittest", file_path],
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
            )
            if unittest_result.returncode == 0:
                return True, "All tests passed"

            failure_output = (
                direct_result.stdout
                + direct_result.stderr
                + unittest_result.stdout
                + unittest_result.stderr
            )
            return False, f"Test failures:\n{failure_output[:300]}"
        except Exception as e:
            return False, f"Error running tests: {str(e)}"

if __name__ == "__main__":
    import tempfile

    # Test 1: Valid code gets high score
    valid_code = "def test_add():\n    assert 1 + 1 == 2\nprint('ALL TESTS PASSED')\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(valid_code)
        valid_path = f.name
    result = SelfEvaluator.evaluate_file(valid_path)
    assert result["score"] > 0.0, f"Valid code should score > 0, got {result['score']}"
    assert "status" in result, "Result must have status key"
    os.unlink(valid_path)

    # Test 2: Syntax error gets score 0
    bad_code = "def test_add(\n    assert 1 + 1 == 2"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(bad_code)
        bad_path = f.name
    result2 = SelfEvaluator.evaluate_file(bad_path)
    assert result2["score"] == 0.0, f"Syntax error should score 0, got {result2['score']}"
    assert result2["status"] == "error", "Syntax error should have error status"
    os.unlink(bad_path)

    # Test 3: Missing file returns error
    result3 = SelfEvaluator.evaluate_file("/nonexistent/path.py")
    assert result3["score"] == 0.0, "Missing file should score 0"
    assert result3["status"] == "error", "Missing file should have error status"

    # Test 4: _check_syntax works standalone
    ok, msg = SelfEvaluator._check_syntax("x = 1")
    assert ok, f"Valid syntax should pass: {msg}"
    ok2, _ = SelfEvaluator._check_syntax("def (broken")
    assert not ok2, "Broken syntax should fail"

    # Test 5: _check_imports returns list
    imports = SelfEvaluator._check_imports("import os\nx = 1")
    assert isinstance(imports, list), "check_imports must return a list"

    # Test 6: Result has recommendation
    assert "recommendation" in result, "Result must have recommendation"
    assert isinstance(result["recommendation"], str), "Recommendation must be string"

    print("ALL TESTS PASSED")