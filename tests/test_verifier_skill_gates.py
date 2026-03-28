"""Tests for skill-module structural gates (assert counting, depth)."""

import ast
import re
import tempfile
import unittest
from pathlib import Path

from src.runtime.verifier import (
    MIN_SKILL_ASSERT_STATEMENTS,
    _assert_statement_count,
    _substantive_depth_gate,
    validate_generated_module,
)


class AssertCountingTests(unittest.TestCase):
    def test_assert_statement_count_uses_ast_not_substrings(self) -> None:
        source = '''
def f():
    x = "not an assert statement here"
    assert(1 + 1 == 2)
    assert True
'''
        tree = ast.parse(source)
        self.assertEqual(_assert_statement_count(tree), 2)


class SubstantiveDepthGateTests(unittest.TestCase):
    def test_allows_thin_helpers_when_bulk_exists(self) -> None:
        source = '''
class Demo:
    def __init__(self):
        self.x = 1

    def meaty(self):
        a = 1
        b = 2
        c = 3
        d = 4
        e = 5
        return a + b + c + d + e

    def tiny(self):
        v = self.x
        return v + 1

    def small(self):
        y = 2
        return y * 2
'''
        tree = ast.parse(source)
        func_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        ok, msg = _substantive_depth_gate(func_nodes, source)
        self.assertTrue(ok, msg=msg)


class ValidateGeneratedModuleTests(unittest.TestCase):
    def test_parenthesized_asserts_count_toward_minimum(self) -> None:
        body = "\n".join(
            [
                "def one():",
                "    a = 1",
                "    b = 2",
                "    c = 3",
                "    d = 4",
                "    e = 5",
                "    return a + b + c + d + e",
                "",
                "def two():",
                "    a = 1",
                "    b = 2",
                "    c = 3",
                "    d = 4",
                "    e = 5",
                "    return a * b * c * d * e",
                "",
                "def three():",
                "    a = 1",
                "    b = 2",
                "    c = 3",
                "    d = 4",
                "    e = 5",
                "    return max(a, b, c, d, e)",
                "",
                'if __name__ == "__main__":',
                "    assert one() == 15",
                "    assert two() == 120",
                "    assert three() == 5",
                "    assert one() > 0",
                "    assert two() > 0",
                "    print('ALL TESTS PASSED')",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample_skill.py"
            path.write_text(body, encoding="utf-8")
            ok, summary = validate_generated_module(str(path), skill_tree=None)
            self.assertTrue(ok, msg=summary)
            match = re.search(r"(\d+)asserts", summary.replace(" ", ""))
            self.assertIsNotNone(match)
            self.assertGreaterEqual(int(match.group(1)), MIN_SKILL_ASSERT_STATEMENTS)

    def test_passes_with_four_asserts_when_depth_ok(self) -> None:
        body = "\n".join(
            [
                "class Box:",
                "    def __init__(self):",
                "        self.n = 0",
                "",
                "    def bump(self):",
                "        self.n += 1",
                "        w = 0",
                "        x = 1",
                "        y = 2",
                "        z = 3",
                "        return self.n + w + x + y + z",
                "",
                "    def read(self):",
                "        a = 1",
                "        b = 0",
                "        return self.n + a + b",
                "",
                "    def reset(self):",
                "        k = 2",
                "        self.n = 0",
                "        return k",
                "",
                'if __name__ == "__main__":',
                "    b = Box()",
                "    assert b.bump() == 7",
                "    assert b.read() == 2",
                "    assert b.reset() == 2",
                "    assert b.read() == 1",
                "    print('ALL TESTS PASSED')",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "four_assert_skill.py"
            path.write_text(body, encoding="utf-8")
            ok, summary = validate_generated_module(str(path), skill_tree=None)
            self.assertTrue(ok, msg=summary)


if __name__ == "__main__":
    unittest.main()
