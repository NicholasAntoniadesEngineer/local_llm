"""Coding challenges that the agent solves using its skills.

Each challenge has:
- A problem description
- Expected output when solved correctly
- A scoring function that measures how well the solution works

The improvement loop picks a challenge, uses skills to solve it,
scores the result, and identifies which skills need improvement.
"""

from __future__ import annotations

import ast
import subprocess
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Challenge:
    """A coding challenge the agent tries to solve."""
    id: str
    name: str
    description: str
    expected_output: str
    difficulty: int  # 1-5
    relevant_skills: list[str]  # skills that should help solve this


@dataclass
class ChallengeResult:
    """Result of attempting a challenge."""
    challenge_id: str
    solved: bool
    score: float  # 0.0 - 1.0
    output: str
    error: str
    time_s: float
    code: str


# ── Challenge definitions ────────────────────────────────────────────────

CHALLENGES = [
    Challenge(
        id="fizzbuzz",
        name="FizzBuzz",
        description="Write a function fizzbuzz(n) that returns a list of strings for 1..n. "
                     "For multiples of 3 return 'Fizz', multiples of 5 return 'Buzz', "
                     "both return 'FizzBuzz', otherwise the number as string.",
        expected_output="1,2,Fizz,4,Buzz,Fizz,7,8,Fizz,Buzz,11,Fizz,13,14,FizzBuzz",
        difficulty=1,
        relevant_skills=["code_validator"],
    ),
    Challenge(
        id="reverse_words",
        name="Reverse Words",
        description="Write a function reverse_words(s) that reverses the order of words in a string. "
                     "Multiple spaces should become single space. Leading/trailing spaces removed.",
        expected_output="world hello",
        difficulty=1,
        relevant_skills=["code_validator"],
    ),
    Challenge(
        id="flatten_nested",
        name="Flatten Nested List",
        description="Write a function flatten(lst) that takes an arbitrarily nested list and "
                     "returns a flat list. Example: flatten([1, [2, [3, 4], 5]]) -> [1, 2, 3, 4, 5]",
        expected_output="[1, 2, 3, 4, 5]",
        difficulty=2,
        relevant_skills=["code_validator", "error_recovery"],
    ),
    Challenge(
        id="word_frequency",
        name="Word Frequency Counter",
        description="Write a function word_freq(text) that returns a dict mapping each word "
                     "(lowercased) to its count. Punctuation should be stripped. "
                     "Example: word_freq('Hello hello world!') -> {'hello': 2, 'world': 1}",
        expected_output="{'hello': 2, 'world': 1}",
        difficulty=2,
        relevant_skills=["code_validator", "search_cache"],
    ),
    Challenge(
        id="lru_cache",
        name="LRU Cache",
        description="Implement a class LRUCache with __init__(capacity), get(key), and put(key, value). "
                     "get returns -1 if key not found. When capacity is exceeded, evict least recently used. "
                     "Both get and put count as 'use'.",
        expected_output="PASS",
        difficulty=3,
        relevant_skills=["search_cache", "memory_compressor"],
    ),
    Challenge(
        id="json_transformer",
        name="JSON Transformer",
        description="Write a function transform_json(data, rules) where data is a dict and rules is a list "
                     "of dicts with 'field', 'action' ('uppercase', 'double', 'remove'). Apply each rule to "
                     "the data dict. Return the transformed dict.",
        expected_output="{'name': 'ALICE', 'age': 60}",
        difficulty=2,
        relevant_skills=["task_planner", "error_recovery"],
    ),
    Challenge(
        id="retry_decorator",
        name="Retry Decorator",
        description="Write a decorator retry(max_attempts=3, backoff=1.0) that retries a function on "
                     "exception up to max_attempts times with exponential backoff (1s, 2s, 4s...). "
                     "Raise the last exception if all attempts fail.",
        expected_output="PASS",
        difficulty=3,
        relevant_skills=["error_recovery", "loop_detector"],
    ),
    Challenge(
        id="dependency_resolver",
        name="Dependency Resolver",
        description="Write a function resolve_deps(deps) where deps is a dict mapping task names to "
                     "lists of dependencies. Return a valid execution order (topological sort). "
                     "Raise ValueError on circular dependencies.",
        expected_output="[valid topological order]",
        difficulty=3,
        relevant_skills=["task_planner", "code_validator"],
    ),
    Challenge(
        id="text_differ",
        name="Text Differ",
        description="Write a function diff_texts(old, new) that returns a list of change tuples: "
                     "('add', line), ('remove', line), ('keep', line) showing the minimal diff "
                     "between two multiline strings.",
        expected_output="PASS",
        difficulty=3,
        relevant_skills=["result_evaluator", "code_validator"],
    ),
    Challenge(
        id="mini_calculator",
        name="Expression Calculator",
        description="Write a function calc(expr) that evaluates simple math expressions with +, -, *, / "
                     "and parentheses. Example: calc('(2 + 3) * 4') -> 20.0. Handle division by zero "
                     "by returning float('inf').",
        expected_output="20.0",
        difficulty=4,
        relevant_skills=["code_validator", "error_recovery", "task_planner"],
    ),
]


# ── Test code templates ──────────────────────────────────────────────────

CHALLENGE_TESTS = {
    "fizzbuzz": """
result = fizzbuzz(15)
expected = ['1','2','Fizz','4','Buzz','Fizz','7','8','Fizz','Buzz','11','Fizz','13','14','FizzBuzz']
assert result == expected, f"Got {result}"
assert fizzbuzz(0) == []
assert fizzbuzz(1) == ['1']
print("PASS")
""",
    "reverse_words": """
assert reverse_words("hello world") == "world hello"
assert reverse_words("  hello   world  ") == "world hello"
assert reverse_words("single") == "single"
assert reverse_words("") == ""
print("PASS")
""",
    "flatten_nested": """
assert flatten([1, [2, [3, 4], 5]]) == [1, 2, 3, 4, 5]
assert flatten([]) == []
assert flatten([1, 2, 3]) == [1, 2, 3]
assert flatten([[[[1]]]]) == [1]
assert flatten([1, [2], [3, [4, [5]]]]) == [1, 2, 3, 4, 5]
print("PASS")
""",
    "word_frequency": """
r = word_freq("Hello hello world!")
assert r == {"hello": 2, "world": 1}, f"Got {r}"
assert word_freq("") == {}
r2 = word_freq("a a a b b c")
assert r2 == {"a": 3, "b": 2, "c": 1}
print("PASS")
""",
    "lru_cache": """
c = LRUCache(2)
c.put(1, 1)
c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)
assert c.get(2) == -1
c.put(4, 4)
assert c.get(1) == -1
assert c.get(3) == 3
assert c.get(4) == 4
print("PASS")
""",
    "json_transformer": """
data = {"name": "alice", "age": 30, "city": "NYC"}
rules = [
    {"field": "name", "action": "uppercase"},
    {"field": "age", "action": "double"},
    {"field": "city", "action": "remove"},
]
r = transform_json(data, rules)
assert r == {"name": "ALICE", "age": 60}, f"Got {r}"
print("PASS")
""",
    "retry_decorator": """
import time as _t
attempts = []
@retry(max_attempts=3, backoff=0.01)
def flaky():
    attempts.append(1)
    if len(attempts) < 3:
        raise ValueError("not yet")
    return "ok"
assert flaky() == "ok"
assert len(attempts) == 3
print("PASS")
""",
    "dependency_resolver": """
deps = {"c": ["a", "b"], "b": ["a"], "a": [], "d": ["c"]}
order = resolve_deps(deps)
assert order.index("a") < order.index("b")
assert order.index("b") < order.index("c")
assert order.index("c") < order.index("d")
try:
    resolve_deps({"a": ["b"], "b": ["a"]})
    assert False, "Should raise ValueError"
except ValueError:
    pass
print("PASS")
""",
    "text_differ": """
old = "line1\\nline2\\nline3"
new = "line1\\nline2_modified\\nline3\\nline4"
d = diff_texts(old, new)
assert any(t[0] == 'remove' for t in d)
assert any(t[0] == 'add' for t in d)
assert any(t[0] == 'keep' for t in d)
print("PASS")
""",
    "mini_calculator": """
assert calc("2 + 3") == 5.0
assert calc("(2 + 3) * 4") == 20.0
assert calc("10 / 2") == 5.0
assert calc("1 / 0") == float("inf")
assert calc("2 + 3 * 4") == 14.0
print("PASS")
""",
}


# ── Challenge runner ─────────────────────────────────────────────────────

def build_challenge_prompt(challenge: Challenge) -> str:
    """Build a prompt for the LLM to solve a coding challenge."""
    test_code = CHALLENGE_TESTS.get(challenge.id, "")
    return f"""Solve this coding challenge. Output ONLY Python code, nothing else.

CHALLENGE: {challenge.name}
{challenge.description}

Your code will be tested with this test code appended to it:
```
{test_code}
```

Write ONLY the Python code needed (imports, function/class definitions).
Include any imports your code needs (like `import time`, `from collections import ...`).
No if __name__ block. No explanation. No markdown. Just valid Python code."""


def run_challenge(challenge: Challenge, code: str) -> ChallengeResult:
    """Run a challenge solution and score it."""
    test_code = CHALLENGE_TESTS.get(challenge.id, "")
    full_code = code + "\n\n" + test_code

    start = time.time()
    try:
        env = os.environ.copy()
        result = subprocess.run(
            ["python3", "-c", full_code],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        elapsed = time.time() - start

        if result.returncode == 0 and "PASS" in result.stdout:
            return ChallengeResult(
                challenge_id=challenge.id,
                solved=True,
                score=1.0,
                output=result.stdout.strip(),
                error="",
                time_s=elapsed,
                code=code,
            )
        return ChallengeResult(
            challenge_id=challenge.id,
            solved=False,
            score=0.0,
            output=result.stdout[:300],
            error=result.stderr[:300],
            time_s=elapsed,
            code=code,
        )
    except subprocess.TimeoutExpired:
        return ChallengeResult(
            challenge_id=challenge.id,
            solved=False,
            score=0.0,
            output="",
            error="Timed out (10s)",
            time_s=10.0,
            code=code,
        )
    except Exception as e:
        return ChallengeResult(
            challenge_id=challenge.id,
            solved=False,
            score=0.0,
            output="",
            error=str(e),
            time_s=time.time() - start,
            code=code,
        )


def get_challenges_by_difficulty(max_difficulty: int = 5) -> list[Challenge]:
    """Return challenges up to the given difficulty level."""
    return [c for c in CHALLENGES if c.difficulty <= max_difficulty]
