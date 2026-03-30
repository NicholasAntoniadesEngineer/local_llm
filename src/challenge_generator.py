"""Dynamic challenge generation — the agent creates new challenges for itself.

When all hardcoded challenges are solved, the agent generates novel challenges
by combining patterns from solved problems. This prevents plateauing on a
fixed problem set.
"""

from __future__ import annotations

import json
import ast
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.challenges import Challenge, CHALLENGE_TESTS, run_challenge
from src.paths import RUNS_DIR

if TYPE_CHECKING:
    from src.agent import MLXAgent


GENERATED_CHALLENGES_FILE = RUNS_DIR / "generated_challenges.json"

# Templates for generating new challenges at each difficulty level
_GENERATION_TEMPLATES = [
    # Difficulty 2: Data structure manipulation
    {
        "pattern": "data_transform",
        "prompt_template": (
            "Invent a NEW Python coding challenge about transforming data structures. "
            "Requirements:\n"
            "- The challenge should involve a function that takes input and returns output\n"
            "- It should be solvable in 10-30 lines of Python\n"
            "- It should test algorithmic thinking, not just library knowledge\n"
            "- Include 3-4 test assertions\n\n"
            "Output ONLY valid JSON with these fields:\n"
            '{{"name": "...", "description": "Write a function ...", '
            '"test_code": "assert func(...) == ...\\nassert func(...) == ...\\nprint(\'PASS\')", '
            '"difficulty": 2}}'
        ),
        "difficulty": 2,
    },
    # Difficulty 3: Algorithm design
    {
        "pattern": "algorithm",
        "prompt_template": (
            "Invent a NEW Python coding challenge about implementing an algorithm. "
            "Requirements:\n"
            "- The challenge should involve a function that solves a classic CS problem variant\n"
            "- It should be solvable in 15-40 lines of Python\n"
            "- It should require careful logic (sorting, searching, graph, DP, or string processing)\n"
            "- Include 4-5 test assertions covering edge cases\n\n"
            "Output ONLY valid JSON with these fields:\n"
            '{{"name": "...", "description": "Write a function ...", '
            '"test_code": "assert func(...) == ...\\nassert func(...) == ...\\nprint(\'PASS\')", '
            '"difficulty": 3}}'
        ),
        "difficulty": 3,
    },
    # Difficulty 4: Design pattern / class
    {
        "pattern": "design",
        "prompt_template": (
            "Invent a NEW Python coding challenge about implementing a design pattern or data structure class. "
            "Requirements:\n"
            "- The challenge should involve a class with 3+ methods\n"
            "- It should be solvable in 20-50 lines of Python\n"
            "- It should test object-oriented thinking and state management\n"
            "- Include 4-5 test assertions\n\n"
            "Output ONLY valid JSON with these fields:\n"
            '{{"name": "...", "description": "Implement a class ...", '
            '"test_code": "obj = ClassName(...)\\nassert obj.method(...) == ...\\nprint(\'PASS\')", '
            '"difficulty": 4}}'
        ),
        "difficulty": 4,
    },
]


def _load_generated_challenges() -> list[dict]:
    """Load previously generated challenges from disk."""
    if not GENERATED_CHALLENGES_FILE.exists():
        return []
    try:
        return json.loads(GENERATED_CHALLENGES_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_generated_challenges(challenges: list[dict]) -> None:
    """Save generated challenges to disk."""
    GENERATED_CHALLENGES_FILE.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_CHALLENGES_FILE.write_text(json.dumps(challenges, indent=2))


def _extract_json_from_response(text: str) -> dict | None:
    """Extract a JSON object from model output.

    Handles common LLM issues: unescaped newlines in strings, surrounding text.
    """
    def _try_parse(s: str) -> dict | None:
        try:
            data = json.loads(s)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
        # LLMs often output literal newlines inside JSON strings — fix them
        try:
            fixed = re.sub(r'(?<!\\)\n', r'\\n', s)
            data = json.loads(fixed)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    # Try the whole response first
    result = _try_parse(text.strip())
    if result:
        return result

    # Find the first { and match to its closing }
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return _try_parse(text[start:i + 1])

    return None


def _validate_generated_challenge(data: dict) -> Challenge | None:
    """Validate that a generated challenge has required fields and working tests."""
    name = data.get("name", "").strip()
    description = data.get("description", "").strip()
    test_code = data.get("test_code", "").strip()
    difficulty = int(data.get("difficulty", 2))

    if not name or not description or not test_code:
        return None
    if len(description) < 20:
        return None
    if "assert" not in test_code or "PASS" not in test_code:
        return None

    # Generate a unique ID
    cid = re.sub(r'[^a-z0-9]', '_', name.lower())[:30].strip('_')
    if not cid:
        return None

    # Verify the test code is valid Python
    try:
        ast.parse(test_code)
    except SyntaxError:
        return None

    return Challenge(
        id=f"gen_{cid}",
        name=name,
        description=description,
        expected_output="PASS",
        difficulty=min(5, max(1, difficulty)),
        relevant_skills=[],
    ), test_code


def generate_new_challenge(agent: "MLXAgent", target_difficulty: int = 2) -> tuple[Challenge, str] | None:
    """Use the LLM to generate a new coding challenge.

    Returns (Challenge, test_code) or None if generation fails.
    """
    from src.runtime.llm_text import strip_thinking_tags

    # Pick the template matching the target difficulty
    template = None
    for t in _GENERATION_TEMPLATES:
        if t["difficulty"] == target_difficulty:
            template = t
            break
    if template is None:
        template = _GENERATION_TEMPLATES[0]

    prompt = template["prompt_template"]

    response = agent.generate_simple(prompt, temperature=0.4)
    response = strip_thinking_tags(response)

    data = _extract_json_from_response(response)
    if data is None:
        print(f"    Challenge generation: failed to extract JSON")
        return None

    result = _validate_generated_challenge(data)
    if result is None:
        print(f"    Challenge generation: validation failed for '{data.get('name', '?')}'")
        return None

    challenge, test_code = result

    # Quick sanity check: make sure the test code doesn't pass with empty code
    from src.challenges import ChallengeResult
    empty_result = run_challenge(challenge, "pass")
    if empty_result.solved:
        print(f"    Challenge generation: trivial test (passes with empty code)")
        return None

    # Save for persistence
    stored = _load_generated_challenges()
    stored.append({
        "id": challenge.id,
        "name": challenge.name,
        "description": challenge.description,
        "test_code": test_code,
        "difficulty": challenge.difficulty,
    })
    _save_generated_challenges(stored)

    # Register the test code so run_challenge can use it
    CHALLENGE_TESTS[challenge.id] = test_code

    print(f"    Generated new challenge: {challenge.name} (difficulty {challenge.difficulty})")
    return challenge, test_code


def load_generated_challenges() -> list[tuple[Challenge, str]]:
    """Load all previously generated challenges and register their tests."""
    stored = _load_generated_challenges()
    results = []
    for data in stored:
        challenge = Challenge(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            expected_output="PASS",
            difficulty=data.get("difficulty", 2),
            relevant_skills=[],
        )
        test_code = data.get("test_code", "")
        if test_code:
            CHALLENGE_TESTS[challenge.id] = test_code
            results.append((challenge, test_code))
    return results
