"""Shared helpers for normalizing raw LLM text output."""

from __future__ import annotations

import re


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
PYTHON_CODE_FENCE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)
BARE_CODE_FENCE_RE = re.compile(r"```\s*(.*?)```", re.DOTALL)


def strip_thinking_tags(text: str) -> str:
    """Remove model thinking tags and normalize surrounding whitespace."""
    return THINK_BLOCK_RE.sub("", text).strip()


def extract_python_code_block(text: str) -> str | None:
    """Extract Python code from model output.

    Priority:
    1. ```python ... ``` fenced block
    2. ``` ... ``` bare fenced block (if content looks like Python)
    3. Raw response if it starts with valid Python (import/def/class/from)
    """
    if not text or not text.strip():
        return None

    # Priority 1: ```python fence
    match = PYTHON_CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()

    # Priority 2: bare ``` fence with Python content
    match = BARE_CODE_FENCE_RE.search(text)
    if match:
        code = match.group(1).strip()
        if "def " in code or "class " in code or "import " in code:
            return code

    # Priority 3: raw unfenced Python — accept if first non-empty line is valid Python
    stripped = text.strip()
    first_line = stripped.split("\n")[0].strip()
    if first_line.startswith(("import ", "from ", "def ", "class ", "#!", "# ")):
        return stripped

    return None


def diagnose_extraction_failure(text: str) -> str:
    """Explain why code extraction failed — for debugging, not recovery."""
    if not text or not text.strip():
        return "EMPTY_RESPONSE: Model returned empty/whitespace-only output"
    if text.startswith("ERROR"):
        return f"GENERATION_ERROR: {text[:200]}"
    if "```" in text and "def " not in text and "class " not in text:
        return f"FENCED_NON_PYTHON: Response has code fences but no def/class. First 200 chars: {text[:200]}"
    if "def " in text or "class " in text:
        return f"UNFENCED_CODE: Response contains Python but not in code fences. Model may need clearer instructions. First 200 chars: {text[:200]}"
    return f"NO_PYTHON_DETECTED: Response doesn't look like Python at all. First 200 chars: {text[:200]}"
