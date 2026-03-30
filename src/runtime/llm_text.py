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
    """Return the first fenced Python code block if present.

    Tries ```python first, falls back to bare ``` if the content looks like Python.
    """
    match = PYTHON_CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()

    # Fallback: bare code fence (```...```) — accept if it looks like Python
    match = BARE_CODE_FENCE_RE.search(text)
    if match:
        code = match.group(1).strip()
        if "def " in code or "class " in code or "import " in code:
            return code

    return None
