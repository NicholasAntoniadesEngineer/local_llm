"""Shared helpers for normalizing raw LLM text output."""

from __future__ import annotations

import re


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
PYTHON_CODE_FENCE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)


def strip_thinking_tags(text: str) -> str:
    """Remove model thinking tags and normalize surrounding whitespace."""
    return THINK_BLOCK_RE.sub("", text).strip()


def extract_python_code_block(text: str) -> str | None:
    """Return the first fenced Python code block if present."""
    match = PYTHON_CODE_FENCE_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()
