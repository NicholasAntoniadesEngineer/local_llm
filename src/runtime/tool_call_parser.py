"""Shared parser for Hermes and fallback tool-call formats."""

from __future__ import annotations

import json
import re


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TOOL_CALL_FALLBACK_RE = re.compile(r"<tool>(\w+)</tool>\s*<args>(\{.*?\})</args>", re.DOTALL)


def extract_tool_calls_from_response(response: str) -> list[dict]:
    """Extract structured tool calls from model output."""
    calls: list[dict] = []

    for match in TOOL_CALL_RE.finditer(response):
        try:
            call = json.loads(match.group(1))
            tool_name = call.get("name", "")
            tool_args = call.get("arguments", {})
            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)
            if tool_name:
                calls.append({"name": tool_name, "arguments": tool_args})
        except json.JSONDecodeError:
            continue
    if calls:
        return calls

    for match in TOOL_CALL_FALLBACK_RE.finditer(response):
        tool_name = match.group(1)
        try:
            tool_args = json.loads(match.group(2))
            calls.append({"name": tool_name, "arguments": tool_args})
        except json.JSONDecodeError:
            continue
    if calls:
        return calls

    if "```python" in response:
        code_match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        if code_match:
            calls.append({"name": "run_python", "arguments": {"code": code_match.group(1).strip()}})
    elif "write_file" in response.lower() or "save" in response.lower():
        code_match = re.search(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
        if code_match:
            calls.append({
                "name": "write_file",
                "arguments": {"path": "solution.py", "content": code_match.group(1).strip()},
            })

    return calls
