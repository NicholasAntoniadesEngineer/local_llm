"""Shared tool taxonomy for controller, verifier, and tool executor."""

from __future__ import annotations


OBSERVATION_TOOLS = frozenset({
    "web_search",
    "read_file",
    "grep_file",
    "list_dir",
})

MUTATION_TOOLS = frozenset({
    "write_file",
    "edit_file",
    "replace_lines",
})

EXECUTION_TOOLS = frozenset({
    "run_python",
    "bash",
})

READ_BATCH_SAFE_TOOLS = OBSERVATION_TOOLS
