import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from memory import SessionMemory
from config import CONFIG


def get_best_next_action(session: SessionMemory) -> Tuple[Optional[str], Optional[Dict]]:
    """Determine the best next action based on session history."""

    # Check if we have enough iterations to make a decision
    if len(session.iterations) < 3:
        return None, None

    # Analyze recent iterations
    recent = session.iterations[-3:]

    # Check for patterns in tool usage
    tool_usage = {}
    for it in recent:
        tool_usage[it.tool_used] = tool_usage.get(it.tool_used, 0) + 1

    # If we've been failing with a particular tool, try something different
    if any(not it.success for it in recent):
        # Try a different tool
        for tool in ['web_search', 'run_python', 'bash', 'read_file', 'write_file']:
            if tool not in tool_usage:
                return tool, {}

    # If we've been successful with a particular tool, continue with it
    if any(it.success for it in recent):
        # Use the most successful tool
        best_tool = max(tool_usage, key=tool_usage.get)
        return best_tool, {}

    # If all recent attempts have failed, try a new approach
    return 'web_search', {'query': 'How to solve this problem'}


def evaluate_action_result(session: SessionMemory, tool: str, args: Dict, result: str, success: bool) -> None:
    """Update the session memory with the result of an action."""

    # Record the iteration
    session.add_iteration(len(session.iterations) + 1, tool, args, result, success)

    # If the action was successful, record it as a discovery
    if success:
        session.record_discovery(f"Used {tool} with args {args} to achieve {result}")
        session.record_success(f"Used {tool} with args {args}", result)
    else:
        session.record_failure(f"Used {tool} with args {args}", result)