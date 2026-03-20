"""Act node: execute selected tools in parallel."""

import asyncio
from datetime import datetime
from typing import Any

import structlog

from src.llm.router import ModelRouter
from src.memory import MemoryManager
from src.tools import execute_tool
from ..state import AgentState

logger = structlog.get_logger(__name__)


async def act_node(
    state: AgentState,
    memory_manager: MemoryManager,
    model_router: ModelRouter,
) -> AgentState:
    """
    Execute selected tools in parallel with retry logic.

    Processes all tool calls from the think node, parallelizing execution
    with asyncio.gather(). Implements exponential backoff retry on failures.

    Max 3 retries per tool with 2^n second delays (2s, 4s, 8s).

    Args:
        state: Agent state with tool calls in last_tool_results
        memory_manager: Memory manager for logging actions
        model_router: Model router for contingency reasoning

    Returns:
        Updated state with tool execution results

    Example:
        Tools: [web_search("quantum computing 2024"), web_search("quantum error correction")]
        Execution: Parallel, max 2-3 concurrent
        Results: [SearchResult(...), SearchResult(...)]
    """
    state["execution_status"] = "researching"
    session_id = state.get("session_id", "unknown")

    logger.info("act_start", session_id=session_id)

    # Get tool calls from think node
    tool_results_dict = state.get("last_tool_results", {})
    actions = tool_results_dict.get("actions", [])

    if not actions:
        logger.info("no_tools_to_execute", session_id=session_id)
        state["last_tool_results"]["executed_tools"] = []
        return state

    # Prepare tool execution tasks
    async def execute_with_retry(tool_name: str, args: dict[str, Any], max_retries: int = 3) -> dict:
        """Execute tool with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                logger.info("tool_execute_start", tool=tool_name, attempt=attempt + 1)

                result = await execute_tool(tool_name, args)

                logger.info("tool_execute_success", tool=tool_name, attempt=attempt + 1)
                return {
                    "tool": tool_name,
                    "args": args,
                    "result": result,
                    "success": True,
                    "attempt": attempt + 1,
                }

            except Exception as e:
                logger.warning("tool_execute_failed", tool=tool_name, attempt=attempt + 1, error=str(e))

                if attempt < max_retries - 1:
                    # Exponential backoff: 2s, 4s, 8s
                    wait_time = 2 ** (attempt + 1)
                    logger.info("tool_retry_wait", tool=tool_name, wait_seconds=wait_time)
                    await asyncio.sleep(wait_time)
                else:
                    # Final attempt failed
                    return {
                        "tool": tool_name,
                        "args": args,
                        "result": None,
                        "success": False,
                        "error": str(e),
                        "attempt": attempt + 1,
                    }

        return {
            "tool": tool_name,
            "args": args,
            "result": None,
            "success": False,
            "error": "Max retries exceeded",
        }

    # Execute all tools in parallel (limit concurrency to 3)
    semaphore = asyncio.Semaphore(3)

    async def bounded_execute(tool_name: str, args: dict) -> dict:
        async with semaphore:
            return await execute_with_retry(tool_name, args)

    tasks = [bounded_execute(action["tool"], action["args"]) for action in actions]

    try:
        executed_results = await asyncio.gather(*tasks)

        logger.info("all_tools_executed", count=len(executed_results))

        # Process results
        successful = [r for r in executed_results if r.get("success")]
        failed = [r for r in executed_results if not r.get("success")]

        if failed:
            logger.warning("tool_failures", count=len(failed))
            state["tool_error"] = f"{len(failed)} tool(s) failed after retries"

        # Update state with results
        state["last_tool_results"]["executed_tools"] = executed_results
        state["last_tool_results"]["successful_count"] = len(successful)
        state["last_tool_results"]["failed_count"] = len(failed)

        # Add to message history
        message = {
            "role": "user",
            "content": f"Tool execution completed: {len(successful)} successful, {len(failed)} failed",
            "executed_tools": executed_results,
            "timestamp": datetime.utcnow().isoformat(),
        }
        state["messages"].append(message)

        return state

    except Exception as e:
        logger.error("act_error", error=str(e))
        state["tool_error"] = str(e)
        state["error_message"] = f"Tool execution failed: {str(e)}"
        state["execution_status"] = "failed"
        return state
