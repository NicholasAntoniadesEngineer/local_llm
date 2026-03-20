"""Think node: reason about current goal and select tools."""

import json
from datetime import datetime

import structlog

from src.llm.router import ModelRouter
from src.llm.base import CompletionRequest
from src.memory import MemoryManager
from src.tools import TOOL_DEFINITIONS, get_tool_schema
from ..state import AgentState

logger = structlog.get_logger(__name__)


def think_node(
    state: AgentState,
    memory_manager: MemoryManager,
    model_router: ModelRouter,
) -> AgentState:
    """
    Reason about current goal and select which tools to use.

    Uses qwen3:32b with thinking enabled to analyze the current sub-goal,
    retrieve relevant prior findings, and decide which tools to call.

    ReAct-style reasoning format:
    - Thought: Analysis of the goal and available tools
    - Tool Selection: Which tools to call and with what arguments
    - Expected Outcome: What we expect the tools to return

    Args:
        state: Current agent state with current_goal_index set
        memory_manager: Memory manager for retrieving past findings
        model_router: Model router for reasoning

    Returns:
        Updated state with tool selection and thinking

    Example:
        Current goal: "Find recent quantum computing breakthroughs"
        Retrieved context: [3 relevant findings from memory]
        Tool selection: web_search + read_url tools
    """
    import asyncio

    async def _think_async() -> dict:
        """Async thinking implementation."""
        state["execution_status"] = "researching"

        current_goal_index = state.get("current_goal_index", 0)
        sub_goals = state.get("sub_goals", [])

        if current_goal_index >= len(sub_goals):
            logger.warning("invalid_goal_index", index=current_goal_index, total=len(sub_goals))
            state["should_continue"] = False
            return state

        current_goal = sub_goals[current_goal_index]
        session_id = state.get("session_id", "unknown")

        logger.info("thinking_start", goal=current_goal, goal_index=current_goal_index)

        # Retrieve relevant context from memory
        prior_context = ""
        try:
            # This would use memory manager's retrieval interface
            # For now, placeholder
            prior_context = f"No prior findings for this goal yet."
            logger.info("memory_retrieved", context_length=len(prior_context))
        except Exception as e:
            logger.warning("memory_retrieval_failed", error=str(e))
            prior_context = "Memory retrieval unavailable."

        # Build tool definitions
        tool_definitions = [get_tool_schema("web_search"), get_tool_schema("read_url")]

        system_prompt = """You are a research assistant with access to web search and content reading tools.
Your task is to analyze a research goal and select the best tools to accomplish it.

Think step-by-step:
1. Understand the goal
2. Consider what information you need
3. Select appropriate tools and search queries
4. Explain why these tools will help

Respond in this format:
<thought>
Your analysis of the goal and approach
</thought>

<tool_calls>
<tool_call>
<tool_name>web_search</tool_name>
<parameters>{"query": "specific search query"}</parameters>
</tool_call>
</tool_calls>

You can call multiple tools in parallel. Focus on 2-3 high-value queries.
"""

        prompt = f"""Current research goal:
{current_goal}

Prior findings context:
{prior_context}

Session ID: {session_id}
Step: {state.get('step_number', 0)}/{state.get('max_steps', 15)}

Analyze this goal and select tools to research it. Include relevant keywords, dates, or specific aspects."""

        try:
            response = await model_router.complete(
                role="reason",
                prompt=prompt,
                system_prompt=system_prompt,
                tools=tool_definitions,
                max_tokens=1500,
                thinking_enabled=True,
                temperature=0.3,
            )

            logger.info(
                "thinking_response_received",
                tokens_out=response.usage.get("output_tokens"),
                thinking_enabled=True,
            )

            # Extract thinking and tool calls
            thinking = response.thinking or ""
            tool_calls = response.tool_calls or []

            # Convert tool calls to action dict
            actions = []
            for tool_call in tool_calls:
                actions.append({
                    "tool": tool_call.name,
                    "args": tool_call.arguments,
                })

            logger.info("thinking_complete", tool_calls_count=len(actions))

            # Update state
            state["last_tool_results"] = {
                "thinking": thinking,
                "actions": actions,
                "model": response.model,
            }

            # If no tools selected, mark decision
            if not actions:
                logger.warning("no_tools_selected", goal=current_goal)
                state["last_tool_results"]["note"] = "Model selected no tools"

            # Add to message history
            message = {
                "role": "assistant",
                "content": f"Analyzed goal: {current_goal}",
                "thinking": thinking[:500] if thinking else "",
                "tool_calls": actions,
                "model": response.model,
                "timestamp": datetime.utcnow().isoformat(),
            }
            state["messages"].append(message)

            # Count tokens
            tokens = response.usage.get("input_tokens", 0) + response.usage.get("output_tokens", 0)
            state["context_tokens"] = state.get("context_tokens", 0) + tokens

            return state

        except Exception as e:
            logger.error("thinking_error", error=str(e))
            state["tool_error"] = str(e)
            state["error_message"] = f"Thinking failed: {str(e)}"
            state["execution_status"] = "failed"
            return state

    # Run async function
    return asyncio.run(_think_async())
