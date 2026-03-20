"""Plan node: decompose objective into sub-goals."""

import json
from datetime import datetime

import structlog

from src.llm.router import ModelRouter
from src.llm.base import CompletionRequest
from src.memory import MemoryManager
from ..state import AgentState

logger = structlog.get_logger(__name__)


def plan_node(
    state: AgentState,
    memory_manager: MemoryManager,
    model_router: ModelRouter,
) -> AgentState:
    """
    Decompose research objective into sub-goals.

    Uses qwen3:8b in JSON mode to generate structured 3-5 sub-goals.
    Each sub-goal is a specific research direction to investigate.

    This is a synchronous wrapper that calls async operations.
    In production, would use asyncio.run() or integrate with async runtime.

    Args:
        state: Current agent state
        memory_manager: Memory manager for persistence
        model_router: Model router for LLM selection

    Returns:
        Updated state with sub-goals and initial message

    Example:
        Input objective: "Find recent advances in quantum computing"
        Output sub-goals:
        - "Research quantum error correction techniques"
        - "Investigate quantum hardware platforms"
        - "Explore quantum software frameworks"
    """
    import asyncio

    async def _plan_async() -> dict:
        """Async planning implementation."""
        state["execution_status"] = "planning"

        objective = state["objective"]
        logger.info("planning_start", objective=objective)

        # Build planning prompt
        system_prompt = """You are a research planning expert. Given a research objective,
decompose it into 3-5 specific, actionable sub-goals that will help thoroughly investigate the topic.

Each sub-goal should:
- Be specific and measurable
- Target different aspects of the objective
- Be achievable in ~2-3 research steps
- Have no overlap with other sub-goals

Output as valid JSON with the following structure:
{
    "sub_goals": [
        "First sub-goal",
        "Second sub-goal",
        ...
    ],
    "reasoning": "Brief explanation of decomposition strategy"
}
"""

        prompt = f"""Decompose this research objective into sub-goals:

Objective: {objective}

Provide 3-5 specific sub-goals as JSON."""

        try:
            # Call model in JSON mode
            response = await model_router.complete(
                role="orchestrate",
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.1,
            )

            logger.info("planning_response_received", tokens_out=response.usage.get("output_tokens"))

            # Parse JSON response
            text = response.text.strip()

            # Try to extract JSON from response (handle markdown formatting)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            plan_data = json.loads(text)
            sub_goals = plan_data.get("sub_goals", [])[:5]  # Max 5 sub-goals

            if not sub_goals:
                logger.warning("planning_no_subgoals", response=response.text[:200])
                sub_goals = [objective]  # Fallback: use objective as single goal

            logger.info("planning_complete", sub_goals_count=len(sub_goals))

            # Update state
            state["sub_goals"] = sub_goals
            state["current_goal_index"] = 0
            state["step_number"] = 1

            # Add to message history
            message = {
                "role": "assistant",
                "content": f"Decomposed objective into {len(sub_goals)} sub-goals",
                "sub_goals": sub_goals,
                "model": response.model,
                "timestamp": datetime.utcnow().isoformat(),
            }
            state["messages"].append(message)

            # Count tokens
            tokens = response.usage.get("input_tokens", 0) + response.usage.get("output_tokens", 0)
            state["context_tokens"] = state.get("context_tokens", 0) + tokens

            return state

        except json.JSONDecodeError as e:
            logger.error("planning_json_error", error=str(e))
            # Fallback: use objective as single goal
            state["sub_goals"] = [objective]
            state["current_goal_index"] = 0
            state["step_number"] = 1
            state["error_message"] = f"Planning JSON parse error: {str(e)}"
            return state

        except Exception as e:
            logger.error("planning_error", error=str(e))
            state["error_message"] = f"Planning failed: {str(e)}"
            state["execution_status"] = "failed"
            return state

    # Run async function
    return asyncio.run(_plan_async())
