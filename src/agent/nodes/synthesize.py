"""Synthesize node: combine findings into coherent response."""

import asyncio
from datetime import datetime

import structlog

from src.llm.router import ModelRouter
from src.memory import MemoryManager
from ..state import AgentState

logger = structlog.get_logger(__name__)


def synthesize_node(
    state: AgentState,
    memory_manager: MemoryManager,
    model_router: ModelRouter,
) -> AgentState:
    """
    Synthesize all research findings into a coherent, well-cited response.

    Uses qwen3:32b (thinking disabled for speed) to:
    1. Organize findings by theme
    2. Synthesize into a cohesive narrative
    3. Add citations for all claims
    4. Highlight key insights
    5. Suggest areas for further research

    Args:
        state: Agent state with all findings and conversation history
        memory_manager: Memory manager for reference
        model_router: Model router for synthesis

    Returns:
        Updated state with final_response field set

    Example:
        Input: 12 findings about quantum computing from web + content reads
        Output: Multi-paragraph synthesis with citations and key insights
    """
    import asyncio

    async def _synthesize_async() -> dict:
        """Async synthesis implementation."""
        state["execution_status"] = "synthesizing"

        objective = state.get("objective", "")
        findings = state.get("findings", [])
        session_id = state.get("session_id", "unknown")

        logger.info(
            "synthesize_start",
            session_id=session_id,
            objective=objective,
            findings_count=len(findings),
        )

        # Format findings for synthesis
        formatted_findings = _format_findings_for_synthesis(findings)

        system_prompt = """You are an expert research synthesizer. Your task is to combine multiple research findings
into a comprehensive, well-organized, and well-cited response.

Guidelines:
1. Organize findings by theme or topic
2. Write in clear, accessible language
3. Add citations in [source N] format
4. Highlight key insights and surprising findings
5. Note any contradictions or areas of uncertainty
6. Suggest areas for further research
7. Maintain objectivity and avoid speculation

Format your response as:
## Overview
<1-2 sentence summary>

## Key Findings
<organized by theme with citations>

## Insights
<synthesized analysis>

## Recommendations for Further Research
<2-3 areas to explore>
"""

        prompt = f"""Research objective:
{objective}

Research findings ({len(findings)} total):
{formatted_findings}

Synthesize these findings into a comprehensive response. Ensure all key information is included and properly cited."""

        try:
            response = await model_router.complete(
                role="synthesize",
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1500,
                thinking_enabled=False,  # Not needed for synthesis
                temperature=0.5,
            )

            logger.info(
                "synthesize_response_received",
                tokens_out=response.usage.get("output_tokens"),
            )

            final_response = response.text.strip()

            # Update state
            state["final_response"] = final_response
            state["execution_status"] = "complete"

            # Add to message history
            message = {
                "role": "assistant",
                "content": final_response,
                "type": "final_synthesis",
                "model": response.model,
                "timestamp": datetime.utcnow().isoformat(),
            }
            state["messages"].append(message)

            # Count tokens
            tokens = response.usage.get("input_tokens", 0) + response.usage.get("output_tokens", 0)
            state["context_tokens"] = state.get("context_tokens", 0) + tokens

            logger.info("synthesize_complete", response_length=len(final_response))

            return state

        except Exception as e:
            logger.error("synthesize_error", error=str(e))
            state["error_message"] = f"Synthesis failed: {str(e)}"
            state["execution_status"] = "failed"
            # Fallback: concatenate findings
            state["final_response"] = "Error during synthesis. Findings:\n\n" + formatted_findings
            return state

    # Run async function
    return asyncio.run(_synthesize_async())


def _format_findings_for_synthesis(findings: list[dict]) -> str:
    """
    Format findings for inclusion in synthesis prompt.

    Args:
        findings: List of finding dicts

    Returns:
        Formatted string for prompt
    """
    if not findings:
        return "No findings available."

    formatted_lines = []
    for i, finding in enumerate(findings, 1):
        title = finding.get("title", "Finding")
        content = finding.get("content", "")[:200]  # Truncate long content
        sources = finding.get("sources", [])
        confidence = finding.get("confidence", 0.5)

        source_str = ", ".join(sources[:2]) if sources else "Unknown source"
        formatted = f"[{i}] {title}\n    Content: {content}\n    Source: {source_str}\n    Confidence: {confidence:.1%}\n"
        formatted_lines.append(formatted)

    return "\n".join(formatted_lines)
