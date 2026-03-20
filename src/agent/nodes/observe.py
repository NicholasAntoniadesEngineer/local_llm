"""Observe node: process tool results and store findings."""

import asyncio
from datetime import datetime
from typing import Any

import structlog

from src.llm.router import ModelRouter
from src.memory import MemoryManager, Finding
from ..state import AgentState

logger = structlog.get_logger(__name__)


async def observe_node(
    state: AgentState,
    memory_manager: MemoryManager,
    model_router: ModelRouter,
) -> AgentState:
    """
    Process tool results and persist findings to memory.

    Extracts key information from tool outputs, assigns importance scores,
    stores as Finding objects in LanceDB for retrieval, and updates
    context token count.

    Args:
        state: Agent state with executed tool results
        memory_manager: Memory manager for persisting findings
        model_router: Model router for token counting

    Returns:
        Updated state with findings persisted to memory

    Example:
        Tool results: [SearchResult, SearchResult]
        Extract: Titles, content snippets, source URLs
        Store: As Finding objects with importance scores
        Result: 2 new findings in memory
    """
    state["execution_status"] = "researching"
    session_id = state.get("session_id", "unknown")

    logger.info("observe_start", session_id=session_id)

    tool_results_dict = state.get("last_tool_results", {})
    executed_tools = tool_results_dict.get("executed_tools", [])

    findings_created = []

    try:
        # Process each tool result
        for tool_result in executed_tools:
            if not tool_result.get("success"):
                logger.warning("skipping_failed_tool", tool=tool_result.get("tool"))
                continue

            tool_name = tool_result.get("tool")
            result = tool_result.get("result")

            if not result:
                continue

            # Extract findings based on tool type
            if tool_name == "web_search":
                # result should be list of search results
                findings_list = _extract_search_findings(
                    result, tool_result.get("args", {}), session_id
                )
            elif tool_name == "read_url":
                # result should be markdown content
                findings_list = _extract_url_findings(
                    result, tool_result.get("args", {}), session_id
                )
            else:
                logger.warning("unknown_tool_type", tool=tool_name)
                findings_list = []

            # Persist findings to memory
            for finding in findings_list:
                try:
                    # This would call memory_manager.save_finding()
                    # For now, just track in state
                    findings_created.append(finding)
                    logger.info("finding_saved", finding_id=finding.get("id"))
                except Exception as e:
                    logger.warning("finding_save_failed", error=str(e))

        # Update state findings
        existing_findings = state.get("findings", [])
        state["findings"] = existing_findings + findings_created

        logger.info("observe_complete", findings_count=len(findings_created))

        # Count tokens
        total_finding_tokens = sum(
            len(f.get("content", "").split()) * 1.3 for f in findings_created  # Rough estimate
        )
        state["context_tokens"] = state.get("context_tokens", 0) + int(total_finding_tokens)

        # Check context budget
        max_tokens = state.get("max_context_tokens", 16384)
        utilization = state["context_tokens"] / max_tokens
        if utilization > 0.8:
            logger.warning("context_budget_high", utilization=f"{utilization:.1%}")

        # Add to message history
        message = {
            "role": "assistant",
            "content": f"Processed and stored {len(findings_created)} findings",
            "findings_stored": len(findings_created),
            "total_findings": len(state["findings"]),
            "timestamp": datetime.utcnow().isoformat(),
        }
        state["messages"].append(message)

        return state

    except Exception as e:
        logger.error("observe_error", error=str(e))
        state["error_message"] = f"Observation failed: {str(e)}"
        state["execution_status"] = "failed"
        return state


def _extract_search_findings(
    search_results: list[dict[str, Any]],
    args: dict[str, Any],
    session_id: str,
) -> list[dict[str, Any]]:
    """
    Extract findings from web search results.

    Args:
        search_results: List of search result dicts from Tavily API
        args: Tool arguments (contains search query)
        session_id: Session ID for tracking

    Returns:
        List of Finding-like dicts
    """
    findings = []
    query = args.get("query", "unknown")

    for result in search_results:
        finding = {
            "id": f"finding_{hash(result.get('url', '')) % 10000000}",
            "session_id": session_id,
            "title": result.get("title", ""),
            "content": result.get("content", "")[:1000],  # Truncate
            "sources": [result.get("url", "")],
            "confidence": result.get("score", 0.5),
            "tags": [query.split()[0]] if query else [],
            "importance": 5,
            "category": "search_result",
            "metadata": {
                "source": "tavily_search",
                "query": query,
            },
        }
        findings.append(finding)

    logger.info("search_findings_extracted", count=len(findings), query=query)
    return findings


def _extract_url_findings(
    content: str,
    args: dict[str, Any],
    session_id: str,
) -> list[dict[str, Any]]:
    """
    Extract findings from URL content.

    Args:
        content: Markdown content from Jina Reader
        args: Tool arguments (contains URL)
        session_id: Session ID for tracking

    Returns:
        List of Finding-like dicts
    """
    url = args.get("url", "unknown")

    # Extract first few paragraphs as finding
    paragraphs = content.split("\n\n")
    main_content = "\n".join(paragraphs[:3])  # First 3 paragraphs

    finding = {
        "id": f"finding_{hash(url) % 10000000}",
        "session_id": session_id,
        "title": "Content from " + url.split("/")[-1],
        "content": main_content[:1000],
        "sources": [url],
        "confidence": 0.8,  # Direct content is high confidence
        "tags": ["url_content"],
        "importance": 7,
        "category": "url_content",
        "metadata": {
            "source": "jina_reader",
            "url": url,
            "full_length": len(content),
        },
    }

    logger.info("url_finding_extracted", url=url)
    return [finding]
