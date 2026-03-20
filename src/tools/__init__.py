"""Tool registry and execution interface for research agent."""

import asyncio
from typing import Any
import structlog

from .web import web_search, read_url

logger = structlog.get_logger(__name__)


# Tool registry
TOOL_REGISTRY = {
    "web_search": web_search,
    "read_url": read_url,
}


def get_tool_schema(tool_name: str) -> dict[str, Any]:
    """
    Get JSON schema for tool (for LLM tool use).

    Args:
        tool_name: Name of tool (web_search, read_url, etc.)

    Returns:
        Tool schema dict for LLM

    Example:
        schema = get_tool_schema("web_search")
        # Returns:
        # {
        #     "name": "web_search",
        #     "description": "Search the web using Tavily API",
        #     "input_schema": {
        #         "type": "object",
        #         "properties": {...}
        #     }
        # }
    """
    schemas = {
        "web_search": {
            "name": "web_search",
            "description": "Search the web for information about a topic. Returns LLM-optimized summaries.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'quantum computing 2024')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
        "read_url": {
            "name": "read_url",
            "description": "Read and extract content from a specific URL. Returns clean markdown.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to read (e.g., 'https://example.com/article')",
                    },
                },
                "required": ["url"],
            },
        },
    }

    if tool_name not in schemas:
        logger.warning("unknown_tool", tool=tool_name)
        raise ValueError(f"Unknown tool: {tool_name}")

    return schemas[tool_name]


async def execute_tool(tool_name: str, args: dict[str, Any]) -> Any:
    """
    Execute a tool with given arguments.

    Args:
        tool_name: Name of tool to execute
        args: Arguments dict

    Returns:
        Tool result (type depends on tool)

    Raises:
        ValueError: If tool not found
        Exception: If tool execution fails
    """
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")

    logger.info("tool_execute", tool=tool_name, args=args)

    tool_func = TOOL_REGISTRY[tool_name]
    result = await tool_func(**args)

    logger.info("tool_result", tool=tool_name, result_type=type(result).__name__)

    return result


__all__ = [
    "TOOL_REGISTRY",
    "get_tool_schema",
    "execute_tool",
    "web_search",
    "read_url",
]
