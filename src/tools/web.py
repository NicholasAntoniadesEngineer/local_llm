"""Web tools: search and content extraction."""

import asyncio
import os
from typing import Any, List
import structlog
import httpx

logger = structlog.get_logger(__name__)


async def web_search(
    query: str,
    max_results: int = 5,
    include_answer: bool = True,
) -> List[dict[str, Any]]:
    """
    Search the web using Tavily API.

    Uses Tavily for LLM-optimized search results with extracted content.

    Args:
        query: Search query
        max_results: Maximum results to return (1-10)
        include_answer: Include AI-generated answer (if available)

    Returns:
        List of search results, each with:
        - url: Source URL
        - title: Page title
        - content: Extracted content summary
        - score: Relevance score (0-1)

    Example:
        results = await web_search("quantum computing breakthroughs 2024")
        # Returns: [{"url": "...", "title": "...", "content": "...", "score": 0.95}, ...]

    Raises:
        ValueError: If query is empty or TAVILY_API_KEY not set
        Exception: If API call fails
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("tavily_api_key_not_found")
        raise ValueError("TAVILY_API_KEY environment variable not set")

    max_results = min(max(max_results, 1), 10)  # Clamp to 1-10

    logger.info("web_search_start", query=query, max_results=max_results)

    try:
        # Call Tavily API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": include_answer,
                },
            )
            response.raise_for_status()

        result_data = response.json()

        # Extract and format results
        results = []
        for result in result_data.get("results", []):
            formatted_result = {
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "content": result.get("content", "")[:500],  # Truncate to 500 chars
                "score": result.get("score", 0.0),
            }
            results.append(formatted_result)

        # Include answer if available
        if include_answer and result_data.get("answer"):
            results.insert(0, {
                "url": "tavily_answer",
                "title": "AI-Generated Answer",
                "content": result_data.get("answer", ""),
                "score": 1.0,
            })

        logger.info("web_search_complete", query=query, results_count=len(results))
        return results

    except httpx.TimeoutException:
        logger.error("web_search_timeout", query=query)
        raise Exception(f"Web search timed out for query: {query}")

    except httpx.HTTPError as e:
        logger.error("web_search_http_error", error=str(e), query=query)
        raise Exception(f"Web search failed: {str(e)}")

    except Exception as e:
        logger.error("web_search_error", error=str(e), query=query)
        raise


async def read_url(url: str) -> str:
    """
    Read and extract content from URL using Jina Reader.

    Jina Reader is a free service that extracts clean markdown from any URL.
    No API key required.

    Args:
        url: URL to read

    Returns:
        Markdown content from the page

    Example:
        content = await read_url("https://example.com/article")
        # Returns: Markdown-formatted content

    Raises:
        ValueError: If URL is invalid
        Exception: If content extraction fails
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty")

    # Validate URL format
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    logger.info("read_url_start", url=url)

    try:
        # Use Jina Reader free endpoint
        # Format: https://r.jina.ai/{original_url}
        jina_url = f"https://r.jina.ai/{url}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                jina_url,
                headers={
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()

        # Jina Reader returns markdown directly
        content = response.text

        if not content or len(content.strip()) < 10:
            logger.warning("read_url_minimal_content", url=url, length=len(content))

        logger.info("read_url_complete", url=url, content_length=len(content))
        return content

    except httpx.TimeoutException:
        logger.error("read_url_timeout", url=url)
        raise Exception(f"URL read timed out: {url}")

    except httpx.HTTPError as e:
        logger.error("read_url_http_error", error=str(e), url=url)
        raise Exception(f"Failed to read URL: {str(e)}")

    except Exception as e:
        logger.error("read_url_error", error=str(e), url=url)
        raise


async def save_finding(
    text: str,
    finding_type: str = "general",
    importance: float = 0.5,
) -> dict[str, Any]:
    """
    Save a finding to research memory.

    Args:
        text: Finding text
        finding_type: Type (research, source, insight, etc.)
        importance: Importance score (0-1)

    Returns:
        Saved finding dict with ID

    Raises:
        ValueError: If text is empty
    """
    if not text or not text.strip():
        raise ValueError("Finding text cannot be empty")

    # This would integrate with MemoryManager
    # For now, just return metadata
    return {
        "id": f"finding_{hash(text) % 10000000}",
        "text": text[:1000],
        "type": finding_type,
        "importance": max(0.0, min(1.0, importance)),
    }


async def update_status(status: str) -> dict[str, Any]:
    """
    Update session execution status.

    Args:
        status: Status string (researching, analyzing, synthesizing, etc.)

    Returns:
        Acknowledgment dict

    Raises:
        ValueError: If status is empty
    """
    if not status or not status.strip():
        raise ValueError("Status cannot be empty")

    logger.info("status_update", status=status)

    return {
        "status": status,
        "acknowledged": True,
    }
