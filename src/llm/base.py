"""Abstract LLM client base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from datetime import datetime


@dataclass
class ToolCall:
    """A single tool invocation."""
    name: str
    arguments: dict[str, Any]


@dataclass
class CompletionRequest:
    """Request to generate a completion."""
    model: str
    prompt: str
    temperature: float = 0.3
    max_tokens: int = 1000
    top_p: float = 0.9
    top_k: int = 40
    stop_sequences: list[str] = field(default_factory=list)
    thinking_enabled: bool = False
    thinking_budget: int = 500
    tools: list[dict[str, Any]] | None = None
    json_mode: bool = False
    system_prompt: str | None = None


@dataclass
class CompletionResponse:
    """Response from LLM completion."""
    text: str
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    thinking: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)  # {input_tokens, output_tokens}
    latency_ms: float = 0.0
    model: str = ""
    cached: bool = False


class LLMClient(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, model: str, base_url: str | None = None):
        self.model = model
        self.base_url = base_url
        self._cache: dict[str, CompletionResponse] = {}

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate completion for the given request.

        Args:
            request: Completion request with model, prompt, and parameters

        Returns:
            CompletionResponse with generated text, tool calls, usage, etc.

        Raises:
            LLMError: If inference fails
        """
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the model's tokenizer.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """
        Check if model is available and ready.

        Returns:
            True if model is loaded and ready
        """
        pass

    async def unload(self) -> None:
        """
        Unload model from memory to free resources.
        Override in subclass if needed.
        """
        pass

    def _get_cache_key(self, request: CompletionRequest) -> str:
        """Generate cache key for request."""
        import hashlib
        key_parts = [
            request.model,
            request.prompt[:100],  # Use first 100 chars
            str(request.temperature),
            str(request.max_tokens),
        ]
        return hashlib.sha256("_".join(key_parts).encode()).hexdigest()

    def _cache_get(self, request: CompletionRequest) -> CompletionResponse | None:
        """Try to get cached response."""
        key = self._get_cache_key(request)
        response = self._cache.get(key)
        if response:
            response.cached = True
        return response

    def _cache_set(self, request: CompletionRequest, response: CompletionResponse) -> None:
        """Cache a response."""
        key = self._get_cache_key(request)
        self._cache[key] = response


class LLMError(Exception):
    """Base exception for LLM operations."""
    pass


class ModelNotAvailableError(LLMError):
    """Model is not available (not loaded, network error, etc)."""
    pass


class ContextLengthExceededError(LLMError):
    """Request exceeds model's context window."""
    pass


class ToolCallParseError(LLMError):
    """Failed to parse tool calls from model response."""
    pass
