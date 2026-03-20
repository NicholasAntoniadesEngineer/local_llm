"""Async Ollama client implementation."""

import asyncio
import json
import re
from time import time

import httpx
import structlog

from .base import (
    LLMClient,
    CompletionRequest,
    CompletionResponse,
    ToolCall,
    LLMError,
    ModelNotAvailableError,
    ContextLengthExceededError,
    ToolCallParseError,
)

logger = structlog.get_logger(__name__)


class OllamaClient(LLMClient):
    """Async client for Ollama local LLM backend."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        super().__init__(model, base_url)
        self.client = httpx.AsyncClient(base_url=base_url, timeout=300.0)
        self._model_loaded = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion via Ollama."""
        start_time = time()

        # Check model is available
        if not await self.is_available():
            raise ModelNotAvailableError(f"Model {request.model} not available")

        # Build Ollama API request
        ollama_request = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
        }

        # Add options for thinking mode if supported
        if request.thinking_enabled and "qwen" in request.model.lower():
            ollama_request["options"] = {
                "num_ctx": 16384,
                "num_predict": request.max_tokens,
            }
        else:
            ollama_request["options"] = {
                "num_ctx": 16384,
                "num_predict": request.max_tokens,
            }

        # Add system prompt if provided
        if request.system_prompt:
            # Include system prompt in the full prompt (Ollama concatenates internally)
            ollama_request["prompt"] = f"{request.system_prompt}\n\n{request.prompt}"

        try:
            response = await self.client.post(
                "/api/generate",
                json=ollama_request,
            )
            response.raise_for_status()

            result = response.json()
            text = result.get("response", "")

            # Try to parse tool calls if tools were provided
            tool_calls = []
            if request.tools and "<tool_call>" in text:
                try:
                    tool_calls = self._parse_tool_calls(text)
                except ToolCallParseError as e:
                    logger.warning("failed_to_parse_tools", error=str(e))

            latency_ms = (time() - start_time) * 1000

            return CompletionResponse(
                text=text,
                stop_reason="end_turn",
                tool_calls=tool_calls,
                usage={
                    "input_tokens": result.get("prompt_eval_count", 0),
                    "output_tokens": result.get("eval_count", 0),
                },
                latency_ms=latency_ms,
                model=request.model,
            )

        except httpx.HTTPError as e:
            logger.error("ollama_http_error", error=str(e))
            raise LLMError(f"Ollama HTTP error: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error("ollama_json_error", error=str(e))
            raise LLMError(f"Ollama returned invalid JSON: {str(e)}")

    async def count_tokens(self, text: str) -> int:
        """Count tokens via tokenize endpoint."""
        try:
            response = await self.client.post(
                "/api/tokenize",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            result = response.json()
            return len(result.get("tokens", []))
        except Exception as e:
            logger.warning("token_count_failed", error=str(e))
            # Fallback: rough estimate (1 token ≈ 4 chars)
            return len(text) // 4

    async def is_available(self) -> bool:
        """Check if model is loaded and available."""
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            result = response.json()
            models = [m["name"].split(":")[0] for m in result.get("models", [])]
            model_base = self.model.split(":")[0]
            return model_base in models
        except Exception as e:
            logger.error("model_availability_check_failed", error=str(e))
            return False

    async def unload(self) -> None:
        """Unload model from memory."""
        try:
            # Send empty prompt with 0 timeout to unload
            await self.client.post(
                "/api/generate",
                json={
                    "model": self.model,
                    "keep_alive": 0,
                    "prompt": "",
                },
                timeout=5.0,
            )
            logger.info("model_unloaded", model=self.model)
        except Exception as e:
            logger.warning("model_unload_failed", error=str(e))

    def _parse_tool_calls(self, text: str) -> list[ToolCall]:
        """Parse tool calls from XML format: <tool_call>...</tool_call>."""
        tool_calls = []
        pattern = r"<tool_call>\s*<tool_name>(.+?)</tool_name>\s*<parameters>(.+?)</parameters>\s*</tool_call>"

        for match in re.finditer(pattern, text, re.DOTALL):
            tool_name = match.group(1).strip()
            params_str = match.group(2).strip()

            try:
                params = json.loads(params_str)
                tool_calls.append(ToolCall(name=tool_name, arguments=params))
            except json.JSONDecodeError as e:
                raise ToolCallParseError(f"Failed to parse params JSON: {str(e)}")

        return tool_calls
