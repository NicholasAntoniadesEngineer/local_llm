"""Smart model router with hardware constraint enforcement."""

import asyncio
from dataclasses import dataclass
from typing import Literal
import yaml
import structlog

from .base import LLMClient, CompletionRequest, CompletionResponse
from .ollama_client import OllamaClient

logger = structlog.get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    type: str  # reasoning, orchestration, moe, code, embedding
    size_gb: float
    tok_s: float  # tokens per second (Ollama)
    tok_s_mlx: float  # tokens per second (MLX)
    backend: str  # ollama, mlx
    supports_thinking: bool


class ModelRouter:
    """Routes requests to optimal model based on role, enforcing hardware constraints."""

    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.clients: dict[str, LLMClient] = {}
        self._lock = asyncio.Lock()  # Prevent concurrent model loads
        self.loaded_models: set[str] = set()  # Track loaded 32B models

    def _load_config(self) -> dict:
        """Load model configuration from YAML."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error("config_not_found", path=self.config_path)
            raise

    async def select_model(
        self,
        role: Literal["orchestrate", "reason", "synthesize", "triage", "code", "embed"],
    ) -> str:
        """
        Select best model for role, respecting hardware constraints.

        Hard constraint: Never load two 32B models simultaneously.
        Strategy: If 32B is already loaded, use fallback model.
        """
        role_config = self.config["roles"][role]
        primary = role_config["primary"]

        # Check if we can load the primary model
        primary_size = self.config["models"][primary]["size_gb"]

        async with self._lock:
            # If primary is 32B and another 32B is loaded, use fallback
            if primary_size >= 19:
                if any(
                    self.config["models"][m]["size_gb"] >= 19 for m in self.loaded_models
                ):
                    logger.warning(
                        "32b_model_already_loaded",
                        primary=primary,
                        using_fallback=True,
                    )
                    return role_config["fallback"][0]  # First fallback

            return primary

    async def complete(
        self,
        role: Literal["orchestrate", "reason", "synthesize", "triage", "code", "embed"],
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> CompletionResponse:
        """
        Generate completion for the given role.

        Selects model, applies role constraints, routes to appropriate backend.
        """
        # Select model respecting constraints
        model = await self.select_model(role)

        # Get role configuration
        role_config = self.config["roles"][role]
        model_config = self.config["models"][model]

        # Build request with role constraints
        request = CompletionRequest(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=kwargs.get("temperature", role_config["constraints"]["temperature"]),
            max_tokens=kwargs.get("max_tokens", 1000),
            thinking_enabled=kwargs.get(
                "thinking_enabled", role_config["constraints"]["thinking_enabled"]
            ),
            thinking_budget=role_config["constraints"].get("thinking_budget", 500),
            tools=kwargs.get("tools"),
            json_mode=role_config.get("json_mode", False),
        )

        # Get or create client
        async with self._lock:
            if model not in self.clients:
                self.clients[model] = OllamaClient(model)
                self.loaded_models.add(model)
                logger.info("model_loaded", model=model, size_gb=model_config["size_gb"])

        client = self.clients[model]

        # Execute completion
        response = await client.complete(request)

        logger.info(
            "completion",
            model=model,
            role=role,
            tokens_out=response.usage.get("output_tokens", 0),
            latency_ms=response.latency_ms,
        )

        return response

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens in text using specified or default model."""
        if model is None:
            model = "orchestrate"  # Use fast model for counting

        model_name = self.config["roles"][model]["primary"]

        if model_name not in self.clients:
            self.clients[model_name] = OllamaClient(model_name)

        return await self.clients[model_name].count_tokens(text)

    async def cleanup(self) -> None:
        """Unload all models to free memory."""
        for model_name, client in self.clients.items():
            try:
                await client.unload()
                logger.info("model_unloaded", model=model_name)
            except Exception as e:
                logger.warning("cleanup_failed", model=model_name, error=str(e))

        self.clients.clear()
        self.loaded_models.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.cleanup()
