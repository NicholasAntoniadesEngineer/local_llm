"""Agent configuration for M4 Max (36GB). Max context = max RAM usage."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    name: str
    max_tokens: int
    context_window: int


@dataclass
class AgentConfig:
    models: dict[str, ModelConfig]
    max_iterations: int
    output_dir: Path
    web_search_timeout: int
    code_execution_timeout: int
    max_search_results: int


CONFIG = AgentConfig(
    models={
        "fast": ModelConfig(
            name="mlx-community/Qwen3.5-9B-MLX-4bit",
            max_tokens=4096,
            context_window=131_072,  # 128K context
        ),
        "balanced": ModelConfig(
            name="mlx-community/Qwen3-30B-A3B-4bit",
            max_tokens=4096,
            context_window=65_536,
        ),
        "quality": ModelConfig(
            name="mlx-community/Qwen3-32B-4bit",
            max_tokens=4096,
            context_window=65_536,
        ),
        # 14B with 128K context - 8GB model + ~16GB KV cache = ~24GB
        # This FILLS the RAM productively
        "tool_calling": ModelConfig(
            name="mlx-community/Qwen3-14B-4bit",
            max_tokens=4096,
            context_window=32_768,
        ),
    },
    max_iterations=25,  # more iterations = more KV cache = more RAM
    output_dir=Path("./agent_outputs"),
    web_search_timeout=10,
    code_execution_timeout=30,
    max_search_results=5,
)
