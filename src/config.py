"""Agent configuration for M4 Max (36GB). Conservative GPU to prevent Metal OOM kernel panics."""

from dataclasses import dataclass
from pathlib import Path
from src.paths import SKILLS_DIR, RUNS_DIR


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
            context_window=32_768,
        ),
        "quality": ModelConfig(
            name="mlx-community/Qwen3-32B-4bit",
            max_tokens=4096,
            context_window=32_768,
        ),
        # Qwen3.5-35B MoE: 18GB weights + 8GB KV = 26GB, leaves 10GB headroom
        # Smartest model that fits on 36GB M4 Max (March 2026)
        "smart": ModelConfig(
            name="mlx-community/Qwen3.5-35B-A3B-4bit",
            max_tokens=4096,
            context_window=32_768,
        ),
        "tool_calling": ModelConfig(
            name="mlx-community/Qwen3-14B-4bit",
            max_tokens=4096,
            context_window=16_384,
        ),
    },
    max_iterations=25,  # more iterations = more KV cache = more RAM
    output_dir=SKILLS_DIR,
    web_search_timeout=10,
    code_execution_timeout=30,
    max_search_results=5,
)
