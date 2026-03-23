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
            context_window=12_288,  # 12K safe limit for 36GB M4 Max
        ),
        # Qwen3-Coder: purpose-built for agentic coding, 70.6% SWE-Bench
        # Same MoE arch as balanced (30B total, 3B active), same memory footprint
        "coder": ModelConfig(
            name="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
            max_tokens=4096,
            context_window=12_288,
        ),
        "quality": ModelConfig(
            name="mlx-community/Qwen3-32B-4bit",
            max_tokens=4096,
            context_window=32_768,
        ),
        # Nemotron-Cascade-2: NVIDIA MoE, 17.8GB, beats Qwen3.5-35B on coding
        # Released March 20, 2026. Gold on IMO+IOI 2025.
        # LiveCodeBench 87.2 vs 74.6, same memory as Qwen3-30B-A3B
        "smart": ModelConfig(
            name="mlx-community/Nemotron-Cascade-2-30B-A3B-4bit",
            max_tokens=4096,
            context_window=32_768,
        ),
        # Qwen3-Coder: purpose-built for coding agents, 256K context
        "coder": ModelConfig(
            name="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
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
