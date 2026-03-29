"""Agent configuration for M4 Max (36GB). Defaults tuned for long self-improve runs on unified memory."""

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
    # Self-improve loop (`tools/improve.py`): TurboQuant off by default for higher decode tok/s; opt in via env.
    self_improve_setdefault_turbo_kv: bool = False
    self_improve_turbo_bits: int = 3
    self_improve_turbo_fp16_edge_layers: int = 4


CONFIG = AgentConfig(
    models={
        "fast": ModelConfig(
            name="mlx-community/Qwen3-14B-4bit",
            max_tokens=8192,
            # Qwen3 dense decoder: TurboQuant-compatible. 40_960 matches HF max_position_embeddings;
            # larger context_window + huge frozen prompts OOMs Metal (malloc/concatenate during prefill).
            context_window=40_960,
        ),
        "balanced": ModelConfig(
            name="mlx-community/Qwen3.5-27B-4bit",
            max_tokens=8192,
            context_window=32_768,
        ),
        "primary": ModelConfig(
            name="mlx-community/Qwen3-32B-4bit",
            max_tokens=8192,
            context_window=32_768,
        ),
        "benchmark_coder": ModelConfig(
            name="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
            max_tokens=8192,
            context_window=32_768,
        ),
        "quality": ModelConfig(
            name="mlx-community/Qwen3-32B-4bit",
            max_tokens=8192,
            context_window=32_768,
        ),
        # Nemotron-Cascade-2: NVIDIA MoE, 17.8GB, beats Qwen3.5-35B on coding
        # Released March 20, 2026. Gold on IMO+IOI 2025.
        # LiveCodeBench 87.2 vs 74.6, same memory as Qwen3-30B-A3B
        "smart": ModelConfig(
            name="mlx-community/Nemotron-Cascade-2-30B-A3B-4bit",
            max_tokens=8192,
            context_window=32_768,
        ),
        "coder": ModelConfig(
            name="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
            max_tokens=8192,
            context_window=32_768,
        ),
        "benchmark_tools": ModelConfig(
            name="mlx-community/gemma-3-27b-it-4bit",
            max_tokens=8192,
            context_window=32_768,
        ),
        "benchmark_judge": ModelConfig(
            name="mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
            max_tokens=8192,
            context_window=32_768,
        ),
        "tool_calling_14b": ModelConfig(
            name="mlx-community/Qwen3-14B-4bit",
            max_tokens=8192,
            context_window=40_960,
        ),
        # Default improve loop: AGENT_MODEL=fast (Qwen3 14B, ~40k context, highest tok/s on M-series).
        # Stronger: AGENT_MODEL=tool_calling (32B). Same weights as fast: tool_calling_14b.
        "tool_calling": ModelConfig(
            name="mlx-community/Qwen3-32B-4bit",
            max_tokens=8192,
            context_window=32_768,
        ),
    },
    max_iterations=40,
    output_dir=SKILLS_DIR,
    web_search_timeout=10,
    code_execution_timeout=30,
    max_search_results=5,
    self_improve_setdefault_turbo_kv=False,
    self_improve_turbo_bits=3,
    self_improve_turbo_fp16_edge_layers=4,
)
