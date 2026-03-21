"""MLX Agent Configuration — Production settings with no magic numbers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class MLXModelConfig:
    """MLX model selection and inference settings."""

    name: str  # HuggingFace model ID
    max_tokens: int  # Max generation length per call
    context_window: int  # Maximum context the model accepts


@dataclass
class ContextBudgetConfig:
    """Token allocation for context window (must sum to context_window)."""

    system_prompt: int  # Fixed system prompt
    tool_definitions: int  # Tool schemas and instructions
    conversation_history: int  # Previous turns in this task
    retrieved_memory: int  # Results from vector search
    workspace: int  # Current work/scratch space
    response_buffer: int  # Reserved for LLM generation

    def total(self) -> int:
        """Total allocated tokens."""
        return (
            self.system_prompt
            + self.tool_definitions
            + self.conversation_history
            + self.retrieved_memory
            + self.workspace
            + self.response_buffer
        )

    def validate(self, max_context: int) -> None:
        """Ensure allocation doesn't exceed context window."""
        total = self.total()
        if total > max_context:
            raise ValueError(
                f"Context budget {total} exceeds model context {max_context}"
            )


@dataclass
class AgentConfig:
    """Complete MLX agent configuration."""

    # Model selection and inference
    models: dict[str, MLXModelConfig]
    default_model: Literal["fast", "balanced", "quality"]

    # Context and memory
    max_iterations: int  # Max ReAct steps before stopping
    context_budget: ContextBudgetConfig

    # Output
    output_dir: Path  # Where agent writes results

    # Tool settings
    web_search_timeout: int  # Seconds to wait for search
    code_execution_timeout: int  # Seconds to wait for Python/bash
    max_search_results: int  # Top N results to return

    def get_model_for_goal(self, goal: str) -> MLXModelConfig:
        """Select model based on goal complexity."""
        complex_keywords = ["analyze", "complex", "sophisticated", "strategy"]
        if any(word in goal.lower() for word in complex_keywords):
            model_name = "balanced"
        else:
            model_name = "fast"
        return self.models[model_name]


# ==============================================================================
# PRODUCTION CONFIGURATION
# ==============================================================================

CONFIG = AgentConfig(
    # Model Configurations (from mlx-community)
    models={
        "fast": MLXModelConfig(
            name="mlx-community/Qwen3-8B-4bit",
            max_tokens=512,
            context_window=16_384,
        ),
        "balanced": MLXModelConfig(
            name="mlx-community/Qwen3-14B-4bit",
            max_tokens=512,
            context_window=16_384,
        ),
        "quality": MLXModelConfig(
            name="mlx-community/Qwen3-32B-4bit",
            max_tokens=512,
            context_window=16_384,
        ),
    },
    default_model="fast",

    # Context Budget (for 16K window, must sum to 16,384)
    context_budget=ContextBudgetConfig(
        system_prompt=1_024,  # Fixed prompt overhead
        tool_definitions=2_048,  # Tool schemas + instructions
        conversation_history=4_096,  # Previous turns
        retrieved_memory=4_096,  # Search/memory results
        workspace=3_072,  # Current task scratch space
        response_buffer=2_048,  # Reserved for generation
    ),

    # Agent behavior
    max_iterations=20,  # Max ReAct steps before stopping

    # File I/O
    output_dir=Path("./agent_outputs"),

    # Tool Timeouts and Limits
    web_search_timeout=10,  # seconds
    code_execution_timeout=30,  # seconds
    max_search_results=5,  # top N results
)

# Validate configuration at import time
CONFIG.context_budget.validate(
    CONFIG.models[CONFIG.default_model].context_window
)
