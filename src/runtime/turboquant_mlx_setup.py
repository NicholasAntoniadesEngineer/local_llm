"""Optional TurboQuant-MLX KV cache (arozanov/turboquant-mlx) for mlx-lm on Apple Silicon.

Set MLX_USE_TURBO_KV=0 to force the default mlx-lm prompt cache.
Environment:
  MLX_TURBO_BITS         — TurboQuant bits for middle layers (default 3)
  MLX_TURBO_FP16_LAYERS  — first/last layers kept FP16 (default 4)
"""

from __future__ import annotations

import os
from typing import Any, Callable

_turboquant_attention_patch_applied = False


def ensure_turboquant_mlx_patch() -> bool:
    """Patch mlx_lm attention for fused TurboQuant decode when the package is installed.

    Safe to call multiple times. Returns True if the patch is active (or was already).
    """
    if _env_flag_disabled("MLX_USE_TURBO_KV"):
        return False
    try:
        from turboquant_mlx import apply_patch
    except ImportError:
        return False
    global _turboquant_attention_patch_applied
    if not _turboquant_attention_patch_applied:
        apply_patch()
        _turboquant_attention_patch_applied = True
    return True


def _env_flag_disabled(name: str) -> bool:
    value = os.environ.get(name, "1").strip().lower()
    return value in ("0", "false", "no", "off")


def _transformer_layer_count(model: Any) -> int:
    if hasattr(model, "layers"):
        return len(model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return len(model.language_model.layers)
    from mlx_lm.models.cache import make_prompt_cache

    return len(make_prompt_cache(model))


def try_make_turboquant_cache_factory(model: Any) -> Callable[[], list[Any]] | None:
    """Return a cache factory using layer-adaptive TurboQuant, or None to use defaults.

    Imports are lazy so the project runs without turboquant-mlx installed.
    """
    if _env_flag_disabled("MLX_USE_TURBO_KV"):
        return None

    if not ensure_turboquant_mlx_patch():
        return None

    try:
        from turboquant_mlx import make_adaptive_cache
    except ImportError:
        return None

    bits = int(os.environ.get("MLX_TURBO_BITS", "3"))
    fp16_layers = int(os.environ.get("MLX_TURBO_FP16_LAYERS", "4"))

    def factory() -> list[Any]:
        num_layers = _transformer_layer_count(model)
        return make_adaptive_cache(
            num_layers,
            bits=bits,
            fp16_layers=fp16_layers,
            fused=True,
            model=model,
        )

    return factory

