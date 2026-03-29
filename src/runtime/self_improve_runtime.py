"""Central environment defaults for the self-improvement loop (`tools/improve.py`).

Call `apply_self_improve_runtime_environment()` before constructing `MLXAgent` so long
multi-step runs get stable MLX macOS flags. TurboQuant-MLX is opt-in via `CONFIG` / env
(higher decode tok/s with default mlx-lm KV). Explicit environment variables always win.

`IMPROVE_LOOP_SLEEP_SEC` (default ``3``): seconds between `--loop` cycles; set ``0`` to
disable the pause.

`AGENT_LOG_RESPONSE_MAX_CHARS`: optional cap on ``response_text`` stored per ``generation`` line
in ``events.jsonl`` (omit for full text).

``MLX_METAL_DECODE_CAP``: optional ceiling (tokens) on per-step decode length; tiered caps also apply
for long prompts to avoid Metal OOM (see ``_metal_safe_max_new_tokens`` in ``mlx_adapter``).
"""

from __future__ import annotations

import os

from src.config import CONFIG
from src.runtime.turboquant_mlx_setup import ensure_turboquant_mlx_patch


def _turbo_kv_env_enabled() -> bool:
    value = os.environ.get("MLX_USE_TURBO_KV", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def apply_self_improve_runtime_environment() -> None:
    """Set process defaults for improve-loop runs (idempotent; respects pre-set env)."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("AGENT_MODEL", "fast")

    if CONFIG.self_improve_setdefault_turbo_kv:
        os.environ.setdefault("MLX_USE_TURBO_KV", "1")
        os.environ.setdefault("MLX_TURBO_BITS", str(CONFIG.self_improve_turbo_bits))
        os.environ.setdefault(
            "MLX_TURBO_FP16_LAYERS",
            str(CONFIG.self_improve_turbo_fp16_edge_layers),
        )
    else:
        os.environ.setdefault("MLX_USE_TURBO_KV", "0")

    ensure_turboquant_mlx_patch()


def print_self_improve_runtime_banner() -> None:
    """One-time friendly summary after `apply_self_improve_runtime_environment`."""
    bits = os.environ.get("MLX_TURBO_BITS", "3")
    edge = os.environ.get("MLX_TURBO_FP16_LAYERS", "4")
    lines = [
        "Self-improve runtime:",
        f"  KMP_DUPLICATE_LIB_OK={os.environ.get('KMP_DUPLICATE_LIB_OK', '')}",
        f"  AGENT_MODEL={os.environ.get('AGENT_MODEL', 'fast')} (tool_calling / quality = 32B, slower)",
    ]
    if _turbo_kv_env_enabled():
        try:
            import turboquant_mlx  # noqa: F401

            lines.append(
                f"  TurboQuant-MLX: on (bits={bits}, fp16_edge_layers={edge}, fused attention patched)"
            )
        except ImportError:
            lines.append(
                "  TurboQuant-MLX: requested but package missing — "
                "pip install -r requirements-turboquant.txt (or tools/install_turboquant_mlx.sh)"
            )
    else:
        lines.append("  TurboQuant-MLX: off (MLX_USE_TURBO_KV)")
    print("\n".join(lines))
