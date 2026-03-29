"""MLX generation adapter extracted from the bootstrap agent."""

from __future__ import annotations

import gc
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from src.runtime.llm_text import strip_thinking_tags


def _mlx_prefill_step_size_from_env() -> int:
    """Chunk size for prompt prefill; smaller values lower peak GPU memory (mlx_lm)."""
    raw = os.environ.get("MLX_PREFILL_STEP_SIZE", "1024").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 1024
    return max(256, min(value, 8192))


def _metal_safe_prompt_token_cap() -> int | None:
    """Extra cap below context_window to avoid Metal command-buffer aborts on long prefills.

    Default 18432 tokens (14B-class + 8k gen headroom on unified memory). Set
    MLX_METAL_SAFE_PROMPT_TOKENS empty to disable, or a higher integer to allow longer prompts.
    """
    default_text = "18432"
    raw = os.environ.get("MLX_METAL_SAFE_PROMPT_TOKENS", default_text)
    stripped = raw.strip() if isinstance(raw, str) else str(raw).strip()
    if stripped == "":
        return None
    try:
        return max(256, int(stripped))
    except ValueError:
        return max(256, int(default_text))


def _mlx_max_kv_size_from_env() -> int | None:
    """Optional KV length cap (sliding window); set MLX_MAX_KV_SIZE to reduce memory."""
    raw = os.environ.get("MLX_MAX_KV_SIZE", "").strip()
    if not raw:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        return None
    return max(512, parsed)


def _metal_safe_max_new_tokens(
    prompt_tokens: int,
    configured_max_tokens: int,
    context_window: int,
) -> int:
    """Cap decode length for long prompts to avoid Metal OOM during generation (unified memory).

    Nominal ``context_window`` can fit prompt+8192 on paper, but long prefills plus a full
    ``max_tokens`` decode often exhaust GPU memory (kIOGPUCommandBufferCallbackErrorOutOfMemory).

    Tiered defaults (override with ``MLX_METAL_DECODE_CAP`` to a single ceiling, or raise tiers):
    - prompt >= 14k → 2048 new tokens
    - prompt >= 10k → 4096
    - prompt >= 8k → 6144
    - else → ``configured_max_tokens`` (still bounded by context headroom)
    """
    context_headroom = max(64, context_window - prompt_tokens - 256)
    if prompt_tokens >= 14_000:
        tier_cap = 2_048
    elif prompt_tokens >= 10_000:
        tier_cap = 4_096
    elif prompt_tokens >= 8_000:
        tier_cap = 6_144
    else:
        tier_cap = configured_max_tokens
    raw_cap = os.environ.get("MLX_METAL_DECODE_CAP", "").strip()
    if raw_cap:
        try:
            env_cap = max(64, int(raw_cap))
            tier_cap = min(tier_cap, env_cap)
        except ValueError:
            pass
    return min(configured_max_tokens, context_headroom, tier_cap)


class MLXGenerationAdapter:
    """Own the streaming generation path and live perf updates."""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        stream_generate: Callable[..., Any],
        generate_fn: Callable[..., str],
        sampler,
        kv_cache_manager,
        status_writer,
        logger,
        perf: dict[str, Any],
        config_model,
        model_size_gb: float,
        draft_model=None,
        clear_cache_fn: Callable[[], None] | None = None,
        collect_garbage_fn: Callable[[], None] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.stream_generate = stream_generate
        self.generate_fn = generate_fn
        self.sampler = sampler
        self.kv_cache_manager = kv_cache_manager
        self.status_writer = status_writer
        self.logger = logger
        self.perf = perf
        self.config_model = config_model
        self.model_size_gb = model_size_gb
        self.draft_model = draft_model
        self.clear_cache_fn = clear_cache_fn or (lambda: None)
        self.collect_garbage_fn = collect_garbage_fn or gc.collect
        self._last_effective_max_tokens = int(config_model.max_tokens)

    def _iteration_timing_live(self, elapsed_so_far: float) -> dict[str, Any]:
        """Wall time for the in-flight LLM call vs last completed and best (min) completed."""
        times = self.perf["step_times"]
        return {
            "this_iteration_s": round(elapsed_so_far, 2),
            "last_iteration_s": round(times[-1], 2) if times else None,
            "best_iteration_s": round(min(times), 2) if times else None,
        }

    def _iteration_timing_after_completion(self) -> dict[str, Any]:
        """After appending to step_times: this = latest wall time; last = previous; best = min."""
        times = self.perf["step_times"]
        if not times:
            return {"this_iteration_s": None, "last_iteration_s": None, "best_iteration_s": None}
        return {
            "this_iteration_s": round(times[-1], 2),
            "last_iteration_s": round(times[-2], 2) if len(times) >= 2 else None,
            "best_iteration_s": round(min(times), 2),
        }

    def generate_response(
        self,
        messages: list[dict[str, Any]],
        format_prompt: Callable[[list[dict[str, Any]]], str],
    ) -> str:
        """Generate a response with streaming updates and perf accounting."""
        self.clear_cache_fn()
        self.collect_garbage_fn()

        prompt = format_prompt(messages)
        stable_prefix = messages[0].get("content", "") if messages else ""
        self.kv_cache_manager.ensure_prefix(stable_prefix)
        prompt_tokens = len(self.tokenizer.encode(prompt)) if hasattr(self.tokenizer, "encode") else len(prompt) // 4
        reserved_for_generation = self.config_model.max_tokens + 1024
        context_prompt_cap = max(256, self.config_model.context_window - reserved_for_generation)
        max_prompt_tokens = context_prompt_cap
        metal_cap = _metal_safe_prompt_token_cap()
        if metal_cap is not None:
            max_prompt_tokens = min(max_prompt_tokens, metal_cap)
        limited_by_metal = metal_cap is not None and max_prompt_tokens < context_prompt_cap
        if prompt_tokens > max_prompt_tokens:
            overshoot = prompt_tokens - max_prompt_tokens
            self.status_writer.write_status(
                status=f"PROMPT TOO LARGE ({prompt_tokens} tok, max {max_prompt_tokens})",
                generating=False,
                perf=self.perf,
                prompt_tokens=prompt_tokens,
                gen_tokens=0,
                decode_tok_s="—",
            )
            hint = (
                " Shrink chat/frozen snapshot, or set MLX_METAL_SAFE_PROMPT_TOKENS higher (empty disables cap)."
                if limited_by_metal
                else " Context shrink should have run; report this as a bug."
            )
            return (
                f"ERROR: Prompt too large for safe prefill ({prompt_tokens} tokens; "
                f"limit {max_prompt_tokens}, over by {overshoot}).{hint}"
            )

        effective_max_tokens = _metal_safe_max_new_tokens(
            prompt_tokens,
            self.config_model.max_tokens,
            self.config_model.context_window,
        )
        self._last_effective_max_tokens = effective_max_tokens
        if effective_max_tokens < self.config_model.max_tokens:
            print(
                f"  ⚠️  MLX Metal safety: max_tokens {self.config_model.max_tokens} → "
                f"{effective_max_tokens} (long prompt {prompt_tokens} tok; avoids GPU OOM)"
            )

        self.perf["prefill_wall_time_start"] = time.time()
        self.status_writer.write_status(
            status=f"PREFILLING {prompt_tokens} tokens...",
            generating=True,
            perf=self.perf,
            prompt_tokens=prompt_tokens,
            gen_tokens=0,
            effective_max_tokens=effective_max_tokens,
            configured_max_tokens=self.config_model.max_tokens,
        )

        prefill_heartbeat_stop = threading.Event()

        def _prefill_heartbeat_loop() -> None:
            wall_start = time.time()
            while not prefill_heartbeat_stop.wait(2.0):
                elapsed_wall = int(time.time() - wall_start)
                self.status_writer.write_status(
                    status=f"PREFILLING {prompt_tokens} tokens ({elapsed_wall}s)…",
                    generating=True,
                    perf=self.perf,
                    prompt_tokens=prompt_tokens,
                    gen_tokens=0,
                    effective_max_tokens=effective_max_tokens,
                    configured_max_tokens=self.config_model.max_tokens,
                )

        heartbeat_thread = threading.Thread(target=_prefill_heartbeat_loop, daemon=True)
        heartbeat_thread.start()

        try:
            start_time = time.perf_counter()
            generation_kwargs = {
                "prompt": prompt,
                "max_tokens": effective_max_tokens,
                "sampler": self.sampler,
                "prefill_step_size": _mlx_prefill_step_size_from_env(),
            }
            max_kv_size = _mlx_max_kv_size_from_env()
            if max_kv_size is not None:
                generation_kwargs["max_kv_size"] = max_kv_size
            if self.kv_cache_manager.prompt_cache:
                generation_kwargs["prompt_cache"] = self.kv_cache_manager.prompt_cache
            if self.draft_model:
                generation_kwargs["draft_model"] = self.draft_model

            response_text, token_count, first_token_time = self._stream_response(generation_kwargs, prompt_tokens, start_time)
            elapsed = time.perf_counter() - start_time
            if token_count == 0:
                token_count = max(1, len(response_text) // 4)
            gen_tokens_per_second = token_count / max(0.01, elapsed)

            self.perf["total_tokens"] += token_count
            self.perf["total_gen_time"] += elapsed
            self.perf["step_times"].append(elapsed)
            self.perf["prompt_tokens"] = self.perf.get("prompt_tokens", 0) + prompt_tokens

            average_tokens_per_second = self.perf["total_tokens"] / max(0.01, self.perf["total_gen_time"])
            actual_prefill = first_token_time if first_token_time else (prompt_tokens / 300.0)
            decode_time = max(0.01, elapsed - actual_prefill)
            decode_tokens_per_second = max(0, token_count - 1) / decode_time if token_count > 1 else gen_tokens_per_second
            final_stats = self._final_generation_stats(
                prompt_tokens=prompt_tokens,
                gen_tokens=token_count,
                elapsed=elapsed,
                average_tokens_per_second=average_tokens_per_second,
                gen_tokens_per_second=gen_tokens_per_second,
                decode_tokens_per_second=decode_tokens_per_second,
                actual_prefill=actual_prefill,
                decode_time=decode_time,
            )
            final_stats.update(self._iteration_timing_after_completion())
            self.perf["peak_tok_s"] = final_stats["peak_tok_s"]
            self.status_writer.write_generation_stats(final_stats)

            print(
                f"  ⚡ {decode_tokens_per_second:.0f} tok/s decode "
                f"({gen_tokens_per_second:.0f} overall) | {prompt_tokens}p+{token_count}g | {elapsed:.1f}s"
            )
            iteration_timing = self._iteration_timing_after_completion()
            self.logger.generation(
                step=len(self.perf["step_times"]),
                prompt_tokens=prompt_tokens,
                gen_tokens=token_count,
                tok_s=decode_tokens_per_second,
                elapsed=elapsed,
                response_text=response_text,
                last_iteration_s=iteration_timing["last_iteration_s"],
                best_iteration_s=iteration_timing["best_iteration_s"],
            )

            self.clear_cache_fn()
            self.collect_garbage_fn()
            return strip_thinking_tags(response_text)
        except Exception as error_value:
            return f"ERROR: {error_value}"
        finally:
            prefill_heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)

    def _stream_response(self, generation_kwargs: dict[str, Any], prompt_tokens: int, start_time: float) -> tuple[str, int, float | None]:
        stream_path = self.logger.run_dir / "stream.txt"
        response_parts: list[str] = []
        token_count = 0
        first_token_time = None
        last_stream_write = start_time
        last_perf_write = start_time

        try:
            for chunk in self.stream_generate(self.model, self.tokenizer, **generation_kwargs):
                token_count += 1
                response_parts.append(chunk.text)
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start_time

                current_time = time.perf_counter()
                elapsed_so_far = current_time - start_time
                if current_time - last_stream_write > 1.0:
                    self._write_text(stream_path, "".join(response_parts))
                    last_stream_write = current_time
                if current_time - last_perf_write > 1.0:
                    self.status_writer.write_generation_stats(
                        self._live_generation_stats(
                            prompt_tokens=prompt_tokens,
                            token_count=token_count,
                            elapsed_so_far=elapsed_so_far,
                            first_token_time=first_token_time,
                        )
                    )
                    last_perf_write = current_time

            response_text = "".join(response_parts)
            self._write_text(stream_path, response_text)
            history_path = self.logger.run_dir / "stream_history.txt"
            self._append_text(
                history_path,
                f"\n--- Step {len(self.perf.get('step_times', [])) + 1} ---\n{response_text}\n",
            )
            return response_text, token_count, first_token_time
        except Exception:
            fallback_response = self.generate_fn(self.model, self.tokenizer, **generation_kwargs)
            return fallback_response, token_count, first_token_time

    def _live_generation_stats(
        self,
        *,
        prompt_tokens: int,
        token_count: int,
        elapsed_so_far: float,
        first_token_time: float | None,
    ) -> dict[str, Any]:
        decode_time = max(0.01, elapsed_so_far - (first_token_time or 0))
        decode_tokens_per_second = (token_count - 1) / decode_time if token_count > 1 else 0
        live_tokens_per_second = token_count / max(0.01, elapsed_so_far)
        out = {
            "gen_tok_s": round(live_tokens_per_second, 1),
            "decode_tok_s": round(decode_tokens_per_second, 1),
            "peak_tok_s": round(max(self.perf.get("peak_tok_s", 0), decode_tokens_per_second), 1),
            "prompt_tokens": prompt_tokens,
            "gen_tokens": token_count,
            "total_gen_tokens": self.perf["total_tokens"] + token_count,
            "total_prompt_tokens": self.perf.get("prompt_tokens", 0) + prompt_tokens,
            "total_all_tokens": self.perf["total_tokens"] + token_count + self.perf.get("prompt_tokens", 0) + prompt_tokens,
            "elapsed": round(elapsed_so_far, 1),
            "total_time": round(self.perf["total_gen_time"] + elapsed_so_far, 1),
            "step": len(self.perf.get("step_times", [])) + 1,
            "context_used": prompt_tokens + token_count,
            "context_window": self.config_model.context_window,
            "context_pct": round((prompt_tokens + token_count) / self.config_model.context_window * 100, 1),
            "model": self.config_model.name.split("/")[-1],
            "model_size_gb": round(self.model_size_gb, 1),
            "max_tokens": self.config_model.max_tokens,
            "configured_max_tokens": self.config_model.max_tokens,
            "effective_max_tokens": self._last_effective_max_tokens,
            "prefill_time_s": round(first_token_time or 0, 2),
            "bandwidth_used_gbs": round(self.model_size_gb * decode_tokens_per_second, 1),
            "generating": True,
        }
        out.update(self._iteration_timing_live(elapsed_so_far))
        total_steps_live = len(self.perf.get("step_times", []))
        tool_total_live = self.perf["tool_success"]["total"]
        tool_ok_live = self.perf["tool_success"]["success"]
        out["tool_calls"] = tool_total_live
        out["tool_success"] = tool_ok_live
        out["tool_success_rate"] = round(tool_ok_live / max(1, tool_total_live) * 100, 0)
        out["avg_step_time"] = (
            round(sum(self.perf["step_times"]) / max(1, total_steps_live), 1) if total_steps_live else 0.0
        )
        out["total_time"] = round(self.perf["total_gen_time"] + elapsed_so_far, 1)
        if first_token_time is not None and first_token_time > 0:
            out["prefill_time_s"] = round(first_token_time, 2)
            out["prefill_tok_s"] = round(prompt_tokens / max(0.01, first_token_time), 0)
            out["decode_time_s"] = round(max(0.01, elapsed_so_far - first_token_time), 2)
        return out

    def _final_generation_stats(
        self,
        *,
        prompt_tokens: int,
        gen_tokens: int,
        elapsed: float,
        average_tokens_per_second: float,
        gen_tokens_per_second: float,
        decode_tokens_per_second: float,
        actual_prefill: float,
        decode_time: float,
    ) -> dict[str, Any]:
        total_steps = len(self.perf["step_times"])
        tool_total = self.perf["tool_success"]["total"]
        tool_ok = self.perf["tool_success"]["success"]
        context_used = prompt_tokens + gen_tokens
        context_pct = (context_used / self.config_model.context_window) * 100
        return {
            "gen_tok_s": round(gen_tokens_per_second, 1),
            "decode_tok_s": round(decode_tokens_per_second, 1),
            "prefill_time_s": round(actual_prefill, 2),
            "prefill_tok_s": round(prompt_tokens / max(0.01, actual_prefill), 0),
            "decode_time_s": round(decode_time, 2),
            "avg_tok_s": round(average_tokens_per_second, 1),
            "peak_tok_s": round(max(self.perf.get("peak_tok_s", 0), decode_tokens_per_second), 1),
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "total_gen_tokens": self.perf["total_tokens"],
            "total_prompt_tokens": self.perf.get("prompt_tokens", 0),
            "total_all_tokens": self.perf["total_tokens"] + self.perf.get("prompt_tokens", 0),
            "elapsed": round(elapsed, 2),
            "total_time": round(self.perf["total_gen_time"], 1),
            "step": total_steps,
            "context_used": context_used,
            "context_window": self.config_model.context_window,
            "context_pct": round(context_pct, 1),
            "model": self.config_model.name.split("/")[-1],
            "model_size_gb": round(self.model_size_gb, 1),
            "max_tokens": self.config_model.max_tokens,
            "configured_max_tokens": self.config_model.max_tokens,
            "effective_max_tokens": self._last_effective_max_tokens,
            "tool_calls": tool_total,
            "tool_success": tool_ok,
            "tool_success_rate": round(tool_ok / max(1, tool_total) * 100, 0),
            "avg_step_time": round(sum(self.perf["step_times"]) / max(1, total_steps), 1),
            "bandwidth_used_gbs": round(self.model_size_gb * decode_tokens_per_second, 1),
            "generating": False,
        }

    def _write_text(self, target_path: Path, text: str) -> None:
        try:
            target_path.write_text(text)
        except Exception:
            pass

    def _append_text(self, target_path: Path, text: str) -> None:
        try:
            with open(target_path, "a") as handle:
                handle.write(text)
        except Exception:
            pass
