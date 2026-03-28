"""MLX generation adapter extracted from the bootstrap agent."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Callable

from src.runtime.llm_text import strip_thinking_tags


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

    def generate_response(
        self,
        messages: list[dict[str, Any]],
        format_prompt: Callable[[list[dict[str, Any]]], str],
    ) -> str:
        """Generate a response with streaming updates and perf accounting."""
        self.clear_cache_fn()
        self.collect_garbage_fn()

        prompt = format_prompt(messages)
        stable_prefix = "\n".join(message.get("content", "") for message in messages[:2])
        self.kv_cache_manager.ensure_prefix(stable_prefix)
        prompt_tokens = len(self.tokenizer.encode(prompt)) if hasattr(self.tokenizer, "encode") else len(prompt) // 4
        self.status_writer.write_status(
            status=f"PREFILLING {prompt_tokens} tokens...",
            generating=True,
            perf=self.perf,
        )

        try:
            start_time = time.perf_counter()
            generation_kwargs = {
                "prompt": prompt,
                "max_tokens": self.config_model.max_tokens,
                "sampler": self.sampler,
                "prefill_step_size": 4096,
            }
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
            self.perf["peak_tok_s"] = final_stats["peak_tok_s"]
            self.status_writer.write_generation_stats(final_stats)

            print(
                f"  ⚡ {decode_tokens_per_second:.0f} tok/s decode "
                f"({gen_tokens_per_second:.0f} overall) | {prompt_tokens}p+{token_count}g | {elapsed:.1f}s"
            )
            self.logger.generation(
                step=len(self.perf["step_times"]),
                prompt_tokens=prompt_tokens,
                gen_tokens=token_count,
                tok_s=decode_tokens_per_second,
                elapsed=elapsed,
                response_preview=response_text[:200],
            )

            self.clear_cache_fn()
            self.collect_garbage_fn()
            return strip_thinking_tags(response_text)
        except Exception as error_value:
            return f"ERROR: {error_value}"

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
        return {
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
            "max_tokens": self.config_model.max_tokens,
            "prefill_time_s": round(first_token_time or 0, 2),
            "bandwidth_used_gbs": round(self.model_size_gb * decode_tokens_per_second, 1),
            "generating": True,
        }

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
