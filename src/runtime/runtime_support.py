"""Shared runtime support utilities used by the MLX bootstrap."""

from __future__ import annotations

import ast
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable


def resource_sampler(run_dir: Path, stop_event: threading.Event) -> None:
    """Sample system resources into the run directory for monitoring."""
    log_path = run_dir / "resources.jsonl"
    try:
        import psutil
    except ImportError:
        return

    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log_handle:
        while not stop_event.is_set():
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                gpu_status = "idle"
                perf_path = run_dir / "perf.json"
                if perf_path.exists():
                    try:
                        perf_payload = json.loads(perf_path.read_text())
                        gpu_status = perf_payload.get("status", "idle")
                    except Exception:
                        gpu_status = "idle"
                entry_payload = json.dumps(
                    {
                        "t": round(time.time(), 2),
                        "cpu": cpu_percent,
                        "mem_gb": round(memory_info.used / 1e9, 1),
                        "free_gb": round(memory_info.available / 1e9, 1),
                        "gpu": gpu_status,
                        "rss_mb": round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 0),
                    }
                )
                log_handle.write(entry_payload + "\n")
                log_handle.flush()
            except Exception:
                pass
            stop_event.wait(0.5)


class IdleScheduler:
    """Runs queued work during CPU-bound tool windows."""

    def __init__(self) -> None:
        self._queue: list[tuple[str, Callable[[], Any]]] = []
        self._results: dict[str, Any] = {}

    def enqueue(self, name: str, fn: Callable[[], Any]) -> None:
        self._queue.append((name, fn))

    def run_pending(self, max_time: float = 3.0) -> None:
        deadline = time.time() + max_time
        while self._queue and time.time() < deadline:
            name, fn = self._queue.pop(0)
            try:
                self._results[name] = fn()
            except Exception as error_value:
                self._results[name] = f"ERROR: {error_value}"

    def get_result(self, name: str) -> Any:
        return self._results.pop(name, None)


class PerfStatusWriter:
    """Write monitor-facing performance status payloads for a run."""

    def __init__(
        self,
        run_dir: Path,
        model_name: str,
        context_window: int,
        max_tokens: int,
        model_size_gb: float,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.model_name = model_name
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.model_size_gb = model_size_gb

    def _read_previous_perf_payload(self) -> dict[str, Any]:
        perf_path = self.run_dir / "perf.json"
        if not perf_path.exists():
            return {}
        try:
            return json.loads(perf_path.read_text())
        except Exception:
            return {}

    def write_status(self, status: str, generating: bool, perf: dict[str, Any], **live_metrics: Any) -> None:
        """Write lightweight live status for the monitor.

        Pass ``prompt_tokens=`` / ``gen_tokens=`` / ``decode_tok_s=`` during prefill so perf.json
        is not overwritten without per-step fields (monitor otherwise shows 0).
        """
        payload: dict[str, Any] = {
            "status": status,
            "generating": generating,
            "model": self.model_name.split("/")[-1],
            "context_window": self.context_window,
            "max_tokens": self.max_tokens,
            "model_size_gb": round(self.model_size_gb, 1),
            "step": len(perf.get("step_times", [])) + 1,
            "total_gen_tokens": perf.get("total_tokens", 0),
            "total_prompt_tokens": perf.get("prompt_tokens", 0),
            "peak_tok_s": perf.get("peak_tok_s", 0),
            "timestamp": time.time(),
        }
        previous_payload = self._read_previous_perf_payload()
        carry_keys = (
            "prompt_tokens",
            "gen_tokens",
            "decode_tok_s",
            "gen_tok_s",
            "context_pct",
            "this_iteration_s",
            "last_iteration_s",
            "best_iteration_s",
            "prefill_time_s",
            "prefill_tok_s",
            "decode_time_s",
            "avg_tok_s",
            "total_gen_tokens",
            "total_prompt_tokens",
            "total_time",
            "avg_step_time",
            "tool_calls",
            "tool_success",
            "tool_success_rate",
            "context_used",
            "elapsed",
            "bandwidth_used_gbs",
            "effective_max_tokens",
            "configured_max_tokens",
        )
        for key_name in carry_keys:
            if key_name in live_metrics:
                payload[key_name] = live_metrics[key_name]
            elif key_name in previous_payload:
                payload[key_name] = previous_payload[key_name]
        if "prompt_tokens" in live_metrics and self.context_window:
            try:
                payload["context_pct"] = round(
                    (int(live_metrics["prompt_tokens"]) / self.context_window) * 100,
                    1,
                )
            except (TypeError, ValueError):
                pass
        elif "context_pct" in live_metrics:
            payload["context_pct"] = live_metrics["context_pct"]
        elif "prompt_tokens" in payload and self.context_window and "context_pct" not in payload:
            try:
                payload["context_pct"] = round(
                    (int(payload["prompt_tokens"]) / self.context_window) * 100,
                    1,
                )
            except (TypeError, ValueError):
                pass
        for key_name, value in live_metrics.items():
            if key_name not in payload:
                payload[key_name] = value
        try:
            prompt_tok = payload.get("prompt_tokens")
            if prompt_tok is not None:
                pt = int(prompt_tok)
                if pt > 0:
                    from src.runtime.mlx_adapter import _metal_safe_max_new_tokens

                    payload["effective_max_tokens"] = _metal_safe_max_new_tokens(
                        pt, self.max_tokens, self.context_window
                    )
                    payload["configured_max_tokens"] = self.max_tokens
        except (TypeError, ValueError, ImportError):
            pass
        self._write_payload(payload)

    def write_generation_stats(self, payload: dict[str, Any]) -> None:
        """Write detailed generation-time metrics."""
        self._write_payload(payload)

    def _write_payload(self, payload: dict[str, Any]) -> None:
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with open(self.run_dir / "perf.json", "w") as perf_handle:
                json.dump(payload, perf_handle)
        except Exception:
            pass


def pre_validate_candidate_file(path_value: str, output_dir: Path) -> str:
    """Quick structural validation for newly written Python files."""
    try:
        candidate_path = Path(path_value)
        if not candidate_path.exists():
            candidate_path = Path(output_dir) / path_value
        if not candidate_path.exists():
            return "OK"
        source_text = candidate_path.read_text()
        parsed_tree = ast.parse(source_text)
        function_count = sum(
            1 for node_value in ast.walk(parsed_tree)
            if isinstance(node_value, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        assertion_count = source_text.count("assert ")
        warnings: list[str] = []
        if function_count < 3:
            warnings.append(f"only {function_count} functions (need >=3)")
        if assertion_count < 5:
            warnings.append(f"only {assertion_count} asserts (need >=5)")
        if "from src." in source_text:
            warnings.append("contains 'from src.' imports (forbidden in skills)")
        if warnings:
            return "WARN: " + "; ".join(warnings)
        return "OK"
    except SyntaxError as error_value:
        return f"WARN: syntax error at line {error_value.lineno}: {error_value.msg}"
    except Exception:
        return "OK"


def evaluate_with_self_evaluator(path_value: Path, load_skill_instance: Callable[[str, str], Any]) -> dict[str, Any]:
    """Evaluate a written file through the self-evaluator skill when available."""
    evaluator = load_skill_instance("self_evaluator", "SelfEvaluator")
    if not evaluator:
        return {}
    try:
        return evaluator.evaluate_file(str(path_value))
    except Exception as error_value:
        return {"status": "error", "recommendation": str(error_value), "score": 0.0}


def record_strategy_outcome(
    *,
    load_skill_instance: Callable[[str, str], Any],
    config_model_name: str,
    goal: str,
    phase: str,
    success: bool,
    perf: dict[str, Any],
    files_written: int,
) -> None:
    """Persist strategy-learner outcomes without local legacy side stores."""
    learner = load_skill_instance("strategy_learner", "StrategyLearner")
    if not learner:
        return
    metrics_map = {
        "tool_success_rate": round(
            perf["tool_success"]["success"] / max(1, perf["tool_success"]["total"]),
            4,
        ),
        "avg_tok_s": round(
            perf["total_tokens"] / max(0.01, perf["total_gen_time"]),
            4,
        ),
        "files_written": float(files_written),
    }
    learner.record_outcome(f"{config_model_name}:{phase}", goal, success, metrics_map)
