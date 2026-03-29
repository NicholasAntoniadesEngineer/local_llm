"""Structured logging for every agent decision, tool call, and result.

``generation`` events include ``response_text`` (full raw model output for that step).
Optional ``AGENT_LOG_RESPONSE_MAX_CHARS`` (positive int) truncates stored text and appends a marker.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import TextIO


from src.paths import RUNS_DIR, get_run_dir


def _agent_log_response_char_limit() -> int | None:
    raw = os.environ.get("AGENT_LOG_RESPONSE_MAX_CHARS", "").strip()
    if not raw:
        return None
    try:
        parsed_limit = int(raw)
    except ValueError:
        return None
    return parsed_limit if parsed_limit > 0 else None


class AgentLogger:
    """Logs every event in a per-run directory."""

    def __init__(self, run_id: str = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = get_run_dir(self.run_id)
        self.log_file = self.run_dir / "events.jsonl"
        self.start_time = time.time()
        self._event_num = 0
        self._events_jsonl_append_handle: TextIO | None = None
        try:
            (self.run_dir / "run_id.txt").write_text(self.run_id)
        except Exception:
            pass

    def _ensure_run_dir(self) -> None:
        """Guarantee the run directory exists before any append/write (handles races or removed dirs)."""
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, event: dict):
        self._ensure_run_dir()
        self._event_num += 1
        event["event_num"] = self._event_num
        event["timestamp"] = datetime.now().isoformat()
        event["elapsed_s"] = round(time.time() - self.start_time, 2)
        event["run_id"] = self.run_id
        if self._events_jsonl_append_handle is None:
            self._events_jsonl_append_handle = open(self.log_file, "a", encoding="utf-8")
        self._events_jsonl_append_handle.write(json.dumps(event, default=str) + "\n")
        self._events_jsonl_append_handle.flush()

    def close(self) -> None:
        """Flush and close the events.jsonl append handle (idempotent)."""
        if self._events_jsonl_append_handle is not None:
            try:
                self._events_jsonl_append_handle.flush()
                self._events_jsonl_append_handle.close()
            except OSError:
                pass
            self._events_jsonl_append_handle = None

    def run_start(self, goal: str, model: str, config: dict):
        import subprocess, platform
        # Capture system specs
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=3).stdout.strip()
        except Exception:
            chip = "unknown"
        try:
            ram_bytes = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                           capture_output=True, text=True, timeout=3).stdout.strip())
            ram_gb = ram_bytes / 1073741824
        except Exception:
            ram_gb = 0
        try:
            cores = int(subprocess.run(["sysctl", "-n", "hw.ncpu"],
                                        capture_output=True, text=True, timeout=3).stdout.strip())
        except Exception:
            cores = 0

        self._write({
            "type": "run_start",
            "goal": goal,
            "model": model,
            "config": config,
            "system": {
                "chip": chip,
                "ram_gb": ram_gb,
                "cores": cores,
                "os": platform.platform(),
                "python": platform.python_version(),
                "bandwidth_gbs": 546 if "Max" in chip else 273,
            },
        })

    def step_start(self, step: int, phase: str, prompt_tokens: int, messages_count: int):
        self._write({
            "type": "step_start",
            "step": step,
            "phase": phase,
            "prompt_tokens": prompt_tokens,
            "messages_in_context": messages_count,
        })

    def generation(
        self,
        step: int,
        prompt_tokens: int,
        gen_tokens: int,
        tok_s: float,
        elapsed: float,
        response_text: str,
        *,
        last_iteration_s: float | None = None,
        best_iteration_s: float | None = None,
    ):
        char_limit = _agent_log_response_char_limit()
        if char_limit is not None and len(response_text) > char_limit:
            stored_response = (
                response_text[:char_limit] + "\n...[truncated: AGENT_LOG_RESPONSE_MAX_CHARS]"
            )
        else:
            stored_response = response_text
        self._write(
            {
                "type": "generation",
                "step": step,
                "prompt_tokens": prompt_tokens,
                "gen_tokens": gen_tokens,
                "tok_s": round(tok_s, 1),
                "elapsed_s": round(elapsed, 2),
                "this_iteration_s": round(elapsed, 2),
                "last_iteration_s": last_iteration_s,
                "best_iteration_s": best_iteration_s,
                "response_preview": response_text[:300],
                "response_text": stored_response,
            }
        )

    def tool_call(self, step: int, tool: str, args: dict):
        self._write({
            "type": "tool_call",
            "step": step,
            "tool": tool,
            "args_preview": {k: str(v)[:100] for k, v in args.items()},
        })

    def tool_result(self, step: int, tool: str, success: bool, result_preview: str):
        self._write({
            "type": "tool_result",
            "step": step,
            "tool": tool,
            "success": success,
            "result_preview": result_preview[:300],
        })

    def decision(self, step: int, decision: str, reason: str):
        self._write({
            "type": "decision",
            "step": step,
            "decision": decision,
            "reason": reason,
        })

    def phase_change(self, step: int, old_phase: str, new_phase: str, reason: str):
        self._write({
            "type": "phase_change",
            "step": step,
            "from": old_phase,
            "to": new_phase,
            "reason": reason,
        })

    def loop_detected(self, step: int, tool: str, times: int):
        self._write({
            "type": "loop_detected",
            "step": step,
            "tool": tool,
            "consecutive_times": times,
        })

    def error(self, step: int, error: str):
        self._write({
            "type": "error",
            "step": step,
            "error": error,
        })

    def run_end(self, steps: int, perf: dict):
        total_tokens = perf.get("total_tokens", 0)
        total_time = perf.get("total_gen_time", 0.01)
        prompt_tokens = perf.get("prompt_tokens", 0)
        tool_total = perf["tool_success"]["total"]
        tool_ok = perf["tool_success"]["success"]
        self._write({
            "type": "run_end",
            "total_steps": steps,
            "summary": {
                "avg_tok_s": round(total_tokens / max(0.01, total_time), 1),
                "peak_tok_s": perf.get("peak_tok_s", 0),
                "total_gen_tokens": total_tokens,
                "total_prompt_tokens": prompt_tokens,
                "total_all_tokens": total_tokens + prompt_tokens,
                "total_gen_time_s": round(total_time, 1),
                "avg_step_time_s": round(total_time / max(1, steps), 1),
                "tool_calls": tool_total,
                "tool_success_rate": round(tool_ok / max(1, tool_total) * 100, 0),
            },
        })

    def validation(self, file: str, passed: bool, reason: str):
        self._write({
            "type": "validation",
            "file": file,
            "passed": passed,
            "reason": reason,
        })

    def write_summary(self, perf: dict, result: dict):
        """Write a human-readable run_summary.json at end of run."""
        total_tokens = perf.get("total_tokens", 0)
        total_time = perf.get("total_gen_time", 0.01)
        prompt_tokens = perf.get("prompt_tokens", 0)
        tool_total = perf["tool_success"]["total"]
        tool_ok = perf["tool_success"]["success"]
        steps = len(perf.get("step_times", []))

        # GPU utilization from resources.jsonl
        gpu_busy_pct = 0
        try:
            res_file = self.run_dir / "resources.jsonl"
            if res_file.exists():
                lines = res_file.read_text().strip().split("\n")
                total_samples = len(lines)
                busy = sum(1 for l in lines if '"GENERATING"' in l or '"PREFILL' in l)
                gpu_busy_pct = round(busy / max(1, total_samples) * 100)
        except Exception:
            pass

        summary = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "started": datetime.fromtimestamp(self.start_time).isoformat(),
            "duration_s": round(time.time() - self.start_time, 1),
            "steps": steps,
            "result": result,
            "tokens": {
                "generated": total_tokens,
                "prompt": prompt_tokens,
                "total": total_tokens + prompt_tokens,
            },
            "performance": {
                "avg_tok_s": round(total_tokens / max(0.01, total_time), 1),
                "peak_tok_s": perf.get("peak_tok_s", 0),
                "avg_step_s": round(total_time / max(1, steps), 1),
                "gpu_busy_pct": gpu_busy_pct,
            },
            "tools": {
                "total_calls": tool_total,
                "success_rate": round(tool_ok / max(1, tool_total) * 100, 0),
            },
        }
        self._ensure_run_dir()
        with open(self.run_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
