"""Structured logging for every agent decision, tool call, and result."""

import json
import time
from pathlib import Path
from datetime import datetime


from src.paths import RUNS_DIR, get_run_dir


class AgentLogger:
    """Logs every event in a per-run directory."""

    def __init__(self, run_id: str = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = get_run_dir(self.run_id)
        self.log_file = self.run_dir / "events.jsonl"
        self.start_time = time.time()
        self._event_num = 0

    def _write(self, event: dict):
        self._event_num += 1
        event["event_num"] = self._event_num
        event["timestamp"] = datetime.now().isoformat()
        event["elapsed_s"] = round(time.time() - self.start_time, 2)
        event["run_id"] = self.run_id
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

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

    def generation(self, step: int, prompt_tokens: int, gen_tokens: int,
                   tok_s: float, elapsed: float, response_preview: str):
        self._write({
            "type": "generation",
            "step": step,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "tok_s": round(tok_s, 1),
            "elapsed_s": round(elapsed, 2),
            "response_preview": response_preview[:300],
        })

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
