#!/usr/bin/env python3
"""
MLX-powered autonomous agent for Apple Silicon.

Uses Qwen3's NATIVE Hermes tool-calling format (<tool_call>) and
tokenizer.apply_chat_template() for reliable tool extraction.

Performance: ~60 tok/s on M4 Max with Qwen3-14B-4bit.
"""

import json
import subprocess
import re
import os
import sys
import gc
import threading
import time as _time_mod
from pathlib import Path
from typing import Optional

import mlx.core as mx

from src.config import CONFIG
from src.memory import MemoryManager
from src.logger import AgentLogger
from src.skill_tree import SkillTree


# ═══════════════════════════════════════════════════════════════════════
# Resource sampler — background thread logging CPU/mem/GPU status
# ═══════════════════════════════════════════════════════════════════════

def _resource_sampler(run_dir: Path, stop_event: threading.Event):
    """Sample system resources every 0.5s into resources.jsonl."""
    log_path = run_dir / "resources.jsonl"
    try:
        import psutil
    except ImportError:
        return  # psutil not available, skip sampling

    with open(log_path, "a") as log:
        while not stop_event.is_set():
            try:
                cpu = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                gpu_status = "idle"
                perf_file = run_dir / "perf.json"
                if perf_file.exists():
                    try:
                        data = json.loads(perf_file.read_text())
                        gpu_status = data.get("status", "idle")
                    except Exception:
                        pass
                entry = json.dumps({
                    "t": round(_time_mod.time(), 2),
                    "cpu": cpu,
                    "mem_gb": round(mem.used / 1e9, 1),
                    "free_gb": round(mem.available / 1e9, 1),
                    "gpu": gpu_status,
                    "rss_mb": round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 0),
                })
                log.write(entry + "\n")
                log.flush()
            except Exception:
                pass
            stop_event.wait(0.5)


# ═══════════════════════════════════════════════════════════════════════
# Idle scheduler — runs background tasks during GPU-idle windows
# ═══════════════════════════════════════════════════════════════════════

class IdleScheduler:
    """Runs queued tasks during tool execution (when GPU is idle)."""

    def __init__(self):
        self._queue: list[tuple[str, callable]] = []
        self._results: dict[str, any] = {}

    def enqueue(self, name: str, fn: callable):
        self._queue.append((name, fn))

    def run_pending(self, max_time: float = 3.0):
        """Run queued tasks up to max_time seconds."""
        deadline = _time_mod.time() + max_time
        while self._queue and _time_mod.time() < deadline:
            name, fn = self._queue.pop(0)
            try:
                self._results[name] = fn()
            except Exception as e:
                self._results[name] = f"ERROR: {e}"

    def get_result(self, name: str):
        return self._results.pop(name, None)


# Tool definitions in OpenAI-compatible format (what Qwen3 was trained on)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using DuckDuckGo. Returns titles, URLs, and snippets. "
                "Use for: finding API docs, understanding libraries, researching algorithms. "
                "Tips: Use specific technical queries like 'python difflib SequenceMatcher tutorial' "
                "not vague ones like 'how to compare strings'. If results are poor, try different keywords. "
                "Results are cached — identical queries return cached results instantly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Specific technical search query (5-10 words)"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code in a subprocess and return stdout/stderr. "
                "The working directory is the project root. PYTHONPATH includes both '.' and 'skills/'. "
                "Use for: testing code, running assertions, validating imports. "
                "CRITICAL: If a test fails, READ the error traceback carefully. Do NOT rerun identical code. "
                "Fix the specific line that failed, then test again. "
                "Timeout: 30 seconds. For long operations, break into smaller steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Complete Python code to execute. Must be self-contained."}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the full contents of a file. Use to understand existing code before modifying it. "
                "Paths are relative to project root: 'src/agent.py', 'skills/metrics.py', etc. "
                "Always read a file before writing a replacement — understand the existing code first. "
                "For large files, consider using grep_file to find specific sections instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path from project root"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write COMPLETE file content. This OVERWRITES the entire file. "
                "For skill modules in skills/: "
                "1. Include ALL imports at the top. Use 'from module_name import ClassName' for prereqs. "
                "   NEVER use 'from src.memory import' or 'from src.config import' — skills run standalone. "
                "2. Write COMPLETE class with all methods. Each method must have real logic (10+ lines), "
                "   not single-line returns or forwarding wrappers. "
                "3. Include 'if __name__ == \"__main__\":' test block with 5+ assert statements. "
                "4. Tests must print 'ALL TESTS PASSED' only if every assertion succeeds. "
                "5. Handle edge cases: empty input, None values, boundary conditions. "
                "For small changes to existing files, prefer edit_file instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path. For skills use just the filename like 'metrics.py'"},
                    "content": {"type": "string", "description": "Complete file content including all imports, classes, and test block"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Make a targeted edit to an existing file by replacing a specific string. "
                "Finds old_content in the file and replaces it with new_content. "
                "Use instead of write_file when you only need to change a few lines. "
                "The old_content must match EXACTLY (including whitespace). "
                "If old_content is not found, the edit fails with an error."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_content": {"type": "string", "description": "Exact string to find and replace"},
                    "new_content": {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old_content", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_file",
            "description": (
                "Search for a pattern in files. Returns matching lines with line numbers. "
                "Use to find specific code patterns, function definitions, imports, or usages "
                "without reading entire files. Much cheaper than read_file for large files. "
                "Searches recursively in directories. Pattern is a Python regex."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "File or directory to search in. Default: 'skills/'"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List files in a directory with sizes. Returns compact output: 'filename (size)' per line. "
                "Use to understand project structure before reading specific files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path. Default: '.'"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a shell command and return output. Use for: git operations, "
                "installing packages, running system commands. "
                "Timeout: 30 seconds. Avoid long-running commands."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to execute"}
                },
                "required": ["cmd"],
            },
        },
    },
]

# Regex for Qwen3's native Hermes tool call format
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Fallback regex for the old format (in case model uses it)
TOOL_CALL_FALLBACK_RE = re.compile(r"<tool>(\w+)</tool>\s*<args>(\{.*?\})</args>", re.DOTALL)


class MLXAgent:
    """MLX-powered agent using Qwen3's native tool-calling format."""

    def __init__(self, config_model_name: str = "balanced", goal: str = "") -> None:
        if config_model_name not in CONFIG.models:
            raise KeyError(f"Model '{config_model_name}' not in CONFIG.models")

        self.config_model = CONFIG.models[config_model_name]
        self.memory_manager = MemoryManager(goal) if goal else None
        self.logger = AgentLogger()
        self.skill_tree = SkillTree()
        self._search_cache: dict[str, str] = {}
        self._idle_scheduler = IdleScheduler()
        self._search_quality: float = 0.5

        # Performance tracking
        self._perf: dict = {
            "total_tokens": 0,
            "total_gen_time": 0.0,
            "step_times": [],
            "tool_success": {"total": 0, "success": 0},
        }

        CONFIG.output_dir.mkdir(parents=True, exist_ok=True)

        # Load MLX model + tokenizer with optimizations
        try:
            from mlx_lm import load, generate, stream_generate
            from mlx_lm.sample_utils import make_sampler

            self._generate_fn = generate
            self._stream_generate = stream_generate
            self.model, self.tokenizer = load(self.config_model.name)

            # OPTIMIZATION: greedy sampler (fastest - single argmax, no sampling)
            self._sampler = make_sampler(temp=0.0)

            # OPTIMIZATION: prompt cache for KV reuse across turns
            try:
                from mlx_lm.models.cache import make_prompt_cache
                self._prompt_cache = make_prompt_cache(self.model)
            except Exception:
                self._prompt_cache = None

            self._draft_model = None

            # Compute actual model size for accurate bandwidth reporting
            try:
                total_bytes = sum(v.nbytes for _, v in mx.utils.tree_flatten(self.model.parameters()))
                self._model_size_gb = total_bytes / 1e9
            except Exception:
                self._model_size_gb = 8.0  # fallback

            print(f"✅ Loaded {self.config_model.name} via MLX ({self._model_size_gb:.1f} GB)")
            print(f"📁 Output folder: {CONFIG.output_dir.resolve()}")
            print(f"📊 Context: {self.config_model.context_window} tokens")
            if self._prompt_cache:
                print(f"⚡ KV cache enabled")
        except ImportError:
            print("❌ MLX not installed. Run: pip install mlx-lm")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)

    def reset_for_new_task(self, goal: str):
        """Reset session state for a new task WITHOUT reloading the model.

        Keeps: model, tokenizer, sampler, prompt_cache (expensive to create)
        Resets: memory, perf counters, logger, goal
        """
        self.goal = goal
        self.memory_manager = MemoryManager(goal)
        self.logger = AgentLogger(goal)
        self._perf = {
            "total_tokens": 0,
            "total_gen_time": 0.0,
            "step_times": [],
            "tool_success": {"total": 0, "success": 0},
            "peak_tok_s": 0.0,
        }
        self._search_cache = {}
        # Recreate prompt cache for fresh KV state
        try:
            from mlx_lm.models.cache import make_prompt_cache
            self._prompt_cache = make_prompt_cache(self.model)
        except Exception:
            pass
        mx.clear_cache()
        gc.collect()

    def _format_prompt(self, messages: list[dict], include_tools: bool = True) -> str:
        """Format messages using tokenizer's native chat template.

        This is the KEY improvement - Qwen3 was trained with apply_chat_template,
        not manual role:content formatting.
        """
        try:
            tools_arg = TOOLS if include_tools else None
            return self.tokenizer.apply_chat_template(
                messages,
                tools=tools_arg,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback if template doesn't support tools parameter
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _write_status(self, status: str, generating: bool = False):
        """Write current status to perf.json so the monitor always has fresh data."""
        try:
            import time as _t
            stats = {
                "status": status,
                "generating": generating,
                "model": self.config_model.name.split("/")[-1],
                "context_window": self.config_model.context_window,
                "max_tokens": self.config_model.max_tokens,
                "model_size_gb": round(self._model_size_gb, 1),
                "step": len(self._perf["step_times"]) + 1,
                "total_gen_tokens": self._perf.get("total_tokens", 0),
                "total_prompt_tokens": self._perf.get("prompt_tokens", 0),
                "peak_tok_s": self._perf.get("peak_tok_s", 0),
                "timestamp": _t.time(),
            }
            with open(self.logger.run_dir / "perf.json", "w") as f:
                json.dump(stats, f)
        except Exception:
            pass

    def _count_tokens(self, messages: list[dict]) -> int:
        """Count actual tokens using the tokenizer (not estimates)."""
        try:
            prompt = self._format_prompt(messages)
            return len(self.tokenizer.encode(prompt))
        except Exception:
            return sum(len(m.get("content", "")) for m in messages) // 4

    def _compress_context(self, messages: list[dict]) -> list[dict]:
        """Tiered context compression to prevent Metal GPU OOM.

        Thresholds based on context_window:
        - 40%: compress tool results to 500 chars
        - 60%: collapse old exchanges to summaries
        - 80%: keep only system + last 2 exchanges
        """
        budget = self.config_model.context_window - self.config_model.max_tokens - 1024
        tokens = self._count_tokens(messages)
        fill_pct = tokens / budget if budget > 0 else 1.0

        if fill_pct < 0.4 or len(messages) <= 4:
            return messages

        # 40-60%: compress tool results in older messages
        if fill_pct < 0.6:
            for i in range(1, len(messages) - 4):
                content = messages[i].get("content", "")
                if messages[i].get("role") == "tool" and len(content) > 500:
                    messages[i]["content"] = content[:250] + "\n...[compressed]...\n" + content[-250:]
            return messages

        # 60-80%: collapse all but last 3 exchanges into summary
        if fill_pct < 0.8:
            if len(messages) > 6:
                summary_parts = []
                for m in messages[1:-6]:
                    role = m.get("role", "?")
                    content = m.get("content", "")[:80]
                    summary_parts.append(f"[{role}] {content}")
                summary = "Previous context:\n" + "\n".join(summary_parts[-10:])
                messages = [messages[0], {"role": "user", "content": summary}] + messages[-6:]
            return messages

        # 80%+: aggressive - keep only system + last 2 exchanges
        if len(messages) > 4:
            messages = [messages[0]] + messages[-4:]
        return messages

    def _generate_response(self, messages: list[dict]) -> str:
        """Generate a response with all MLX optimizations + performance tracking."""
        import time as _time

        # Clear GPU memory BEFORE generation to prevent Metal OOM
        mx.clear_cache()
        gc.collect()

        prompt = self._format_prompt(messages)

        # Count prompt tokens (real count, not estimate)
        prompt_tokens = len(self.tokenizer.encode(prompt)) if hasattr(self.tokenizer, 'encode') else len(prompt) // 4

        # Write PREFILLING status so monitor shows activity during the silent prefill phase
        self._write_status(f"PREFILLING {prompt_tokens} tokens...", generating=True)

        try:
            t0 = _time.perf_counter()

            # Build kwargs with all optimizations for max tok/s
            kwargs = {
                "prompt": prompt,
                "max_tokens": self.config_model.max_tokens,
                "sampler": self._sampler,           # greedy argmax = fastest decode
                "prefill_step_size": 4096,          # 2x faster prompt processing vs default 512
            }
            if self._prompt_cache:
                kwargs["prompt_cache"] = self._prompt_cache
            if self._draft_model:
                kwargs["draft_model"] = self._draft_model

            # Stream tokens to file for real-time monitoring
            stream_file = self.logger.run_dir / "stream.txt"
            response_parts = []
            token_count = 0
            first_token_time = None
            last_stream_write = t0
            last_perf_write = t0

            try:
                for chunk in self._stream_generate(self.model, self.tokenizer, **kwargs):
                    token_count += 1
                    response_parts.append(chunk.text)

                    # Measure time-to-first-token (= actual prefill time)
                    if first_token_time is None:
                        first_token_time = _time.perf_counter() - t0

                    now = _time.perf_counter()
                    elapsed_so_far = now - t0

                    # Time-based stream write (~1s interval)
                    if now - last_stream_write > 1.0:
                        try:
                            with open(stream_file, "w") as sf:
                                sf.write("".join(response_parts))
                        except Exception:
                            pass
                        last_stream_write = now

                    # Perf stats (~1s interval for real-time monitoring)
                    if now - last_perf_write > 1.0:
                        decode_time = max(0.01, elapsed_so_far - (first_token_time or 0))
                        decode_tok_s = (token_count - 1) / decode_time if token_count > 1 else 0
                        live_tok_s = token_count / max(0.01, elapsed_so_far)
                        try:
                            live_stats = {
                                "gen_tok_s": round(live_tok_s, 1),
                                "decode_tok_s": round(decode_tok_s, 1),
                                "peak_tok_s": round(max(self._perf.get("peak_tok_s", 0), decode_tok_s), 1),
                                "prompt_tokens": prompt_tokens,
                                "gen_tokens": token_count,
                                "total_gen_tokens": self._perf["total_tokens"] + token_count,
                                "total_prompt_tokens": self._perf.get("prompt_tokens", 0) + prompt_tokens,
                                "total_all_tokens": self._perf["total_tokens"] + token_count + self._perf.get("prompt_tokens", 0) + prompt_tokens,
                                "elapsed": round(elapsed_so_far, 1),
                                "total_time": round(self._perf["total_gen_time"] + elapsed_so_far, 1),
                                "step": len(self._perf["step_times"]) + 1,
                                "context_used": prompt_tokens + token_count,
                                "context_window": self.config_model.context_window,
                                "context_pct": round((prompt_tokens + token_count) / self.config_model.context_window * 100, 1),
                                "model": self.config_model.name.split("/")[-1],
                                "max_tokens": self.config_model.max_tokens,
                                "prefill_time_s": round(first_token_time or 0, 2),
                                "bandwidth_used_gbs": round(self._model_size_gb * decode_tok_s, 1),
                                "generating": True,
                            }
                            with open(self.logger.run_dir / "perf.json", "w") as pf:
                                json.dump(live_stats, pf)
                        except Exception:
                            pass
                        last_perf_write = now
                response = "".join(response_parts)
                # Final write + append to full history
                try:
                    with open(stream_file, "w") as sf:
                        sf.write(response)
                    with open(self.logger.run_dir / "stream_history.txt", "a") as hf:
                        hf.write(f"\n--- Step {len(self._perf['step_times'])+1} ---\n")
                        hf.write(response)
                        hf.write("\n")
                except Exception:
                    pass
            except Exception:
                # Fallback to non-streaming
                response = self._generate_fn(self.model, self.tokenizer, **kwargs)

            elapsed = _time.perf_counter() - t0

            # Token counting - use stream token_count if available, else estimate
            gen_tokens = token_count if token_count > 0 else max(1, len(response) // 4)
            gen_tok_s = gen_tokens / max(0.01, elapsed)

            self._perf["total_tokens"] += gen_tokens
            self._perf["total_gen_time"] += elapsed
            self._perf["step_times"].append(elapsed)
            self._perf["prompt_tokens"] = self._perf.get("prompt_tokens", 0) + prompt_tokens

            avg_tok_s = self._perf["total_tokens"] / max(0.01, self._perf["total_gen_time"])

            # True decode speed using measured time-to-first-token
            actual_prefill = first_token_time if first_token_time else (prompt_tokens / 300.0)
            decode_time = max(0.01, elapsed - actual_prefill)
            true_decode_tok_s = max(0, token_count - 1) / decode_time if token_count > 1 else gen_tok_s

            # Write stats to file for monitor to read
            total_steps = len(self._perf["step_times"])
            tool_total = self._perf["tool_success"]["total"]
            tool_ok = self._perf["tool_success"]["success"]
            context_used = prompt_tokens + gen_tokens
            context_pct = (context_used / self.config_model.context_window) * 100

            stats = {
                "gen_tok_s": round(gen_tok_s, 1),
                "decode_tok_s": round(true_decode_tok_s, 1),
                "prefill_time_s": round(actual_prefill, 2),
                "prefill_tok_s": round(prompt_tokens / max(0.01, actual_prefill), 0),
                "decode_time_s": round(decode_time, 2),
                "avg_tok_s": round(avg_tok_s, 1),
                "peak_tok_s": round(max(self._perf.get("peak_tok_s", 0), true_decode_tok_s), 1),
                "prompt_tokens": prompt_tokens,
                "gen_tokens": gen_tokens,
                "total_gen_tokens": self._perf["total_tokens"],
                "total_prompt_tokens": self._perf.get("prompt_tokens", 0),
                "total_all_tokens": self._perf["total_tokens"] + self._perf.get("prompt_tokens", 0),
                "elapsed": round(elapsed, 2),
                "total_time": round(self._perf["total_gen_time"], 1),
                "step": total_steps,
                "context_used": context_used,
                "context_window": self.config_model.context_window,
                "context_pct": round(context_pct, 1),
                "model": self.config_model.name.split("/")[-1],
                "model_size_gb": round(self._model_size_gb, 1),
                "max_tokens": self.config_model.max_tokens,
                "tool_calls": tool_total,
                "tool_success": tool_ok,
                "tool_success_rate": round(tool_ok / max(1, tool_total) * 100, 0),
                "avg_step_time": round(sum(self._perf["step_times"]) / max(1, total_steps), 1),
                "bandwidth_used_gbs": round(self._model_size_gb * true_decode_tok_s, 1),
                "generating": False,
            }
            self._perf["peak_tok_s"] = stats["peak_tok_s"]
            try:
                with open(self.logger.run_dir / "perf.json", "w") as f:
                    json.dump(stats, f)
            except Exception:
                pass

            print(f"  ⚡ {true_decode_tok_s:.0f} tok/s decode ({gen_tok_s:.0f} overall) | {prompt_tokens}p+{gen_tokens}g | {elapsed:.1f}s")
            self.logger.generation(
                step=len(self._perf["step_times"]),
                prompt_tokens=prompt_tokens, gen_tokens=gen_tokens,
                tok_s=true_decode_tok_s, elapsed=elapsed,
                response_preview=response[:200],
            )

            # Free GPU memory to prevent Metal kernel panic (IOGPUMemory OOM)
            mx.clear_cache()
            gc.collect()

            # Strip thinking blocks
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            return response
        except Exception as e:
            return f"ERROR: {e}"

    def _extract_tool_calls(self, response: str) -> list[dict]:
        """Extract tool calls from model output.

        Supports:
        1. Qwen3 native: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        2. Fallback: <tool>name</tool><args>{...}</args>
        3. Last resort: regex patterns from narrative
        """
        calls = []

        # 1. Native Hermes format (what Qwen3 was trained on)
        for match in TOOL_CALL_RE.finditer(response):
            try:
                call = json.loads(match.group(1))
                name = call.get("name", "")
                args = call.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                if name:
                    calls.append({"name": name, "arguments": args})
            except json.JSONDecodeError:
                continue

        if calls:
            return calls

        # 2. Old format fallback
        for match in TOOL_CALL_FALLBACK_RE.finditer(response):
            name = match.group(1)
            try:
                args = json.loads(match.group(2))
                calls.append({"name": name, "arguments": args})
            except json.JSONDecodeError:
                continue

        if calls:
            return calls

        # 3. Extract from narrative (model wrote prose instead of tool call)
        # Check for python code FIRST (most common case for self-improvement)
        if "```python" in response:
            code_match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
            if code_match:
                calls.append({"name": "run_python", "arguments": {"code": code_match.group(1).strip()}})
        elif "write_file" in response.lower() or "save" in response.lower():
            code_match = re.search(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
            if code_match:
                calls.append({
                    "name": "write_file",
                    "arguments": {"path": "solution.py", "content": code_match.group(1).strip()},
                })

        return calls

    # ── Tool implementations ──────────────────────────────────────────────

    def _edit_file(self, path: str, old_content: str, new_content: str) -> str:
        """Structured file edit — find and replace a specific string."""
        try:
            p = Path(path)
            if not p.exists():
                # Try in skills/
                p = CONFIG.output_dir / path
            if not p.exists():
                return f"File not found: {path}"
            text = p.read_text()
            if old_content not in text:
                return f"old_content not found in {path}. Read the file first to get the exact text."
            new_text = text.replace(old_content, new_content, 1)
            p.write_text(new_text)
            return f"Edited {path}: replaced {len(old_content)} chars with {len(new_content)} chars"
        except Exception as e:
            return f"Edit failed: {e}"

    def _grep_file(self, pattern: str, path: str = "skills/") -> str:
        """Search for a regex pattern in files."""
        import re as _re
        try:
            results = []
            p = Path(path)
            files = [p] if p.is_file() else sorted(p.rglob("*.py"))
            for f in files[:20]:
                try:
                    for i, line in enumerate(f.read_text().splitlines(), 1):
                        if _re.search(pattern, line):
                            results.append(f"{f}:{i}: {line.strip()}")
                            if len(results) >= 20:
                                break
                except Exception:
                    continue
                if len(results) >= 20:
                    break
            return "\n".join(results) if results else f"No matches for '{pattern}' in {path}"
        except Exception as e:
            return f"Grep failed: {e}"

    def _list_dir(self, path: str = ".") -> str:
        """List directory contents with sizes."""
        try:
            p = Path(path)
            if not p.is_dir():
                return f"Not a directory: {path}"
            entries = []
            for f in sorted(p.iterdir()):
                if f.name.startswith(".") or f.name == "__pycache__":
                    continue
                if f.is_file():
                    size = f.stat().st_size
                    entries.append(f"{f.name} ({size:,}B)")
                elif f.is_dir():
                    entries.append(f"{f.name}/")
            return "\n".join(entries) if entries else "(empty)"
        except Exception as e:
            return f"List failed: {e}"

    def execute_tool(self, name: str, args: dict) -> str:
        """Route and execute a tool call."""
        dispatch = {
            "web_search": lambda a: self._web_search(a.get("query", "")),
            "run_python": lambda a: self._run_python(a.get("code", "")),
            "bash": lambda a: self._bash(a.get("cmd", "")),
            "read_file": lambda a: self._read_file(a.get("path", "")),
            "write_file": lambda a: self._write_file(
                a.get("path") or a.get("file_path") or a.get("file_name", ""),
                a.get("content", ""),
            ),
            "edit_file": lambda a: self._edit_file(
                a.get("path", ""), a.get("old_content", ""), a.get("new_content", ""),
            ),
            "grep_file": lambda a: self._grep_file(a.get("pattern", ""), a.get("path", "skills/")),
            "list_dir": lambda a: self._list_dir(a.get("path", ".")),
        }
        handler = dispatch.get(name)
        if not handler:
            return f"Unknown tool: {name}"
        try:
            # Write tool-executing status so monitor always has fresh data
            self._write_status(f"TOOL: {name}", generating=False)
            result = handler(args)
            # GPU is idle during tool execution — run background tasks
            self._idle_scheduler.run_pending(max_time=2.0)
            return result
        except Exception as e:
            return f"ERROR in {name}: {e}"

    def _web_search(self, query: str) -> str:
        """Search via DuckDuckGo with caching and relevance scoring."""
        if not query:
            return "ERROR: empty query"

        # Check cache
        if query in self._search_cache:
            return f"[CACHED] {self._search_cache[query]}"

        try:
            import httpx
            from bs4 import BeautifulSoup
            import urllib.parse

            url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

            with httpx.Client(timeout=CONFIG.web_search_timeout) as client:
                resp = client.get(url, headers=headers, follow_redirects=True)
                if resp.status_code != 200:
                    return f"Search failed (HTTP {resp.status_code})"

                soup = BeautifulSoup(resp.text, "html.parser")
                results = []

                for block in soup.find_all("div", class_="result"):
                    link = block.find("a", href=True)
                    if not link:
                        continue

                    href = link.get("href", "")
                    title = link.get_text(strip=True)

                    # Resolve DuckDuckGo redirect URLs
                    if "uddg=" in href:
                        try:
                            href = urllib.parse.parse_qs(
                                urllib.parse.urlparse(href).query
                            ).get("uddg", [href])[0]
                        except Exception:
                            pass

                    if not href.startswith("http") or not title or len(title) < 4:
                        continue

                    # Extract snippet
                    snippet_el = block.find("div", class_="result__snippet")
                    snippet = snippet_el.get_text(strip=True)[:150] if snippet_el else ""

                    results.append(f"- {title}\n  {href}\n  {snippet}")

                    if len(results) >= CONFIG.max_search_results:
                        break

                if not results:
                    return f"No results for '{query}'"

                output = f"Results for '{query}':\n\n" + "\n\n".join(results)

                # Cache it
                self._search_cache[query] = output
                return output

        except ImportError:
            return "ERROR: pip install httpx beautifulsoup4"
        except Exception as e:
            return f"Search error: {e}"

    def _run_python(self, code: str) -> str:
        """Execute Python code with timeout."""
        if not code or code.strip() == "":
            return "ERROR: empty code"
        try:
            # Run from agent_outputs so imports work for generated files
            env = os.environ.copy()
            env["PYTHONPATH"] = str(CONFIG.output_dir.resolve()) + ":" + env.get("PYTHONPATH", "")
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=CONFIG.code_execution_timeout,
                env=env,
            )
            output = result.stdout + result.stderr
            return output if output.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: timed out after {CONFIG.code_execution_timeout}s"
        except Exception as e:
            return f"ERROR: {e}"

    def _bash(self, cmd: str) -> str:
        """Execute shell command with timeout."""
        if not cmd:
            return "ERROR: empty command"
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=CONFIG.code_execution_timeout,
            )
            output = result.stdout + result.stderr
            return output if output.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: timed out after {CONFIG.code_execution_timeout}s"
        except Exception as e:
            return f"ERROR: {e}"

    def _read_file(self, path: str) -> str:
        """Read file contents."""
        try:
            with open(path) as f:
                return f.read()
        except FileNotFoundError:
            return f"ERROR: file not found: {path}"
        except Exception as e:
            return f"ERROR: {e}"

    def _write_file(self, path: str, content: str) -> str:
        """Write file to output directory."""
        if not path:
            return "ERROR: path required"
        if not content:
            return "ERROR: content is empty"

        # Resolve path
        if os.path.isabs(path):
            full_path = Path(path)
        elif "skills" in path:
            full_path = Path(path)
        else:
            full_path = CONFIG.output_dir / path

        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, "w") as f:
            n = f.write(content)

        return f"Wrote {n} bytes to {full_path}"

    # ── Memory context ────────────────────────────────────────────────────

    def _build_memory_context(self) -> str:
        """Build concise memory summary for the model."""
        if not self.memory_manager:
            return ""

        mem = self.memory_manager.memory
        lines = [f"[Memory: {len(mem.iterations)} steps completed]"]

        for it in mem.iterations[-4:]:
            status = "OK" if it.success else "FAIL"
            preview = it.result[:60].replace("\n", " ") if it.result else ""
            lines.append(f"  {status} {it.tool_used}: {preview}")

        if mem.discoveries:
            lines.append(f"  Discoveries: {len(mem.discoveries)}")

        return "\n".join(lines)

    # ── Main ReAct loop ───────────────────────────────────────────────────

    def run_loop(self, goal: str) -> None:
        """Execute ReAct loop with native Qwen3 tool calling."""
        if not self.memory_manager:
            self.memory_manager = MemoryManager(goal)

        print(f"\n🚀 Agent starting:\n  Goal: {goal}\n")
        self.logger.run_start(goal, self.config_model.name, {
            "max_tokens": self.config_model.max_tokens,
            "context_window": self.config_model.context_window,
            "max_iterations": CONFIG.max_iterations,
        })

        system_msg = (
            f"/nothink\n"
            f"You are an autonomous coding agent. Goal: {goal}\n\n"
            f"Work naturally: read code to understand, write code to build, test to verify. "
            f"Use grep_file and list_dir to explore efficiently. Use edit_file for small changes. "
            f"Fix errors by reading the traceback and changing the broken line — never rerun identical code. "
            f"Do NOT import agent.py or load MLX models in tests. Say DONE when tests pass."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Begin working on: {goal}"},
        ]

        phase = "research"
        consecutive_no_tool = 0
        consecutive_stuck = 0  # Hard escape after 3 consecutive stuck detections
        # Content-hash loop detection: catches write→test→write→test oscillation
        _seen_writes: dict[str, int] = {}   # hash(content) → count
        _seen_errors: dict[str, int] = {}   # hash(error[:200]) → count

        # Try to use the LoopDetector skill if available
        try:
            from skills import get_skill
            _loop_detector = get_skill("loop_detector")
        except Exception:
            _loop_detector = None

        # Start resource sampler thread for real-time utilization tracking
        _sampler_stop = threading.Event()
        _sampler_thread = threading.Thread(
            target=_resource_sampler,
            args=(self.logger.run_dir, _sampler_stop),
            daemon=True,
        )
        _sampler_thread.start()

        for step in range(1, CONFIG.max_iterations + 1):
            # Inject skill focus every 5 steps or on failure
            if step % 5 == 1 or (step > 1 and not self._perf["tool_success"].get("last_ok", True)):
                try:
                    nxt = self.skill_tree.get_next_skill()
                    if nxt and nxt.get("current_impact", 0) > 7:
                        messages.append({"role": "user", "content":
                            f"Focus on mastering: {nxt['name']} — {nxt.get('description', '')}"})
                except Exception:
                    pass

            # Check for real-time user input (inside current run dir)
            _input_file = self.logger.run_dir / "user_input.txt"
            try:
                if _input_file.exists() and _input_file.stat().st_size > 0:
                    user_hint = _input_file.read_text().strip()
                    _input_file.write_text("")  # Clear after reading
                    if user_hint:
                        print(f"  📨 USER INPUT: {user_hint[:100]}")
                        messages.append({"role": "user", "content": f"USER INSTRUCTION: {user_hint}"})
                        self.logger.event("user_input", {"hint": user_hint[:500]})
            except Exception:
                pass

            # Generate response
            response = self._generate_response(messages)

            print(f"\n[Step {step}] Phase: {phase}")
            # Show first 300 chars of response
            display = response[:300].replace("\n", "\n  ")
            print(f"  {display}")

            # Check for completion
            if "DONE" in response and step > 3:
                print("\n✅ Agent signaled completion.")
                break

            # Extract tool calls
            tool_calls = self._extract_tool_calls(response)

            if not tool_calls:
                consecutive_no_tool += 1
                print(f"  ⚠️ No tool call detected ({consecutive_no_tool}/3)")

                if consecutive_no_tool >= 3:
                    print("  🔨 Forcing tool execution")
                    # Force based on phase
                    if phase == "research":
                        tool_calls = [{"name": "web_search", "arguments": {"query": goal[:80]}}]
                    elif phase == "code":
                        tool_calls = [{"name": "run_python", "arguments": {"code": "print('test')"}}]
                    else:
                        tool_calls = [{"name": "write_file", "arguments": {"path": "output.py", "content": "# TODO"}}]
                    consecutive_no_tool = 0
                else:
                    # Add response and ask model to use tools
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": "Please call a tool to make progress. Use web_search, run_python, or write_file."})
                    continue
            else:
                consecutive_no_tool = 0

            # Execute each tool call
            for call in tool_calls[:1]:  # Execute one at a time for control
                name = call["name"]
                args = call["arguments"]

                print(f"  🔧 {name}({json.dumps(args)[:80]})")
                self.logger.tool_call(step, name, args)
                result = self.execute_tool(name, args)
                result_preview = result[:200].replace("\n", " ")
                print(f"  → {result_preview}")
                success = not result.startswith("ERROR")
                self.logger.tool_result(step, name, success, result_preview)

                # Track in memory
                success = not result.startswith("ERROR")
                self._perf["tool_success"]["total"] += 1
                if success:
                    self._perf["tool_success"]["success"] += 1
                if self.memory_manager:
                    self.memory_manager.record_attempt(
                        step=step, tool=name, args=args,
                        result=result[:500], success=success,
                        learning="OK" if success else result[:80],
                    )
                    if success and name == "web_search":
                        self.memory_manager.record_discovery(result[:200])

                # ── Loop detection (skill-based + hash fallback) ────────
                import hashlib as _hl
                _stuck = False

                # Record in LoopDetector skill if available
                if _loop_detector and hasattr(_loop_detector, 'record'):
                    _loop_detector.record(name, json.dumps(args)[:200], result[:200])
                    if _loop_detector.is_stuck():
                        _stuck = True
                        escape = _loop_detector.suggest_escape(name)
                        print(f"  🔄 STUCK (skill-detected): {escape or 'try different approach'}")

                # Hash-based fallback
                if not _stuck and name == "write_file" and success:
                    h = _hl.md5(args.get("content", "")[:500].encode()).hexdigest()
                    _seen_writes[h] = _seen_writes.get(h, 0) + 1
                    if _seen_writes[h] >= 2:
                        _stuck = True
                        print(f"  🔄 STUCK: wrote identical file {_seen_writes[h]}x")

                if not _stuck and not success:
                    h = _hl.md5(result[:200].encode()).hexdigest()
                    _seen_errors[h] = _seen_errors.get(h, 0) + 1
                    if _seen_errors[h] >= 2:
                        _stuck = True
                        print(f"  🔄 STUCK: same error {_seen_errors[h]}x")

                if _stuck:
                    consecutive_stuck += 1
                    self.logger.loop_detected(step, name, consecutive_stuck)

                    if consecutive_stuck >= 3:
                        # HARD ESCAPE: 3 stuck in a row = abort this cycle
                        print(f"  💀 HARD ESCAPE: stuck {consecutive_stuck}x, aborting cycle")
                        break

                    messages.append({"role": "user", "content": (
                        f"STOP. You are stuck in a loop ({consecutive_stuck}/3 before abort).\n"
                        f"LATEST RESULT: {result[:300]}\n\n"
                        f"CHANGE YOUR APPROACH COMPLETELY. If importing fails, use only stdlib. "
                        f"If testing fails, read the error and fix the specific line. "
                        f"Do NOT repeat the same tool call."
                    )})
                    _seen_writes.clear()
                    _seen_errors.clear()
                else:
                    consecutive_stuck = 0

                # Phase transitions
                if name == "web_search":
                    phase = "research"
                elif name == "run_python":
                    phase = "code"
                elif name == "write_file":
                    phase = "save"

                # Add to conversation as tool response
                messages.append({"role": "assistant", "content": response})
                # Truncate long tool results to save context
                max_result = min(4000, self.config_model.context_window // 4)
                messages.append({"role": "tool", "content": result[:max_result]})

            # Tiered context compression (prevents Metal GPU OOM)
            messages = self._compress_context(messages)

            # Force save phase after enough code runs (count from messages)
            recent_tools = [m.get("content", "")[:20] for m in messages[-12:] if m.get("role") == "assistant"]
            code_runs = sum(1 for t in recent_tools if "run_python" in t)
            if code_runs >= 4 and phase == "code":
                print(f"  ⚠️ 4+ code runs - moving to save phase")
                phase = "save"
                messages.append({
                    "role": "user",
                    "content": "You've tested enough. Now SAVE your best code to a file using write_file. Include the complete working code.",
                })

        # Save session
        if self.memory_manager:
            self.memory_manager.memory.save(self.memory_manager.memory_file)

        # Log run end
        self.logger.run_end(step, self._perf)

        # Performance summary
        p = self._perf
        avg_tok_s = p["total_tokens"] / max(0.01, p["total_gen_time"])
        success_rate = p["tool_success"]["success"] / max(1, p["tool_success"]["total"])
        avg_step = sum(p["step_times"]) / max(1, len(p["step_times"]))
        print(f"\n📊 Performance: {avg_tok_s:.0f} tok/s avg | {p['total_tokens']} tokens | {success_rate:.0%} tool success | {avg_step:.1f}s/step")
        print("🛑 Session complete.")

        # Stop resource sampler
        _sampler_stop.set()
        _sampler_thread.join(timeout=2)


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python agent.py '<goal>'")
        sys.exit(1)

    goal = " ".join(sys.argv[1:])
    # Default to tool_calling (Qwen3-14B, proven 0.971 F1)
    model = os.environ.get("AGENT_MODEL", "tool_calling")
    agent = MLXAgent(config_model_name=model, goal=goal)
    agent.run_loop(goal)


if __name__ == "__main__":
    main()
