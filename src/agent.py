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
from pathlib import Path
from typing import Optional

import mlx.core as mx

from src.config import CONFIG
from src.memory import MemoryManager
from src.logger import AgentLogger
from src.skill_tree import SkillTree


# Tool definitions in OpenAI-compatible format (what Qwen3 was trained on)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information on any topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code and return output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to run"}
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

            print(f"✅ Loaded {self.config_model.name} via MLX")
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

    def _generate_response(self, messages: list[dict]) -> str:
        """Generate a response with all MLX optimizations + performance tracking."""
        import time as _time
        prompt = self._format_prompt(messages)

        # Count prompt tokens
        prompt_tokens = len(self.tokenizer.encode(prompt)) if hasattr(self.tokenizer, 'encode') else len(prompt) // 4

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

            try:
                for chunk in self._stream_generate(self.model, self.tokenizer, **kwargs):
                    token_count += 1
                    response_parts.append(chunk.text)
                    # Stream file every 10 tokens, perf stats every 100
                    if token_count % 10 == 0:
                        elapsed_so_far = _time.perf_counter() - t0
                        live_tok_s = token_count / max(0.01, elapsed_so_far)
                        try:
                            with open(stream_file, "w") as sf:
                                sf.write("".join(response_parts))
                        except Exception:
                            pass
                    # Perf stats less frequently (heavy JSON write)
                    if token_count % 100 == 0:
                        try:
                            live_stats = {
                                "gen_tok_s": round(live_tok_s, 1),
                                "avg_tok_s": round(live_tok_s, 1),
                                "peak_tok_s": round(max(self._perf.get("peak_tok_s", 0), live_tok_s), 1),
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
                                "tool_calls": self._perf["tool_success"]["total"],
                                "tool_success": self._perf["tool_success"]["success"],
                                "tool_success_rate": round(self._perf["tool_success"]["success"] / max(1, self._perf["tool_success"]["total"]) * 100, 0),
                                "avg_step_time": round((self._perf["total_gen_time"] + elapsed_so_far) / max(1, len(self._perf["step_times"]) + 1), 1),
                                "bandwidth_used_gbs": round(8.0 * live_tok_s, 1),
                                "generating": True,
                            }
                            with open(self.logger.run_dir / "perf.json", "w") as pf:
                                json.dump(live_stats, pf)
                        except Exception:
                            pass
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

            # Estimate true decode speed (exclude prefill time)
            est_prefill_time = prompt_tokens / 286.0
            est_decode_time = max(0.1, elapsed - est_prefill_time)
            true_decode_tok_s = gen_tokens / est_decode_time

            # Write stats to file for monitor to read
            total_steps = len(self._perf["step_times"])
            tool_total = self._perf["tool_success"]["total"]
            tool_ok = self._perf["tool_success"]["success"]
            context_used = prompt_tokens + gen_tokens
            context_pct = (context_used / self.config_model.context_window) * 100

            stats = {
                "gen_tok_s": round(gen_tok_s, 1),
                "decode_tok_s": round(true_decode_tok_s, 1),
                "decode_max": 41.0,
                "decode_efficiency": round(true_decode_tok_s / 41.0 * 100, 0),
                "prefill_time_s": round(est_prefill_time, 1),
                "decode_time_s": round(est_decode_time, 1),
                "avg_tok_s": round(avg_tok_s, 1),
                "peak_tok_s": round(max(
                    self._perf.get("peak_tok_s", 0), gen_tok_s
                ), 1),
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
                "max_tokens": self.config_model.max_tokens,
                "tool_calls": tool_total,
                "tool_success": tool_ok,
                "tool_success_rate": round(tool_ok / max(1, tool_total) * 100, 0),
                "avg_step_time": round(sum(self._perf["step_times"]) / max(1, total_steps), 1),
                "messages_in_context": len(messages) if 'messages' in dir() else 0,
                # Bandwidth = model_size_bytes × tokens_generated / time
                # 14B at 4-bit ≈ 8GB weights read per token generated
                "bandwidth_used_gbs": round(8.0 * gen_tok_s, 1),
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
            mx.metal.clear_cache()
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
        }
        handler = dispatch.get(name)
        if not handler:
            return f"Unknown tool: {name}"
        try:
            return handler(args)
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
            f"You are an autonomous agent. Your goal: {goal}\n\n"
            f"DECISION FRAMEWORK - Before each action, ask yourself:\n"
            f"1. Do I KNOW enough to write the code? If yes -> write_file or run_python\n"
            f"2. Do I need to READ a file to understand something? If yes -> read_file\n"
            f"3. Is my code WORKING? If no -> fix the specific error shown in the result\n"
            f"4. Is my code TESTED and passing? If yes -> say DONE\n\n"
            f"RULES:\n"
            f"- Do NOT search the web. You already know how to write Python.\n"
            f"- After reading files, write code IMMEDIATELY. Don't over-research.\n"
            f"- When a test fails, read the ERROR MESSAGE and fix that SPECIFIC bug.\n"
            f"- Don't rewrite the same code if it failed. Change the broken part only.\n"
            f"- When testing, do NOT import agent.py or load MLX models (too slow).\n"
            f"- Include 'ALL TESTS PASSED' print in your test block.\n"
            f"- Call ONE tool per response.\n"
            f"- When tests pass, say DONE."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Begin working on: {goal}"},
        ]

        phase = "research"
        consecutive_no_tool = 0
        tool_history: list[str] = []  # Track recent tool names for loop detection

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

                # Update tool history for loop detection
                tool_history.append(f"{name}:{'FAIL' if not success else 'OK'}")

                # Error loop detection: same tool failing 2+ times = stuck
                recent_fails = [h for h in tool_history[-3:] if h.endswith(":FAIL")]
                if len(recent_fails) >= 2 and all(h.startswith(f"{name}:") for h in tool_history[-2:]):
                    print(f"  🔄 ERROR LOOP: {name} failing repeatedly - injecting fix guidance")
                    self.logger.loop_detected(step, name, len(recent_fails))
                    # Inject the error directly so the model SEES it and fixes it
                    messages.append({"role": "user", "content": (
                        f"STOP. You've called {name} {len(recent_fails)} times and it keeps failing with:\n"
                        f"ERROR: {result[:300]}\n\n"
                        f"You MUST fix the bug. If you can't fix it in one more try, "
                        f"rewrite the ENTIRE file from scratch with simpler code. "
                        f"Do NOT repeat the same code."
                    )})
                    tool_history.clear()

                # General loop: same tool 3+ times regardless of success
                tool_names_only = [h.split(":")[0] for h in tool_history]
                if len(tool_names_only) >= 3 and len(set(tool_names_only[-3:])) == 1:
                    print(f"  🔄 LOOP: {tool_names_only[-1]} x3 - forcing phase switch")
                    self.logger.loop_detected(step, tool_names_only[-1], 3)
                    old_phase = phase
                    if name == "web_search":
                        phase = "code"
                    elif name == "run_python":
                        phase = "save"
                    self.logger.phase_change(step, old_phase, phase, f"loop on {name}")
                    tool_history.clear()

                # Phase transitions
                if name == "web_search":
                    phase = "research"
                elif name == "run_python":
                    phase = "code"
                elif name == "write_file":
                    phase = "save"

                # Add to conversation as tool response
                messages.append({"role": "assistant", "content": response})
                # Give model FULL file contents - we have 128K context, use it
                messages.append({"role": "tool", "content": result[:20000]})

            # Keep conversation manageable but preserve enough context
            if len(messages) > 60:
                # Keep system + last 56 messages - we have 128K context, USE IT ALL
                mem_ctx = self._build_memory_context()
                messages = messages[:1] + [
                    {"role": "user", "content": f"CONTEXT: {mem_ctx}\nDo NOT re-read files you already read. Use the information you have."}
                ] + messages[-56:]

            # Phase forcing after enough research
            research_count = sum(1 for t in tool_history[-5:] if t == "web_search")
            if research_count >= 3 and phase == "research":
                print(f"  ⚠️ 3+ searches done - moving to code phase")
                phase = "code"
                messages.append({
                    "role": "user",
                    "content": "You've done enough research. Now write Python code to implement what you learned. Use run_python.",
                })

            # Force save phase after enough code runs
            code_count = sum(1 for t in tool_history[-6:] if t == "run_python")
            if code_count >= 4 and phase == "code":
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
