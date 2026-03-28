#!/usr/bin/env python3
"""MLX-powered bootstrap for the verifier-driven runtime."""

import os
import sys
import gc
from pathlib import Path
from typing import Any

import mlx.core as mx

from src.config import CONFIG
from src.context_manager import ContextBudgetGuard, EpisodicBuffer, KVCacheManager
from src.memory import MemoryManager
from src.logger import AgentLogger
from src.runtime.controller import AgentController
from src.runtime.mlx_adapter import MLXGenerationAdapter
from src.runtime.policy import PolicyEngine
from src.runtime.runtime_support import (
    IdleScheduler,
    PerfStatusWriter,
    evaluate_with_self_evaluator,
    pre_validate_candidate_file,
    record_strategy_outcome,
    resource_sampler,
)
from src.runtime.state_store import PersistentStateStore
from src.runtime.tool_call_parser import extract_tool_calls_from_response
from src.runtime.tools import TOOL_DEFINITIONS, ToolExecutor
from src.runtime.verifier import RuntimeVerifier
from src.skill_tree import SkillTree


# Tool definitions in OpenAI-compatible format (what Qwen3 was trained on)
TOOLS = TOOL_DEFINITIONS

class MLXAgent:
    """MLX-powered agent bootstrap using the shared runtime."""

    def __init__(self, config_model_name: str = "balanced", goal: str = "") -> None:
        if config_model_name not in CONFIG.models:
            raise KeyError(f"Model '{config_model_name}' not in CONFIG.models")

        self.config_model_name = config_model_name
        self.config_model = CONFIG.models[config_model_name]
        self.goal = goal
        self.memory_manager = MemoryManager(goal) if goal else None
        self.logger = AgentLogger()
        self.skill_tree = SkillTree()
        self._idle_scheduler = IdleScheduler()
        self._episodic_buffer = EpisodicBuffer()
        self._skill_instances: dict[str, Any] = {}
        self._files_written = 0
        self._cache_factory = None
        self.state_store = PersistentStateStore(self.logger.run_id, goal, self.logger.run_dir)
        self._status_writer = PerfStatusWriter(
            run_dir=self.logger.run_dir,
            model_name=self.config_model.name,
            context_window=self.config_model.context_window,
            max_tokens=self.config_model.max_tokens,
            model_size_gb=0.0,
        )
        self.tool_executor = ToolExecutor(CONFIG.output_dir, write_status=self._write_status)
        self.verifier = RuntimeVerifier(self.skill_tree)
        self.policy_engine = PolicyEngine(self._load_skill_module, goal)

        # Performance tracking
        self._perf: dict = {
            "total_tokens": 0,
            "total_gen_time": 0.0,
            "step_times": [],
            "tool_success": {"total": 0, "success": 0},
        }
        self._mlx_adapter = None

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
                self._cache_factory = lambda: make_prompt_cache(self.model)
            except Exception:
                self._cache_factory = None
            self._kv_cache_manager = KVCacheManager(self._cache_factory)
            self._context_guard = ContextBudgetGuard(
                tokenizer=self.tokenizer,
                context_window=self.config_model.context_window,
                max_tokens=self.config_model.max_tokens,
                episodic_buffer=self._episodic_buffer,
            )

            self._draft_model = None

            # Compute actual model size for accurate bandwidth reporting
            try:
                total_bytes = sum(v.nbytes for _, v in mx.utils.tree_flatten(self.model.parameters()))
                self._model_size_gb = total_bytes / 1e9
            except Exception:
                self._model_size_gb = 8.0  # fallback
            self._status_writer = PerfStatusWriter(
                run_dir=self.logger.run_dir,
                model_name=self.config_model.name,
                context_window=self.config_model.context_window,
                max_tokens=self.config_model.max_tokens,
                model_size_gb=self._model_size_gb,
            )
            self._mlx_adapter = MLXGenerationAdapter(
                model=self.model,
                tokenizer=self.tokenizer,
                stream_generate=self._stream_generate,
                generate_fn=self._generate_fn,
                sampler=self._sampler,
                kv_cache_manager=self._kv_cache_manager,
                status_writer=self._status_writer,
                logger=self.logger,
                perf=self._perf,
                config_model=self.config_model,
                model_size_gb=self._model_size_gb,
                draft_model=self._draft_model,
                clear_cache_fn=mx.clear_cache,
                collect_garbage_fn=gc.collect,
            )

            print(f"✅ Loaded {self.config_model.name} via MLX ({self._model_size_gb:.1f} GB)")
            print(f"📁 Output folder: {CONFIG.output_dir.resolve()}")
            print(f"📊 Context: {self.config_model.context_window} tokens")
            if self._kv_cache_manager.prompt_cache:
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
        self._files_written = 0
        self.state_store = PersistentStateStore(self.logger.run_id, goal, self.logger.run_dir)
        self._status_writer = PerfStatusWriter(
            run_dir=self.logger.run_dir,
            model_name=self.config_model.name,
            context_window=self.config_model.context_window,
            max_tokens=self.config_model.max_tokens,
            model_size_gb=self._model_size_gb,
        )
        self.tool_executor = ToolExecutor(CONFIG.output_dir, write_status=self._write_status)
        self.verifier = RuntimeVerifier(self.skill_tree)
        self.policy_engine = PolicyEngine(self._load_skill_module, goal)
        self._perf = {
            "total_tokens": 0,
            "total_gen_time": 0.0,
            "step_times": [],
            "tool_success": {"total": 0, "success": 0},
            "peak_tok_s": 0.0,
        }
        self._mlx_adapter = MLXGenerationAdapter(
            model=self.model,
            tokenizer=self.tokenizer,
            stream_generate=self._stream_generate,
            generate_fn=self._generate_fn,
            sampler=self._sampler,
            kv_cache_manager=self._kv_cache_manager,
            status_writer=self._status_writer,
            logger=self.logger,
            perf=self._perf,
            config_model=self.config_model,
            model_size_gb=self._model_size_gb,
            draft_model=self._draft_model,
            clear_cache_fn=mx.clear_cache,
            collect_garbage_fn=gc.collect,
        )
        self._kv_cache_manager.invalidate()
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
        self._status_writer.write_status(status=status, generating=generating, perf=self._perf)

    def _pre_validate(self, path: str) -> str:
        """Quick AST validation during idle time. Returns 'OK' or 'WARN: ...'."""
        return pre_validate_candidate_file(path, CONFIG.output_dir)

    def _count_tokens(self, messages: list[dict]) -> int:
        """Count actual tokens using the tokenizer (not estimates)."""
        try:
            prompt = self._format_prompt(messages)
            return len(self.tokenizer.encode(prompt))
        except Exception:
            return sum(len(m.get("content", "")) for m in messages) // 4

    def _load_skill_instance(self, skill_name: str, class_name: str) -> Any:
        """Load a skill class lazily and cache the instance."""
        cache_key = f"{skill_name}:{class_name}"
        if cache_key in self._skill_instances:
            return self._skill_instances[cache_key]

        try:
            from skills import get_skill

            skill_module = get_skill(skill_name)
            skill_class = getattr(skill_module, class_name, None)
            if skill_class is None:
                return None
            skill_instance = skill_class()
            self._skill_instances[cache_key] = skill_instance
            return skill_instance
        except Exception:
            return None

    def _load_skill_module(self, skill_name: str, _unused_class_name: str | None = None) -> Any:
        """Load a skill module lazily and cache it in the instance map."""
        cache_key = f"module:{skill_name}"
        if cache_key in self._skill_instances:
            return self._skill_instances[cache_key]
        try:
            from skills import get_skill

            skill_module = get_skill(skill_name)
            self._skill_instances[cache_key] = skill_module
            return skill_module
        except Exception:
            return None

    def _evaluate_written_file(self, path_value: Path) -> dict[str, Any]:
        """Evaluate a written file with the self-evaluator skill when available."""
        return evaluate_with_self_evaluator(path_value, self._load_skill_instance)

    def _record_strategy_outcome(self, success: bool, phase: str) -> None:
        """Persist a skill-based strategy summary for future runs."""
        record_strategy_outcome(
            load_skill_instance=self._load_skill_instance,
            config_model_name=self.config_model_name,
            goal=self.goal,
            phase=phase,
            success=success,
            perf=self._perf,
            files_written=self._files_written,
        )

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
            return self._episodic_buffer.compress_messages(messages)

        # 80%+: aggressive - keep only system + last 2 exchanges
        if len(messages) > 4:
            messages = [messages[0]] + messages[-4:]
        return messages

    def _generate_response(self, messages: list[dict]) -> str:
        """Generate a response through the shared MLX adapter."""
        if not self._mlx_adapter:
            return "ERROR: MLX adapter not initialized"
        return self._mlx_adapter.generate_response(messages, self._format_prompt)

    def _extract_tool_calls(self, response: str) -> list[dict]:
        """Extract tool calls from model output."""
        return extract_tool_calls_from_response(response)

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
        """Execute the verifier-driven runtime controller."""
        self.goal = goal
        if not self.memory_manager:
            self.memory_manager = MemoryManager(goal)

        print(f"\n🚀 Agent starting:\n  Goal: {goal}\n")
        self.logger.run_start(goal, self.config_model.name, {
            "max_tokens": self.config_model.max_tokens,
            "context_window": self.config_model.context_window,
            "max_iterations": CONFIG.max_iterations,
        })

        controller = AgentController(
            goal=goal,
            config_model=self.config_model,
            logger=self.logger,
            memory_manager=self.memory_manager,
            state_store=self.state_store,
            policy_engine=self.policy_engine,
            verifier=self.verifier,
            tool_executor=self.tool_executor,
            skill_tree=self.skill_tree,
            idle_scheduler=self._idle_scheduler,
            context_guard=self._context_guard,
            compress_context=self._compress_context,
            format_prompt=self._format_prompt,
            generate_response=self._generate_response,
            extract_tool_calls=self._extract_tool_calls,
            build_memory_context=self._build_memory_context,
            load_skill_instance=self._load_skill_instance,
            pre_validate=self._pre_validate,
            evaluate_written_file=self._evaluate_written_file,
            perf=self._perf,
            max_iterations=CONFIG.max_iterations,
            resource_sampler=resource_sampler,
        )
        controller_summary = controller.run(resume=False)
        self._files_written = self.tool_executor.files_written

        if self.memory_manager:
            self.memory_manager.memory.save(self.memory_manager.memory_file)

        self.logger.run_end(controller_summary.steps_used, self._perf)
        average_tokens_per_second = self._perf["total_tokens"] / max(0.01, self._perf["total_gen_time"])
        success_rate = self._perf["tool_success"]["success"] / max(1, self._perf["tool_success"]["total"])
        average_step_time = sum(self._perf["step_times"]) / max(1, len(self._perf["step_times"]))
        print(
            f"\n📊 Performance: {average_tokens_per_second:.0f} tok/s avg | "
            f"{self._perf['total_tokens']} tokens | {success_rate:.0%} tool success | "
            f"{average_step_time:.1f}s/step"
        )
        print("🛑 Session complete.")

        self._record_strategy_outcome(controller_summary.accepted, controller_summary.final_phase)
        self.state_store.update_run_status(
            completed=controller_summary.completed,
            accepted=controller_summary.accepted,
            final_phase=controller_summary.final_phase,
            steps_used=controller_summary.steps_used,
            active_skill_id=controller_summary.active_skill_id,
        )
        self.state_store.record_strategy_outcome(
            strategy_name=f"{self.config_model_name}:{controller_summary.final_phase}",
            success=controller_summary.accepted,
            metrics={
                "tool_success_rate": round(success_rate, 4),
                "avg_tok_s": round(average_tokens_per_second, 4),
                "steps_used": float(controller_summary.steps_used),
            },
        )

        try:
            self.logger.write_summary(self._perf, {
                "completed": controller_summary.completed,
                "accepted": controller_summary.accepted,
                "steps_used": controller_summary.steps_used,
                "max_steps": CONFIG.max_iterations,
                "final_phase": controller_summary.final_phase,
            })
        except Exception:
            pass


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python agent.py '<goal>' [--model profile]")
        sys.exit(1)

    args = sys.argv[1:]
    model = os.environ.get("AGENT_MODEL", "tool_calling")
    if "--model" in args:
        model_index = args.index("--model")
        if model_index + 1 < len(args):
            model = args[model_index + 1]
            del args[model_index:model_index + 2]

    goal = " ".join(args)
    agent = MLXAgent(config_model_name=model, goal=goal)
    agent.run_loop(goal)


if __name__ == "__main__":
    main()
