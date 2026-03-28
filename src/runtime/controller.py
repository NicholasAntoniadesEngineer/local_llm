"""Controller loop for the autonomous agent runtime."""

from __future__ import annotations

import json
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.runtime.prompt_builder import (
    build_plan_items_from_policy,
    build_prompt_messages,
    build_task_hypothesis,
)
from src.runtime.policy import PolicyEngine, skill_relative_path_from_goal
from src.runtime.state_store import PersistentStateStore
from src.runtime.task_state import (
    TASK_PHASE_ACCEPT,
    TASK_PHASE_ABORT,
    TASK_PHASE_INSPECT,
    TASK_PHASE_PLAN,
    TASK_PHASE_VERIFY,
    TaskState,
)
from src.runtime.tool_kinds import EXECUTION_TOOLS, MUTATION_TOOLS, OBSERVATION_TOOLS
from src.runtime.verifier import RuntimeVerifier, VerificationResult


@dataclass
class ControllerSummary:
    """Summary returned by the controller after a run."""

    completed: bool
    accepted: bool
    final_phase: str
    steps_used: int
    last_verification: VerificationResult | None
    active_skill_id: str


class AgentController:
    """Verifier-driven controller for long-running agent execution."""

    def __init__(
        self,
        *,
        goal: str,
        config_model,
        logger,
        memory_manager,
        state_store: PersistentStateStore,
        policy_engine: PolicyEngine,
        verifier: RuntimeVerifier,
        tool_executor,
        skill_tree,
        idle_scheduler,
        context_guard,
        compress_context: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        format_prompt: Callable[[list[dict[str, Any]]], str],
        generate_response: Callable[[list[dict[str, Any]]], str],
        extract_tool_calls: Callable[[str], list[dict[str, Any]]],
        build_memory_context: Callable[[], str],
        load_skill_instance: Callable[[str, str], Any],
        pre_validate: Callable[[str], str],
        evaluate_written_file: Callable[[Path], dict[str, Any]],
        perf: dict[str, Any],
        max_iterations: int,
        resource_sampler: Callable[[Path, threading.Event], None],
    ) -> None:
        self.goal = goal
        self.config_model = config_model
        self.logger = logger
        self.memory_manager = memory_manager
        self.state_store = state_store
        self.policy_engine = policy_engine
        self.verifier = verifier
        self.tool_executor = tool_executor
        self.skill_tree = skill_tree
        self.idle_scheduler = idle_scheduler
        self.context_guard = context_guard
        self.compress_context = compress_context
        self.format_prompt = format_prompt
        self.generate_response = generate_response
        self.extract_tool_calls = extract_tool_calls
        self.build_memory_context = build_memory_context
        self.load_skill_instance = load_skill_instance
        self.pre_validate = pre_validate
        self.evaluate_written_file = evaluate_written_file
        self.perf = perf
        self.max_iterations = max_iterations
        self.resource_sampler = resource_sampler
        self._frozen_static_block: str | None = None

    def _ensure_frozen_static_block(self) -> str:
        if self._frozen_static_block is not None:
            return self._frozen_static_block
        from src.paths import ROOT, SKILLS_DIR
        from src.runtime.repo_bootstrap import build_frozen_static_prompt_block

        tree_text_value = ""
        if self.skill_tree is not None:
            try:
                full_tree_text = self.skill_tree._tree_text()
                tree_text_value = full_tree_text[:14000] + (
                    "\n...(truncated skill tree)\n" if len(full_tree_text) > 14000 else ""
                )
            except Exception:
                tree_text_value = ""
        self._frozen_static_block = build_frozen_static_prompt_block(
            ROOT, SKILLS_DIR, skill_tree_text=tree_text_value
        )
        return self._frozen_static_block

    def _new_task_state(self) -> TaskState:
        return TaskState(task_id=self.state_store.run_id, goal_text=self.goal)

    def _restore_task_state(self, checkpoint_payload: dict[str, Any] | None) -> TaskState:
        if checkpoint_payload and checkpoint_payload.get("task_state"):
            return TaskState.from_dict(checkpoint_payload["task_state"])
        return self._new_task_state()

    def _resume_verification(self, task_state: TaskState) -> VerificationResult | None:
        if not task_state.verifier_status or task_state.verifier_status == "pending":
            return None
        return VerificationResult(
            status=task_state.verifier_status,
            accepted=task_state.last_verification_accepted,
            should_stop=task_state.accepted,
            summary=task_state.last_verification_summary,
            target_path=task_state.target_files[-1] if task_state.target_files else "",
            failure_type=task_state.last_failure_type,
        )

    def _build_prompt_messages(self, task_state: TaskState, policy) -> list[dict[str, Any]]:
        memory_context = self.build_memory_context()
        retrieval_context = self.state_store.build_retrieval_context(self.goal)
        past_failures = self.state_store.get_recent_failures()
        frozen_static_block = self._ensure_frozen_static_block()
        prompt_messages = build_prompt_messages(
            task_state,
            policy,
            memory_context=memory_context,
            retrieval_context=retrieval_context,
            past_failures=past_failures,
            static_context_block=frozen_static_block,
        )
        return self.compress_context(prompt_messages)

    def _log_phase_transition(self, task_state: TaskState, next_phase: str, reason: str) -> None:
        previous_phase = task_state.phase
        changed = task_state.transition_phase(next_phase, reason)
        if changed:
            self.logger.phase_change(task_state.step, previous_phase, next_phase, reason)

    def _phase_after_no_tool(self, task_state: TaskState) -> tuple[str, str]:
        if task_state.budget_state.no_tool_retries >= self.policy_engine.config.no_tool_retry_limit:
            return TASK_PHASE_PLAN, "No tool call emitted repeatedly; inject fallback action."
        return TASK_PHASE_PLAN, "No tool call emitted; replan for explicit tool use."

    def _phase_after_tool(
        self,
        tool_name: str,
        execution_result,
        verification: VerificationResult,
    ) -> tuple[str, str]:
        if verification.should_stop or verification.accepted and verification.status == "accepted_completion":
            return TASK_PHASE_ACCEPT, "Verifier accepted completion."
        if tool_name in OBSERVATION_TOOLS - {"read_file"}:
            return TASK_PHASE_INSPECT, f"{tool_name} gathered external or workspace evidence."
        if tool_name == "read_file":
            return TASK_PHASE_PLAN, "Source inspection complete; update the edit plan."
        if tool_name in MUTATION_TOOLS:
            if verification.accepted:
                return TASK_PHASE_VERIFY, "Mutation accepted by write verifier; run verification checks next."
            return TASK_PHASE_PLAN, f"Mutation rejected with {verification.failure_type or 'verification'} failure."
        if tool_name in EXECUTION_TOOLS:
            if verification.accepted:
                return TASK_PHASE_VERIFY, "Runtime validation succeeded; check whether acceptance criteria are met."
            return TASK_PHASE_PLAN, f"Runtime validation failed with {verification.failure_type or 'runtime'} failure."
        if not execution_result.success:
            return TASK_PHASE_PLAN, f"Tool execution failed for {tool_name}; choose a different action."
        return TASK_PHASE_PLAN, f"{tool_name} completed; re-evaluate the next step."

    def _checkpoint_payload(self, task_state: TaskState) -> dict[str, Any]:
        return {
            "step": task_state.step,
            "task_state": task_state.to_dict(),
        }

    def _recent_tool_names(self, task_state: TaskState) -> list[str]:
        return [record.tool_name for record in task_state.action_history]

    def _coerce_repeated_web_search_tool_calls(
        self,
        task_state: TaskState,
        tool_calls: list[dict[str, Any]],
        goal_text: str,
    ) -> list[dict[str, Any]]:
        if not tool_calls:
            return tool_calls
        if not all(call.get("name") == "web_search" for call in tool_calls):
            return tool_calls
        history_tools = [record.tool_name for record in task_state.action_history[-6:]]
        if sum(1 for name in history_tools if name == "web_search") < 1:
            return tool_calls
        path_guess = skill_relative_path_from_goal(goal_text)
        if path_guess:
            return [{"name": "read_file", "arguments": {"path": path_guess}}]
        return [{"name": "list_dir", "arguments": {"path": "skills"}}]

    def _run_single_tool_iteration(
        self,
        step: int,
        task_state: TaskState,
        tool_call: dict[str, Any],
        loop_detector: Any,
    ) -> tuple[bool, VerificationResult | None]:
        """Execute one tool call through verify/commit/phase. Returns (should_abort_run, verification_or_none).

        On any unexpected exception: log to events.jsonl, record failure, move to PLAN, return (False, None).
        """
        tool_name_safe = str(tool_call.get("name", "unknown_tool"))
        try:
            if not isinstance(tool_call.get("arguments"), dict):
                raise ValueError(f"Tool call missing dict arguments: {tool_call!r}")
            tool_name = str(tool_call["name"])
            tool_args = tool_call["arguments"]
            args_preview = json.dumps(tool_args, default=str)[:240]
            self.logger.tool_call(step, tool_name, tool_args)
            execution_result = self.tool_executor.execute(tool_name, tool_args)
            self.logger.tool_result(step, tool_name, execution_result.success, execution_result.summary)

            tool_total = self.perf.setdefault("tool_success", {"total": 0, "success": 0})
            tool_total["total"] = int(tool_total.get("total", 0)) + 1
            if execution_result.success:
                tool_total["success"] = int(tool_total.get("success", 0)) + 1

            if "path" in tool_args and isinstance(tool_args["path"], str):
                task_state.add_target_file(tool_args["path"])

            if tool_name in {"write_file", "edit_file", "replace_lines"} and execution_result.success and execution_result.written_path:
                self.idle_scheduler.enqueue(
                    "pre_validate",
                    lambda path_value=str(execution_result.written_path): self.pre_validate(path_value),
                )
                self.idle_scheduler.enqueue(
                    "validate_writes",
                    lambda path_value=execution_result.written_path: self.evaluate_written_file(path_value),
                )
            self.idle_scheduler.run_pending(max_time=self.policy_engine.config.idle_budget_s)

            self.state_store.record_tool_attempt(
                step=step,
                phase=task_state.phase,
                tool_name=tool_name,
                args=tool_args,
                result_text=execution_result.output,
                success=execution_result.success,
            )

            task_state.add_action_record(
                tool_name=tool_name,
                success=execution_result.success,
                args_preview=args_preview,
                result_preview=f"{execution_result.result_kind}: {execution_result.summary}",
            )
            task_state.last_result_summary = execution_result.output[:500]

            if self.memory_manager:
                self.memory_manager.record_attempt(
                    step=step,
                    tool=tool_name,
                    args=tool_args,
                    result=execution_result.output[:500],
                    success=execution_result.success,
                    learning="OK" if execution_result.success else execution_result.output[:120],
                )
                if execution_result.success and tool_name == "web_search":
                    self.memory_manager.record_discovery(execution_result.output[:200])

            verification = self.verifier.evaluate_tool_result(
                tool_name,
                execution_result.output,
                execution_result.written_path,
            )
            task_state.update_verification(
                verification.status,
                verification.accepted,
                verification.summary,
                verification.failure_type,
                verification.target_path,
            )
            if execution_result.written_path:
                task_state.add_artifact(
                    str(execution_result.written_path),
                    verification.accepted,
                    verification.status,
                    verification.summary,
                )
                if verification.accepted:
                    try:
                        commit_message = self.tool_executor.commit_mutation(execution_result.written_path)
                        task_state.add_action_record(
                            tool_name="commit_mutation",
                            success=True,
                            args_preview=str(execution_result.written_path),
                            result_preview=commit_message,
                        )
                    except Exception as commit_error:
                        self.logger.error(
                            step,
                            f"commit_mutation failed for {execution_result.written_path}: {commit_error}",
                        )
                        task_state.add_failure_reason(f"commit_mutation failed: {commit_error}")
                else:
                    try:
                        rollback_message = self.tool_executor.rollback_mutation(execution_result.written_path)
                        task_state.add_failure_reason(rollback_message)
                        task_state.add_action_record(
                            tool_name="rollback_mutation",
                            success=True,
                            args_preview=str(execution_result.written_path),
                            result_preview=rollback_message,
                        )
                    except Exception as rollback_error:
                        self.logger.error(
                            step,
                            f"rollback_mutation failed for {execution_result.written_path}: {rollback_error}",
                        )
                        task_state.add_failure_reason(f"rollback_mutation failed: {rollback_error}")

            try:
                self.state_store.record_validation(
                    step=step,
                    target_path=verification.target_path,
                    accepted=verification.accepted,
                    summary=verification.summary,
                    reward=verification.reward,
                )
            except Exception as persist_error:
                self.logger.error(step, f"record_validation failed: {persist_error}")
                task_state.add_failure_reason(f"record_validation failed: {persist_error}")

            self.logger.validation(verification.target_path or tool_name, verification.accepted, verification.summary)

            if task_state.active_skill_id:
                try:
                    reward_value = self.policy_engine.reward_from_outcome(verification)
                    self.skill_tree.update_impact_from_result(task_state.active_skill_id, reward_value)
                    self.state_store.record_reward(step, task_state.active_skill_id, reward_value, verification.summary)
                except Exception as reward_error:
                    self.logger.error(step, f"reward/skill_tree update failed: {reward_error}")
                    task_state.add_failure_reason(f"Non-fatal: reward update failed ({reward_error})")

            pre_validate_message = self.idle_scheduler.get_result("pre_validate")
            if isinstance(pre_validate_message, str) and pre_validate_message.startswith("WARN:"):
                task_state.add_failure_reason(f"Pre-validation warning: {pre_validate_message}")

            next_phase, reason = self._phase_after_tool(tool_name, execution_result, verification)

            iteration_should_stop = False
            if loop_detector and hasattr(loop_detector, "record"):
                loop_detector.record(tool_name, args_preview, execution_result.output[:200])
                if loop_detector.is_stuck():
                    task_state.budget_state.consecutive_stuck_cycles += 1
                    self.logger.loop_detected(step, tool_name, task_state.budget_state.consecutive_stuck_cycles)
                    task_state.add_failure_reason("Loop detector triggered after repeated similar actions.")
                    reward_value = self.policy_engine.reward_from_outcome(verification, loop_detected=True)
                    if task_state.active_skill_id:
                        try:
                            self.state_store.record_reward(step, task_state.active_skill_id, reward_value, "loop_detected")
                        except Exception as loop_reward_error:
                            self.logger.error(step, f"record_reward (loop) failed: {loop_reward_error}")
                    next_phase = TASK_PHASE_PLAN
                    reason = "Loop detected; replan before taking another mutation step."
                    if task_state.budget_state.consecutive_stuck_cycles >= self.policy_engine.config.stuck_abort_limit:
                        next_phase = TASK_PHASE_ABORT
                        reason = "Loop detector exceeded abort limit."
                        iteration_should_stop = True
                else:
                    task_state.budget_state.consecutive_stuck_cycles = 0

            self._log_phase_transition(task_state, next_phase, reason)
            return iteration_should_stop, verification

        except Exception as pipeline_exc:
            tail = traceback.format_exc()[-3500:]
            self.logger.error(
                step,
                f"Tool pipeline crashed ({tool_name_safe}): {pipeline_exc}\n{tail}",
            )
            task_state.add_failure_reason(f"Tool pipeline error ({tool_name_safe}): {pipeline_exc}")
            tool_total = self.perf.setdefault("tool_success", {"total": 0, "success": 0})
            tool_total["total"] = int(tool_total.get("total", 0)) + 1
            task_state.add_action_record(
                tool_name=tool_name_safe,
                success=False,
                args_preview=json.dumps(tool_call.get("arguments", {}), default=str)[:240],
                result_preview=f"pipeline_exception: {pipeline_exc}",
            )
            self._log_phase_transition(
                task_state,
                TASK_PHASE_PLAN,
                "Recovered from internal error during tool handling; try a different tool or smaller edit.",
            )
            return False, None

    def run(self, resume: bool = False) -> ControllerSummary:
        checkpoint_payload = self.state_store.load_latest_checkpoint() if resume else None
        task_state = self._restore_task_state(checkpoint_payload)
        last_verification = self._resume_verification(task_state)

        self.state_store.register_run(
            model_name=self.config_model.name,
            config={
                "max_tokens": self.config_model.max_tokens,
                "context_window": self.config_model.context_window,
                "max_iterations": self.max_iterations,
            },
            resumed=resume,
        )

        loop_detector = self.load_skill_instance("loop_detector", "LoopDetector")
        active_skill = None

        sampler_stop = threading.Event()
        sampler_thread = threading.Thread(
            target=self.resource_sampler,
            args=(self.logger.run_dir, sampler_stop),
            daemon=True,
        )
        sampler_thread.start()

        try:
            for step in range(task_state.step + 1, self.max_iterations + 1):
                task_state.mark_step(step)

                if step == 1 or step % 5 == 1 or not active_skill:
                    active_skill = self.skill_tree.peek_next_skill()
                    if active_skill:
                        task_state.active_skill_id = active_skill.get("id", "")
                        task_state.active_skill_name = active_skill.get("name", "")

                recent_tool_names = self._recent_tool_names(task_state)
                policy = self.policy_engine.build_step_policy(
                    phase=task_state.phase,
                    step=step,
                    max_iterations=self.max_iterations,
                    perf=self.perf,
                    files_written=getattr(self.tool_executor, "files_written", 0),
                    last_result=task_state.last_result_summary,
                    memory_manager=self.memory_manager,
                    current_skill=active_skill,
                    recent_tool_names=recent_tool_names,
                )
                task_state.set_plan_items(build_plan_items_from_policy(policy))
                task_state.current_hypothesis = build_task_hypothesis(task_state, policy)

                prompt_messages = self._build_prompt_messages(task_state, policy)
                budget_result = self.context_guard.enforce_budget(
                    prompt_messages,
                    self.format_prompt,
                )
                task_state.update_budget(budget_result.prompt_tokens, budget_result.action)
                self.logger.step_start(step, task_state.phase, budget_result.prompt_tokens, len(budget_result.messages))

                response = self.generate_response(budget_result.messages)
                completion_result = self.verifier.evaluate_completion_signal(response, last_verification)
                if completion_result.should_stop:
                    task_state.update_verification(
                        completion_result.status,
                        completion_result.accepted,
                        completion_result.summary,
                        completion_result.failure_type,
                        completion_result.target_path,
                    )
                    self._log_phase_transition(task_state, TASK_PHASE_ACCEPT, "Verifier-approved completion signal.")
                    self.state_store.record_validation(
                        step=step,
                        target_path=completion_result.target_path,
                        accepted=True,
                        summary=completion_result.summary,
                        reward=completion_result.reward,
                    )
                    self.state_store.save_checkpoint(step, self._checkpoint_payload(task_state))
                    return ControllerSummary(
                        completed=True,
                        accepted=True,
                        final_phase=task_state.phase,
                        steps_used=step,
                        last_verification=completion_result,
                        active_skill_id=task_state.active_skill_id,
                    )

                tool_calls = self.extract_tool_calls(response)
                if not tool_calls:
                    task_state.budget_state.no_tool_retries += 1
                    task_state.add_failure_reason("Model response did not emit a valid tool call.")
                    next_phase, reason = self._phase_after_no_tool(task_state)
                    self._log_phase_transition(task_state, next_phase, reason)
                    if task_state.budget_state.no_tool_retries >= self.policy_engine.config.no_tool_retry_limit:
                        tool_calls = [
                            self.policy_engine.fallback_tool_call(
                                task_state.phase,
                                self.goal,
                                policy.suggested_tool,
                                recent_tool_names=self._recent_tool_names(task_state),
                            )
                        ]
                        task_state.budget_state.no_tool_retries = 0
                    else:
                        self.state_store.save_checkpoint(step, self._checkpoint_payload(task_state))
                        continue
                else:
                    task_state.budget_state.no_tool_retries = 0

                tool_calls = self._coerce_repeated_web_search_tool_calls(task_state, tool_calls, self.goal)

                selected_calls = self.policy_engine.select_tool_batch(tool_calls)
                should_stop = False
                for tool_call in selected_calls:
                    batch_abort, ver = self._run_single_tool_iteration(
                        step, task_state, tool_call, loop_detector
                    )
                    if ver is not None:
                        last_verification = ver
                    if batch_abort:
                        should_stop = True
                        break

                self.state_store.save_checkpoint(step, self._checkpoint_payload(task_state))
                if should_stop:
                    break

            return ControllerSummary(
                completed=task_state.completed,
                accepted=bool(last_verification and last_verification.accepted) or task_state.accepted,
                final_phase=task_state.phase,
                steps_used=task_state.step,
                last_verification=last_verification,
                active_skill_id=task_state.active_skill_id,
            )
        finally:
            sampler_stop.set()
            sampler_thread.join(timeout=2)
