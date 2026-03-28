"""Policy engine for controller decisions, rewards, and prompt guidance."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def skill_relative_path_from_goal(goal_text: str) -> str | None:
    """Resolve skills/foo.py from a skill-tree goal line 'File: foo.py'."""
    match = re.search(r"File:\s*(\S+\.py)", goal_text)
    if not match:
        return None
    return f"skills/{match.group(1)}"

from src.paths import RUNS_DIR
from src.runtime.tool_kinds import READ_BATCH_SAFE_TOOLS
from src.write_guard import AtomicWriter


POLICY_FILE = RUNS_DIR / "controller_policy.json"


@dataclass
class ControllerPolicyConfig:
    """Mutable policy configuration that can be improved without editing code."""

    low_confidence_threshold: float = 0.3
    no_tool_retry_limit: int = 3
    stuck_abort_limit: int = 3
    read_batch_limit: int = 3
    max_tool_calls_per_step: int = 2
    idle_budget_s: float = 5.0


@dataclass
class StepPolicy:
    """Structured policy output for one controller step."""

    confidence: float
    action: str
    phase: str
    suggested_tool: str
    guidance_messages: list[str]
    active_skill_id: str
    active_skill_name: str
    completed_tasks: list[str]


class PolicyEngine:
    """Skill-aware policy engine for the controller."""

    def __init__(self, load_skill_instance, goal: str) -> None:
        self.load_skill_instance = load_skill_instance
        self.goal = goal
        self._writer = AtomicWriter(minimum_python_bytes=32)
        self.config = self._load_config()

    def _load_config(self) -> ControllerPolicyConfig:
        if POLICY_FILE.exists():
            try:
                return ControllerPolicyConfig(**json.loads(POLICY_FILE.read_text()))
            except Exception:
                pass
        config = ControllerPolicyConfig()
        self.save_config(config)
        return config

    def save_config(self, config: ControllerPolicyConfig | None = None) -> None:
        selected_config = config or self.config
        POLICY_FILE.parent.mkdir(parents=True, exist_ok=True)
        result = self._writer.write_text(POLICY_FILE, json.dumps(asdict(selected_config), indent=2))
        if not result.success:
            raise RuntimeError(result.message)

    def _task_plan(self) -> tuple[Any, list[dict[str, Any]]]:
        planner_module = self.load_skill_instance("task_planner", None)
        if not planner_module or not hasattr(planner_module, "TaskPlanner"):
            return None, []
        try:
            planner = planner_module.TaskPlanner(self.goal)
            return planner, planner.decompose()
        except Exception:
            return None, []

    def confidence_snapshot(
        self,
        step: int,
        max_iterations: int,
        perf: dict[str, Any],
        files_written: int,
        last_result: str,
        memory_manager,
    ) -> dict[str, Any]:
        scorer_module = self.load_skill_instance("confidence_scorer", None)
        if not scorer_module or not hasattr(scorer_module, "ConfidenceScorer") or not memory_manager:
            return {"knowledge": 0.0, "capability": 0.0, "progress": 0.0, "overall": 0.0, "action": "research"}

        scorer = scorer_module.ConfidenceScorer()
        tool_total = perf["tool_success"]["total"]
        tool_success = perf["tool_success"]["success"]
        errors_count = max(0, tool_total - tool_success)
        relevant_keywords = [
            word_value
            for word_value in set(self.goal.lower().split())
            if len(word_value) > 3 and word_value in last_result.lower()
        ]
        knowledge_score = scorer.score_knowledge(
            len(memory_manager.memory.discoveries),
            relevant_keywords,
        )
        capability_score = scorer.score_capability(
            (tool_success / max(1, tool_total)) * 100.0,
            errors_count,
        )
        progress_score = scorer.score_progress(step - 1, max(1, max_iterations), files_written)
        overall_score = scorer.overall(knowledge_score, capability_score, progress_score)
        return {
            "knowledge": knowledge_score,
            "capability": capability_score,
            "progress": progress_score,
            "overall": overall_score,
            "action": scorer.should_act(knowledge_score, capability_score, progress_score),
        }

    def build_step_policy(
        self,
        phase: str,
        step: int,
        max_iterations: int,
        perf: dict[str, Any],
        files_written: int,
        last_result: str,
        memory_manager,
        current_skill: dict[str, Any] | None,
        recent_tool_names: list[str] | None = None,
    ) -> StepPolicy:
        confidence = self.confidence_snapshot(step, max_iterations, perf, files_written, last_result, memory_manager)
        guidance_messages: list[str] = []
        planner, tasks = self._task_plan()
        completed_task_names = []

        if planner and memory_manager:
            completed_task_names = [
                iteration.tool_used
                for iteration in memory_manager.memory.iterations[-5:]
                if iteration.success
            ]
            try:
                next_task = planner.next_task(completed_task_names)
            except Exception:
                next_task = None
            if next_task:
                guidance_messages.append(
                    f"Planner next task: {next_task.get('task', 'unknown')} using {next_task.get('tool', 'unknown')}."
                )

        router_module = self.load_skill_instance("smart_router", None)
        suggested_tool = ""
        if router_module and hasattr(router_module, "SmartRouter"):
            try:
                router = router_module.SmartRouter()
                router.current_phase = phase
                suggested_tool = router.pick_tool(
                    confidence["knowledge"],
                    confidence["capability"],
                    confidence["progress"],
                ) or ""
                if suggested_tool:
                    guidance_messages.append(f"Router suggests tool: {suggested_tool}.")
            except Exception:
                suggested_tool = ""

        recent_names = list(recent_tool_names or [])
        web_search_count = sum(1 for name in recent_names if name == "web_search")
        read_like_count = sum(1 for name in recent_names if name in ("read_file", "list_dir"))
        if web_search_count >= 1 and read_like_count < 1:
            suggested_tool = "read_file"
            guidance_messages.append(
                "External search already used this task: next tool must be read_file (or list_dir) on the target skill file, not web_search."
            )
        elif web_search_count >= 2 and files_written == 0 and phase in ("inspect", "plan", "discover"):
            target_hint = skill_relative_path_from_goal(self.goal)
            if target_hint:
                guidance_messages.append(
                    f"Enough web_search steps: open {target_hint} with read_file, then replace_lines, edit_file, or write_file."
                )

        if confidence["overall"] < self.config.low_confidence_threshold:
            guidance_messages.append(
                "Confidence is low. Read code or gather evidence before writing more code."
            )

        active_skill_id = current_skill.get("id", "") if current_skill else ""
        active_skill_name = current_skill.get("name", "") if current_skill else ""
        if active_skill_name:
            guidance_messages.append(f"Active skill target: {active_skill_name}.")

        return StepPolicy(
            confidence=confidence["overall"],
            action=confidence["action"],
            phase=phase,
            suggested_tool=suggested_tool,
            guidance_messages=guidance_messages,
            active_skill_id=active_skill_id,
            active_skill_name=active_skill_name,
            completed_tasks=completed_task_names,
        )

    def select_tool_batch(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not tool_calls:
            return []
        selected_calls = []
        for call in tool_calls:
            if len(selected_calls) >= self.config.max_tool_calls_per_step:
                break
            if selected_calls and call["name"] not in READ_BATCH_SAFE_TOOLS:
                break
            if not selected_calls or call["name"] in READ_BATCH_SAFE_TOOLS:
                selected_calls.append(call)
            if len(selected_calls) >= self.config.read_batch_limit and call["name"] in READ_BATCH_SAFE_TOOLS:
                break
        return selected_calls or tool_calls[:1]

    def fallback_tool_call(
        self,
        phase: str,
        goal: str,
        suggested_tool: str = "",
        recent_tool_names: list[str] | None = None,
    ) -> dict[str, Any]:
        phase_defaults = {
            "discover": "web_search",
            "inspect": "read_file",
            "plan": "read_file",
            "patch": "write_file",
            "verify": "run_python",
            "research": "web_search",
            "code": "run_python",
            "save": "write_file",
        }
        preferred_tool = suggested_tool or phase_defaults.get(phase, "read_file")
        recent_names = list(recent_tool_names or [])
        if preferred_tool == "web_search" and any(name == "web_search" for name in recent_names[-6:]):
            skill_path = skill_relative_path_from_goal(goal)
            if skill_path:
                return {"name": "read_file", "arguments": {"path": skill_path}}
            return {"name": "list_dir", "arguments": {"path": "skills"}}
        if preferred_tool == "web_search":
            return {"name": "web_search", "arguments": {"query": goal[:80]}}
        if preferred_tool == "read_file":
            return {"name": "list_dir", "arguments": {"path": "."}}
        if preferred_tool == "run_python":
            return {"name": "run_python", "arguments": {"code": "print('validation probe')"}}
        return {"name": "write_file", "arguments": {"path": "output.py", "content": "print('checkpoint artifact')"}}

    def reward_from_outcome(self, verification_result, loop_detected: bool = False) -> float:
        reward_value = float(getattr(verification_result, "reward", 0.0))
        if loop_detected:
            reward_value -= 0.5
        return reward_value
