"""Pure prompt-assembly helpers for the verifier-driven controller."""

from __future__ import annotations

from typing import Any, Protocol

from src.runtime.task_state import TaskState


class StepPolicyView(Protocol):
    """Minimal policy shape required for prompt construction."""

    guidance_messages: list[str]
    suggested_tool: str
    confidence: float
    action: str
    active_skill_name: str


def protected_message(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content, "protected": True}


def build_plan_items_from_policy(policy: StepPolicyView) -> list[str]:
    plan_items: list[str] = []
    if policy.active_skill_name:
        plan_items.append(f"Advance active skill target: {policy.active_skill_name}")
    if policy.suggested_tool:
        plan_items.append(f"Preferred next tool: {policy.suggested_tool}")
    if policy.action:
        plan_items.append(f"Primary controller action: {policy.action}")
    for guidance_message in policy.guidance_messages[:4]:
        plan_items.append(guidance_message)
    return plan_items[:6]


def build_task_hypothesis(task_state: TaskState, policy: StepPolicyView) -> str:
    plan_preview = policy.guidance_messages[0] if policy.guidance_messages else "Gather evidence before mutating files."
    return (
        f"Phase={task_state.phase}; confidence={policy.confidence:.2f}; "
        f"next={policy.suggested_tool or policy.action or 'inspect'}; {plan_preview[:160]}"
    )


def build_prompt_messages(
    task_state: TaskState,
    policy: StepPolicyView,
    *,
    memory_context: str,
    retrieval_context: list[str],
    past_failures: list[str],
) -> list[dict[str, Any]]:
    system_parts = [
        "/nothink",
        "You are an autonomous coding agent operating inside a strict verifier-driven runtime.",
        f"Goal: {task_state.goal_text}",
        f"Current phase: {task_state.phase}",
        "You must make progress through tool calls. Do not claim success until the verifier has accepted an artifact.",
        "Respond with precise tool calls. Keep reasoning concise and action-oriented.",
    ]
    if task_state.target_files:
        system_parts.append("Tracked files:\n" + "\n".join(f"- {item}" for item in task_state.target_files[-6:]))
    if task_state.plan_items:
        system_parts.append("Current task plan:\n" + "\n".join(f"- {item}" for item in task_state.plan_items))
    if task_state.current_hypothesis:
        system_parts.append("Current hypothesis:\n- " + task_state.current_hypothesis)
    if task_state.last_verification_summary:
        system_parts.append(
            "Latest verifier result:\n"
            f"- status={task_state.verifier_status}\n"
            f"- accepted={task_state.last_verification_accepted}\n"
            f"- failure_type={task_state.last_failure_type or 'none'}\n"
            f"- summary={task_state.last_verification_summary}"
        )
    if task_state.failure_reasons:
        system_parts.append("Recent failures:\n" + "\n".join(f"- {item}" for item in task_state.failure_reasons[-5:]))
    if task_state.action_history:
        action_lines = [
            f"- step={item.step} phase={item.phase} tool={item.tool_name} ok={item.success} "
            f"args={item.args_preview} result={item.result_preview}"
            for item in task_state.action_history[-5:]
        ]
        system_parts.append("Recent actions:\n" + "\n".join(action_lines))
    if retrieval_context:
        system_parts.append("Relevant past records:\n" + "\n".join(f"- {line}" for line in retrieval_context))
    if past_failures:
        system_parts.append("Historical verifier failures:\n" + "\n".join(f"- {line}" for line in past_failures[:4]))
    if memory_context:
        system_parts.append(memory_context)

    user_parts = [
        f"Work the current phase: {task_state.phase}.",
        "If information is missing, inspect files first. If a failure repeats, choose a materially different action.",
    ]
    if policy.guidance_messages:
        user_parts.append("Controller guidance:\n" + "\n".join(f"- {item}" for item in policy.guidance_messages[:4]))

    return [
        protected_message("system", "\n\n".join(system_parts)),
        protected_message("user", "\n\n".join(user_parts)),
    ]
