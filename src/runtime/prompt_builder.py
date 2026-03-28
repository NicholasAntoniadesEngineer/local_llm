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
    static_context_block: str = "",
) -> list[dict[str, Any]]:
    web_search_used = any(item.tool_name == "web_search" for item in task_state.action_history)
    repo_inspection_used = any(
        item.tool_name in ("read_file", "list_dir") for item in task_state.action_history
    )

    directive_block = "\n".join(
        [
            "/nothink",
            "You are an autonomous coding agent operating inside a strict verifier-driven runtime.",
            "You must make progress through tool calls. Do not claim success until the verifier has accepted an artifact.",
            "Respond with precise tool calls. Keep reasoning concise and action-oriented.",
            "Editing: use read_file with start_line/end_line for large skills; use replace_lines for multi-line changes; "
            "use edit_file only when replacing a unique exact substring.",
            "After at least one web_search, read the target skill file (read_file) or list skills/; "
            "do not repeat web_search with the same query.",
        ]
    )

    frozen_prefix = static_context_block.strip()
    if not frozen_prefix:
        frozen_prefix = "## Static context\n(Repo bootstrap did not supply a block; KV reuse may be limited.)\n"

    situation_block = "\n".join(
        [
            f"Goal: {task_state.goal_text}",
            f"Current phase: {task_state.phase}",
            f"Task signals: web_search_used={web_search_used}; repo_inspection_used={repo_inspection_used}.",
        ]
    )

    working_memory_sections: list[str] = []
    if task_state.target_files:
        working_memory_sections.append(
            "Tracked files:\n" + "\n".join(f"- {item}" for item in task_state.target_files[-6:])
        )
    if task_state.plan_items:
        working_memory_sections.append(
            "Current task plan:\n" + "\n".join(f"- {item}" for item in task_state.plan_items)
        )
    if task_state.current_hypothesis:
        working_memory_sections.append("Current hypothesis:\n- " + task_state.current_hypothesis)
    working_memory_block = "\n\n".join(working_memory_sections) if working_memory_sections else ""

    verifier_sections: list[str] = []
    if task_state.last_verification_summary:
        verifier_sections.append(
            "Latest verifier result:\n"
            f"- status={task_state.verifier_status}\n"
            f"- accepted={task_state.last_verification_accepted}\n"
            f"- failure_type={task_state.last_failure_type or 'none'}\n"
            f"- summary={task_state.last_verification_summary}"
        )
    if task_state.failure_reasons:
        verifier_sections.append(
            "Recent failures:\n" + "\n".join(f"- {item}" for item in task_state.failure_reasons[-5:])
        )
    verifier_block = "\n\n".join(verifier_sections) if verifier_sections else ""

    action_block = ""
    if task_state.action_history:
        action_lines = [
            f"- step={item.step} phase={item.phase} tool={item.tool_name} ok={item.success} "
            f"args={item.args_preview} result={item.result_preview}"
            for item in task_state.action_history[-5:]
        ]
        action_block = "Recent actions:\n" + "\n".join(action_lines)

    history_block_parts: list[str] = []
    if retrieval_context:
        history_block_parts.append(
            "Relevant past records:\n" + "\n".join(f"- {line}" for line in retrieval_context)
        )
    if past_failures:
        history_block_parts.append(
            "Historical verifier failures:\n" + "\n".join(f"- {line}" for line in past_failures[:4])
        )
    if memory_context:
        history_block_parts.append(memory_context)
    history_block = "\n\n".join(history_block_parts) if history_block_parts else ""

    dynamic_sections: list[str] = ["## Situation\n" + situation_block]
    if working_memory_block:
        dynamic_sections.append("## Working memory\n" + working_memory_block)
    if verifier_block:
        dynamic_sections.append("## Verifier feedback\n" + verifier_block)
    if action_block:
        dynamic_sections.append("## Recent tool trace\n" + action_block)
    if history_block:
        dynamic_sections.append("## Long-term context\n" + history_block)

    user_sections = dynamic_sections + [
        "## Step instruction\n"
        + f"Work the current phase: {task_state.phase}.\n"
        + "If information is missing, inspect files first. If a failure repeats, choose a materially different action.",
    ]
    if policy.guidance_messages:
        user_sections.append(
            "## Controller guidance\n" + "\n".join(f"- {item}" for item in policy.guidance_messages[:4])
        )

    system_content = frozen_prefix + "\n\n## Directive\n" + directive_block

    return [
        protected_message("system", system_content),
        protected_message("user", "\n\n".join(user_sections)),
    ]
