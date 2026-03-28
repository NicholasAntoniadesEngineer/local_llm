"""Explicit task-state objects for the verifier-driven controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


TASK_PHASE_DISCOVER = "discover"
TASK_PHASE_INSPECT = "inspect"
TASK_PHASE_PLAN = "plan"
TASK_PHASE_PATCH = "patch"
TASK_PHASE_VERIFY = "verify"
TASK_PHASE_ACCEPT = "accept"
TASK_PHASE_ABORT = "abort"

TERMINAL_TASK_PHASES = {
    TASK_PHASE_ACCEPT,
    TASK_PHASE_ABORT,
}


@dataclass
class TaskBudgetState:
    """Budget state tracked for each controller step."""

    prompt_tokens: int = 0
    budget_action: str = "none"
    no_tool_retries: int = 0
    consecutive_stuck_cycles: int = 0


@dataclass
class TaskArtifact:
    """Artifact written or validated during a task."""

    path: str
    step: int
    verifier_status: str = ""
    accepted: bool = False
    summary: str = ""


@dataclass
class TaskActionRecord:
    """Structured record of a recent tool action."""

    step: int
    phase: str
    tool_name: str
    success: bool
    args_preview: str
    result_preview: str


@dataclass
class PhaseTransitionRecord:
    """Structured phase transition for auditability and resume."""

    step: int
    from_phase: str
    to_phase: str
    reason: str


@dataclass
class TaskState:
    """Persisted controller state, independent from prompt transcript growth."""

    task_id: str
    goal_text: str
    phase: str = TASK_PHASE_DISCOVER
    step: int = 0
    verifier_status: str = "pending"
    last_verification_summary: str = ""
    last_verification_accepted: bool = False
    last_failure_type: str = ""
    last_result_summary: str = ""
    current_hypothesis: str = ""
    active_skill_id: str = ""
    active_skill_name: str = ""
    target_files: list[str] = field(default_factory=list)
    plan_items: list[str] = field(default_factory=list)
    completed_plan_items: list[str] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    artifacts_written: list[TaskArtifact] = field(default_factory=list)
    action_history: list[TaskActionRecord] = field(default_factory=list)
    phase_transitions: list[PhaseTransitionRecord] = field(default_factory=list)
    budget_state: TaskBudgetState = field(default_factory=TaskBudgetState)
    accepted: bool = False
    completed: bool = False

    def mark_step(self, step: int) -> None:
        """Advance the state to the current controller step."""
        self.step = step

    def transition_phase(self, next_phase: str, reason: str) -> bool:
        """Change phase with an explicit, persisted reason."""
        if next_phase == self.phase:
            return False
        self.phase_transitions.append(
            PhaseTransitionRecord(
                step=self.step,
                from_phase=self.phase,
                to_phase=next_phase,
                reason=reason[:240],
            )
        )
        self.phase = next_phase
        if next_phase in TERMINAL_TASK_PHASES:
            self.completed = True
            self.accepted = next_phase == TASK_PHASE_ACCEPT
        return True

    def add_target_file(self, path_value: str) -> None:
        """Track the distinct files touched by the controller."""
        if path_value and path_value not in self.target_files:
            self.target_files.append(path_value)

    def add_failure_reason(self, reason_text: str, max_items: int = 8) -> None:
        """Keep a compact list of recent failure reasons."""
        if not reason_text:
            return
        self.failure_reasons.append(reason_text[:240])
        self.failure_reasons = self.failure_reasons[-max_items:]

    def add_action_record(
        self,
        tool_name: str,
        success: bool,
        args_preview: str,
        result_preview: str,
        max_items: int = 8,
    ) -> None:
        """Store a compact action log for prompt assembly and resume."""
        self.action_history.append(
            TaskActionRecord(
                step=self.step,
                phase=self.phase,
                tool_name=tool_name,
                success=success,
                args_preview=args_preview[:240],
                result_preview=result_preview[:320],
            )
        )
        self.action_history = self.action_history[-max_items:]

    def add_artifact(self, path_value: str, accepted: bool, verifier_status: str, summary: str) -> None:
        """Track each written artifact and its latest verifier state."""
        if not path_value:
            return
        self.add_target_file(path_value)
        self.artifacts_written.append(
            TaskArtifact(
                path=path_value,
                step=self.step,
                verifier_status=verifier_status,
                accepted=accepted,
                summary=summary[:320],
            )
        )
        self.artifacts_written = self.artifacts_written[-12:]

    def update_verification(
        self,
        status: str,
        accepted: bool,
        summary: str,
        failure_type: str,
        target_path: str = "",
    ) -> None:
        """Update the latest verifier state."""
        self.verifier_status = status
        self.last_verification_accepted = accepted
        self.last_verification_summary = summary[:320]
        self.last_failure_type = failure_type[:80]
        if target_path:
            self.add_target_file(target_path)
        if not accepted and summary:
            self.add_failure_reason(summary)

    def update_budget(self, prompt_tokens: int, budget_action: str) -> None:
        """Persist the current prompt budget status."""
        self.budget_state.prompt_tokens = prompt_tokens
        self.budget_state.budget_action = budget_action

    def set_plan_items(self, items: list[str]) -> None:
        """Persist the current plan items."""
        self.plan_items = [item[:200] for item in items[:8]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a checkpoint-safe dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskState":
        """Hydrate from a checkpoint dictionary."""
        budget_payload = payload.get("budget_state", {})
        artifact_payloads = payload.get("artifacts_written", [])
        action_payloads = payload.get("action_history", [])
        phase_payloads = payload.get("phase_transitions", [])
        return cls(
            task_id=payload["task_id"],
            goal_text=payload["goal_text"],
            phase=payload.get("phase", TASK_PHASE_DISCOVER),
            step=payload.get("step", 0),
            verifier_status=payload.get("verifier_status", "pending"),
            last_verification_summary=payload.get("last_verification_summary", ""),
            last_verification_accepted=payload.get("last_verification_accepted", False),
            last_failure_type=payload.get("last_failure_type", ""),
            last_result_summary=payload.get("last_result_summary", ""),
            current_hypothesis=payload.get("current_hypothesis", ""),
            active_skill_id=payload.get("active_skill_id", ""),
            active_skill_name=payload.get("active_skill_name", ""),
            target_files=list(payload.get("target_files", [])),
            plan_items=list(payload.get("plan_items", [])),
            completed_plan_items=list(payload.get("completed_plan_items", [])),
            failure_reasons=list(payload.get("failure_reasons", [])),
            artifacts_written=[TaskArtifact(**artifact_payload) for artifact_payload in artifact_payloads],
            action_history=[TaskActionRecord(**action_payload) for action_payload in action_payloads],
            phase_transitions=[PhaseTransitionRecord(**phase_payload) for phase_payload in phase_payloads],
            budget_state=TaskBudgetState(**budget_payload),
            accepted=payload.get("accepted", False),
            completed=payload.get("completed", False),
        )
