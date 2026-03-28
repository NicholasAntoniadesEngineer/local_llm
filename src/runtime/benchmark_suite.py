"""Fixed benchmark slices for controller regression tracking."""

from __future__ import annotations


FIXED_BENCHMARK_SLICE_NAME = "v2_fixed_slice"

FIXED_BENCHMARK_CASES = [
    {
        "id": "inspect_controller_phase_machine",
        "goal": (
            "Inspect the runtime controller phase machine and produce a verifier-accepted "
            "artifact that reports the current phase flow and the most likely replanning failure."
        ),
    },
    {
        "id": "verifier_failure_recovery",
        "goal": (
            "Trigger a small runtime verification workflow, recover from one rejected change, "
            "and produce a verifier-accepted artifact that records the failure class and retry path."
        ),
    },
    {
        "id": "task_state_resume_snapshot",
        "goal": (
            "Exercise task-state persistence and checkpoint behavior, then produce a verifier-accepted "
            "artifact that summarizes step, phase, target files, and last failure type."
        ),
    },
]


def get_fixed_benchmark_cases() -> list[dict[str, str]]:
    """Return the canonical fixed benchmark slice for v2 regression tracking."""
    return [dict(case_record) for case_record in FIXED_BENCHMARK_CASES]
