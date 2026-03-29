"""Runtime-backed improvement cycle runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import shutil
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.paths import IMPROVE_SESSION_FILE
from src.runtime.verifier import validate_generated_module
from src.skill_tree import SkillTree

if TYPE_CHECKING:
    from src.agent import MLXAgent


@dataclass
class ImprovementScenario:
    """Selected improvement scenario for one cycle."""

    cycle_num: int
    skill_id: str
    skill_name: str
    action: str
    goal_text: str
    target_path: Path


@dataclass
class ImprovementCycleResult:
    """Outcome of one shared-runtime improvement cycle."""

    scenario: ImprovementScenario | None
    accepted: bool
    summary: str
    outcome: str


def _append_improve_journal(record: dict[str, Any]) -> None:
    IMPROVE_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {**record, "timestamp": datetime.now().isoformat()}
    with IMPROVE_SESSION_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def select_improvement_scenario(cycle_num: int, skill_tree: SkillTree) -> ImprovementScenario | None:
    """Choose the next skill build or upgrade scenario from the skill tree."""
    skill_tree.evolve_tree()
    new_skill = skill_tree.peek_next_skill()
    weak_skill = skill_tree.get_weakest_skill()

    if new_skill:
        selected_skill = new_skill
        action_name = "BUILDING"
        goal_text = skill_tree.build_goal_for_skill(selected_skill)
        skill_tree.record_pull(selected_skill["id"])
    elif weak_skill and weak_skill.get("quality_score", 999) < 200:
        selected_skill = weak_skill
        action_name = "UPGRADING"
        goal_text = skill_tree.build_upgrade_goal(selected_skill)
    else:
        return None

    return ImprovementScenario(
        cycle_num=cycle_num,
        skill_id=selected_skill["id"],
        skill_name=selected_skill["name"],
        action=action_name,
        goal_text=goal_text,
        target_path=Path("skills") / selected_skill["file"],
    )


def _restore_target_file(target_path: Path, backup_path: Path) -> None:
    """Restore the last known-good target file when post-run validation fails."""
    if backup_path.exists():
        shutil.copy2(backup_path, target_path)
        backup_path.unlink()
        return
    if target_path.exists():
        target_path.unlink()


def run_improvement_cycle(
    cycle_num: int,
    model_name: str,
    agent: "MLXAgent | None" = None,
) -> ImprovementCycleResult:
    """Run one improvement cycle using the shared runtime controller.

    Callers should invoke ``apply_self_improve_runtime_environment()`` once before
    the first cycle (``tools/improve.py`` does this); it is not repeated here to
    avoid redundant env work every cycle.
    """
    # Reuse the agent's SkillTree when looping so the controller, verifier, and
    # scenario picker share one DB handle + in-memory graph (avoids "no such table"
    # / stale-graph races from a second SkillTree() on the same file).
    skill_tree = agent.skill_tree if agent is not None else SkillTree()
    scenario = select_improvement_scenario(cycle_num, skill_tree)
    if scenario is None:
        result = ImprovementCycleResult(
            scenario=None,
            accepted=False,
            summary="All skills complete and no weak completed skill requires upgrade.",
            outcome="idle",
        )
        _append_improve_journal({"cycle_num": cycle_num, "outcome": result.outcome, "summary": result.summary})
        return result

    scenario.target_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = scenario.target_path.with_suffix(scenario.target_path.suffix + ".bak")
    if scenario.target_path.exists():
        shutil.copy2(scenario.target_path, backup_path)

    pre_ok = False
    pre_message = ""
    if scenario.target_path.exists():
        pre_ok, pre_message = validate_generated_module(str(scenario.target_path), skill_tree=skill_tree)
    if pre_ok:
        skill_tree.mark_completed(scenario.skill_id, pre_message)
        if backup_path.exists():
            backup_path.unlink()
        result = ImprovementCycleResult(
            scenario=scenario,
            accepted=True,
            summary=f"Pre-validated: {pre_message}",
            outcome="pre_validated",
        )
        _append_improve_journal(
            {
                "cycle_num": cycle_num,
                "outcome": result.outcome,
                "skill_id": scenario.skill_id,
                "target_path": str(scenario.target_path),
                "summary": result.summary,
            }
        )
        return result

    if agent is None:
        from src.agent import MLXAgent

        agent = MLXAgent(config_model_name=model_name, goal=scenario.goal_text)
    else:
        agent.reset_for_new_task(scenario.goal_text)
    agent.run_loop(scenario.goal_text)

    accepted, summary = validate_generated_module(str(scenario.target_path), skill_tree=skill_tree)
    agent.state_store.record_validation(
        step=cycle_num,
        target_path=str(scenario.target_path),
        accepted=accepted,
        summary=summary,
        reward=1.0 if accepted else -1.0,
    )
    agent.state_store.record_reward(
        step=cycle_num,
        skill_id=scenario.skill_id,
        reward=1.0 if accepted else -1.0,
        reason=summary,
    )

    if accepted:
        skill_tree.mark_completed(scenario.skill_id, summary)
        if backup_path.exists():
            backup_path.unlink()
    else:
        skill_tree.mark_failed(scenario.skill_id, summary)
        _restore_target_file(scenario.target_path, backup_path)

    agent.state_store.update_run_status(
        improvement_cycle=cycle_num,
        improvement_action=scenario.action,
        improvement_skill_id=scenario.skill_id,
        improvement_target_path=str(scenario.target_path),
        post_validation_accepted=accepted,
    )
    outcome = "accepted" if accepted else "failed"
    result = ImprovementCycleResult(
        scenario=scenario,
        accepted=accepted,
        summary=summary,
        outcome=outcome,
    )
    _append_improve_journal(
        {
            "cycle_num": cycle_num,
            "outcome": outcome,
            "skill_id": scenario.skill_id,
            "target_path": str(scenario.target_path),
            "summary": summary,
        }
    )
    return result
