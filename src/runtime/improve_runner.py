"""Runtime-backed improvement cycle runner.

v2: Direct generation mode — bypasses the multi-step controller loop.
Generates complete skill modules in 1-3 attempts with retry + temperature.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import shutil
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.paths import IMPROVE_SESSION_FILE, RUNS_DIR, SKILLS_DIR
from src.runtime.verifier import validate_generated_module
from src.skill_tree import SkillTree

if TYPE_CHECKING:
    from src.agent import MLXAgent


# ── Paths for feedback loop ──────────────────────────────────────────────

SUCCESSFUL_GENERATIONS_FILE = RUNS_DIR / "successful_generations.jsonl"
METRICS_FILE = RUNS_DIR / "metrics.json"


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


def _load_metrics() -> dict[str, Any]:
    """Load or initialize the metrics tracker."""
    if METRICS_FILE.exists():
        try:
            return json.loads(METRICS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "real_generation_count": 0,
        "real_pass_count": 0,
        "total_attempts": 0,
        "skills_completed_by_llm": [],
    }


def _save_metrics(metrics: dict[str, Any]) -> None:
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    derived = {**metrics}
    gen_count = derived.get("real_generation_count", 0)
    derived["real_pass_rate"] = (
        round(derived.get("real_pass_count", 0) / gen_count, 3) if gen_count > 0 else 0.0
    )
    METRICS_FILE.write_text(json.dumps(derived, indent=2, default=str))


def _save_successful_generation(skill_id: str, prompt: str, code: str) -> None:
    """Save a successful (prompt, code) pair for future few-shot use."""
    SUCCESSFUL_GENERATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "skill_id": skill_id,
        "prompt_preview": prompt[:500],
        "code": code,
        "timestamp": datetime.now().isoformat(),
        "lines": len(code.splitlines()),
    }
    with SUCCESSFUL_GENERATIONS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _load_few_shot_example() -> str:
    """Load the best recent successful generation as a few-shot example."""
    if not SUCCESSFUL_GENERATIONS_FILE.exists():
        return _builtin_few_shot_example()
    try:
        lines = SUCCESSFUL_GENERATIONS_FILE.read_text().strip().splitlines()
        if not lines:
            return _builtin_few_shot_example()
        # Use the most recent success, preferring shorter ones (cleaner examples)
        best = None
        for line in reversed(lines[-10:]):
            record = json.loads(line)
            if best is None or record.get("lines", 999) < best.get("lines", 999):
                best = record
        if best and best.get("code"):
            return f"# ── Example of a passing skill module ──\n{best['code']}"
    except (json.JSONDecodeError, OSError):
        pass
    return _builtin_few_shot_example()


def _builtin_few_shot_example() -> str:
    """Fallback few-shot example using an existing passing skill."""
    example_path = SKILLS_DIR / "error_recovery.py"
    if example_path.exists():
        source = example_path.read_text()
        return f"# ── Example of a passing skill module (error_recovery.py) ──\n{source}"
    return ""


def select_improvement_scenario(cycle_num: int, skill_tree: SkillTree) -> ImprovementScenario | None:
    """Choose the next skill build or upgrade scenario from the skill tree."""
    skill_tree.evolve_tree()
    new_skill = skill_tree.peek_next_skill()

    if new_skill:
        selected_skill = new_skill
        action_name = "BUILDING"
        goal_text = skill_tree.build_goal_for_skill(selected_skill)
        skill_tree.record_pull(selected_skill["id"])
    else:
        # Only upgrade if there's a genuinely weak skill (not already strong)
        weak_skill = skill_tree.get_weakest_skill()
        if weak_skill and weak_skill.get("quality_score", 999) < 150:
            selected_skill = weak_skill
            action_name = "UPGRADING"
            goal_text = skill_tree.build_upgrade_goal(selected_skill)
        else:
            return None

    # Record cooldown to prevent re-selecting same skill next cycle
    skill_tree.record_attempt(selected_skill["id"])

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


def _build_direct_generation_prompt(
    scenario: ImprovementScenario,
    skill_tree: SkillTree,
    failure_context: str = "",
) -> str:
    """Build a focused prompt for single-shot skill generation."""
    skill = skill_tree._skill_dict(scenario.skill_id)
    if not skill:
        return scenario.goal_text

    # Get full prerequisite source code (not just signatures)
    prereq_source = skill_tree.read_prereq_full_source(skill)

    # Get a few-shot example
    few_shot = _load_few_shot_example()

    # Build compact, focused prompt
    fail_block = ""
    if failure_context:
        fail_block = f"""
=== PREVIOUS ATTEMPT FAILED ===
{failure_context}
Fix this specific issue. Do not repeat the same mistake.
"""

    return f"""/nothink
Write a complete, standalone Python module.
{fail_block}
=== TASK ===
Module: {skill['name']}
File: skills/{skill['file']}
Description: {skill.get('description', '')}
Specification: {skill.get('spec', '')}

=== RULES (code MUST pass ALL or it gets deleted) ===
1. MINIMUM 3 functions/methods (not counting __init__)
2. MINIMUM 4 assert statements (real Python `assert`)
3. No 'from src.' imports — standalone only
4. Must print 'ALL TESTS PASSED' when run with python3
5. Real logic in each method (not stubs/pass)
6. Handle edge cases: empty inputs, None values
7. If >100 lines: must include try/except error handling
8. Include 'if __name__ == "__main__":' test block

=== PREREQUISITE CODE (import with `from <module> import *`) ===
{prereq_source}

=== IMPORT RULES ===
- Import prereqs: from <module_name> import <ClassName>
- NEVER import from src.*
- Each prereq must be USED (not just imported)
- Catch ImportError and provide standalone fallback

=== TEST HINTS ===
{skill.get('test_hint', 'Write thorough tests')}

{few_shot}

Output ONLY the complete Python code for {skill['file']}. No explanation, no markdown fences."""


def run_direct_generation(
    scenario: ImprovementScenario,
    skill_tree: SkillTree,
    agent: "MLXAgent",
    max_attempts: int = 3,
) -> tuple[bool, str, int]:
    """Generate a skill module using direct single-shot LLM generation with retries.

    Returns (accepted, summary, attempts_used).
    """
    from src.runtime.llm_text import extract_python_code_block, strip_thinking_tags

    temperatures = [0.0, 0.3, 0.6]
    failure_context = ""

    for attempt in range(max_attempts):
        temp = temperatures[min(attempt, len(temperatures) - 1)]
        prompt = _build_direct_generation_prompt(scenario, skill_tree, failure_context)

        print(f"  Attempt {attempt + 1}/{max_attempts} (temp={temp})...")

        response = agent.generate_simple(prompt, temperature=temp)
        response = strip_thinking_tags(response)

        # Extract Python code from response
        code = extract_python_code_block(response)
        if not code:
            # Try using the raw response if it looks like Python
            if "def " in response or "class " in response:
                code = response
            else:
                failure_context = "Model did not output valid Python code. Output ONLY Python code, no markdown."
                print(f"    No code extracted from response")
                continue

        # Write to target file
        scenario.target_path.parent.mkdir(parents=True, exist_ok=True)
        scenario.target_path.write_text(code)

        # Validate
        accepted, summary = validate_generated_module(
            str(scenario.target_path), skill_tree=skill_tree
        )

        if accepted:
            print(f"    PASSED: {summary}")
            _save_successful_generation(scenario.skill_id, prompt[:500], code)
            return True, summary, attempt + 1

        # Prepare failure context for next attempt
        failure_context = summary
        print(f"    FAILED: {summary}")

    return False, failure_context, max_attempts


def run_improvement_cycle(
    cycle_num: int,
    model_name: str,
    agent: "MLXAgent | None" = None,
) -> ImprovementCycleResult:
    """Run one improvement cycle using direct generation.

    Callers should invoke ``apply_self_improve_runtime_environment()`` once before
    the first cycle (``tools/improve.py`` does this); it is not repeated here to
    avoid redundant env work every cycle.
    """
    skill_tree = agent.skill_tree if agent is not None else SkillTree()
    scenario = select_improvement_scenario(cycle_num, skill_tree)
    if scenario is None:
        result = ImprovementCycleResult(
            scenario=None,
            accepted=False,
            summary="All skills complete and no weak skill requires upgrade.",
            outcome="idle",
        )
        _append_improve_journal({"cycle_num": cycle_num, "outcome": result.outcome, "summary": result.summary})
        return result

    scenario.target_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = scenario.target_path.with_suffix(scenario.target_path.suffix + ".bak")
    if scenario.target_path.exists():
        shutil.copy2(scenario.target_path, backup_path)

    # Skip pre-validation for skills on cooldown — force actual LLM generation
    pre_ok = False
    pre_message = ""
    if scenario.target_path.exists() and scenario.skill_id not in skill_tree._recently_attempted:
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

    # ── Direct generation mode ──────────────────────────────────────────
    if agent is None:
        from src.agent import MLXAgent
        agent = MLXAgent(config_model_name=model_name, goal=scenario.goal_text)

    accepted, summary, attempts_used = run_direct_generation(
        scenario, skill_tree, agent, max_attempts=3
    )

    # ── Track metrics ───────────────────────────────────────────────────
    metrics = _load_metrics()
    metrics["real_generation_count"] = metrics.get("real_generation_count", 0) + 1
    metrics["total_attempts"] = metrics.get("total_attempts", 0) + attempts_used
    if accepted:
        metrics["real_pass_count"] = metrics.get("real_pass_count", 0) + 1
        completed_list = metrics.get("skills_completed_by_llm", [])
        if scenario.skill_id not in completed_list:
            completed_list.append(scenario.skill_id)
        metrics["skills_completed_by_llm"] = completed_list
    _save_metrics(metrics)

    # ── Update skill tree ───────────────────────────────────────────────
    if accepted:
        skill_tree.mark_completed(scenario.skill_id, summary)
        if backup_path.exists():
            backup_path.unlink()
    else:
        skill_tree.mark_failed(scenario.skill_id, summary)
        _restore_target_file(scenario.target_path, backup_path)

    outcome = "accepted" if accepted else "failed"
    result = ImprovementCycleResult(
        scenario=scenario,
        accepted=accepted,
        summary=f"[{attempts_used} attempts] {summary}",
        outcome=outcome,
    )
    _append_improve_journal(
        {
            "cycle_num": cycle_num,
            "outcome": outcome,
            "skill_id": scenario.skill_id,
            "target_path": str(scenario.target_path),
            "summary": result.summary,
            "attempts": attempts_used,
        }
    )
    return result
