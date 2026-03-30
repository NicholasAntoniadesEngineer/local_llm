"""Runtime-backed improvement cycle runner.

v3: Challenge-driven improvement. The agent solves coding challenges,
measures results, and upgrades skills that are weakest at helping it succeed.
Falls back to skill building/upgrading when no challenges remain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import shutil
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.challenges import (
    CHALLENGES,
    Challenge,
    ChallengeResult,
    build_challenge_prompt,
    run_challenge,
)
from src.paths import IMPROVE_SESSION_FILE, RUNS_DIR, SKILLS_DIR
from src.runtime.verifier import validate_generated_module
from src.skill_tree import SkillTree

if TYPE_CHECKING:
    from src.agent import MLXAgent


# ── Paths for feedback loop (all absolute via src.paths) ─────────────────

SUCCESSFUL_GENERATIONS_FILE = RUNS_DIR / "successful_generations.jsonl"
METRICS_FILE = RUNS_DIR / "metrics.json"
CHALLENGE_RESULTS_FILE = RUNS_DIR / "challenge_results.jsonl"

# Cap JSONL files to prevent unbounded growth
_MAX_JSONL_LINES = 2000


def _ensure_dirs() -> None:
    """Create required directories upfront. Called once at module load."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)


_ensure_dirs()


@dataclass
class ImprovementScenario:
    """Selected improvement scenario for one cycle."""

    cycle_num: int
    skill_id: str
    skill_name: str
    action: str  # "BUILDING", "UPGRADING", "CHALLENGE"
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
    _append_jsonl(IMPROVE_SESSION_FILE, {**record, "timestamp": datetime.now().isoformat()})


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
        "challenges_attempted": 0,
        "challenges_solved": 0,
        "challenge_scores": {},
    }


def _save_metrics(metrics: dict[str, Any]) -> None:
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    derived = {**metrics}
    gen_count = derived.get("real_generation_count", 0)
    derived["real_pass_rate"] = (
        round(derived.get("real_pass_count", 0) / gen_count, 3) if gen_count > 0 else 0.0
    )
    ch_attempted = derived.get("challenges_attempted", 0)
    ch_solved = derived.get("challenges_solved", 0)
    derived["challenge_solve_rate"] = (
        round(ch_solved / ch_attempted, 3) if ch_attempted > 0 else 0.0
    )
    METRICS_FILE.write_text(json.dumps(derived, indent=2, default=str))


def _save_successful_generation(skill_id: str, prompt: str, code: str) -> None:
    """Save a successful (prompt, code) pair for future few-shot use."""
    _append_jsonl(SUCCESSFUL_GENERATIONS_FILE, {
        "skill_id": skill_id,
        "prompt_preview": prompt[:500],
        "code": code,
        "timestamp": datetime.now().isoformat(),
        "lines": len(code.splitlines()),
    })


def _save_challenge_result(result: ChallengeResult) -> None:
    """Save a challenge result for tracking progress over time."""
    _append_jsonl(CHALLENGE_RESULTS_FILE, {
        "challenge_id": result.challenge_id,
        "solved": result.solved,
        "score": result.score,
        "time_s": result.time_s,
        "error": result.error[:200] if result.error else "",
        "code_lines": len(result.code.splitlines()),
        "timestamp": datetime.now().isoformat(),
    })


def _load_few_shot_example() -> str:
    """Load the best recent successful generation as a few-shot example."""
    if not SUCCESSFUL_GENERATIONS_FILE.exists():
        return _builtin_few_shot_example()
    try:
        lines = SUCCESSFUL_GENERATIONS_FILE.read_text().strip().splitlines()
        if not lines:
            return _builtin_few_shot_example()
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


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skipping corrupted lines instead of aborting."""
    if not path.exists():
        return []
    records = []
    try:
        for line in path.read_text().strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Skip corrupted line, don't abort entire file
    except OSError:
        pass
    return records


def _append_jsonl(path: Path, record: dict) -> None:
    """Append a record to a JSONL file, capping at _MAX_JSONL_LINES."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    # Cap file size: keep only the last _MAX_JSONL_LINES
    try:
        lines = path.read_text().strip().splitlines()
        if len(lines) > _MAX_JSONL_LINES:
            path.write_text("\n".join(lines[-_MAX_JSONL_LINES:]) + "\n")
    except OSError:
        pass


def _get_solved_challenge_ids() -> set[str]:
    """Get IDs of challenges that have been solved at least once."""
    solved = set()
    for record in _read_jsonl(CHALLENGE_RESULTS_FILE):
        if record.get("solved"):
            solved.add(record.get("challenge_id", ""))
    return solved


def _get_attempt_counts() -> dict[str, int]:
    """Count how many times each challenge has been attempted."""
    counts: dict[str, int] = {}
    for record in _read_jsonl(CHALLENGE_RESULTS_FILE):
        cid = record.get("challenge_id", "")
        counts[cid] = counts.get(cid, 0) + 1
    return counts


def _load_challenge_few_shot() -> str:
    """Load the most recent successful challenge solution for in-context learning."""
    records = _read_jsonl(SUCCESSFUL_GENERATIONS_FILE)
    # Find most recent challenge solution (not skill)
    for record in reversed(records):
        code = record.get("code", "")
        if code and record.get("lines", 0) < 80:  # Prefer compact solutions
            return code
    return ""


def _get_common_failure_patterns() -> str:
    """Analyze past failures and return a hint block for the prompt."""
    records = _read_jsonl(CHALLENGE_RESULTS_FILE)
    error_counts: dict[str, int] = {}
    error_examples: dict[str, str] = {}

    for record in records:
        if record.get("solved"):
            continue
        error = record.get("error", "")
        # Extract error type from bracketed prefix: [syntax_error] ...
        if error.startswith("[") and "]" in error:
            error_type = error[1:error.index("]")]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            if error_type not in error_examples:
                error_examples[error_type] = error[error.index("]") + 2:][:80]

    if not error_counts:
        return ""

    # Build hint block from top 3 most common errors
    sorted_errors = sorted(error_counts.items(), key=lambda x: -x[1])[:3]
    lines = ["AVOID THESE COMMON MISTAKES:"]
    for error_type, count in sorted_errors:
        example = error_examples.get(error_type, "")
        label = error_type.replace("_", " ").title()
        lines.append(f"- {label} ({count}x): {example}")

    return "\n".join(lines)


def _get_solve_rates_by_difficulty() -> dict[int, float]:
    """Compute solve rate per difficulty level from history."""
    records = _read_jsonl(CHALLENGE_RESULTS_FILE)
    by_difficulty: dict[int, list[bool]] = {}

    challenge_map = {c.id: c.difficulty for c in CHALLENGES}
    for record in records:
        cid = record.get("challenge_id", "")
        diff = challenge_map.get(cid, 0)
        if diff not in by_difficulty:
            by_difficulty[diff] = []
        by_difficulty[diff].append(bool(record.get("solved")))

    return {
        diff: sum(results) / len(results) if results else 0.0
        for diff, results in by_difficulty.items()
    }


def _pick_next_challenge() -> Challenge | None:
    """Pick the next challenge using metrics-driven difficulty progression.

    Strategy:
    1. Pick unsolved challenges at the current difficulty level
    2. If all at current level solved with >80% rate, move to next level
    3. If current level has <30% rate after 5+ attempts, drop back
    4. Within a level, pick the least-attempted challenge
    """
    if not CHALLENGES:
        return None

    solved = _get_solved_challenge_ids()
    solve_rates = _get_solve_rates_by_difficulty()
    counts = _get_attempt_counts()

    # Determine target difficulty based on solve rates
    target_difficulty = 1
    for diff in sorted(solve_rates.keys()):
        rate = solve_rates[diff]
        attempts_at_diff = sum(
            counts.get(c.id, 0) for c in CHALLENGES if c.difficulty == diff
        )
        if rate >= 0.8 and attempts_at_diff >= 3:
            target_difficulty = diff + 1  # Ready for harder challenges
        elif rate < 0.3 and attempts_at_diff >= 5:
            target_difficulty = max(1, diff)  # Stay at this level
            break

    # Pick unsolved at target difficulty (or lower)
    candidates = [
        c for c in CHALLENGES
        if c.id not in solved and c.difficulty <= target_difficulty
    ]
    if candidates:
        candidates.sort(key=lambda c: (c.difficulty, counts.get(c.id, 0)))
        return candidates[0]

    # All at target solved — pick least-attempted at any level
    return min(CHALLENGES, key=lambda c: counts.get(c.id, 0))


# ── Skill improvement path ───────────────────────────────────────────────

def select_improvement_scenario(cycle_num: int, skill_tree: SkillTree) -> ImprovementScenario | None:
    """Choose the next skill build or upgrade scenario from the skill tree."""
    skill_tree.evolve_tree()
    new_skill = skill_tree.peek_next_skill()

    if new_skill:
        selected_skill = new_skill
        action_name = "BUILDING"
        goal_text = skill_tree.build_goal_for_skill(selected_skill)
        skill_tree.record_pull(selected_skill["id"])
        return ImprovementScenario(
            cycle_num=cycle_num,
            skill_id=selected_skill["id"],
            skill_name=selected_skill["name"],
            action=action_name,
            goal_text=goal_text,
            target_path=SKILLS_DIR / selected_skill["file"],
        )

    # No new skills to build — try upgrading weakest
    weak_skill = skill_tree.get_weakest_skill()
    if weak_skill:
        selected_skill = weak_skill
        action_name = "UPGRADING"
        goal_text = skill_tree.build_upgrade_goal(selected_skill)
        return ImprovementScenario(
            cycle_num=cycle_num,
            skill_id=selected_skill["id"],
            skill_name=selected_skill["name"],
            action=action_name,
            goal_text=goal_text,
            target_path=SKILLS_DIR / selected_skill["file"],
        )

    return None


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

    prereq_source = skill_tree.read_prereq_full_source(skill)
    few_shot = _load_few_shot_example()

    fail_block = ""
    if failure_context:
        fail_block = f"""
=== PREVIOUS ATTEMPT FAILED ===
{failure_context}
Fix this specific issue. Do not repeat the same mistake.
"""

    return f"""Write a complete, standalone Python module.
{fail_block}
=== TASK ===
Module: {skill['name']}
File: skills/{skill['file']}
Description: {skill.get('description', '')}
Specification: {skill.get('spec', '')}

=== RULES (code gets deleted if ANY rule fails) ===
1. At least 3 functions/methods with real logic (no pass/stubs)
2. At least 4 `assert` statements in an `if __name__ == "__main__":` test block
3. Print 'ALL TESTS PASSED' at the end if all tests pass
4. No 'from src.' imports — use only stdlib + prereqs listed below

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

Output ONLY the complete Python code for {skill['file']}. No explanation."""


def run_direct_generation(
    scenario: ImprovementScenario,
    skill_tree: SkillTree,
    agent: "MLXAgent",
    max_attempts: int = 3,
) -> tuple[bool, str, int]:
    """Generate a skill module using direct single-shot LLM generation with retries."""
    from src.runtime.llm_text import extract_python_code_block, diagnose_extraction_failure, strip_thinking_tags

    temperatures = [0.0, 0.3, 0.6]
    failure_context = ""

    for attempt in range(max_attempts):
        temp = temperatures[min(attempt, len(temperatures) - 1)]
        prompt = _build_direct_generation_prompt(scenario, skill_tree, failure_context)

        print(f"  Attempt {attempt + 1}/{max_attempts} (temp={temp})...")

        response = agent.generate_simple(prompt, temperature=temp)
        response = strip_thinking_tags(response)

        code = extract_python_code_block(response)
        if not code:
            diagnosis = diagnose_extraction_failure(response)
            failure_context = f"Code extraction failed: {diagnosis}"
            print(f"    EXTRACTION FAILED: {diagnosis}")
            continue

        scenario.target_path.parent.mkdir(parents=True, exist_ok=True)
        scenario.target_path.write_text(code)

        accepted, summary = validate_generated_module(
            str(scenario.target_path), skill_tree=skill_tree
        )

        if accepted:
            print(f"    PASSED: {summary}")
            _save_successful_generation(scenario.skill_id, prompt[:500], code)
            return True, summary, attempt + 1

        failure_context = summary
        print(f"    FAILED: {summary}")

    return False, failure_context, max_attempts


# ── Challenge execution path ─────────────────────────────────────────────

def run_challenge_cycle(
    cycle_num: int,
    agent: "MLXAgent",
) -> ImprovementCycleResult:
    """Pick a coding challenge, solve it with the LLM, score the result.

    Feedback loops active:
    1. Few-shot: includes best recent successful solution in prompt
    2. Failure patterns: warns about common mistake types
    3. Difficulty: picks challenges based on actual solve rates
    4. Error classification: structured error types for pattern tracking
    """
    from src.runtime.llm_text import extract_python_code_block, diagnose_extraction_failure, strip_thinking_tags

    challenge = _pick_next_challenge()
    if challenge is None:
        return ImprovementCycleResult(
            scenario=None,
            accepted=False,
            summary="No challenges available.",
            outcome="idle",
        )

    # FEEDBACK LOOP 1: Load a successful solution as few-shot context
    few_shot = _load_challenge_few_shot()

    # FEEDBACK LOOP 2: Load common failure patterns as hints
    failure_hints = _get_common_failure_patterns()

    solve_rates = _get_solve_rates_by_difficulty()
    rate_str = ", ".join(f"d{d}={r:.0%}" for d, r in sorted(solve_rates.items()) if r > 0)
    print(f"\n  CHALLENGE: {challenge.name} (difficulty {challenge.difficulty}/5)")
    if rate_str:
        print(f"  Solve rates: {rate_str}")
    if few_shot:
        print(f"  Few-shot: {len(few_shot)} chars from prior success")
    if failure_hints:
        print(f"  Failure hints injected")

    temperatures = [0.0, 0.3, 0.6]
    best_result: ChallengeResult | None = None

    for attempt in range(3):
        temp = temperatures[attempt]
        prompt = build_challenge_prompt(challenge, few_shot=few_shot, failure_hints=failure_hints)

        if best_result and not best_result.solved:
            prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {best_result.error}\nFix the issue."

        print(f"  Attempt {attempt + 1}/3 (temp={temp})...")
        response = agent.generate_simple(prompt, temperature=temp)
        response = strip_thinking_tags(response)

        code = extract_python_code_block(response)
        if not code:
            diagnosis = diagnose_extraction_failure(response)
            print(f"    EXTRACTION FAILED: {diagnosis}")
            continue

        result = run_challenge(challenge, code)
        _save_challenge_result(result)

        if result.solved:
            print(f"    SOLVED in {result.time_s:.1f}s")
            # FEEDBACK LOOP 1: Save solution for future few-shot context
            _save_successful_generation(challenge.id, prompt[:300], code)
            best_result = result
            break
        else:
            print(f"    FAILED: {result.error[:100]}")
            best_result = result

    # Update metrics
    metrics = _load_metrics()
    metrics["challenges_attempted"] = metrics.get("challenges_attempted", 0) + 1
    solved = best_result is not None and best_result.solved
    if solved:
        metrics["challenges_solved"] = metrics.get("challenges_solved", 0) + 1
    scores = metrics.get("challenge_scores", {})
    scores[challenge.id] = 1.0 if solved else 0.0
    metrics["challenge_scores"] = scores
    _save_metrics(metrics)

    scenario = ImprovementScenario(
        cycle_num=cycle_num,
        skill_id=challenge.id,
        skill_name=challenge.name,
        action="CHALLENGE",
        goal_text=challenge.description,
        target_path=RUNS_DIR / f"challenge_{challenge.id}.py",
    )

    outcome = "challenge_solved" if solved else "challenge_failed"
    summary = f"Challenge '{challenge.name}': {'SOLVED' if solved else 'FAILED'}"
    if not solved and best_result:
        summary += f" — {best_result.error[:100]}"
    elif not solved:
        summary += " — no valid code generated in any attempt"

    _append_improve_journal({
        "cycle_num": cycle_num,
        "outcome": outcome,
        "skill_id": challenge.id,
        "summary": summary,
    })

    return ImprovementCycleResult(
        scenario=scenario,
        accepted=solved,
        summary=summary,
        outcome=outcome,
    )


# ── Main cycle ───────────────────────────────────────────────────────────

def run_improvement_cycle(
    cycle_num: int,
    model_name: str,
    agent: "MLXAgent | None" = None,
) -> ImprovementCycleResult:
    """Run one improvement cycle: challenge or skill building.

    Priority:
    1. Build any new locked skills (rarely — all 13 seeds are built)
    2. Solve coding challenges (primary work)
    3. Upgrade weakest skill (when all challenges solved recently)
    """
    skill_tree = agent.skill_tree if agent is not None else SkillTree()

    # Step 1: Check if any new skills need building
    skill_tree.evolve_tree()
    new_skill = skill_tree.peek_next_skill()
    if new_skill:
        scenario = ImprovementScenario(
            cycle_num=cycle_num,
            skill_id=new_skill["id"],
            skill_name=new_skill["name"],
            action="BUILDING",
            goal_text=skill_tree.build_goal_for_skill(new_skill),
            target_path=SKILLS_DIR / new_skill["file"],
        )
        skill_tree.record_pull(new_skill["id"])

        # Pre-validate
        if scenario.target_path.exists():
            pre_ok, pre_msg = validate_generated_module(str(scenario.target_path), skill_tree=skill_tree)
            if pre_ok:
                skill_tree.mark_completed(scenario.skill_id, pre_msg)
                return ImprovementCycleResult(
                    scenario=scenario, accepted=True,
                    summary=f"Pre-validated: {pre_msg}", outcome="pre_validated",
                )

        # Build with LLM
        if agent is None:
            from src.agent import MLXAgent
            agent = MLXAgent(config_model_name=model_name, goal=scenario.goal_text)

        skill_tree.record_attempt(scenario.skill_id)
        accepted, summary, attempts = run_direct_generation(scenario, skill_tree, agent)

        metrics = _load_metrics()
        metrics["real_generation_count"] = metrics.get("real_generation_count", 0) + 1
        metrics["total_attempts"] = metrics.get("total_attempts", 0) + attempts
        if accepted:
            metrics["real_pass_count"] = metrics.get("real_pass_count", 0) + 1
            skill_tree.mark_completed(scenario.skill_id, summary)
        else:
            skill_tree.mark_failed(scenario.skill_id, summary)
        _save_metrics(metrics)

        _append_improve_journal({
            "cycle_num": cycle_num,
            "outcome": "accepted" if accepted else "failed",
            "skill_id": scenario.skill_id,
            "summary": f"[{attempts} attempts] {summary}",
        })
        return ImprovementCycleResult(
            scenario=scenario, accepted=accepted,
            summary=f"[{attempts} attempts] {summary}",
            outcome="accepted" if accepted else "failed",
        )

    # Step 2: Solve coding challenges (primary work loop)
    if agent is None:
        from src.agent import MLXAgent
        agent = MLXAgent(config_model_name=model_name, goal="challenge")

    return run_challenge_cycle(cycle_num, agent)
