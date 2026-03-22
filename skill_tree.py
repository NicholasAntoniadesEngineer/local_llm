"""
Skill Tree: A directed graph of agent capabilities.

Each skill has:
- Prerequisites (what must be built first)
- Priority tier (foundation → intelligence → integration → optimization)
- A concrete spec (exactly what to build, what to test)
- Impact score (how much it improves the system)

The tree is the SINGLE SOURCE OF TRUTH for what the agent builds next.
It picks the highest-impact unblocked skill automatically.
"""

import os
import subprocess
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Skill:
    """One buildable capability."""
    id: str
    name: str
    file: str                           # output filename
    tier: int                           # 1=foundation, 2=intelligence, 3=integration, 4=optimization
    impact: int                         # 1-10, higher = more important
    prereqs: list[str] = field(default_factory=list)  # skill IDs that must be PASSING first
    description: str = ""               # what this does for the system
    spec: str = ""                      # EXACT instructions for the LLM
    test_hint: str = ""                 # what the test should verify


# ═══════════════════════════════════════════════════════════════════════════
# THE SKILL TREE
# ═══════════════════════════════════════════════════════════════════════════

SKILLS = {

    # ── TIER 1: FOUNDATION (no prereqs) ───────────────────────────────────

    "search_cache": Skill(
        id="search_cache",
        name="Search Result Cache",
        file="search_cache.py",
        tier=1, impact=7,
        description="Cache web search results with TTL to avoid duplicate API calls",
        spec=(
            "Class SearchCache with: get(key)->Optional[str], "
            "set(key, value, ttl_seconds=300), cleanup() removes expired, "
            "stats() returns dict with hit_count, miss_count, size. "
            "Use time.time() for TTL tracking."
        ),
        test_hint="Set 3 values, verify hits, verify miss on unknown key, verify expiry after TTL",
    ),

    "metrics": Skill(
        id="metrics",
        name="Performance Metrics",
        file="metrics.py",
        tier=1, impact=6,
        description="Track tool calls, success rates, timing, token counts per session",
        spec=(
            "Class AgentMetrics with: record_step(tool, success, duration, tokens=0), "
            "summary()->dict with tool_counts/success_rate/avg_time/total_tokens, "
            "report()->str formatted summary. Track per-tool breakdowns."
        ),
        test_hint="Simulate 20 steps across 4 tools, verify counts and rates are correct",
    ),

    "memory_compressor": Skill(
        id="memory_compressor",
        name="Memory Compression",
        file="memory_compressor.py",
        tier=1, impact=6,
        description="Compress old session iterations to keep memory lean",
        spec=(
            "Function compress_session(iterations: list, keep_recent=10) -> tuple[list, str]. "
            "Keep the most recent 'keep_recent' iterations unchanged. "
            "Summarize older ones into a single string: count by tool, success rate, key findings. "
            "Return (recent_iterations, summary_string)."
        ),
        test_hint="Create 50 fake iterations, compress, verify <=10 remain plus summary string",
    ),

    "error_recovery": Skill(
        id="error_recovery",
        name="Error Recovery",
        file="error_recovery.py",
        tier=1, impact=8,
        description="Retry failed operations with exponential backoff and error classification",
        spec=(
            "Class ErrorRecovery with: classify_error(error_str)->str returning one of "
            "'timeout','import','syntax','network','permission','unknown'. "
            "suggest_fix(error_type)->str returning actionable advice. "
            "should_retry(error_type, attempt_num)->bool (max 3 retries, no retry on syntax). "
            "backoff_seconds(attempt_num)->float (exponential: 1, 2, 4)."
        ),
        test_hint="Test each error type classification, verify retry logic, verify backoff times",
    ),

    "code_validator": Skill(
        id="code_validator",
        name="Code Validator",
        file="code_validator.py",
        tier=1, impact=9,
        description="Validate generated Python code: syntax, imports, test execution",
        spec=(
            "Class CodeValidator with: check_syntax(code_str)->tuple[bool, str], "
            "check_imports(code_str)->list[str] returning missing modules, "
            "run_tests(file_path)->tuple[bool, str] executing the file and checking output, "
            "validate_all(file_path)->dict with syntax_ok, imports_ok, tests_ok, issues list."
        ),
        test_hint="Test with valid code, code with syntax error, code with missing import, code with failing test",
    ),

    # ── TIER 2: INTELLIGENCE (require foundation) ─────────────────────────

    "loop_detector": Skill(
        id="loop_detector",
        name="Loop Detector",
        file="loop_detector.py",
        tier=2, impact=8,
        prereqs=["metrics"],
        description="Detect when agent is stuck repeating the same action with similar results",
        spec=(
            "Class LoopDetector with: record(tool, args_str, result_str), "
            "is_stuck()->bool True if last 3 calls same tool AND results >80% similar, "
            "suggest_escape(current_tool)->str suggesting different tool/approach, "
            "similarity(a,b)->float using difflib.SequenceMatcher. "
            "Track a sliding window of last 10 actions."
        ),
        test_hint="Test: 3 identical calls=stuck, 3 different calls=not stuck, similar but not identical results",
    ),

    "confidence_scorer": Skill(
        id="confidence_scorer",
        name="Confidence Scorer",
        file="confidence_scorer.py",
        tier=2, impact=9,
        prereqs=["metrics", "error_recovery"],
        description="Score agent confidence: should it act, research more, or ask for help?",
        spec=(
            "Class ConfidenceScorer with: "
            "score_knowledge(discoveries_count, relevant_keywords)->float 0-1, "
            "score_capability(tool_success_rate, errors_count)->float 0-1, "
            "score_progress(steps_taken, max_steps, files_written)->float 0-1, "
            "should_act(knowledge, capability, progress)->str returning 'research','code','save','abort', "
            "overall(knowledge, capability, progress)->float 0-1 combined score."
        ),
        test_hint="Test: low knowledge=research, high everything=save, many errors=abort, balanced=code",
    ),

    "result_evaluator": Skill(
        id="result_evaluator",
        name="Result Evaluator",
        file="result_evaluator.py",
        tier=2, impact=7,
        prereqs=["search_cache"],
        description="Evaluate quality of tool results - are they useful or junk?",
        spec=(
            "Class ResultEvaluator with: "
            "score_search_result(query, title, snippet)->float 0-1 relevance, "
            "score_code_output(code, output, has_error)->float 0-1 quality, "
            "is_duplicate(new_result, previous_results)->bool, "
            "summarize_quality(scores_list)->dict with avg/min/max/assessment."
        ),
        test_hint="Test: relevant search=high score, junk=low, working code=high, error output=low, exact duplicate detection",
    ),

    "task_planner": Skill(
        id="task_planner",
        name="Task Planner",
        file="task_planner.py",
        tier=2, impact=8,
        prereqs=["error_recovery"],
        description="Decompose complex goals into ordered subtasks",
        spec=(
            "Class TaskPlanner with: "
            "decompose(goal_str)->list[dict] each with 'task','tool','priority','depends_on', "
            "next_task(completed_tasks)->dict or None, "
            "is_complete(completed_tasks, all_tasks)->bool, "
            "replan(failed_task, error)->list[dict] alternative approach. "
            "Handle common patterns: read->analyze->code->test->save."
        ),
        test_hint="Test: decompose a coding goal into 5+ steps, verify ordering, verify replan on failure",
    ),

    # ── TIER 3: INTEGRATION (require intelligence) ────────────────────────

    "smart_router": Skill(
        id="smart_router",
        name="Smart Tool Router",
        file="smart_router.py",
        tier=3, impact=9,
        prereqs=["confidence_scorer", "loop_detector", "task_planner"],
        description="Given current state, pick the optimal next tool and arguments",
        spec=(
            "Class SmartRouter with: "
            "pick_tool(phase, confidence, loop_status, task_plan)->dict with 'tool','args','reason', "
            "should_change_phase(current_phase, confidence, steps_in_phase)->tuple[bool, str], "
            "format_tool_prompt(tool, args, context)->str generating the tool call string. "
            "Integrate confidence scoring, loop detection, and task planning."
        ),
        test_hint="Test: research phase+low confidence=web_search, code phase+loop=phase change, high confidence+code done=save",
    ),

    "self_evaluator": Skill(
        id="self_evaluator",
        name="Self Evaluator",
        file="self_evaluator.py",
        tier=3, impact=10,
        prereqs=["code_validator", "result_evaluator", "metrics"],
        description="Agent evaluates its own output quality before saving",
        spec=(
            "Class SelfEvaluator with: "
            "evaluate_file(file_path)->dict with scores for: "
            "syntax(0-1), has_tests(bool), tests_pass(bool), "
            "code_size(bytes), docstrings(bool), type_hints(bool), "
            "overall_quality(0-1), recommendation('keep','fix','discard'). "
            "Use CodeValidator for syntax, run tests, check for docstrings and type hints."
        ),
        test_hint="Test with: good quality file=keep, syntax error=discard, no tests=fix, tiny placeholder=discard",
    ),

    # ── TIER 4: OPTIMIZATION (require integration) ────────────────────────

    "orchestrator": Skill(
        id="orchestrator",
        name="Agent Orchestrator",
        file="orchestrator.py",
        tier=4, impact=10,
        prereqs=["smart_router", "self_evaluator"],
        description="Master controller that wires all modules together into a smarter agent loop",
        spec=(
            "Class Orchestrator with: "
            "plan_cycle(goal, codebase_state)->list of planned steps, "
            "decide_next(current_state, history)->dict with tool/args/reason, "
            "evaluate_progress(steps_done, goal)->dict with completion_pct/quality/should_continue, "
            "post_mortem(run_log)->dict analyzing what worked, what didn't, what to try next time. "
            "This is the brain that coordinates all other modules."
        ),
        test_hint="Test: given a goal and empty state, produces sensible plan. Given completed state, recommends save. Given loop state, recommends escape.",
    ),

    "strategy_learner": Skill(
        id="strategy_learner",
        name="Strategy Learner",
        file="strategy_learner.py",
        tier=4, impact=10,
        prereqs=["orchestrator", "self_evaluator"],
        description="Learn from past runs: what strategies work for what kinds of tasks",
        spec=(
            "Class StrategyLearner with: "
            "record_outcome(task_type, strategy_used, success, quality_score), "
            "best_strategy(task_type)->str the approach with highest historical success, "
            "avoid_strategy(task_type)->str the approach with lowest success, "
            "success_rate_by_strategy()->dict mapping strategy->rate, "
            "recommend(task_type, available_strategies)->str picking the best option. "
            "Persist data to a JSON file."
        ),
        test_hint="Record 20 outcomes across 3 task types, verify best/worst strategies match expected, verify persistence to disk",
    ),
}


def get_system_state() -> dict:
    """Scan what's built and what's passing."""
    output_dir = Path("./agent_outputs")
    state = {"passing": set(), "failing": set(), "missing": set()}

    for skill_id, skill in SKILLS.items():
        fpath = output_dir / skill.file
        if not fpath.exists():
            state["missing"].add(skill_id)
            continue

        # Test it
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(".").resolve()) + ":" + str(output_dir.resolve())
            result = subprocess.run(
                ["python3", str(fpath)],
                capture_output=True, text=True, timeout=10, env=env,
            )
            if "PASSED" in result.stdout:
                state["passing"].add(skill_id)
            else:
                state["failing"].add(skill_id)
        except Exception:
            state["failing"].add(skill_id)

    return state


def get_next_skill() -> Skill:
    """Pick the highest-impact unblocked skill to build next."""
    load_proposals()  # Merge any agent-proposed skills
    state = get_system_state()

    candidates = []
    for skill_id, skill in SKILLS.items():
        # Skip already passing
        if skill_id in state["passing"]:
            continue

        # Check prereqs are met
        prereqs_met = all(p in state["passing"] for p in skill.prereqs)
        if not prereqs_met:
            continue

        # Boost priority for failing files (fix > build new)
        priority = skill.impact * 10 + (5 if skill_id in state["failing"] else 0)

        # Lower tiers first (foundation before intelligence)
        priority += (5 - skill.tier) * 20

        candidates.append((priority, skill))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def get_full_codebase_dump() -> str:
    """Dump the FULL content of every essential file for injection into context.

    Skips non-essential files (monitor.py, setup.py, test stubs) to stay
    within context budget. The agent sees REAL CODE, not summaries.
    """
    # Core files the agent MUST understand to improve the system
    essential_core = ["agent.py", "config.py", "memory.py", "skill_tree.py", "improve.py", "logger.py"]
    # Skip: monitor.py (display only), setup.py (one-time), test stubs

    lines = []

    for fname in essential_core:
        f = Path(fname)
        if not f.exists():
            continue
        try:
            content = f.read_text()
            lines.append(f"\n{'='*60}")
            lines.append(f"FILE: {fname} ({len(content.splitlines())} lines)")
            lines.append(f"{'='*60}")
            lines.append(content)
        except Exception:
            pass

    # ALL agent outputs - full content
    output_dir = Path("./agent_outputs")
    if output_dir.exists():
        for f in sorted(output_dir.glob("*.py")):
            if f.name.startswith("."):
                continue
            try:
                content = f.read_text()
                lines.append(f"\n{'='*60}")
                lines.append(f"FILE: agent_outputs/{f.name} ({len(content.splitlines())} lines)")
                lines.append(f"{'='*60}")
                lines.append(content)
            except Exception:
                pass

    return "\n".join(lines)


def get_full_codebase_summary() -> str:
    """Short summary for logging/display (not for agent context)."""
    lines = []
    for f in sorted(Path(".").glob("*.py")):
        try:
            content = f.read_text()
            loc = len(content.splitlines())
            lines.append(f"  {f.name} ({loc} lines)")
        except Exception:
            pass
    output_dir = Path("./agent_outputs")
    if output_dir.exists():
        for f in sorted(output_dir.glob("*.py")):
            if f.name.startswith("."):
                continue
            lines.append(f"  agent_outputs/{f.name}")
    return "\n".join(lines)


def get_tree_as_text() -> str:
    """Render the full skill tree as text for the agent to see."""
    state = get_system_state()
    tier_names = {1: "FOUNDATION", 2: "INTELLIGENCE", 3: "INTEGRATION", 4: "OPTIMIZATION"}
    lines = []

    for tier in [1, 2, 3, 4]:
        lines.append(f"\nTIER {tier}: {tier_names[tier]}")
        for sid, skill in SKILLS.items():
            if skill.tier != tier:
                continue
            if sid in state["passing"]:
                status = "DONE"
            elif sid in state["failing"]:
                status = "BROKEN"
            else:
                prereqs_met = all(p in state["passing"] for p in skill.prereqs)
                status = "READY" if prereqs_met else "LOCKED"
            deps = f" (needs: {', '.join(skill.prereqs)})" if skill.prereqs else ""
            lines.append(f"  [{status}] {skill.name} -> {skill.file} (impact:{skill.impact}/10){deps}")

    return "\n".join(lines)


def build_goal_for_skill(skill: Skill, state: dict) -> str:
    """Build a goal with the ENTIRE codebase injected into context."""
    full_code = get_full_codebase_dump()
    tree_text = get_tree_as_text()
    tier_name = {1: "Foundation", 2: "Intelligence", 3: "Integration", 4: "Optimization"}

    return f"""You are a self-improving AI agent. Below is your ENTIRE codebase.
You already have all the code - do NOT use read_file for these files.
Go straight to writing your improvement.

{full_code}

=== YOUR SKILL TREE ===
{tree_text}

=== YOUR CURRENT TASK ===
BUILD: {skill.name} (Tier {skill.tier}: {tier_name[skill.tier]}, Impact: {skill.impact}/10)
FILE: {skill.file}
WHY: {skill.description}

SPEC:
{skill.spec}

TEST REQUIREMENTS:
{skill.test_hint}

=== EXPANDING THE SYSTEM ===
After building this skill, if you see an opportunity to improve the skill tree
itself (add new skills, refine specs, identify missing capabilities), you can
write suggestions to agent_outputs/tree_proposals.txt using write_file.
Each proposal should have: skill name, file, tier, prereqs, description, spec.

=== RULES ===
1. Read ALL source files you need to understand the full system
2. Read existing agent_outputs/*.py to understand what's built
3. Write COMPLETE working code in {skill.file}
4. Include 'if __name__ == "__main__":' test block
5. Tests MUST print 'ALL TESTS PASSED' when everything works
6. Don't import agent.py in tests (loads MLX model, too slow)
7. CAN import from memory.py, config.py, or other agent_outputs/*.py
8. Fix errors until tests pass, then say DONE"""


def load_proposals():
    """Load skill proposals written by the agent and merge into SKILLS."""
    proposals_file = Path("./agent_outputs/tree_proposals.txt")
    if not proposals_file.exists():
        return

    try:
        content = proposals_file.read_text()
        # Simple parsing: look for structured proposals
        # Agent writes: SKILL: name | FILE: x.py | TIER: N | PREREQS: a,b | DESC: ... | SPEC: ...
        for line in content.split("\n"):
            if not line.startswith("SKILL:"):
                continue
            parts = {}
            for segment in line.split("|"):
                segment = segment.strip()
                if ":" in segment:
                    key, val = segment.split(":", 1)
                    parts[key.strip().lower()] = val.strip()

            if "skill" in parts and "file" in parts:
                sid = parts["skill"].lower().replace(" ", "_")
                if sid not in SKILLS:
                    SKILLS[sid] = Skill(
                        id=sid,
                        name=parts["skill"],
                        file=parts["file"],
                        tier=int(parts.get("tier", 2)),
                        impact=int(parts.get("impact", 7)),
                        prereqs=[p.strip() for p in parts.get("prereqs", "").split(",") if p.strip()],
                        description=parts.get("desc", parts.get("description", "")),
                        spec=parts.get("spec", "Build this module with tests."),
                        test_hint="Tests must print ALL TESTS PASSED",
                    )
    except Exception:
        pass


def print_tree():
    """Print the full skill tree with current status."""
    state = get_system_state()
    tier_names = {1: "FOUNDATION", 2: "INTELLIGENCE", 3: "INTEGRATION", 4: "OPTIMIZATION"}

    for tier in [1, 2, 3, 4]:
        print(f"\n{'─'*40}")
        print(f"TIER {tier}: {tier_names[tier]}")
        print(f"{'─'*40}")
        for sid, skill in SKILLS.items():
            if skill.tier != tier:
                continue
            if sid in state["passing"]:
                status = "✅ PASSING"
            elif sid in state["failing"]:
                status = "❌ FAILING"
            else:
                prereqs_met = all(p in state["passing"] for p in skill.prereqs)
                status = "⬜ READY" if prereqs_met else "🔒 LOCKED"
            deps = f" (needs: {', '.join(skill.prereqs)})" if skill.prereqs else ""
            print(f"  {status} [{skill.impact}/10] {skill.name} -> {skill.file}{deps}")

    next_skill = get_next_skill()
    if next_skill:
        print(f"\n→ NEXT: {next_skill.name} ({next_skill.file})")
    else:
        print(f"\n→ ALL SKILLS COMPLETE!")


if __name__ == "__main__":
    print_tree()
