#!/usr/bin/env python3
"""
Self-aware improvement loop.

Each cycle the agent:
1. Reads the ENTIRE codebase
2. Reads what it has already built (skills/)
3. Reads its own history of successes and failures
4. DECIDES what to improve next based on what would have the most impact
5. Builds it, tests it, keeps it only if it passes

The agent sees everything. It chooses what matters.
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path
from src.agent import MLXAgent
from src.logger import AgentLogger
from src.skill_tree import SkillTree


HISTORY_FILE = Path("./runs/history.json")


def load_history() -> dict:
    """Load improvement history - what worked, what failed, what exists."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return {"cycles": [], "total_passed": 0, "total_failed": 0}


def save_history(history: dict):
    """Persist improvement history."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def scan_codebase() -> str:
    """Build a map of the entire codebase for the agent to see."""
    lines = []

    # Core source files
    lines.append("=== CORE SOURCE FILES ===")
    for f in sorted(Path(".").glob("*.py")):
        size = f.stat().st_size
        with open(f) as fh:
            first_line = fh.readline().strip()
            # Count classes and functions
            content = fh.read()
            classes = content.count("\nclass ")
            funcs = content.count("\ndef ")
        lines.append(f"  {f.name} ({size:,}B) - {classes} classes, {funcs} functions - {first_line}")

    # Agent outputs (what's already been built)
    lines.append("\n=== ALREADY BUILT (skills/) ===")
    output_dir = Path("./skills")
    if output_dir.exists():
        py_files = sorted(output_dir.glob("*.py"))
        if py_files:
            for f in py_files:
                if f.name.startswith("."):
                    continue
                size = f.stat().st_size
                # Quick test
                try:
                    env = os.environ.copy()
                    env["PYTHONPATH"] = str(Path(".").resolve()) + ":" + str(output_dir.resolve())
                    result = subprocess.run(
                        ["python3", str(f)], capture_output=True, text=True,
                        timeout=10, env=env,
                    )
                    status = "PASSING" if "PASSED" in result.stdout else "FAILING"
                except Exception:
                    status = "ERROR"
                lines.append(f"  {f.name} ({size:,}B) [{status}]")
        else:
            lines.append("  (nothing built yet)")
    else:
        lines.append("  (directory doesn't exist)")

    return "\n".join(lines)


def scan_history(history: dict) -> str:
    """Summarize what's been attempted before."""
    if not history["cycles"]:
        return "No previous improvement attempts."

    lines = [f"Previous attempts: {history['total_passed']} passed, {history['total_failed']} failed\n"]
    lines.append("Recent cycles:")
    for cycle in history["cycles"][-8:]:
        status = "PASS" if cycle["passed"] else "FAIL"
        lines.append(f"  [{status}] {cycle['target']} - {cycle.get('reason', '')[:60]}")

    return "\n".join(lines)


def build_goal(cycle_num: int, codebase_map: str, history_summary: str) -> dict:
    """Build an intelligent, context-aware goal for the agent.

    The agent sees the full codebase, what's already built, and what failed before.
    It must decide what would be most impactful to build next.
    """

    # Figure out what's already passing so the agent doesn't rebuild it
    existing_passing = []
    existing_failing = []
    output_dir = Path("./skills")
    for f in sorted(output_dir.glob("*.py")):
        if f.name.startswith("."):
            continue
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(".").resolve()) + ":" + str(output_dir.resolve())
            r = subprocess.run(["python3", str(f)], capture_output=True, text=True, timeout=10, env=env)
            if "PASSED" in r.stdout:
                existing_passing.append(f.name)
            else:
                existing_failing.append(f.name)
        except Exception:
            existing_failing.append(f.name)

    passing_str = ", ".join(existing_passing) if existing_passing else "none"
    failing_str = ", ".join(existing_failing) if existing_failing else "none"

    goal = f"""Improve this MLX agent system. Cycle #{cycle_num}.

SOURCE FILES: {codebase_map}

ALREADY BUILT AND PASSING (do NOT rebuild these): {passing_str}
BUILT BUT FAILING (fix or replace): {failing_str}

HISTORY: {history_summary}

You MUST build something NEW that doesn't exist yet, or fix a failing file.
Do NOT recreate {passing_str}.

Ideas for new modules:
- error_recovery.py: retry failed tool calls with backoff
- code_validator.py: syntax check + test runner for generated code
- task_planner.py: decompose complex goals into steps
- confidence_scorer.py: score how confident the agent is in its output
- result_evaluator.py: compare tool results to detect duplicates
- tool_router.py: pick the best tool for the current situation

Read source files, write code, test it, save it.
Tests must print 'ALL TESTS PASSED'. Don't import agent.py in tests.
Say DONE when tests pass."""

    return {
        "goal": goal,
        "test_cmd": None,  # Discovered dynamically
        "success_marker": "PASSED",
    }


def discover_output_file():
    """Find the most recently created/modified .py file in skills."""
    output_dir = Path("./skills")
    py_files = sorted(output_dir.glob("*.py"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in py_files:
        if f.name.startswith("."):
            continue
        return str(f)
    return None


def validate_output(filepath: str) -> tuple[bool, str]:
    """Validate generated code — syntax, tests, AND structural quality."""
    path = Path(filepath)
    if not path.exists():
        return False, "File not created"

    size = path.stat().st_size
    if size < 200:
        return False, f"Too small ({size}B) - placeholder"

    source = path.read_text()

    # Structural quality gate
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # Count functions, classes, asserts
    func_count = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    class_count = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    assert_count = source.count("assert ")

    if func_count < 3:
        return False, f"Too few functions ({func_count}). Need ≥3 for production quality."
    if assert_count < 5:
        return False, f"Too few assertions ({assert_count}). Need ≥5 to verify correctness."

    # Reject files that import from src.* (skills must be standalone)
    if "from src." in source:
        return False, "Bad import: 'from src.*' — skills must be standalone"

    # Run tests
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(".").resolve()) + ":" + str(Path("./skills").resolve())
        result = subprocess.run(
            ["python3", str(path)],
            capture_output=True, text=True, timeout=30, env=env,
        )
        if (result.returncode == 0
                and "ALL TESTS PASSED" in result.stdout
                and "Traceback" not in result.stderr):
            return True, f"Tests passed ({size:,}B, {func_count}fn, {assert_count}asserts)"
        else:
            output = result.stdout + result.stderr
            return False, f"Tests failed: {output[:200]}"
    except subprocess.TimeoutExpired:
        return False, "Timed out (30s)"
    except Exception as e:
        return False, f"Error: {e}"


def run_cycle(cycle_num: int, agent=None) -> bool:
    """Run one self-aware improvement cycle: build new OR upgrade weak."""
    history = load_history()

    tree = SkillTree()
    tree.evolve_tree()

    # Decide: build new skill or upgrade weakest existing one
    new_skill = tree.get_next_skill()
    weak_skill = tree.get_weakest_skill()
    upgrading = False

    if new_skill:
        # Prioritize new skills, but upgrade every 3rd cycle if all built
        skill = new_skill
        goal_text = tree.build_goal_for_skill(skill)
        action = "BUILDING"
    elif weak_skill and weak_skill.get("quality_score", 999) < 200:
        # No new skills to build - upgrade the weakest
        skill = weak_skill
        goal_text = tree.build_upgrade_goal(skill)
        upgrading = True
        action = f"UPGRADING (quality: {skill.get('quality_score', '?')})"
    else:
        print("\n🎉 ALL SKILLS COMPLETE AND HIGH QUALITY!")
        tree.print_tree()
        return True

    print(f"\n{'='*70}")
    print(f"CYCLE #{cycle_num} | {datetime.now().strftime('%H:%M:%S')}")
    print(f"{action}: {skill['name']} (Tier {skill.get('tier','?')}, Impact {skill.get('current_impact', skill.get('impact','?'))}/10)")
    print(f"FILE: {skill.get('file','?')}")
    print(f"{'='*70}")
    tree.print_tree()
    print()

    output_dir = Path("./skills")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backup existing file before upgrade (restore if upgrade breaks it)
    target_file_path = output_dir / skill["file"]
    backup_path = target_file_path.with_suffix(".py.bak")
    if target_file_path.exists():
        import shutil
        shutil.copy2(target_file_path, backup_path)

    target_file = str(output_dir / skill["file"])
    target_name = skill["file"]
    max_retries = 3
    ok = False
    msg = ""

    for attempt in range(1, max_retries + 1):
        try:
            if attempt == 1:
                # First attempt: use the full goal
                current_goal = goal_text
            else:
                # Retry: tell the agent exactly what failed and how to fix it
                current_goal = (
                    f"Your previous code was REJECTED by validation.\n\n"
                    f"REJECTION REASON: {msg}\n\n"
                    f"Fix this SPECIFIC issue. The validation requires:\n"
                    f"- MINIMUM 3 functions or methods (not counting test code)\n"
                    f"- MINIMUM 5 assert statements in the test block\n"
                    f"- No 'from src.' imports\n"
                    f"- Must print 'ALL TESTS PASSED'\n\n"
                    f"Read the current file, fix the issue, test it, say DONE."
                )

            if agent is None:
                model_name = os.environ.get("AGENT_MODEL", "tool_calling")
                agent = MLXAgent(config_model_name=model_name, goal=f"Build: {skill['name']}")
            else:
                agent.reset_for_new_task(f"{'Fix' if attempt > 1 else 'Build'}: {skill['name']}")
            agent.run_loop(current_goal)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"\n❌ Agent error: {e}")
            msg = str(e)[:200]
            continue

        # Validate the output
        if not Path(target_file).exists():
            msg = "No file produced"
            print(f"\n❌ Attempt {attempt}/{max_retries}: {msg}")
            continue

        print(f"\n{'─'*40}")
        print(f"VALIDATING (attempt {attempt}/{max_retries}): {target_file}")
        ok, msg = validate_output(target_file)

        if ok:
            print(f"✅ PASSED: {target_name} - {msg}")
            break
        else:
            print(f"❌ FAILED (attempt {attempt}/{max_retries}): {msg}")
            if attempt < max_retries:
                print(f"🔄 Retrying with validation feedback...")

    # Log result
    cycle_logger = AgentLogger(f"cycle_{cycle_num}")
    cycle_logger.validation(target_name, ok, msg)

    if ok:
        tree.mark_completed(skill["id"], msg)
        if backup_path.exists():
            backup_path.unlink()
        history["cycles"].append({
            "cycle": cycle_num, "target": target_name, "passed": True,
            "reason": msg[:100], "timestamp": datetime.now().isoformat(),
        })
        history["total_passed"] += 1
    else:
        tree.mark_failed(skill["id"], msg)
        if backup_path.exists():
            import shutil
            shutil.copy2(backup_path, target_file_path)
            print(f"🔄 Restored backup: {target_name}")
        elif Path(target_file).exists():
            Path(target_file).unlink()
            print(f"🗑️  Deleted: {target_name}")
        history["cycles"].append({
            "cycle": cycle_num, "target": target_name, "passed": False,
            "reason": msg[:100], "timestamp": datetime.now().isoformat(),
        })
        history["total_failed"] += 1

    save_history(history)
    return ok


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python improve.py 1           # Run cycle 1")
        print("  python improve.py 1 --loop    # Run continuously")
        print("")
        print("The agent sees the ENTIRE codebase each cycle and")
        print("DECIDES what to improve based on what would have the most impact.")
        sys.exit(1)

    cycle_num = int(sys.argv[1])
    loop_mode = "--loop" in sys.argv

    # Pre-load model ONCE for all cycles (saves 5-10s per cycle)
    model_name = os.environ.get("AGENT_MODEL", "tool_calling")
    persistent_agent = MLXAgent(config_model_name=model_name, goal="init") if loop_mode else None

    if loop_mode:
        current = cycle_num
        while True:
            try:
                ok = run_cycle(current, agent=persistent_agent)
                h = load_history()
                rate = h["total_passed"] / max(1, h["total_passed"] + h["total_failed"])
                print(f"\n📊 Overall: {h['total_passed']} passed, {h['total_failed']} failed ({rate:.0%} success rate)")
                current += 1
                time.sleep(3)
            except KeyboardInterrupt:
                h = load_history()
                print(f"\n\nFinal: {h['total_passed']} passed, {h['total_failed']} failed")
                break
    else:
        ok = run_cycle(cycle_num)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
