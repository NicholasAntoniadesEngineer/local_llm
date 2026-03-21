#!/usr/bin/env python3
"""
Self-Improvement Loop: Agent analyzes its own code and improves itself iteratively.

The agent's PRIMARY goal: Become more capable by:
1. Reading its own source code
2. Identifying limitations and improvements
3. Implementing code changes
4. Testing those changes
5. Measuring improvement
6. Iterating
"""

import sys
import subprocess
from agent import MLXAgent
from datetime import datetime

def commit_improvements(cycle_num: int) -> bool:
    """Commit improvements to git if there are changes."""
    try:
        # Check if there are changes
        status = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if not status.stdout.strip():
            print("No changes to commit")
            return True

        # Stage changes
        subprocess.run(["git", "add", "-A"], capture_output=True, timeout=5)

        # Create commit message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_msg = f"Self-improvement cycle #{cycle_num} - {timestamp}"

        # Commit
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            print(f"✅ Committed: {commit_msg}")
            return True
        else:
            print(f"⚠️  Commit failed: {result.stderr[:100]}")
            return False

    except Exception as e:
        print(f"⚠️  Could not commit: {str(e)}")
        return False


def run_self_improvement_cycle(cycle_num: int = 1):
    """Run one cycle of agent self-improvement.

    Args:
        cycle_num: Which cycle this is (for naming/tracking)
    """

    self_improvement_goal = f"""
You are a self-improving AI agent. CYCLE #{cycle_num}.
Your PRIMARY goal is to make YOURSELF better.

WHAT YOU DO:
1. **ANALYZE** - Read your own source code (agent.py, memory.py, reflection.py)
2. **IDENTIFY** - Find ONE specific limitation or improvement opportunity
3. **PLAN** - Design exactly what to change and why it will help
4. **IMPLEMENT** - Write the code change
5. **TEST** - Verify the change works correctly
6. **MEASURE** - Show how the improvement helps you be better at your job
7. **ITERATE** - Repeat for different improvements

AREAS TO IMPROVE:
- Search quality detection (already have this - now enhance further)
- Loop detection accuracy (detect patterns more reliably)
- Memory efficiency (compress old discoveries)
- Phase transition logic (smarter decisions about when to code)
- Tool effectiveness (better at writing Python code)
- Reflection accuracy (better at learning from failures)

START HERE:
1. Read agent.py completely using read_file
2. Read memory.py to understand memory tracking
3. Read reflection.py to understand learning
4. Identify ONE clear improvement you could make to any of these
5. Implement it
6. Write a test for it
7. Show the improvement works

Your first improvement goal: Enhance the loop detection to also track
QUALITY OF LOOPS. Not just "same tool 3 times" but "same tool 3 times
producing similar results that aren't working". This would prevent
wasteful repetition earlier.

Go!
"""

    print("=" * 80)
    print("🤖 SELF-IMPROVEMENT CYCLE: Agent improves its own capabilities")
    print("=" * 80)
    print(f"\nGoal:\n{self_improvement_goal[:300]}...\n")

    # Create agent with self-improvement goal
    agent = MLXAgent(config_model_name="balanced", goal="Self-improvement: Enhance own capabilities")

    print("Starting self-improvement loop...\n")
    agent.run_loop(self_improvement_goal)

    print("\n" + "=" * 80)
    print(f"✅ Cycle #{cycle_num} complete.")
    print("=" * 80)

    # Commit improvements to git
    commit_improvements(cycle_num)


if __name__ == "__main__":
    import sys

    # Get cycle number from command line if provided
    cycle_num = 1
    if len(sys.argv) > 1:
        try:
            cycle_num = int(sys.argv[1])
        except ValueError:
            cycle_num = 1

    run_self_improvement_cycle(cycle_num)
