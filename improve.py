#!/usr/bin/env python3
"""
Self-Improving Agent: Single entry point for agent improvement cycles.

Each cycle: Agent builds ONE concrete feature that extends capabilities.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from agent import MLXAgent


class ImprovementCycle:
    """Manages a single improvement cycle."""

    # Define what to build in each cycle
    FEATURES = {
        1: {
            "name": "Search Result Caching",
            "description": "Cache API results to avoid duplicate calls",
            "file": "cache.py",
        },
        2: {
            "name": "Performance Metrics",
            "description": "Track and log agent performance metrics",
            "file": "metrics.py",
        },
        3: {
            "name": "Memory Compression",
            "description": "Automatically compress old discoveries",
            "file": "compression.py",
        },
    }

    def __init__(self, cycle_num: int):
        """Initialize improvement cycle."""
        self.cycle_num = cycle_num
        self.feature = self.FEATURES.get(cycle_num)

    def run(self) -> bool:
        """Run the improvement cycle."""

        if not self.feature:
            print(f"\n✅ Predefined cycles complete. Agent can now choose improvements.")
            return True

        print(f"\n{'='*80}")
        print(f"CYCLE #{self.cycle_num}: {self.feature['name']}")
        print(f"{'='*80}")

        # Create goal for agent
        goal = self._create_goal()

        # Run agent
        try:
            agent = MLXAgent(
                config_model_name="balanced",
                goal=f"Build: {self.feature['name']}"
            )
            agent.run_loop(goal)

            # Commit changes
            self._commit_changes()
            return True

        except Exception as e:
            print(f"\n❌ Cycle failed: {e}")
            return False

    def _create_goal(self) -> str:
        """Create the improvement goal for the agent."""

        return f"""
CYCLE #{self.cycle_num}: Build {self.feature['name']}

TASK:
Create a new file '{self.feature['file']}' that implements {self.feature['description']}.

REQUIREMENTS:
1. Write COMPLETE, working Python code
2. Include docstrings and type hints
3. Make it production-quality
4. Test it works with run_python
5. Save to {self.feature['file']}

EXAMPLE STRUCTURE:
- Class or functions with clear purpose
- Docstrings on all public methods
- Type hints on parameters and returns
- Working example/test at the bottom

Build this now.
"""

    def _commit_changes(self) -> bool:
        """Commit the new feature to git."""
        try:
            file_path = self.feature['file']

            # Check if file was created
            if not Path(file_path).exists():
                print(f"⚠️  File {file_path} not created")
                return False

            # Stage and commit
            subprocess.run(["git", "add", file_path], timeout=5, check=True)

            commit_msg = (
                f"feat(cycle{self.cycle_num}): {self.feature['name']}\n\n"
                f"- {self.feature['description']}\n"
                f"- Timestamp: {datetime.now().isoformat()}"
            )

            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                timeout=5,
                check=True,
                capture_output=True
            )

            print(f"\n✅ Committed: {self.feature['name']}")
            return True

        except Exception as e:
            print(f"⚠️  Commit failed: {e}")
            return False


def main():
    """Main entry point."""

    if len(sys.argv) < 2:
        print("Usage: python improve.py <cycle_number>")
        print("\nExample:")
        print("  python improve.py 1    # Run cycle 1 (Search caching)")
        print("  python improve.py 2    # Run cycle 2 (Metrics)")
        sys.exit(1)

    try:
        cycle_num = int(sys.argv[1])
    except ValueError:
        print("Error: cycle_number must be an integer")
        sys.exit(1)

    # Run the improvement cycle
    cycle = ImprovementCycle(cycle_num)
    success = cycle.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
