#!/usr/bin/env python3
"""
Infinite Self-Improvement Loop

The agent will continuously improve itself, cycle after cycle,
creating an ever-improving system with no stopping point.

Each cycle:
1. Analyzes its own code
2. Finds ONE improvement
3. Implements it
4. Tests it
5. Commits it
6. Repeats forever

The system should compound improvements:
- Cycle 1-3: Basic optimizations (caching, memory)
- Cycle 4-6: Pattern recognition improvements
- Cycle 7-10: Auto-tuning and specialized learning
- Cycle 10+: Emergent capabilities you couldn't predict
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

def run_improvement_cycle(cycle_num: int) -> bool:
    """Run one self-improvement cycle with real-time output.

    Args:
        cycle_num: Which cycle number this is

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"🔄 IMPROVEMENT CYCLE #{cycle_num}")
    print(f"{'='*80}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("Running agent...\n")
    sys.stdout.flush()

    try:
        # Run the self-improvement script with real-time output
        process = subprocess.Popen(
            [sys.executable, "self_improve.py", str(cycle_num)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line-buffered
        )

        # Show output in real-time
        last_lines = []
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
                sys.stdout.flush()
                last_lines.append(line.rstrip())
                # Keep last 100 lines in memory
                if len(last_lines) > 100:
                    last_lines.pop(0)

        returncode = process.wait()

        if returncode == 0:
            print(f"\n✅ Cycle #{cycle_num} completed successfully")
        else:
            print(f"\n⚠️  Cycle #{cycle_num} completed with return code {returncode}")

        return returncode == 0

    except subprocess.TimeoutExpired:
        print(f"\n⏱️  Cycle #{cycle_num} timed out")
        process.kill()
        return False
    except Exception as e:
        print(f"\n❌ Cycle #{cycle_num} failed with error: {e}")
        return False


def main():
    """Run infinite improvement cycles."""

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                   🚀 INFINITE SELF-IMPROVEMENT LOOP 🚀                     ║
║                                                                            ║
║  The agent will continuously improve itself, forever.                      ║
║                                                                            ║
║  Each cycle the agent will:                                               ║
║  1. Read its own code                                                      ║
║  2. Identify ONE improvement to make                                       ║
║  3. Implement the improvement                                              ║
║  4. Test that it works                                                     ║
║  5. Commit the change                                                      ║
║  6. Go to step 1                                                           ║
║                                                                            ║
║  Expected progression:                                                     ║
║  • Cycles 1-2:   Basic optimizations                                      ║
║  • Cycles 3-5:   Pattern detection & learning                             ║
║  • Cycles 6-10:  Domain-specific improvements                             ║
║  • Cycles 10+:   Emergent capabilities                                    ║
║                                                                            ║
║  Press Ctrl+C to stop at any time.                                         ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

    print("Starting infinite improvement loop immediately...\n")
    time.sleep(1)  # Brief pause so message is visible

    cycle_num = 1
    successful_cycles = 0
    failed_cycles = 0

    while True:
        try:
            # Run one improvement cycle
            success = run_improvement_cycle(cycle_num)

            if success:
                successful_cycles += 1
            else:
                failed_cycles += 1

            # Summary
            print(f"\n📊 Progress: {successful_cycles} successful, {failed_cycles} failed")
            print(f"   Total improvements made: {successful_cycles}")
            print(f"   System uptime: ~{cycle_num * 5} minutes")

            # Check git status
            try:
                status = subprocess.run(
                    ["git", "status", "--short"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if status.stdout.strip():
                    print(f"   Uncommitted changes: {len(status.stdout.strip().split(chr(10)))} files")
            except:
                pass

            # Wait before next cycle
            print(f"\n⏳ Waiting 5 seconds before cycle #{cycle_num + 1}...")
            time.sleep(5)

            cycle_num += 1

        except KeyboardInterrupt:
            print(f"\n\n{'='*80}")
            print("🛑 Infinite improvement interrupted by user")
            print(f"{'='*80}")
            print(f"\n📈 Final Statistics:")
            print(f"   Total cycles completed: {cycle_num - 1}")
            print(f"   Successful: {successful_cycles}")
            print(f"   Failed: {failed_cycles}")
            print(f"   Success rate: {100*successful_cycles/(successful_cycles+failed_cycles):.1f}%")
            print(f"\n💾 All improvements committed to git.")
            print(f"   View changes: git log --oneline | head -20")
            print(f"   View diffs:   git diff HEAD~{successful_cycles}...HEAD")
            print()
            break


if __name__ == "__main__":
    main()
