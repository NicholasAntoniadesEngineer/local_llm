#!/usr/bin/env python3
"""Send real-time messages to the running agent. Run in a separate terminal."""

import sys
sys.path.insert(0, ".")

from pathlib import Path
from src.paths import RUNS_DIR


def get_input_file() -> Path:
    """Find user_input.txt in the latest run directory."""
    if RUNS_DIR.exists():
        run_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
        if run_dirs:
            latest = max(run_dirs, key=lambda d: d.stat().st_mtime)
            return latest / "user_input.txt"
    return Path("/tmp/agent_input.txt")


print("=" * 50)
print("  Agent Input Console")
print("  Type a message and press Enter to send.")
print("  The agent reads it at the start of each step.")
print("  Type 'quit' to exit.")
print("=" * 50)

while True:
    try:
        msg = input("\n> ")
        if msg.strip().lower() in ("quit", "exit", "q"):
            break
        if msg.strip():
            f = get_input_file()
            f.write_text(msg.strip())
            print(f"  Sent to {f.parent.name}/user_input.txt")
    except (KeyboardInterrupt, EOFError):
        break

print("\nInput console closed.")
