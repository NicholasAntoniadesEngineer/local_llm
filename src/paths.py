"""Central path definitions. Every file references paths from here."""

from pathlib import Path

# Project root (parent of src/)
ROOT = Path(__file__).parent.parent

# Core source
SRC_DIR = ROOT / "src"

# Agent-built skill modules
SKILLS_DIR = ROOT / "skills"

# Per-run artifacts (logs, model output, perf stats)
RUNS_DIR = ROOT / "runs"

# Run history (tracks pass/fail across all cycles)
HISTORY_FILE = RUNS_DIR / "history.json"

# Proposals file (agent can expand the skill tree)
PROPOSALS_FILE = SKILLS_DIR / "tree_proposals.txt"


def get_run_dir(run_id: str) -> Path:
    """Get or create a timestamped directory for a specific run.

    Format: runs/YYYYMMDD_HHMMSS_short_goal/
    Sanitizes the run_id to be filesystem-safe.
    """
    from datetime import datetime
    import re as _re

    # Sanitize: remove non-alphanumeric, collapse spaces, truncate
    safe = _re.sub(r'[^a-zA-Z0-9_\- ]', '', run_id).strip()
    safe = _re.sub(r'\s+', '_', safe)[:40]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"{timestamp}_{safe}" if safe else timestamp

    d = RUNS_DIR / dirname
    d.mkdir(parents=True, exist_ok=True)
    return d
