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
    """Get or create a directory for a specific run."""
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d
