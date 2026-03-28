"""Central path definitions. Every file references paths from here."""

from pathlib import Path

# Project root (parent of src/)
ROOT = Path(__file__).parent.parent

# Core source
SRC_DIR = ROOT / "src"

# Agent-built skill modules
SKILLS_DIR = ROOT / "skills"

# Canonical rule text for frozen system prompt (KV-stable message 0)
AGENT_RULES_FILE = ROOT / "AGENT_RULES.md"

# Per-run artifacts (logs, model output, perf stats)
RUNS_DIR = ROOT / "run_output_data"

# Structured improve-loop journal (shared by tools/improve.py and tools/monitor.py)
IMPROVE_SESSION_FILE = RUNS_DIR / "improve_session.jsonl"

# Run history (tracks pass/fail across all cycles)
HISTORY_FILE = RUNS_DIR / "history.json"

# Proposals file (agent can expand the skill tree)
PROPOSALS_FILE = SKILLS_DIR / "tree_proposals.txt"


def get_run_dir(run_id: str) -> Path:
    """Get or create a timestamped directory for a specific run.

    Format: run_output_data/YYYYMMDD_HHMMSS_<suffix>/
    where suffix is a sanitized goal slug, or a short hex token when run_id is empty or
    duplicates the wall-clock second stamp (default AgentLogger run ids).
    """
    from datetime import datetime
    import re as _re

    # Sanitize: remove non-alphanumeric, collapse spaces, truncate
    safe = _re.sub(r'[^a-zA-Z0-9_\- ]', '', run_id).strip()
    safe = _re.sub(r'\s+', '_', safe)[:40]

    import secrets as _secrets

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_suffix = _secrets.token_hex(3)
    if not safe:
        dirname = f"{timestamp}_{unique_suffix}"
    elif safe == timestamp:
        # Default AgentLogger run_id is the same pattern as timestamp; avoid YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS
        dirname = f"{timestamp}_{unique_suffix}"
    else:
        dirname = f"{timestamp}_{safe}"

    d = RUNS_DIR / dirname
    d.mkdir(parents=True, exist_ok=True)
    return d
