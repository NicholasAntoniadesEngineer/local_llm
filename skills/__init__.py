"""Auto-register all skill modules in this directory."""

from pathlib import Path

SKILL_MODULES = [
    f.stem for f in Path(__file__).parent.glob("*.py")
    if not f.name.startswith("_")
]
