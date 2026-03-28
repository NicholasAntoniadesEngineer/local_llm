"""Build the frozen static prompt block (message 0) for KV-cache stability."""

from __future__ import annotations

from pathlib import Path

from src.paths import AGENT_RULES_FILE
from src.runtime.verifier import (
    MIN_SKILL_ASSERT_STATEMENTS,
    MIN_SUBSTANTIVE_LINES_BEST_METHOD,
    MIN_SUBSTANTIVE_LINES_PER_METHOD,
    MIN_SUBSTANTIVE_LINES_SUM_NON_INIT,
)

_SNAPSHOT_MAX_CHARS = 85_000
_AGENT_RULES_MAX_CHARS = 32_000
_SKILL_FILE_MAX_LINES = 140
_VERIFIER_SNIPPET_MAX_LINES = 160
_README_MAX_LINES = 120


def _read_text_limited(path: Path, *, max_chars: int | None = None) -> str:
    if not path.is_file():
        return ""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if max_chars is not None and len(raw) > max_chars:
        return raw[:max_chars] + "\n\n...(truncated)\n"
    return raw


def _head_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + "\n\n...(truncated)\n"


def build_frozen_static_prompt_block(
    root: Path,
    skills_dir: Path,
    *,
    skill_tree_text: str = "",
) -> str:
    """Assemble KV-stable context: rules, verifier summary, tree, and skill/repo excerpts."""
    chunks: list[str] = [
        "## Frozen static context",
        "The next message is dynamic (goal, phase, verifier feedback, traces). "
        "Treat this block as the long-lived project contract.\n",
    ]

    rules_body = _read_text_limited(AGENT_RULES_FILE, max_chars=_AGENT_RULES_MAX_CHARS)
    if rules_body.strip():
        chunks.append("## AGENT_RULES.md\n" + rules_body.strip())
    else:
        chunks.append(
            "## AGENT_RULES.md\n(missing at repo root; add AGENT_RULES.md for full rule text)\n"
        )

    chunks.append(
        "## Verifier numeric gates (summary)\n"
        f"- assert statements (AST): ≥ {MIN_SKILL_ASSERT_STATEMENTS}\n"
        f"- substantive lines per non-__init__ method: ≥ {MIN_SUBSTANTIVE_LINES_PER_METHOD}\n"
        f"- best method substantive lines: ≥ {MIN_SUBSTANTIVE_LINES_BEST_METHOD}\n"
        f"- sum substantive lines (non-__init__): ≥ {MIN_SUBSTANTIVE_LINES_SUM_NON_INIT}\n"
        "- prereqs: import + non-import usage required when skill has dependencies\n"
        "- no `from src.` imports; standalone skills/\n"
    )

    verifier_path = root / "src" / "runtime" / "verifier.py"
    verifier_src = _read_text_limited(verifier_path)
    if verifier_src:
        chunks.append(
            "## Verifier module (excerpt)\n```python\n"
            + _head_lines(verifier_src, _VERIFIER_SNIPPET_MAX_LINES)
            + "\n```\n"
        )

    readme_path = root / "README.md"
    readme_src = _read_text_limited(readme_path)
    if readme_src.strip():
        chunks.append("## README (excerpt)\n" + _head_lines(readme_src, _README_MAX_LINES))

    tree_trimmed = skill_tree_text.strip()
    if tree_trimmed:
        if len(tree_trimmed) > 16_000:
            tree_trimmed = tree_trimmed[:16_000] + "\n...(truncated skill tree)\n"
        chunks.append("## Skill tree snapshot\n" + tree_trimmed)

    skill_chunks: list[str] = []
    for skill_path in sorted(skills_dir.glob("*.py")):
        name = skill_path.name
        if name.startswith("_"):
            continue
        body = _read_text_limited(skill_path)
        excerpt = _head_lines(body, _SKILL_FILE_MAX_LINES)
        skill_chunks.append(f"### skills/{name}\n```python\n{excerpt}\n```\n")

    if skill_chunks:
        chunks.append("## Skill modules (prefix excerpts)\n" + "\n".join(skill_chunks))

    assembled = "\n\n".join(part.strip() for part in chunks if part.strip())
    if len(assembled) > _SNAPSHOT_MAX_CHARS:
        assembled = assembled[:_SNAPSHOT_MAX_CHARS] + "\n\n...(frozen block truncated for size)\n"
    return assembled
