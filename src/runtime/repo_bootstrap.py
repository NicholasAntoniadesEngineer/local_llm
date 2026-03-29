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

# Global cap (suffix sections drop first: README → tools → tail of src excerpts).
# ~48k chars keeps frozen block closer to ~12–18k tokens; larger snapshots risk Metal abort mid-prefill.
_SNAPSHOT_MAX_CHARS = 48_000
_AGENT_RULES_MAX_CHARS = 32_000
_SKILL_FILE_MAX_LINES = 260
_VERIFIER_SNIPPET_MAX_LINES = 380
_README_MAX_LINES = 120
_SRC_FILE_MAX_LINES_PRIORITY = 180
_SRC_FILE_MAX_LINES_OTHER = 110
_TOOLS_FILE_MAX_LINES = 120

# Earlier = more important within the src excerpt section (after skills + verifier).
_SRC_EXCERPT_PRIORITY: tuple[str, ...] = (
    "src/runtime/tools.py",
    "src/runtime/tool_kinds.py",
    "src/runtime/policy.py",
    "src/runtime/prompt_builder.py",
    "src/runtime/controller.py",
    "src/runtime/task_state.py",
    "src/runtime/state_store.py",
    "src/agent.py",
    "src/config.py",
    "src/paths.py",
    "src/skill_tree.py",
    "src/runtime/improve_runner.py",
    "src/runtime/self_improve_runtime.py",
    "src/runtime/agent_runtime.py",
    "src/runtime/mlx_adapter.py",
    "src/context_manager.py",
    "src/memory.py",
    "src/logger.py",
    "src/write_guard.py",
    "src/runtime/runtime_support.py",
    "src/runtime/tool_call_parser.py",
    "src/runtime/llm_text.py",
    "src/runtime/patcher.py",
    "src/runtime/benchmark_suite.py",
    "src/runtime/turboquant_mlx_setup.py",
)

_SKIP_SRC_EXCERPT_REL = frozenset(
    {
        "src/runtime/repo_bootstrap.py",
        "src/runtime/verifier.py",
    }
)


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


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def _iter_src_py_paths(root: Path) -> list[Path]:
    src_root = root / "src"
    if not src_root.is_dir():
        return []
    return [
        path_value
        for path_value in src_root.rglob("*.py")
        if "__pycache__" not in path_value.parts
    ]


def _sort_src_paths_for_excerpts(root: Path, paths: list[Path]) -> list[Path]:
    rank_map = {rel_path: index for index, rel_path in enumerate(_SRC_EXCERPT_PRIORITY)}

    def sort_key(path_value: Path) -> tuple[int, str]:
        relative_posix = path_value.relative_to(root).as_posix()
        if relative_posix in _SKIP_SRC_EXCERPT_REL:
            return (10_000, relative_posix)
        priority_rank = rank_map.get(relative_posix, 500)
        return (priority_rank, relative_posix)

    ordered = sorted(paths, key=sort_key)
    return [path_value for path_value in ordered if sort_key(path_value)[0] < 10_000]


def _build_src_inventory_lines(root: Path, paths: list[Path]) -> str:
    lines_out: list[str] = ["## src/ inventory (line counts)", ""]
    for path_value in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        relative_posix = path_value.relative_to(root).as_posix()
        try:
            line_total = _line_count(path_value)
        except OSError:
            line_total = -1
        lines_out.append(f"- `{relative_posix}` ({line_total} lines)")
    return "\n".join(lines_out)


def _build_src_excerpt_chunk(root: Path, path_value: Path) -> str:
    relative_posix = path_value.relative_to(root).as_posix()
    max_lines = (
        _SRC_FILE_MAX_LINES_PRIORITY
        if relative_posix in _SRC_EXCERPT_PRIORITY
        else _SRC_FILE_MAX_LINES_OTHER
    )
    body = _read_text_limited(path_value)
    excerpt = _head_lines(body, max_lines)
    return f"### {relative_posix}\n```python\n{excerpt}\n```\n"


def _build_tools_excerpts(root: Path) -> str:
    tools_dir = root / "tools"
    if not tools_dir.is_dir():
        return ""
    parts: list[str] = ["## tools/ (prefix excerpts)", ""]
    for path_value in sorted(tools_dir.glob("*.py")):
        body = _read_text_limited(path_value)
        excerpt = _head_lines(body, _TOOLS_FILE_MAX_LINES)
        parts.append(f"### tools/{path_value.name}\n```python\n{excerpt}\n```\n")
    return "\n".join(parts).strip()


def build_frozen_static_prompt_block(
    root: Path,
    skills_dir: Path,
    *,
    skill_tree_text: str = "",
) -> str:
    """Assemble KV-stable context: rules, tree, skills, verifier, then broader repo excerpts."""
    header_chunk = "\n\n".join(
        [
            "## Frozen static context",
            "The next message is dynamic (goal, phase, verifier feedback, traces). "
            "Treat this block as the long-lived project contract.",
        ]
    )

    rules_body = _read_text_limited(AGENT_RULES_FILE, max_chars=_AGENT_RULES_MAX_CHARS)
    if rules_body.strip():
        rules_chunk = "## AGENT_RULES.md\n" + rules_body.strip()
    else:
        rules_chunk = (
            "## AGENT_RULES.md\n(missing at repo root; add AGENT_RULES.md for full rule text)\n"
        )

    gates_chunk = (
        "## Verifier numeric gates (summary)\n"
        f"- assert statements (AST): ≥ {MIN_SKILL_ASSERT_STATEMENTS}\n"
        f"- substantive lines per non-__init__ method: ≥ {MIN_SUBSTANTIVE_LINES_PER_METHOD}\n"
        f"- best method substantive lines: ≥ {MIN_SUBSTANTIVE_LINES_BEST_METHOD}\n"
        f"- sum substantive lines (non-__init__): ≥ {MIN_SUBSTANTIVE_LINES_SUM_NON_INIT}\n"
        "- prereqs: import + non-import usage required when skill has dependencies\n"
        "- no `from src.` imports; standalone skills/\n"
    )

    tree_trimmed = skill_tree_text.strip()
    if tree_trimmed:
        if len(tree_trimmed) > 20_000:
            tree_trimmed = tree_trimmed[:20_000] + "\n...(truncated skill tree)\n"
        tree_chunk = "## Skill tree snapshot\n" + tree_trimmed
    else:
        tree_chunk = ""

    skill_chunks: list[str] = []
    for skill_path in sorted(skills_dir.glob("*.py")):
        skill_name = skill_path.name
        if skill_name.startswith("_"):
            continue
        body = _read_text_limited(skill_path)
        excerpt = _head_lines(body, _SKILL_FILE_MAX_LINES)
        skill_chunks.append(f"### skills/{skill_name}\n```python\n{excerpt}\n```\n")
    skills_section = (
        "## Skill modules (prefix excerpts)\n" + "\n".join(skill_chunks) if skill_chunks else ""
    )

    verifier_path = root / "src" / "runtime" / "verifier.py"
    verifier_src = _read_text_limited(verifier_path)
    verifier_chunk = ""
    if verifier_src:
        verifier_chunk = (
            "## Verifier module (excerpt)\n```python\n"
            + _head_lines(verifier_src, _VERIFIER_SNIPPET_MAX_LINES)
            + "\n```\n"
        )

    src_paths = _iter_src_py_paths(root)
    inventory_chunk = _build_src_inventory_lines(root, src_paths) if src_paths else ""

    excerpt_paths = _sort_src_paths_for_excerpts(root, src_paths)
    src_excerpt_parts = [_build_src_excerpt_chunk(root, p) for p in excerpt_paths]
    src_excerpts_chunk = (
        "## src/ excerpts (prioritized)\n" + "\n".join(src_excerpt_parts) if src_excerpt_parts else ""
    )

    tools_chunk = _build_tools_excerpts(root)

    readme_path = root / "README.md"
    readme_src = _read_text_limited(readme_path)
    readme_chunk = (
        "## README (excerpt)\n" + _head_lines(readme_src, _README_MAX_LINES)
        if readme_src.strip()
        else ""
    )

    ordered_sections = [
        header_chunk,
        rules_chunk,
        gates_chunk,
        tree_chunk,
        skills_section,
        verifier_chunk,
        inventory_chunk,
        src_excerpts_chunk,
        tools_chunk,
        readme_chunk,
    ]
    assembled = "\n\n".join(section.strip() for section in ordered_sections if section.strip())
    if len(assembled) > _SNAPSHOT_MAX_CHARS:
        assembled = assembled[:_SNAPSHOT_MAX_CHARS] + "\n\n...(frozen block truncated for size)\n"
    return assembled
