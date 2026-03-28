# Agent rules (frozen context)

This file is loaded into the **first** LLM message (KV-stable). The **user** message carries goal, phase, verifier results, and history—do not assume those appear here.

## Verifier alignment

- **Asserts**: At least **4** real Python `assert` statements (AST), not only the word in docstrings.
- **Depth**: Besides `__init__`, every function needs substantive body lines; at least one method **≥5** substantive lines; sum of substantive lines across those methods **≥10** (see runtime verifier).
- **Size**: Skill modules must not be tiny placeholders (minimum file size enforced).
- **Imports**: No `from src.` — skills are standalone under `skills/`.
- **Tests**: `python skills/<file>.py` must print `ALL TESTS PASSED` when successful.
- **Prerequisites**: If the skill tree lists prereqs for this skill:
  - **Import** each prereq module (e.g. `from error_recovery import ...`).
  - **Use** each prereq in code that is **not** an import line: call a class/function, or reference a name in expressions. Import-only fails with **Missing prereq usage**.

## File size and errors

- If the module will exceed **~100 lines**, structure with **`try` / `except`** around risky or external-facing sections so failures are handled, not bare tracebacks in normal paths.
- Prefer explicit checks and asserts over silent failure.

## Tools and editing

- Use `read_file` with line ranges for large files; `replace_lines` for multi-line edits; `edit_file` only when replacing a unique exact substring.
- After `web_search`, read the target skill with `read_file` or `list_dir`; do not repeat the same search query.

## Completion

- Do not claim the task is done until the **verifier** has accepted the artifact.
- Emit concrete tool calls each turn when action is needed.
