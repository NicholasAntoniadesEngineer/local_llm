"""Atomic file writes with syntax and truncation protection."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WriteResult:
    """Outcome of an attempted guarded write."""

    success: bool
    message: str
    path: Path


class AtomicWriter:
    """Write files safely with validation, backup, and rollback."""

    def __init__(self, minimum_python_bytes: int = 200, size_ratio_floor: float = 0.8) -> None:
        self.minimum_python_bytes = minimum_python_bytes
        self.size_ratio_floor = size_ratio_floor

    def _validate_python(self, target_path: Path, content_text: str) -> None:
        if target_path.suffix == ".py":
            ast.parse(content_text)

    def _minimum_size(self, target_path: Path, existing_text: str) -> int:
        if target_path.suffix != ".py":
            return 1
        existing_threshold = int(len(existing_text) * self.size_ratio_floor) if existing_text else 0
        return max(self.minimum_python_bytes, existing_threshold)

    def write_text(self, target_path: Path, content_text: str) -> WriteResult:
        """Write content to disk only if validation succeeds."""
        resolved_path = Path(target_path)
        if not content_text:
            return WriteResult(False, "ERROR: write rejected - content is empty", resolved_path)

        temp_path = resolved_path.with_suffix(resolved_path.suffix + ".tmp")
        backup_path = resolved_path.with_suffix(resolved_path.suffix + ".bak")
        existing_text = resolved_path.read_text() if resolved_path.exists() else ""

        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            self._validate_python(resolved_path, content_text)

            minimum_size = self._minimum_size(resolved_path, existing_text)
            if len(content_text) < minimum_size:
                raise ValueError(
                    f"content too small ({len(content_text)}B < required {minimum_size}B)"
                )

            temp_path.write_text(content_text)

            if resolved_path.exists():
                backup_path.write_text(existing_text)

            temp_path.replace(resolved_path)
            if backup_path.exists():
                backup_path.unlink()

            return WriteResult(
                True,
                f"Wrote {len(content_text)} bytes to {resolved_path}",
                resolved_path,
            )
        except Exception as error_value:
            if temp_path.exists():
                temp_path.unlink()
            if backup_path.exists():
                backup_path.replace(resolved_path)
            return WriteResult(
                False,
                f"ERROR: write rejected - {error_value}",
                resolved_path,
            )
