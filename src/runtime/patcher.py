"""Guarded mutation pipeline with explicit commit and rollback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.write_guard import AtomicWriter, WriteResult


@dataclass
class PendingMutation:
    """Mutation staged on disk and awaiting verifier commit."""

    target_path: Path
    mutation_kind: str
    existed_before: bool
    previous_text: str
    updated_text: str


class MutationCoordinator:
    """Apply guarded mutations and revert them when verification fails."""

    def __init__(self, atomic_writer: AtomicWriter) -> None:
        self.atomic_writer = atomic_writer
        self._pending_mutations: dict[str, PendingMutation] = {}

    def apply_mutation(self, target_path: Path, updated_text: str, mutation_kind: str) -> tuple[WriteResult, PendingMutation | None]:
        """Write the new content and register it for verifier commit/rollback."""
        resolved_path = Path(target_path)
        existed_before = resolved_path.exists()
        previous_text = resolved_path.read_text() if existed_before else ""
        write_result = self.atomic_writer.write_text(resolved_path, updated_text)
        if not write_result.success:
            return write_result, None
        mutation_record = PendingMutation(
            target_path=resolved_path,
            mutation_kind=mutation_kind,
            existed_before=existed_before,
            previous_text=previous_text,
            updated_text=updated_text,
        )
        self._pending_mutations[str(resolved_path)] = mutation_record
        return write_result, mutation_record

    def has_pending_mutation(self, target_path: Path | str) -> bool:
        """Return whether a path currently has an uncommitted mutation."""
        return str(Path(target_path)) in self._pending_mutations

    def commit_mutation(self, target_path: Path | str) -> str:
        """Commit a pending mutation after verifier acceptance."""
        resolved_key = str(Path(target_path))
        mutation_record = self._pending_mutations.pop(resolved_key, None)
        if mutation_record is None:
            return f"No pending mutation for {resolved_key}"
        return f"Committed {mutation_record.mutation_kind} on {resolved_key}"

    def rollback_mutation(self, target_path: Path | str) -> str:
        """Rollback a pending mutation after verifier rejection."""
        resolved_path = Path(target_path)
        resolved_key = str(resolved_path)
        mutation_record = self._pending_mutations.pop(resolved_key, None)
        if mutation_record is None:
            return f"No pending mutation for {resolved_key}"
        if mutation_record.existed_before:
            restore_result = self.atomic_writer.write_text(resolved_path, mutation_record.previous_text)
            if not restore_result.success:
                raise RuntimeError(restore_result.message)
            return f"Rolled back {mutation_record.mutation_kind} on {resolved_key}"
        if resolved_path.exists():
            resolved_path.unlink()
        return f"Removed newly created file at {resolved_key}"
