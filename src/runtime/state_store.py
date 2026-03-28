"""Unified persistent state for runs, memory, rewards, and checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.paths import RUNS_DIR
from src.write_guard import AtomicWriter


STATE_FILE = RUNS_DIR / "controller_state.json"


@dataclass
class MemoryRecord:
    """Structured record for a tool attempt."""

    run_id: str
    step: int
    phase: str
    goal: str
    tool_name: str
    success: bool
    input_summary: str
    result_summary: str
    timestamp: str


@dataclass
class ValidationRecord:
    """Structured record for a verifier outcome."""

    run_id: str
    step: int
    target_path: str
    accepted: bool
    summary: str
    reward: float
    timestamp: str


@dataclass
class StrategyRecord:
    """Structured record for a controller strategy outcome."""

    run_id: str
    goal: str
    strategy_name: str
    success: bool
    metrics: dict[str, float]
    timestamp: str


@dataclass
class RewardRecord:
    """Structured reward record for skill/bandit shaping."""

    run_id: str
    step: int
    skill_id: str
    reward: float
    reason: str
    timestamp: str


@dataclass
class BenchmarkRecord:
    """Stored benchmark summary for a model/profile run."""

    benchmark_name: str
    profile_name: str
    model_name: str
    metrics: dict[str, float]
    timestamp: str


@dataclass
class TaskSnapshotRecord:
    """Stored task-state snapshot for promptless resume and monitoring."""

    run_id: str
    step: int
    phase: str
    accepted: bool
    completed: bool
    verifier_status: str
    failure_type: str
    target_files: list[str]
    timestamp: str


class PersistentStateStore:
    """Repo-local durable state store for the controller."""

    def __init__(self, run_id: str, goal: str, run_dir: Path) -> None:
        self.run_id = run_id
        self.goal = goal
        self.run_dir = Path(run_dir)
        self._writer = AtomicWriter(minimum_python_bytes=32)
        self._checkpoint_dir = self.run_dir / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_state_file()

    def _ensure_state_file(self) -> None:
        if STATE_FILE.exists():
            return
        payload = {
            "version": 1,
            "runs": [],
            "memory_records": [],
            "validation_records": [],
            "strategy_records": [],
            "reward_records": [],
            "benchmark_records": [],
            "task_snapshots": [],
        }
        self._write_state(payload)

    def _read_state(self) -> dict[str, Any]:
        self._ensure_state_file()
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {
                "version": 1,
                "runs": [],
                "memory_records": [],
                "validation_records": [],
                "strategy_records": [],
                "reward_records": [],
                "benchmark_records": [],
                "task_snapshots": [],
            }

    def _write_state(self, payload: dict[str, Any]) -> None:
        result = self._writer.write_text(STATE_FILE, json.dumps(payload, indent=2))
        if not result.success:
            raise RuntimeError(result.message)

    def _append_record(self, key: str, record: dict[str, Any], limit: int = 2000) -> None:
        payload = self._read_state()
        records = payload.setdefault(key, [])
        records.append(record)
        payload[key] = records[-limit:]
        self._write_state(payload)

    def register_run(self, model_name: str, config: dict[str, Any], resumed: bool = False) -> None:
        payload = self._read_state()
        runs = payload.setdefault("runs", [])
        current_timestamp = datetime.now().isoformat()
        runs = [run for run in runs if run.get("run_id") != self.run_id]
        runs.append(
            {
                "run_id": self.run_id,
                "goal": self.goal,
                "model": model_name,
                "config": config,
                "run_dir": str(self.run_dir),
                "resumed": resumed,
                "started_at": current_timestamp,
                "last_checkpoint": "",
                "updated_at": current_timestamp,
            }
        )
        payload["runs"] = runs[-200:]
        self._write_state(payload)

    def update_run_status(self, **fields: Any) -> None:
        payload = self._read_state()
        current_timestamp = datetime.now().isoformat()
        for run_record in payload.get("runs", []):
            if run_record.get("run_id") == self.run_id:
                run_record.update(fields)
                run_record["updated_at"] = current_timestamp
                break
        self._write_state(payload)

    def record_tool_attempt(
        self,
        step: int,
        phase: str,
        tool_name: str,
        args: dict[str, Any],
        result_text: str,
        success: bool,
    ) -> None:
        record = MemoryRecord(
            run_id=self.run_id,
            step=step,
            phase=phase,
            goal=self.goal,
            tool_name=tool_name,
            success=success,
            input_summary=json.dumps(args, default=str)[:300],
            result_summary=result_text[:400],
            timestamp=datetime.now().isoformat(),
        )
        self._append_record("memory_records", asdict(record))

    def record_validation(
        self,
        step: int,
        target_path: str,
        accepted: bool,
        summary: str,
        reward: float,
    ) -> None:
        record = ValidationRecord(
            run_id=self.run_id,
            step=step,
            target_path=target_path,
            accepted=accepted,
            summary=summary[:400],
            reward=reward,
            timestamp=datetime.now().isoformat(),
        )
        self._append_record("validation_records", asdict(record))

    def record_strategy_outcome(
        self,
        strategy_name: str,
        success: bool,
        metrics: dict[str, float],
    ) -> None:
        record = StrategyRecord(
            run_id=self.run_id,
            goal=self.goal,
            strategy_name=strategy_name,
            success=success,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
        )
        self._append_record("strategy_records", asdict(record))

    def record_reward(
        self,
        step: int,
        skill_id: str,
        reward: float,
        reason: str,
    ) -> None:
        record = RewardRecord(
            run_id=self.run_id,
            step=step,
            skill_id=skill_id,
            reward=reward,
            reason=reason[:300],
            timestamp=datetime.now().isoformat(),
        )
        self._append_record("reward_records", asdict(record))

    def record_benchmark(
        self,
        benchmark_name: str,
        profile_name: str,
        model_name: str,
        metrics: dict[str, float],
    ) -> None:
        record = BenchmarkRecord(
            benchmark_name=benchmark_name,
            profile_name=profile_name,
            model_name=model_name,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
        )
        self._append_record("benchmark_records", asdict(record), limit=500)

    def record_task_snapshot(self, task_state: dict[str, Any]) -> None:
        """Store a compact task-state snapshot for monitoring and resume analysis."""
        record = TaskSnapshotRecord(
            run_id=self.run_id,
            step=int(task_state.get("step", 0)),
            phase=str(task_state.get("phase", "")),
            accepted=bool(task_state.get("accepted", False)),
            completed=bool(task_state.get("completed", False)),
            verifier_status=str(task_state.get("verifier_status", "")),
            failure_type=str(task_state.get("last_failure_type", "")),
            target_files=list(task_state.get("target_files", []))[-6:],
            timestamp=datetime.now().isoformat(),
        )
        self._append_record("task_snapshots", asdict(record), limit=500)

    def save_checkpoint(self, step: int, payload: dict[str, Any]) -> Path:
        checkpoint_path = self._checkpoint_dir / f"step_{step:04d}.json"
        result = self._writer.write_text(checkpoint_path, json.dumps(payload, indent=2, default=str))
        if not result.success:
            raise RuntimeError(result.message)
        if "task_state" in payload:
            self.record_task_snapshot(payload["task_state"])
        self.update_run_status(last_checkpoint=str(checkpoint_path))
        return checkpoint_path

    def load_latest_checkpoint(self) -> dict[str, Any] | None:
        checkpoint_files = sorted(self._checkpoint_dir.glob("step_*.json"))
        if not checkpoint_files:
            return None
        try:
            return json.loads(checkpoint_files[-1].read_text())
        except Exception:
            return None

    def get_recent_failures(self, max_items: int = 5) -> list[str]:
        payload = self._read_state()
        failure_records = [
            record
            for record in payload.get("validation_records", [])
            if not record.get("accepted", False)
        ]
        failure_records = sorted(failure_records, key=lambda record: record.get("timestamp", ""), reverse=True)
        return [record.get("summary", "") for record in failure_records[:max_items]]

    def build_retrieval_context(self, goal: str, max_items: int = 6) -> list[str]:
        payload = self._read_state()
        goal_terms = set(goal.lower().split())
        candidate_records: list[tuple[float, str]] = []

        def score_text(text_value: str) -> float:
            text_terms = set(text_value.lower().split())
            if not text_terms:
                return 0.0
            overlap = len(goal_terms & text_terms)
            union = len(goal_terms | text_terms)
            return overlap / max(1, union)

        for record in payload.get("validation_records", [])[-200:]:
            summary_text = record.get("summary", "")
            score_value = score_text(summary_text) + (0.2 if not record.get("accepted", True) else 0.05)
            if score_value > 0:
                candidate_records.append((score_value, f"Validation: {summary_text}"))

        for record in payload.get("memory_records", [])[-200:]:
            summary_text = f"{record.get('tool_name', '')} {record.get('result_summary', '')}"
            score_value = score_text(summary_text) + (0.1 if record.get("success", False) else 0.15)
            if score_value > 0:
                candidate_records.append((score_value, f"Memory: {summary_text[:220]}"))

        for record in payload.get("strategy_records", [])[-100:]:
            summary_text = f"{record.get('strategy_name', '')} {record.get('goal', '')}"
            score_value = score_text(summary_text) + (0.2 if record.get("success", False) else 0.0)
            if score_value > 0:
                candidate_records.append((score_value, f"Strategy: {summary_text[:220]}"))

        candidate_records.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in candidate_records[:max_items]]

    def get_monitor_metrics(self, run_id: str | None = None) -> dict[str, Any]:
        payload = self._read_state()
        target_run_id = run_id or self.run_id
        validation_records = [
            record for record in payload.get("validation_records", [])
            if record.get("run_id") == target_run_id
        ]
        reward_records = [
            record for record in payload.get("reward_records", [])
            if record.get("run_id") == target_run_id
        ]
        accepted_count = sum(1 for record in validation_records if record.get("accepted"))
        total_validations = len(validation_records)
        reward_total = sum(float(record.get("reward", 0.0)) for record in reward_records)
        task_snapshots = [
            record for record in payload.get("task_snapshots", [])
            if record.get("run_id") == target_run_id
        ]
        latest_snapshot = task_snapshots[-1] if task_snapshots else {}
        last_checkpoint = ""
        for run_record in payload.get("runs", []):
            if run_record.get("run_id") == target_run_id:
                last_checkpoint = run_record.get("last_checkpoint", "")
                break
        return {
            "validation_pass_rate": round(accepted_count / max(1, total_validations) * 100, 1),
            "validation_count": total_validations,
            "reward_total": round(reward_total, 2),
            "last_checkpoint": last_checkpoint,
            "task_phase": latest_snapshot.get("phase", ""),
            "task_step": latest_snapshot.get("step", 0),
            "last_failure_type": latest_snapshot.get("failure_type", ""),
            "task_target_files": latest_snapshot.get("target_files", []),
        }

    def get_latest_run_record(self, run_id: str | None = None) -> dict[str, Any]:
        """Return the latest matching run record."""
        payload = self._read_state()
        if run_id:
            matching_runs = [
                run_record for run_record in payload.get("runs", [])
                if run_record.get("run_id") == run_id
            ]
            return matching_runs[-1] if matching_runs else {}
        runs = payload.get("runs", [])
        return runs[-1] if runs else {}

    def get_latest_benchmark_record(
        self,
        profile_name: str = "",
        benchmark_prefix: str = "",
    ) -> dict[str, Any]:
        """Return the latest benchmark record matching an optional profile and prefix."""
        payload = self._read_state()
        benchmark_records = payload.get("benchmark_records", [])
        filtered_records = []
        for record in benchmark_records:
            if profile_name and record.get("profile_name") != profile_name:
                continue
            benchmark_name = str(record.get("benchmark_name", ""))
            if benchmark_prefix and not benchmark_name.startswith(benchmark_prefix):
                continue
            filtered_records.append(record)
        return filtered_records[-1] if filtered_records else {}
