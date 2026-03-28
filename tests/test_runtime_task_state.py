import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.runtime.benchmark_suite import FIXED_BENCHMARK_SLICE_NAME
from src.runtime.state_store import PersistentStateStore
from src.runtime.task_state import TASK_PHASE_ACCEPT, TASK_PHASE_PLAN, TaskState
from src.runtime.tools import ToolExecutor
from src.runtime.verifier import RuntimeVerifier, VerificationResult


class TaskStateTests(unittest.TestCase):
    def test_task_state_round_trip_preserves_phase_and_completion(self) -> None:
        task_state = TaskState(task_id="run-1", goal_text="refactor controller")
        task_state.mark_step(3)
        task_state.set_plan_items(["Inspect controller", "Patch verifier"])
        task_state.add_target_file("src/runtime/controller.py")
        task_state.update_verification("rejected_write", False, "Syntax error: bad indent", "syntax")
        task_state.transition_phase(TASK_PHASE_PLAN, "Need replanning after syntax failure.")
        task_state.transition_phase(TASK_PHASE_ACCEPT, "Verifier accepted final artifact.")

        restored_state = TaskState.from_dict(task_state.to_dict())

        self.assertEqual(restored_state.phase, TASK_PHASE_ACCEPT)
        self.assertTrue(restored_state.accepted)
        self.assertTrue(restored_state.completed)
        self.assertEqual(restored_state.last_failure_type, "syntax")
        self.assertIn("src/runtime/controller.py", restored_state.target_files)


class RuntimeVerifierTests(unittest.TestCase):
    def test_completion_requires_prior_accepted_verification(self) -> None:
        verifier = RuntimeVerifier()
        accepted_verification = VerificationResult(
            status="validated_write",
            accepted=True,
            should_stop=False,
            summary="Tests passed",
            target_path="artifact.py",
        )

        accepted_completion = verifier.evaluate_completion_signal("DONE", accepted_verification)
        rejected_completion = verifier.evaluate_completion_signal("DONE", None)

        self.assertTrue(accepted_completion.should_stop)
        self.assertEqual(rejected_completion.failure_type, "completion")
        self.assertFalse(rejected_completion.should_stop)


class StateStoreTaskSnapshotTests(unittest.TestCase):
    def test_save_checkpoint_records_task_snapshot_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            state_file = Path(temp_dir) / "controller_state.json"

            with patch("src.runtime.state_store.STATE_FILE", state_file):
                state_store = PersistentStateStore("run-2", "improve verifier", run_dir)
                state_store.register_run("model-name", {"max_iterations": 4}, resumed=False)
                task_state = TaskState(task_id="run-2", goal_text="improve verifier")
                task_state.mark_step(2)
                task_state.add_target_file("src/runtime/verifier.py")
                task_state.update_verification("rejected_write", False, "Tests failed: bad output", "test")
                state_store.save_checkpoint(2, {"step": 2, "task_state": task_state.to_dict()})

                metrics = state_store.get_monitor_metrics("run-2")
                payload = json.loads(state_file.read_text())

                self.assertEqual(metrics["task_phase"], task_state.phase)
                self.assertEqual(metrics["task_step"], 2)
                self.assertEqual(metrics["last_failure_type"], "test")
                self.assertTrue(payload["task_snapshots"])

    def test_latest_benchmark_record_filters_by_profile_and_slice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            state_file = Path(temp_dir) / "controller_state.json"

            with patch("src.runtime.state_store.STATE_FILE", state_file):
                state_store = PersistentStateStore("run-3", "benchmark runtime", run_dir)
                state_store.record_benchmark(
                    benchmark_name=f"{FIXED_BENCHMARK_SLICE_NAME}:inspect_controller_phase_machine",
                    profile_name="balanced",
                    model_name="model-a",
                    metrics={"avg_tok_s": 55.5, "elapsed_s": 12.0},
                )
                state_store.record_benchmark(
                    benchmark_name="ad_hoc",
                    profile_name="balanced",
                    model_name="model-a",
                    metrics={"avg_tok_s": 10.0, "elapsed_s": 99.0},
                )

                benchmark_record = state_store.get_latest_benchmark_record(
                    profile_name="balanced",
                    benchmark_prefix=FIXED_BENCHMARK_SLICE_NAME,
                )

                self.assertTrue(benchmark_record["benchmark_name"].startswith(FIXED_BENCHMARK_SLICE_NAME))
                self.assertEqual(benchmark_record["metrics"]["avg_tok_s"], 55.5)


class ToolExecutorMutationTests(unittest.TestCase):
    def test_rollback_restores_previous_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            target_path = output_dir / "sample.txt"
            target_path.write_text("before")
            tool_executor = ToolExecutor(output_dir=output_dir)

            execution_result = tool_executor.execute(
                "write_file",
                {"path": str(target_path), "content": "after"},
            )

            self.assertTrue(execution_result.success)
            self.assertEqual(target_path.read_text(), "after")

            rollback_message = tool_executor.rollback_mutation(target_path)

            self.assertIn("Rolled back", rollback_message)
            self.assertEqual(target_path.read_text(), "before")


if __name__ == "__main__":
    unittest.main()
