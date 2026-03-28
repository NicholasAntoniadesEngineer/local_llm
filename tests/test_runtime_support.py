import json
import tempfile
import unittest
from pathlib import Path

from src.runtime.runtime_support import IdleScheduler, PerfStatusWriter, pre_validate_candidate_file


class RuntimeSupportTests(unittest.TestCase):
    def test_idle_scheduler_runs_enqueued_work_and_returns_result(self) -> None:
        scheduler = IdleScheduler()
        scheduler.enqueue("sample", lambda: "done")

        scheduler.run_pending(max_time=0.1)

        self.assertEqual(scheduler.get_result("sample"), "done")

    def test_pre_validate_candidate_file_flags_shallow_python_module(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "sample.py"
            target_path.write_text("def f():\n    return 1\n")

            message = pre_validate_candidate_file(str(target_path), Path(temp_dir))

            self.assertTrue(message.startswith("WARN:"))
            self.assertIn("functions", message)

    def test_perf_status_writer_writes_perf_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            writer = PerfStatusWriter(
                run_dir=run_dir,
                model_name="mlx-community/Qwen3-32B-4bit",
                context_window=32768,
                max_tokens=4096,
                model_size_gb=17.5,
            )

            writer.write_status(
                status="TOOL: read_file",
                generating=False,
                perf={"step_times": [1.0], "total_tokens": 10, "prompt_tokens": 20, "peak_tok_s": 30.0},
            )

            payload = json.loads((run_dir / "perf.json").read_text())
            self.assertEqual(payload["status"], "TOOL: read_file")
            self.assertEqual(payload["model"], "Qwen3-32B-4bit")
            self.assertEqual(payload["step"], 2)


if __name__ == "__main__":
    unittest.main()
