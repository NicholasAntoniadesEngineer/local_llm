"""Tests for read_file ranges and replace_lines (self-improvement editing)."""

import tempfile
import unittest
from pathlib import Path

from src.runtime.tools import ToolExecutor


class ReadFileRangeTests(unittest.TestCase):
    def test_read_file_line_range_is_numbered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_file = Path(temp_dir) / "sample.txt"
            lines = [f"line {index}" for index in range(1, 21)]
            root_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            executor = ToolExecutor(Path(temp_dir))
            result = executor.execute(
                "read_file",
                {"path": str(root_file), "start_line": 5, "end_line": 7},
            )
            self.assertIsNone(result.written_path)
            self.assertTrue(result.output.startswith("(lines 5-7 of 20)"), msg=result.output)
            self.assertIn("    5| line 5", result.output)
            self.assertIn("    7| line 7", result.output)

    def test_replace_lines_replaces_inclusive_range(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # .txt avoids AtomicWriter's 200B minimum for .py skill files.
            skill_file = Path(temp_dir) / "mod.txt"
            skill_file.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
            executor = ToolExecutor(Path(temp_dir))
            result = executor.execute(
                "replace_lines",
                {"path": str(skill_file), "start_line": 2, "end_line": 4, "content": "X\nY"},
            )
            self.assertTrue(result.success, msg=result.output)
            self.assertEqual(skill_file.read_text(), "a\nX\nY\ne\n")

    def test_large_file_without_range_returns_truncation_notice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            big = Path(temp_dir) / "big.py"
            big.write_text("\n".join(f"x = {index}" for index in range(950)) + "\n", encoding="utf-8")
            executor = ToolExecutor(Path(temp_dir))
            result = executor.execute("read_file", {"path": str(big)})
            self.assertIsNone(result.written_path)
            self.assertIn("[Truncated:", result.output)
            self.assertIn("950 lines", result.output)


if __name__ == "__main__":
    unittest.main()
