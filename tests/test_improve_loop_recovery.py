"""Regression tests for improve-loop recovery (peek/pull, policy, goal path)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from src.runtime.policy import PolicyEngine, skill_relative_path_from_goal
from src.skill_tree import SkillTree


class SkillTreePeekPullTests(unittest.TestCase):
    def test_peek_next_skill_does_not_increment_pull_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "skill_tree_test.db"
            tree = SkillTree(db_path=str(db_path))
            peeked = tree.peek_next_skill()
            self.assertIsNotNone(peeked)
            skill_id = peeked["id"]
            count_before = tree.graph.nodes[skill_id].get("pull_count", 0)
            tree.peek_next_skill()
            count_after_peek = tree.graph.nodes[skill_id].get("pull_count", 0)
            self.assertEqual(count_after_peek, count_before)
            tree.record_pull(skill_id)
            count_after_record = tree.graph.nodes[skill_id].get("pull_count", 0)
            self.assertEqual(count_after_record, count_before + 1)


class SkillRelativePathFromGoalTests(unittest.TestCase):
    def test_extracts_skills_relative_path(self) -> None:
        goal = "=== TASK ===\nModule: X\nFile: code_validator.py\nImpact: 9"
        self.assertEqual(skill_relative_path_from_goal(goal), "skills/code_validator.py")

    def test_returns_none_without_file_line(self) -> None:
        self.assertIsNone(skill_relative_path_from_goal("no file line here"))


class PolicyWebSearchPivotTests(unittest.TestCase):
    def test_forces_read_file_after_web_search_in_history(self) -> None:
        def load_skill_instance(skill_name: str, _class_name: str | None = None):
            return None

        engine = PolicyEngine(load_skill_instance, "=== TASK ===\nFile: error_recovery.py\n")
        policy = engine.build_step_policy(
            phase="inspect",
            step=2,
            max_iterations=20,
            perf={"tool_success": {"total": 1, "success": 1}},
            files_written=0,
            last_result="cached search",
            memory_manager=None,
            current_skill={"id": "e", "name": "Error Recovery"},
            recent_tool_names=["web_search"],
        )
        self.assertEqual(policy.suggested_tool, "read_file")


if __name__ == "__main__":
    unittest.main()
