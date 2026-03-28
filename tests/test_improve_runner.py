import unittest
from pathlib import Path

from src.runtime.improve_runner import select_improvement_scenario


class FakeSkillTree:
    def __init__(self, new_skill=None, weak_skill=None):
        self._new_skill = new_skill
        self._weak_skill = weak_skill
        self.evolve_calls = 0

    def evolve_tree(self):
        self.evolve_calls += 1

    def get_next_skill(self):
        return self._new_skill

    def get_weakest_skill(self):
        return self._weak_skill

    def build_goal_for_skill(self, skill):
        return f"build {skill['name']}"

    def build_upgrade_goal(self, skill):
        return f"upgrade {skill['name']}"


class ImproveRunnerScenarioTests(unittest.TestCase):
    def test_select_improvement_prefers_new_skill(self):
        fake_tree = FakeSkillTree(
            new_skill={
                "id": "skill-1",
                "name": "New Skill",
                "file": "new_skill.py",
            }
        )

        scenario = select_improvement_scenario(4, fake_tree)

        self.assertIsNotNone(scenario)
        self.assertEqual(fake_tree.evolve_calls, 1)
        self.assertEqual(scenario.skill_id, "skill-1")
        self.assertEqual(scenario.action, "BUILDING")
        self.assertEqual(scenario.goal_text, "build New Skill")
        self.assertEqual(scenario.target_path, Path("skills") / "new_skill.py")

    def test_select_improvement_falls_back_to_upgrade(self):
        fake_tree = FakeSkillTree(
            weak_skill={
                "id": "skill-2",
                "name": "Weak Skill",
                "file": "weak_skill.py",
                "quality_score": 150,
            }
        )

        scenario = select_improvement_scenario(7, fake_tree)

        self.assertIsNotNone(scenario)
        self.assertEqual(scenario.skill_id, "skill-2")
        self.assertEqual(scenario.action, "UPGRADING")
        self.assertEqual(scenario.goal_text, "upgrade Weak Skill")

    def test_select_improvement_returns_none_when_no_work_remains(self):
        fake_tree = FakeSkillTree(
            weak_skill={
                "id": "skill-3",
                "name": "Healthy Skill",
                "file": "healthy_skill.py",
                "quality_score": 300,
            }
        )

        scenario = select_improvement_scenario(9, fake_tree)

        self.assertIsNone(scenario)


if __name__ == "__main__":
    unittest.main()
