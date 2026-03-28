import unittest

from src.paths import ROOT, SKILLS_DIR
from src.runtime.repo_bootstrap import build_frozen_static_prompt_block


class RepoBootstrapTests(unittest.TestCase):
    def test_build_frozen_block_includes_rules_and_skills(self) -> None:
        block = build_frozen_static_prompt_block(
            ROOT, SKILLS_DIR, skill_tree_text="TIER 1\n- smoke skill row"
        )
        self.assertIn("Frozen static context", block)
        self.assertIn("AGENT_RULES", block)
        self.assertIn("Verifier numeric gates", block)
        self.assertIn("TIER 1", block)
        self.assertIn("skills/", block)


if __name__ == "__main__":
    unittest.main()
