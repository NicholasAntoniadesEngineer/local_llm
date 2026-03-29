"""Tests for ContextBudgetGuard hard shrink (oversize system / frozen static)."""

import os
import unittest
from unittest.mock import patch

from src.context_manager import ContextBudgetGuard, EpisodicBuffer


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(max(1, len(text) // 3)))


class ContextBudgetTests(unittest.TestCase):
    def test_hard_shrink_truncates_system_to_fit(self):
        tokenizer = FakeTokenizer()

        def formatter(messages: list[dict]) -> str:
            return "\n\n".join(str(message.get("content", "")) for message in messages)

        guard = ContextBudgetGuard(
            tokenizer=tokenizer,
            context_window=2048,
            max_tokens=256,
            episodic_buffer=EpisodicBuffer(),
        )
        huge_system = "line\n" * 8000
        messages = [
            {"role": "system", "content": huge_system, "protected": True},
            {"role": "user", "content": "short user tail", "protected": True},
        ]
        result = guard.enforce_budget(messages, formatter)
        self.assertLessEqual(result.prompt_tokens, guard.context_window - guard.max_tokens - 512)
        self.assertIn("truncated for token budget", result.messages[0]["content"])

    def test_metal_safe_env_tightens_hard_limit(self):
        tokenizer = FakeTokenizer()

        def formatter(messages: list[dict]) -> str:
            return "\n\n".join(str(message.get("content", "")) for message in messages)

        guard = ContextBudgetGuard(
            tokenizer=tokenizer,
            context_window=50_000,
            max_tokens=64,
            episodic_buffer=EpisodicBuffer(),
        )
        huge_system = "line\n" * 12_000
        messages = [{"role": "system", "content": huge_system}]
        with patch.dict(os.environ, {"MLX_METAL_SAFE_PROMPT_TOKENS": "500"}, clear=False):
            result = guard.enforce_budget(messages, formatter)
        self.assertLessEqual(result.prompt_tokens, 500)


if __name__ == "__main__":
    unittest.main()
