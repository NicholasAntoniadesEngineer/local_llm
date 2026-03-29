"""Tests for self-improve loop runtime defaults."""

import os
import unittest
from unittest.mock import patch

from src.runtime.self_improve_runtime import apply_self_improve_runtime_environment


class SelfImproveRuntimeTests(unittest.TestCase):
    def test_setdefault_kmp_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            apply_self_improve_runtime_environment()
            self.assertEqual(os.environ.get("KMP_DUPLICATE_LIB_OK"), "TRUE")

    def test_respects_preset_turbo_kv_off(self) -> None:
        with patch.dict(os.environ, {"MLX_USE_TURBO_KV": "0"}, clear=False):
            apply_self_improve_runtime_environment()
            self.assertEqual(os.environ.get("MLX_USE_TURBO_KV"), "0")

    def test_sets_turbo_bits_from_config_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.runtime.self_improve_runtime.CONFIG") as mock_config:
                mock_config.self_improve_setdefault_turbo_kv = True
                mock_config.self_improve_turbo_bits = 3
                mock_config.self_improve_turbo_fp16_edge_layers = 4
                apply_self_improve_runtime_environment()
            self.assertEqual(os.environ.get("MLX_USE_TURBO_KV"), "1")
            self.assertEqual(os.environ.get("MLX_TURBO_BITS"), "3")
            self.assertEqual(os.environ.get("MLX_TURBO_FP16_LAYERS"), "4")

    def test_defaults_turbo_kv_off_and_agent_model_fast(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            apply_self_improve_runtime_environment()
            self.assertEqual(os.environ.get("MLX_USE_TURBO_KV"), "0")
            self.assertEqual(os.environ.get("AGENT_MODEL"), "fast")

    def test_respects_preset_agent_model(self) -> None:
        with patch.dict(os.environ, {"AGENT_MODEL": "tool_calling"}, clear=True):
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            apply_self_improve_runtime_environment()
            self.assertEqual(os.environ.get("AGENT_MODEL"), "tool_calling")


if __name__ == "__main__":
    unittest.main()
