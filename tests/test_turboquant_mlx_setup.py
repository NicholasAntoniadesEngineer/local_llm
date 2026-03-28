"""Tests for optional TurboQuant-MLX cache wiring (no heavy model load)."""

import os
import unittest
from unittest.mock import MagicMock, patch

from src.runtime.turboquant_mlx_setup import try_make_turboquant_cache_factory


class TurboquantSetupTests(unittest.TestCase):
    def test_disabled_by_env_returns_none(self) -> None:
        mock_model = MagicMock()
        mock_model.layers = [object()] * 3
        with patch.dict(os.environ, {"MLX_USE_TURBO_KV": "0"}, clear=False):
            self.assertIsNone(try_make_turboquant_cache_factory(mock_model))


if __name__ == "__main__":
    unittest.main()
