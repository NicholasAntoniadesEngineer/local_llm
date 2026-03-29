import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.runtime.mlx_adapter import MLXGenerationAdapter


class FakeChunk:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeTokenizer:
    def encode(self, text: str):
        return text.split()


class FakeKVCacheManager:
    def __init__(self) -> None:
        self.prompt_cache = None
        self.prefixes = []

    def ensure_prefix(self, prefix_text: str) -> bool:
        self.prefixes.append(prefix_text)
        return True


class FakeStatusWriter:
    def __init__(self) -> None:
        self.status_calls = []
        self.generation_payloads = []

    def write_status(self, status, generating, perf, **kwargs):
        self.status_calls.append((status, generating, perf.get("total_tokens", 0)))

    def write_generation_stats(self, payload):
        self.generation_payloads.append(payload)


class FakeLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.generation_calls = []

    def generation(self, **kwargs):
        self.generation_calls.append(kwargs)


class MLXAdapterTests(unittest.TestCase):
    def test_streaming_generation_updates_perf_and_strips_thinking(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            perf = {"total_tokens": 0, "total_gen_time": 0.0, "step_times": [], "tool_success": {"total": 0, "success": 0}}
            status_writer = FakeStatusWriter()
            logger = FakeLogger(run_dir)
            kv_cache_manager = FakeKVCacheManager()

            def fake_stream_generate(model, tokenizer, **kwargs):
                yield FakeChunk("<think>hidden</think>Hello ")
                yield FakeChunk("world")

            adapter = MLXGenerationAdapter(
                model=object(),
                tokenizer=FakeTokenizer(),
                stream_generate=fake_stream_generate,
                generate_fn=lambda model, tokenizer, **kwargs: "fallback response",
                sampler=object(),
                kv_cache_manager=kv_cache_manager,
                status_writer=status_writer,
                logger=logger,
                perf=perf,
                config_model=type("ConfigModel", (), {"max_tokens": 64, "context_window": 1024, "name": "mlx-community/test-model"})(),
                model_size_gb=8.0,
                clear_cache_fn=lambda: None,
                collect_garbage_fn=lambda: None,
            )

            response = adapter.generate_response(
                [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
                lambda messages: "formatted prompt",
            )

            self.assertEqual(response, "Hello world")
            self.assertEqual(kv_cache_manager.prefixes, ["sys"])
            self.assertGreater(perf["total_tokens"], 0)
            self.assertEqual(len(perf["step_times"]), 1)
            self.assertTrue(status_writer.status_calls)
            self.assertTrue(status_writer.generation_payloads)
            self.assertTrue(logger.generation_calls)
            self.assertTrue((run_dir / "stream.txt").exists())

    def test_oversize_prompt_skips_stream_and_returns_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            perf = {"total_tokens": 0, "total_gen_time": 0.0, "step_times": [], "tool_success": {"total": 0, "success": 0}}
            status_writer = FakeStatusWriter()
            logger = FakeLogger(run_dir)
            kv_cache_manager = FakeKVCacheManager()

            def should_not_run(model, tokenizer, **kwargs):
                raise AssertionError("stream_generate must not run for oversize prompt")

            huge = "word " * 400

            adapter = MLXGenerationAdapter(
                model=object(),
                tokenizer=FakeTokenizer(),
                stream_generate=should_not_run,
                generate_fn=lambda model, tokenizer, **kwargs: "fallback response",
                sampler=object(),
                kv_cache_manager=kv_cache_manager,
                status_writer=status_writer,
                logger=logger,
                perf=perf,
                config_model=type("ConfigModel", (), {"max_tokens": 64, "context_window": 1024, "name": "mlx-community/test-model"})(),
                model_size_gb=8.0,
                clear_cache_fn=lambda: None,
                collect_garbage_fn=lambda: None,
            )

            response = adapter.generate_response(
                [{"role": "user", "content": "x"}],
                lambda messages: huge,
            )

            self.assertIn("ERROR: Prompt too large", response)
            self.assertEqual(perf["step_times"], [])
            self.assertTrue(any("TOO LARGE" in str(call[0]) for call in status_writer.status_calls))

    def test_stream_generate_receives_prefill_and_max_kv_from_env(self):
        captured: dict = {}

        def fake_stream_generate(model, tokenizer, **kwargs):
            captured.clear()
            captured.update(kwargs)
            yield FakeChunk("ok")

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            perf = {"total_tokens": 0, "total_gen_time": 0.0, "step_times": [], "tool_success": {"total": 0, "success": 0}}
            adapter = MLXGenerationAdapter(
                model=object(),
                tokenizer=FakeTokenizer(),
                stream_generate=fake_stream_generate,
                generate_fn=lambda model, tokenizer, **kwargs: "fallback",
                sampler=object(),
                kv_cache_manager=FakeKVCacheManager(),
                status_writer=FakeStatusWriter(),
                logger=FakeLogger(run_dir),
                perf=perf,
                config_model=type(
                    "ConfigModel", (), {"max_tokens": 64, "context_window": 8192, "name": "mlx-community/test-model"}
                )(),
                model_size_gb=8.0,
                clear_cache_fn=lambda: None,
                collect_garbage_fn=lambda: None,
            )
            with patch.dict(
                os.environ,
                {"MLX_PREFILL_STEP_SIZE": "512", "MLX_MAX_KV_SIZE": "4096"},
                clear=False,
            ):
                response = adapter.generate_response(
                    [{"role": "user", "content": "hello"}],
                    lambda messages: "short prompt",
                )
            self.assertEqual(response, "ok")
            self.assertEqual(captured.get("prefill_step_size"), 512)
            self.assertEqual(captured.get("max_kv_size"), 4096)

    def test_mlx_prefill_and_kv_helpers(self):
        from src.runtime.mlx_adapter import _mlx_max_kv_size_from_env, _mlx_prefill_step_size_from_env

        with patch.dict(os.environ, {"MLX_PREFILL_STEP_SIZE": "2048"}, clear=False):
            self.assertEqual(_mlx_prefill_step_size_from_env(), 2048)
        with patch.dict(os.environ, {"MLX_PREFILL_STEP_SIZE": "100"}, clear=False):
            self.assertEqual(_mlx_prefill_step_size_from_env(), 256)
        with patch.dict(os.environ, {"MLX_PREFILL_STEP_SIZE": "not-a-number"}, clear=False):
            self.assertEqual(_mlx_prefill_step_size_from_env(), 1024)
        with patch.dict(os.environ, {"MLX_MAX_KV_SIZE": "1024"}, clear=False):
            self.assertEqual(_mlx_max_kv_size_from_env(), 1024)
        with patch.dict(os.environ, {"MLX_MAX_KV_SIZE": ""}, clear=False):
            self.assertIsNone(_mlx_max_kv_size_from_env())

    def test_metal_safe_prompt_token_cap_env(self):
        from src.runtime.mlx_adapter import _metal_safe_prompt_token_cap

        with patch.dict(os.environ, {"MLX_METAL_SAFE_PROMPT_TOKENS": ""}, clear=False):
            self.assertIsNone(_metal_safe_prompt_token_cap())
        with patch.dict(os.environ, {"MLX_METAL_SAFE_PROMPT_TOKENS": "5000"}, clear=False):
            self.assertEqual(_metal_safe_prompt_token_cap(), 5000)
        with patch.dict(os.environ, {"MLX_METAL_SAFE_PROMPT_TOKENS": "18432"}, clear=False):
            self.assertEqual(_metal_safe_prompt_token_cap(), 18432)

    def test_metal_safe_max_new_tokens_tiers_and_env(self):
        from src.runtime.mlx_adapter import _metal_safe_max_new_tokens

        self.assertEqual(_metal_safe_max_new_tokens(15_019, 8192, 40_960), 2048)
        self.assertEqual(_metal_safe_max_new_tokens(11_000, 8192, 40_960), 4096)
        self.assertEqual(_metal_safe_max_new_tokens(9000, 8192, 40_960), 6144)
        self.assertEqual(_metal_safe_max_new_tokens(4000, 8192, 40_960), 8192)
        with patch.dict(os.environ, {"MLX_METAL_DECODE_CAP": "1024"}, clear=False):
            self.assertEqual(_metal_safe_max_new_tokens(15_019, 8192, 40_960), 1024)
        # Context headroom clamps when prompt nears context_window
        self.assertEqual(_metal_safe_max_new_tokens(40_500, 8192, 40_960), 204)

    def test_metal_safe_cap_rejects_before_stream(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            perf = {"total_tokens": 0, "total_gen_time": 0.0, "step_times": [], "tool_success": {"total": 0, "success": 0}}

            def should_not_run(model, tokenizer, **kwargs):
                raise AssertionError("stream_generate must not run when metal cap exceeded")

            long_prompt = " ".join([f"t{i}" for i in range(400)])
            adapter = MLXGenerationAdapter(
                model=object(),
                tokenizer=FakeTokenizer(),
                stream_generate=should_not_run,
                generate_fn=lambda model, tokenizer, **kwargs: "fallback",
                sampler=object(),
                kv_cache_manager=FakeKVCacheManager(),
                status_writer=FakeStatusWriter(),
                logger=FakeLogger(run_dir),
                perf=perf,
                config_model=type(
                    "ConfigModel", (), {"max_tokens": 64, "context_window": 50_000, "name": "mlx-community/test-model"}
                )(),
                model_size_gb=8.0,
                clear_cache_fn=lambda: None,
                collect_garbage_fn=lambda: None,
            )
            with patch.dict(os.environ, {"MLX_METAL_SAFE_PROMPT_TOKENS": "256"}, clear=False):
                response = adapter.generate_response(
                    [{"role": "user", "content": "x"}],
                    lambda messages: long_prompt,
                )
            self.assertIn("ERROR: Prompt too large", response)
            self.assertIn("MLX_METAL_SAFE_PROMPT_TOKENS", response)


if __name__ == "__main__":
    unittest.main()
