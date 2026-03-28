import tempfile
import unittest
from pathlib import Path

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

    def write_status(self, status, generating, perf):
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
            self.assertGreater(perf["total_tokens"], 0)
            self.assertEqual(len(perf["step_times"]), 1)
            self.assertTrue(status_writer.status_calls)
            self.assertTrue(status_writer.generation_payloads)
            self.assertTrue(logger.generation_calls)
            self.assertTrue((run_dir / "stream.txt").exists())


if __name__ == "__main__":
    unittest.main()
