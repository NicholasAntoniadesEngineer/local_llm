#!/usr/bin/env python3
"""Benchmark configured local model profiles on representative agent tasks."""

from __future__ import annotations

import argparse
import time

from src.agent import MLXAgent
from src.config import CONFIG
from src.runtime.benchmark_suite import FIXED_BENCHMARK_SLICE_NAME, get_fixed_benchmark_cases


DEFAULT_PROFILES = [
    "benchmark_coder",
    "primary",
    "balanced",
    "coder",
]


def benchmark_profile(profile_name: str, goal: str, benchmark_name: str) -> dict[str, float]:
    if profile_name not in CONFIG.models:
        raise KeyError(f"Unknown profile: {profile_name}")

    start_time = time.perf_counter()
    agent = MLXAgent(config_model_name=profile_name, goal=goal)
    agent.run_loop(goal)
    elapsed_s = time.perf_counter() - start_time
    total_tokens = float(agent._perf.get("total_tokens", 0))
    prompt_tokens = float(agent._perf.get("prompt_tokens", 0))
    tool_calls = float(agent._perf["tool_success"]["total"])
    tool_success = float(agent._perf["tool_success"]["success"])
    metrics = {
        "elapsed_s": round(elapsed_s, 2),
        "generated_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "avg_tok_s": round(total_tokens / max(0.01, agent._perf.get("total_gen_time", 0.01)), 2),
        "tool_success_rate": round(tool_success / max(1.0, tool_calls) * 100.0, 2),
    }
    agent.state_store.record_benchmark(
        benchmark_name=benchmark_name,
        profile_name=profile_name,
        model_name=CONFIG.models[profile_name].name,
        metrics=metrics,
    )
    return metrics


def run_fixed_benchmark_slice(profile_name: str) -> list[tuple[str, dict[str, float]]]:
    """Run the canonical v2 benchmark slice for one model profile."""
    slice_results: list[tuple[str, dict[str, float]]] = []
    for case_record in get_fixed_benchmark_cases():
        benchmark_name = f"{FIXED_BENCHMARK_SLICE_NAME}:{case_record['id']}"
        metrics = benchmark_profile(profile_name, case_record["goal"], benchmark_name)
        slice_results.append((benchmark_name, metrics))
    return slice_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local MLX agent model profiles.")
    parser.add_argument("goal", nargs="?", help="Representative agent goal to execute for the benchmark")
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=DEFAULT_PROFILES,
        help="Profile names from src.config.CONFIG.models",
    )
    parser.add_argument(
        "--slice",
        choices=["fixed", "ad_hoc"],
        default="fixed",
        help="Run the canonical fixed slice or a single ad hoc goal.",
    )
    args = parser.parse_args()

    for profile_name in args.profiles:
        print(f"\n=== Benchmarking {profile_name} ===")
        try:
            if args.slice == "fixed":
                for benchmark_name, metrics in run_fixed_benchmark_slice(profile_name):
                    print(f"{benchmark_name}: {metrics}")
                continue
            if not args.goal:
                raise ValueError("goal is required when --slice=ad_hoc")
            metrics = benchmark_profile(profile_name, args.goal, benchmark_name=args.goal[:80])
            print(metrics)
        except Exception as error_value:
            print(f"FAILED: {error_value}")


if __name__ == "__main__":
    main()
