#!/usr/bin/env python3
"""Shared-runtime improvement scenario runner."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from src.agent import MLXAgent
from src.runtime.improve_runner import run_improvement_cycle, METRICS_FILE
from src.runtime.self_improve_runtime import (
    apply_self_improve_runtime_environment,
    print_self_improve_runtime_banner,
)


def run_cycle(cycle_num: int, agent: MLXAgent | None = None) -> str:
    """Run one improvement cycle. Returns idle | pass | fail."""
    model_name = os.environ.get("AGENT_MODEL", "fast")
    cycle_result = run_improvement_cycle(cycle_num=cycle_num, model_name=model_name, agent=agent)
    if cycle_result.outcome == "idle":
        print(f"\n⏸ Idle: {cycle_result.summary}")
        return "idle"

    scenario = cycle_result.scenario
    assert scenario is not None
    print(f"\n{'=' * 70}")
    print(f"CYCLE #{cycle_num}")
    print(f"{scenario.action}: {scenario.skill_name}")
    print(f"TARGET: {scenario.target_path}")
    print(f"{'=' * 70}")
    if cycle_result.outcome in ("accepted", "pre_validated", "challenge_solved"):
        print(f"✅ PASSED: {cycle_result.summary}")
        return "pass"
    print(f"❌ FAILED: {cycle_result.summary}")
    return "fail"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shared-runtime improvement cycles.")
    parser.add_argument("cycle_num", type=int, help="Starting cycle number")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    apply_self_improve_runtime_environment()
    print_self_improve_runtime_banner()

    model_name = os.environ.get("AGENT_MODEL", "fast")
    persistent_agent = MLXAgent(config_model_name=model_name, goal="self-improve") if args.loop else None
    passed_cycles = 0
    failed_cycles = 0

    if args.loop:
        current_cycle = args.cycle_num
        while True:
            try:
                status = run_cycle(current_cycle, agent=persistent_agent)
                if status == "pass":
                    passed_cycles += 1
                elif status == "fail":
                    failed_cycles += 1
                total_completed = passed_cycles + failed_cycles
                success_rate = passed_cycles / max(1, total_completed)

                # Show live metrics from the feedback loops
                metrics_str = ""
                try:
                    if METRICS_FILE.exists():
                        m = json.loads(METRICS_FILE.read_text())
                        ch_rate = m.get("challenge_solve_rate", 0)
                        ch_total = m.get("challenges_attempted", 0)
                        rates = m.get("solve_rates_by_difficulty", {})
                        rates_str = " ".join(f"d{k}={float(v):.0%}" for k, v in sorted(rates.items()))
                        metrics_str = f" | challenges: {ch_rate:.0%} of {ch_total}"
                        if rates_str:
                            metrics_str += f" [{rates_str}]"
                except (json.JSONDecodeError, OSError):
                    pass

                print(
                    f"\n📊 Session: {passed_cycles}/{total_completed} "
                    f"({success_rate:.0%}){metrics_str}"
                )
                current_cycle += 1
                sleep_s = float(os.environ.get("IMPROVE_LOOP_SLEEP_SEC", "3"))
                if sleep_s > 0:
                    time.sleep(sleep_s)
            except KeyboardInterrupt:
                print(f"\n\nFinal session: {passed_cycles} passed, {failed_cycles} failed")
                break
        return

    status = run_cycle(args.cycle_num, agent=None)
    sys.exit(0 if status != "fail" else 1)


if __name__ == "__main__":
    main()
