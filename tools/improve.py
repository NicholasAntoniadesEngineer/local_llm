#!/usr/bin/env python3
"""Shared-runtime improvement scenario runner."""

from __future__ import annotations

import argparse
import os
import sys
import time

from src.agent import MLXAgent
from src.runtime.improve_runner import run_improvement_cycle


def run_cycle(cycle_num: int, agent: MLXAgent | None = None) -> bool:
    """Run one improvement cycle through the shared runtime."""
    model_name = os.environ.get("AGENT_MODEL", "tool_calling")
    cycle_result = run_improvement_cycle(cycle_num=cycle_num, model_name=model_name, agent=agent)
    if cycle_result.scenario is None:
        print(f"\n🎉 {cycle_result.summary}")
        return True

    scenario = cycle_result.scenario
    print(f"\n{'=' * 70}")
    print(f"CYCLE #{cycle_num}")
    print(f"{scenario.action}: {scenario.skill_name}")
    print(f"TARGET: {scenario.target_path}")
    print(f"{'=' * 70}")
    if cycle_result.accepted:
        print(f"✅ PASSED: {cycle_result.summary}")
    else:
        print(f"❌ FAILED: {cycle_result.summary}")
    return cycle_result.accepted


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shared-runtime improvement cycles.")
    parser.add_argument("cycle_num", type=int, help="Starting cycle number")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    args = parser.parse_args()

    model_name = os.environ.get("AGENT_MODEL", "tool_calling")
    persistent_agent = MLXAgent(config_model_name=model_name, goal="init") if args.loop else None
    passed_cycles = 0
    failed_cycles = 0

    if args.loop:
        current_cycle = args.cycle_num
        while True:
            try:
                accepted = run_cycle(current_cycle, agent=persistent_agent)
                if accepted:
                    passed_cycles += 1
                else:
                    failed_cycles += 1
                total_cycles = passed_cycles + failed_cycles
                success_rate = passed_cycles / max(1, total_cycles)
                print(
                    f"\n📊 Session: {passed_cycles} passed, {failed_cycles} failed "
                    f"({success_rate:.0%} success rate)"
                )
                current_cycle += 1
                time.sleep(3)
            except KeyboardInterrupt:
                print(f"\n\nFinal session: {passed_cycles} passed, {failed_cycles} failed")
                break
        return

    accepted = run_cycle(args.cycle_num, agent=None)
    sys.exit(0 if accepted else 1)


if __name__ == "__main__":
    main()
