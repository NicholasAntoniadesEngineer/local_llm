"""Strategy Learner - tracks what approaches work for what tasks and recommends best strategies."""

import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class StrategyLearner:
    """Learn from outcomes to recommend effective strategies and avoid bad ones."""

    def __init__(self):
        self.history: List[Dict] = []
        self.avoid_list: Dict[str, set] = defaultdict(set)

    def record_outcome(self, strategy: str, task: str, success: bool,
                       metrics: Optional[Dict[str, float]] = None) -> None:
        """Record the outcome of applying a strategy to a task."""
        if not strategy or not task:
            raise ValueError("strategy and task must be non-empty strings")
        if not isinstance(success, bool):
            raise TypeError("success must be a boolean")

        entry = {
            "strategy": strategy,
            "task": task,
            "success": success,
            "metrics": metrics or {},
        }
        self.history.append(entry)

    def best_strategy(self, task: str, top_n: int = 1) -> List[Tuple[str, float]]:
        """Return top_n strategies for a task ranked by success rate and avg metric score."""
        if not task:
            return []

        # Group outcomes by strategy for this task
        strat_outcomes: Dict[str, List[Dict]] = defaultdict(list)
        for entry in self.history:
            if entry["task"] == task:
                strat_outcomes[entry["strategy"]].append(entry)

        if not strat_outcomes:
            return []

        scores = {}
        for strat, outcomes in strat_outcomes.items():
            if strat in self.avoid_list.get(task, set()):
                continue
            n = len(outcomes)
            success_rate = sum(1 for o in outcomes if o["success"]) / n
            avg_metric = 0.0
            metric_vals = [v for o in outcomes for v in o["metrics"].values()]
            if metric_vals:
                avg_metric = sum(metric_vals) / len(metric_vals)
            # Weighted score: 70% success rate + 30% avg metrics
            scores[strat] = round(0.7 * success_rate + 0.3 * avg_metric, 4)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def avoid_strategy(self, task: str, strategy: str) -> None:
        """Mark a strategy to never recommend for a task."""
        if not task or not strategy:
            raise ValueError("task and strategy must be non-empty strings")
        self.avoid_list[task].add(strategy)

    def recommend(self, task: str) -> Optional[str]:
        """Recommend the best strategy for a task, or None if no data."""
        results = self.best_strategy(task, top_n=1)
        if results:
            return results[0][0]
        return None

    def win_rate(self, strategy: str) -> float:
        """Overall win rate for a strategy across all tasks."""
        relevant = [e for e in self.history if e["strategy"] == strategy]
        if not relevant:
            return 0.0
        return sum(1 for e in relevant if e["success"]) / len(relevant)

    def summary(self) -> Dict:
        """Summary stats across all recorded outcomes."""
        total = len(self.history)
        wins = sum(1 for e in self.history if e["success"])
        strategies = set(e["strategy"] for e in self.history)
        tasks = set(e["task"] for e in self.history)
        return {
            "total_outcomes": total,
            "win_rate": round(wins / total, 4) if total else 0.0,
            "unique_strategies": len(strategies),
            "unique_tasks": len(tasks),
            "avoided": {t: list(s) for t, s in self.avoid_list.items()},
        }


if __name__ == "__main__":
    sl = StrategyLearner()

    # Record 20 outcomes across strategies and tasks
    for i in range(10):
        sl.record_outcome("aggressive", "code_gen", i % 3 != 0, {"quality": 0.7 + i * 0.02})
        sl.record_outcome("conservative", "code_gen", i % 2 == 0, {"quality": 0.5 + i * 0.01})

    # Test 1: best_strategy returns aggressive (higher win rate)
    best = sl.best_strategy("code_gen", top_n=2)
    assert len(best) == 2, f"Expected 2, got {len(best)}"
    assert best[0][0] == "aggressive", f"Expected aggressive first, got {best[0][0]}"
    assert best[0][1] > best[1][1], "Best should score higher than second"

    # Test 2: avoid_strategy removes it from recommendations
    sl.avoid_strategy("code_gen", "aggressive")
    best_after = sl.best_strategy("code_gen", top_n=1)
    assert best_after[0][0] == "conservative", "After avoiding aggressive, conservative should be best"

    # Test 3: recommend returns top strategy
    sl2 = StrategyLearner()
    sl2.record_outcome("fast", "search", True, {"relevance": 0.9})
    sl2.record_outcome("slow", "search", False, {"relevance": 0.3})
    assert sl2.recommend("search") == "fast"

    # Test 4: recommend returns None for unknown task
    assert sl2.recommend("unknown_task") is None

    # Test 5: win_rate calculation
    assert sl2.win_rate("fast") == 1.0
    assert sl2.win_rate("slow") == 0.0
    assert sl2.win_rate("nonexistent") == 0.0

    # Test 6: summary stats
    s = sl2.summary()
    assert s["total_outcomes"] == 2
    assert s["unique_strategies"] == 2
    assert s["unique_tasks"] == 1

    # Test 7: empty inputs raise errors
    try:
        sl2.record_outcome("", "task", True)
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    # Test 8: edge case - best_strategy on empty task
    assert sl2.best_strategy("") == []

    print("ALL TESTS PASSED")
