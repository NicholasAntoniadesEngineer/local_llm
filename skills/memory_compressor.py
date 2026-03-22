"""Compress old session iterations to save context window tokens."""

from dataclasses import dataclass, field
from typing import List, Dict, Any


def compress_session(iterations: List[Dict[str, Any]], keep_recent: int = 10) -> Dict[str, Any]:
    """Compress a list of iteration dicts, keeping only the most recent ones.

    Returns dict with 'recent' (full iterations) and 'summary' (compressed old ones).
    """
    if len(iterations) <= keep_recent:
        return {"recent": iterations, "summary": "", "total": len(iterations), "compressed": 0, "kept": len(iterations)}

    recent = iterations[-keep_recent:]
    older = iterations[:-keep_recent]

    # Summarize older iterations: tool name + success/fail + first 40 chars of result
    summary_lines = []
    for it in older:
        tool = it.get("tool", it.get("tool_used", "?"))
        ok = "OK" if it.get("success", False) else "FAIL"
        result = str(it.get("result", ""))[:40]
        summary_lines.append(f"{tool}:{ok} {result}")

    return {
        "recent": recent,
        "summary": "\n".join(summary_lines),
        "total": len(iterations),
        "compressed": len(older),
        "kept": len(recent),
    }


if __name__ == "__main__":
    # Test with 50 iterations
    iters = [
        {"tool": f"tool_{i % 5}", "result": f"result_{i}", "success": i % 3 != 0}
        for i in range(50)
    ]

    result = compress_session(iters, keep_recent=10)
    assert len(result["recent"]) == 10, f"Expected 10 recent, got {len(result['recent'])}"
    assert result["compressed"] == 40, f"Expected 40 compressed, got {result['compressed']}"
    assert result["total"] == 50
    assert len(result["summary"]) > 0, "Summary should not be empty"

    # Test with fewer than keep_recent
    small = compress_session(iters[:5], keep_recent=10)
    assert len(small["recent"]) == 5
    assert small["summary"] == ""
    assert small["compressed"] == 0

    # Test with exactly keep_recent
    exact = compress_session(iters[:10], keep_recent=10)
    assert len(exact["recent"]) == 10
    assert exact["summary"] == ""

    print('ALL TESTS PASSED')
