"""Memory Compressor - compress old session iterations to save context window space."""

import json
import hashlib
from typing import List, Dict, Any, Optional
from collections import Counter


class MemoryCompressor:
    """Compresses session iteration history by keeping recent entries intact
    and summarizing older ones into compact representations."""

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

    def compress_session(self, iterations: List[Any], keep_recent: int = 10) -> List[Any]:
        """Compress iterations: keep recent N intact, summarize the rest.

        Returns: [summary_of_old, ...recent_iterations]
        """
        if not iterations or not isinstance(iterations, list):
            return []
        if keep_recent < 0:
            keep_recent = 0
        if len(iterations) <= keep_recent:
            return list(iterations)

        old = iterations[:-keep_recent] if keep_recent > 0 else iterations
        recent = iterations[-keep_recent:] if keep_recent > 0 else []
        summary = self.summarize_old(old)
        return [summary] + list(recent)

    def summarize_old(self, iterations: List[Any]) -> Dict[str, Any]:
        """Extract key facts from old iterations into a compact summary."""
        if not iterations:
            return {"type": "summary", "count": 0, "tools": {}, "outcomes": {}}

        tools_used = Counter()
        successes = 0
        failures = 0
        discoveries = []
        errors = []

        for it in iterations:
            if isinstance(it, dict):
                tool = it.get("tool", it.get("name", "unknown"))
                tools_used[tool] += 1
                if it.get("success", False):
                    successes += 1
                else:
                    failures += 1
                    err = it.get("error", it.get("result", ""))
                    if err and isinstance(err, str) and len(err) > 5:
                        errors.append(err[:100])
                disc = it.get("discovery", it.get("learning", ""))
                if disc and isinstance(disc, str) and len(disc) > 5:
                    discoveries.append(disc[:100])
            else:
                tools_used["unknown"] += 1

        return {
            "type": "summary",
            "count": len(iterations),
            "tools": dict(tools_used.most_common(5)),
            "outcomes": {"success": successes, "fail": failures},
            "key_discoveries": discoveries[:5],
            "common_errors": list(set(errors))[:3],
        }

    def merge_similar(self, iterations: List[Any], threshold: float = None) -> List[Any]:
        """Deduplicate near-identical iterations using content hashing."""
        if not iterations or not isinstance(iterations, list):
            return []

        seen: Dict[str, Dict] = {}
        result = []

        for it in iterations:
            if isinstance(it, dict):
                fp = hashlib.md5(
                    f"{it.get('tool','')}{str(it.get('result',''))[:200]}".encode()
                ).hexdigest()[:12]
            else:
                fp = hashlib.md5(str(it)[:200].encode()).hexdigest()[:12]

            if fp in seen:
                seen[fp]["merged_count"] = seen[fp].get("merged_count", 1) + 1
            else:
                entry = dict(it) if isinstance(it, dict) else {"value": it}
                entry["merged_count"] = 1
                seen[fp] = entry
                result.append(entry)

        return result

    def estimate_tokens(self, iterations: List[Any]) -> int:
        """Estimate token count (~4 chars per token)."""
        total_chars = sum(len(json.dumps(it, default=str)) for it in iterations)
        return total_chars // 4


if __name__ == "__main__":
    mc = MemoryCompressor()

    # Test 1: compress keeps recent and summarizes old
    data = [{"tool": "search", "result": f"r{i}"} for i in range(50)]
    compressed = mc.compress_session(data, keep_recent=10)
    assert len(compressed) == 11, f"Expected 11, got {len(compressed)}"
    assert compressed[0]["type"] == "summary"
    assert compressed[0]["count"] == 40

    # Test 2: fewer than keep_recent returns all
    assert len(mc.compress_session([{"tool": "t"}] * 3, keep_recent=10)) == 3

    # Test 3: empty/None input
    assert mc.compress_session([]) == []
    assert mc.compress_session(None) == []

    # Test 4: summarize_old extracts stats
    old = [
        {"tool": "search", "success": True, "discovery": "found API"},
        {"tool": "search", "success": True, "discovery": "found docs"},
        {"tool": "write", "success": False, "error": "syntax error"},
        {"tool": "run", "success": True},
    ]
    s = mc.summarize_old(old)
    assert s["count"] == 4
    assert s["tools"]["search"] == 2
    assert len(s["key_discoveries"]) == 2

    # Test 5: merge_similar deduplicates
    dupes = [
        {"tool": "s", "result": "hello"},
        {"tool": "s", "result": "hello"},
        {"tool": "s", "result": "hello"},
        {"tool": "w", "result": "done"},
    ]
    merged = mc.merge_similar(dupes)
    assert len(merged) == 2
    assert merged[0]["merged_count"] == 3

    # Test 6: merge empty
    assert mc.merge_similar([]) == []

    # Test 7: estimate_tokens > 0
    assert mc.estimate_tokens(data[:5]) > 0

    # Test 8: compress with keep_recent=0
    c0 = mc.compress_session(data, keep_recent=0)
    assert len(c0) == 1
    assert c0[0]["type"] == "summary"

    print("ALL TESTS PASSED")
