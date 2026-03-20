"""Test fixtures and utilities for research agent tests."""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List


def get_sample_embeddings(count: int = 3, dim: int = 768) -> List[List[float]]:
    """Generate sample 768-dimensional embeddings.

    Args:
        count: Number of embeddings to generate
        dim: Embedding dimensionality (default 768)

    Returns:
        List of embeddings
    """
    embeddings = []
    for i in range(count):
        embedding = np.random.randn(dim) * 0.1 + (i * 0.01)
        embeddings.append(embedding.tolist())
    return embeddings


def load_sample_rules() -> Dict[str, Any]:
    """Load sample rules configuration.

    Returns:
        Rules dictionary with hard, soft, and meta rules
    """
    return {
        "hard_rules": [
            {
                "id": "H1",
                "rule": "Verify claims against minimum 2 independent sources",
                "enforcement": "block_if_violated",
                "rationale": "Single-source claims have high failure rate",
            },
            {
                "id": "H2",
                "rule": "Explicitly cite all sources used; include URL or publication details",
                "enforcement": "block_if_violated",
                "rationale": "Enables verification and tracks information provenance",
            },
        ],
        "soft_rules": [
            {
                "id": "S1",
                "priority": "high",
                "confidence": 0.85,
                "rule": "Prefer primary sources (papers, official reports) over secondary summaries",
                "rationale": "Reduces distortion from interpretive layers",
                "effectiveness_score": 0.82,
            },
            {
                "id": "S2",
                "priority": "high",
                "confidence": 0.78,
                "rule": "For technical topics, prefer sources published in last 18 months",
                "rationale": "Rapidly evolving fields; old papers may be superseded",
                "effectiveness_score": 0.76,
            },
        ],
        "meta_rules": [
            {
                "id": "M1",
                "priority": "critical",
                "rule": "When rules conflict, prefer accuracy over completeness",
                "rationale": "False information is worse than incomplete information",
            },
        ],
    }


def load_sample_model_config() -> Dict[str, Any]:
    """Load sample model configuration.

    Returns:
        Model configuration dictionary
    """
    return {
        "roles": {
            "orchestration": {
                "model": "qwen3:8b",
                "description": "Fast routing and planning decisions",
                "context_window": 8192,
                "max_batch_size": 4,
            },
            "reasoning": {
                "model": "qwen3:32b",
                "description": "Deep research and analysis",
                "context_window": 16384,
                "max_batch_size": 1,
            },
            "code_generation": {
                "model": "qwen2.5-coder:32b",
                "description": "Code analysis and generation",
                "context_window": 16384,
                "max_batch_size": 1,
            },
        },
        "model_defaults": {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 2000,
        },
    }


def create_test_findings_jsonl() -> str:
    """Create sample findings in JSONL format.

    Returns:
        JSONL-formatted string of findings
    """
    findings = [
        {
            "id": "finding_001",
            "session_id": "session_001",
            "title": "Quantum Error Correction Breakthrough",
            "content": "Surface codes achieve error correction",
            "confidence": 0.92,
            "sources": ["https://arxiv.org/abs/2401.xxxxx"],
        },
        {
            "id": "finding_002",
            "session_id": "session_001",
            "title": "Neural Network Scaling",
            "content": "Large models continue to improve",
            "confidence": 0.85,
            "sources": ["https://neurips.org/paper/xxxxx"],
        },
    ]

    return "\n".join(json.dumps(f) for f in findings)


__all__ = [
    "get_sample_embeddings",
    "load_sample_rules",
    "load_sample_model_config",
    "create_test_findings_jsonl",
]
