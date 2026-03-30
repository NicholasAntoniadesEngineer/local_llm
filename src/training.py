"""LoRA fine-tuning pipeline for self-improvement.

Collects successful (prompt, code) pairs from challenge solving and
periodically fine-tunes the model using MLX's LoRA support. This is
the mechanism by which the MODEL actually improves, not just the prompts.

Usage:
    python -m src.training                    # Run fine-tuning on collected data
    python -m src.training --check            # Check if enough data exists
    python -m src.training --export-only      # Export training data without fine-tuning
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

from src.paths import RUNS_DIR


# ── Paths ────────────────────────────────────────────────────────────────

TRAINING_DATA_FILE = RUNS_DIR / "training_data.jsonl"
TRAINING_LOG_FILE = RUNS_DIR / "training_log.json"
SUCCESSFUL_GENERATIONS_FILE = RUNS_DIR / "successful_generations.jsonl"
CHALLENGE_RESULTS_FILE = RUNS_DIR / "challenge_results.jsonl"
ADAPTER_DIR = RUNS_DIR / "lora_adapters"

# Minimum examples needed before fine-tuning is worthwhile
MIN_TRAINING_EXAMPLES = 15


def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL file, skip corrupted lines."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def collect_training_data() -> list[dict[str, str]]:
    """Collect (prompt, completion) pairs from successful generations.

    Returns list of dicts with 'prompt' and 'completion' keys,
    formatted for chat fine-tuning.
    """
    examples = []

    # Source 1: Successful challenge solutions
    for record in _read_jsonl(CHALLENGE_RESULTS_FILE):
        if not record.get("solved"):
            continue
        code = record.get("code", "")
        if not code or len(code) < 20:
            continue
        challenge_id = record.get("challenge_id", "unknown")
        # Build a training prompt similar to what the model saw
        examples.append({
            "prompt": f"Solve this coding challenge. Output ONLY Python code.\n\nCHALLENGE: {challenge_id}",
            "completion": code,
            "source": "challenge",
            "quality": 1.0,
        })

    # Source 2: Successful skill generations
    for record in _read_jsonl(SUCCESSFUL_GENERATIONS_FILE):
        code = record.get("code", "")
        if not code or len(code) < 50:
            continue
        prompt_preview = record.get("prompt_preview", "")
        examples.append({
            "prompt": prompt_preview if prompt_preview else "Write a Python module.",
            "completion": code,
            "source": "skill",
            "quality": 1.0,
        })

    return examples


def export_training_data(output_path: Path | None = None) -> Path:
    """Export collected data as JSONL formatted for MLX LoRA fine-tuning.

    MLX LoRA expects JSONL with {"text": "<prompt>\\n<completion>"} format,
    or chat format with {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}.
    """
    examples = collect_training_data()
    if not examples:
        print("No training data available yet. Solve some challenges first.")
        return TRAINING_DATA_FILE

    path = output_path or TRAINING_DATA_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a Python code generator. Output ONLY valid Python code. No explanations, no markdown fences, no commentary.",
                    },
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["completion"]},
                ],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Exported {len(examples)} training examples to {path}")
    return path


def check_training_readiness() -> dict[str, Any]:
    """Check if we have enough data for fine-tuning."""
    examples = collect_training_data()
    challenge_count = sum(1 for e in examples if e["source"] == "challenge")
    skill_count = sum(1 for e in examples if e["source"] == "skill")

    ready = len(examples) >= MIN_TRAINING_EXAMPLES
    return {
        "total_examples": len(examples),
        "challenge_examples": challenge_count,
        "skill_examples": skill_count,
        "min_required": MIN_TRAINING_EXAMPLES,
        "ready": ready,
        "message": (
            f"Ready for fine-tuning ({len(examples)} examples)"
            if ready
            else f"Need {MIN_TRAINING_EXAMPLES - len(examples)} more examples "
            f"(have {len(examples)}, need {MIN_TRAINING_EXAMPLES})"
        ),
    }


def run_lora_finetuning(
    model_name: str = "mlx-community/Qwen3-14B-4bit",
    num_epochs: int = 1,
    learning_rate: float = 1e-5,
    lora_rank: int = 8,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Run LoRA fine-tuning on collected training data.

    Uses mlx_lm's built-in LoRA support. Saves adapters to ADAPTER_DIR.
    """
    readiness = check_training_readiness()
    if not readiness["ready"]:
        return {"success": False, "error": readiness["message"]}

    # Export training data
    data_path = export_training_data()

    # Check if mlx_lm.lora is available
    try:
        from mlx_lm import lora as mlx_lora
    except ImportError:
        # Fallback: try the standalone lora module
        try:
            import subprocess
            ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable, "-m", "mlx_lm.lora",
                "--model", model_name,
                "--data", str(data_path.parent),
                "--train",
                "--adapter-path", str(ADAPTER_DIR),
                "--lora-rank", str(lora_rank),
                "--num-layers", "8",
                "--iters", str(readiness["total_examples"] * num_epochs),
                "--learning-rate", str(learning_rate),
                "--batch-size", str(batch_size),
            ]

            print(f"Running LoRA fine-tuning: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max
                cwd=str(Path(__file__).parent.parent),
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Fine-tuning failed: {result.stderr[:500]}",
                }

            # Log the training run
            log = _load_training_log()
            log["runs"].append({
                "timestamp": datetime.now().isoformat(),
                "examples": readiness["total_examples"],
                "epochs": num_epochs,
                "lr": learning_rate,
                "rank": lora_rank,
                "adapter_path": str(ADAPTER_DIR),
                "stdout": result.stdout[:500],
            })
            _save_training_log(log)

            return {
                "success": True,
                "adapter_path": str(ADAPTER_DIR),
                "examples_used": readiness["total_examples"],
                "message": f"LoRA adapters saved to {ADAPTER_DIR}",
            }

        except Exception as e:
            return {"success": False, "error": f"LoRA training error: {e}"}

    return {"success": False, "error": "mlx_lm.lora module not found"}


def _load_training_log() -> dict:
    if TRAINING_LOG_FILE.exists():
        try:
            return json.loads(TRAINING_LOG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"runs": []}


def _save_training_log(log: dict) -> None:
    TRAINING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_LOG_FILE.write_text(json.dumps(log, indent=2))


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoRA fine-tuning from successful generations")
    parser.add_argument("--check", action="store_true", help="Check if enough data exists")
    parser.add_argument("--export-only", action="store_true", help="Export training data without fine-tuning")
    parser.add_argument("--model", default="mlx-community/Qwen3-14B-4bit", help="Model to fine-tune")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    args = parser.parse_args()

    if args.check:
        r = check_training_readiness()
        print(json.dumps(r, indent=2))
    elif args.export_only:
        export_training_data()
    else:
        r = run_lora_finetuning(
            model_name=args.model,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            lora_rank=args.rank,
        )
        print(json.dumps(r, indent=2))
