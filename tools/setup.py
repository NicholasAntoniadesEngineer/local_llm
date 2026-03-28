#!/usr/bin/env python3
"""
Setup script for MLX Agent on Apple Silicon.

Installs all dependencies, downloads models, validates environment.
Run once on a new machine: python setup.py
"""

import subprocess
import sys
import os
from pathlib import Path


# Models to download (ordered by priority)
MODELS = [
    (
        "mlx-community/Qwen3-14B-4bit",
        "~8GB",
        "Default improve: AGENT_MODEL=fast (TurboQuant-compatible); tool_calling_14b uses same weights",
    ),
    ("mlx-community/Qwen3-32B-4bit", "~18GB", "Stronger codegen: AGENT_MODEL=tool_calling"),
    ("mlx-community/Qwen3-30B-A3B-4bit", "16GB", "MoE coder: AGENT_MODEL=coder"),
]

# Python packages
PACKAGES = [
    "mlx-lm",
    "httpx",
    "beautifulsoup4",
    "rich",
]

# Brew packages
BREW_PACKAGES = [
    ("vladkens/tap/macmon", "GPU/memory monitor (no sudo)"),
]


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)


def check_hardware():
    """Verify Apple Silicon."""
    print("\n1. Checking hardware...")
    result = run("sysctl -n machdep.cpu.brand_string", check=False)
    chip = result.stdout.strip()
    print(f"  Chip: {chip}")

    result = run("sysctl -n hw.memsize", check=False)
    ram_gb = int(result.stdout.strip()) / 1073741824
    print(f"  RAM: {ram_gb:.0f} GB")

    if "Apple" not in chip:
        print("  ⚠️  Not Apple Silicon - MLX won't work")
        return False
    if ram_gb < 16:
        print("  ⚠️  Less than 16GB RAM - may struggle with larger models")

    print("  ✅ Hardware OK")
    return True


def setup_venv():
    """Create virtual environment."""
    print("\n2. Setting up Python environment...")
    venv_path = Path("mlx_agent_env")
    if not venv_path.exists():
        run(f"python3 -m venv {venv_path}")
        print("  ✅ Created virtual environment")
    else:
        print("  ✅ Virtual environment exists")


def install_packages():
    """Install Python packages."""
    print("\n3. Installing Python packages...")
    for pkg in PACKAGES:
        result = run(f"pip install {pkg}", check=False)
        if result.returncode == 0:
            print(f"  ✅ {pkg}")
        else:
            print(f"  ❌ {pkg}: {result.stderr[:100]}")


def install_brew_packages():
    """Install brew packages."""
    print("\n4. Installing system tools...")
    for pkg, desc in BREW_PACKAGES:
        result = run(f"brew install {pkg}", check=False)
        if result.returncode == 0:
            print(f"  ✅ {pkg} ({desc})")
        else:
            print(f"  ⚠️  {pkg}: skipped (brew not available?)")


def download_models():
    """Download MLX models."""
    print("\n5. Downloading models...")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    for model_id, size, desc in MODELS:
        print(f"\n  Downloading: {model_id} ({size})")
        print(f"  Purpose: {desc}")

        # Check if already cached
        cache_name = model_id.replace("/", "--")
        cache_path = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{cache_name}"
        if cache_path.exists():
            actual_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
            print(f"  ✅ Already cached ({actual_size / 1073741824:.1f} GB)")
            continue

        try:
            result = run(
                f'python3 -c "from mlx_lm import load; load(\'{model_id}\')"',
                check=False,
            )
            if result.returncode == 0:
                print(f"  ✅ Downloaded")
            else:
                print(f"  ❌ Failed: {result.stderr[:100]}")
        except Exception as e:
            print(f"  ❌ Error: {e}")


def create_directories():
    """Create required directories."""
    print("\n6. Creating directories...")
    dirs = [
        Path("./skills"),
        Path.home() / ".claude" / "sessions",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {d}")


def validate():
    """Run validation checks."""
    print("\n7. Validating...")

    # Check imports
    try:
        result = run('python3 -c "from mlx_lm import load; print(\'mlx-lm OK\')"', check=False)
        print(f"  ✅ mlx-lm: {result.stdout.strip()}")
    except Exception:
        print("  ❌ mlx-lm import failed")

    # Check agent compiles
    for f in ["agent.py", "config.py", "memory.py", "improve.py", "monitor.py"]:
        result = run(f"python3 -m py_compile {f}", check=False)
        if result.returncode == 0:
            print(f"  ✅ {f} compiles")
        else:
            print(f"  ❌ {f}: {result.stderr[:100]}")


def print_usage():
    """Print usage instructions."""
    print(f"""
{'='*60}
✅ SETUP COMPLETE
{'='*60}

Run the self-improving agent:
  KMP_DUPLICATE_LIB_OK=TRUE python improve.py 1 --loop

Monitor performance (separate terminal):
  python monitor.py

Monitor hardware (separate terminal):
  macmon

Available models (set AGENT_MODEL env var):
  fast         - Qwen3 14B (~8GB, default improve loop, TurboQuant + ~40k context)
  balanced     - Qwen3 30B MoE (16GB, best quality/speed)
  quality      - Qwen3.5 27B (14GB, highest quality)
  tool_calling - Qwen3 32B (~18GB, strongest codegen)
  tool_calling_14b - Qwen3 14B (8GB, lighter)

Example:
  AGENT_MODEL=tool_calling KMP_DUPLICATE_LIB_OK=TRUE python improve.py 1 --loop

Agent rules and frozen prompt (KV message 0): see AGENT_RULES.md at repo root.

Optional TurboQuant-MLX KV cache (longer context / lower memory on M-series):
  ./tools/install_turboquant_mlx.sh
  (Do not use raw pip install git+... on Python 3.14 — use this script; see requirements-turboquant.txt.)
  MLX_USE_TURBO_KV=0  # to disable and use default mlx-lm KV cache

Self-improve loop (`tools/improve.py`) applies defaults automatically (see src/config.py:
  self_improve_setdefault_turbo_kv, self_improve_turbo_bits, self_improve_turbo_fp16_edge_layers).
""")


def main():
    print("="*60)
    print("MLX Agent Setup for Apple Silicon")
    print("="*60)

    check_hardware()
    create_directories()
    install_packages()
    install_brew_packages()
    download_models()
    validate()
    print_usage()


if __name__ == "__main__":
    main()
