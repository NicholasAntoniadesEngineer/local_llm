#!/usr/bin/env python3
"""turboquant-mlx uses setuptools.backends._legacy, removed in current setuptools (e.g. Python 3.14).

Patch checkout to use setuptools.build_meta before `pip install -e`.
Usage: python tools/patch_turboquant_pyproject.py /path/to/turboquant-mlx
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: patch_turboquant_pyproject.py <turboquant-mlx-root>", file=sys.stderr)
        sys.exit(2)
    root = Path(sys.argv[1]).resolve()
    path = root / "pyproject.toml"
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        sys.exit(1)
    text = path.read_text(encoding="utf-8")
    legacy = 'build-backend = "setuptools.backends._legacy:_Backend"'
    modern = 'build-backend = "setuptools.build_meta"'
    if legacy not in text:
        if modern in text:
            print("turboquant-mlx pyproject already uses setuptools.build_meta")
            return
        print("turboquant-mlx pyproject.toml has no legacy backend line; not patched", file=sys.stderr)
        sys.exit(1)
    path.write_text(text.replace(legacy, modern), encoding="utf-8")
    print("Patched pyproject.toml: setuptools.build_meta (PEP 517)")


if __name__ == "__main__":
    main()
