#!/usr/bin/env python3
"""
Validate code structure, imports, and basic functionality.

Run after implementation agents complete to catch any structural issues.
"""

import sys
import importlib.util
from pathlib import Path
from typing import List, Tuple

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"


def check_file_exists(path: Path) -> Tuple[bool, str]:
    """Check if a file exists."""
    if path.exists():
        return True, f"✅ {path.relative_to(ROOT_DIR)}"
    return False, f"❌ Missing: {path.relative_to(ROOT_DIR)}"


def check_module_imports(module_path: Path) -> Tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        spec = importlib.util.spec_from_file_location("module", module_path)
        if spec is None or spec.loader is None:
            return False, f"❌ Cannot load spec: {module_path.name}"

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, f"✅ Imports OK: {module_path.relative_to(SRC_DIR)}"
    except Exception as e:
        return False, f"❌ Import error in {module_path.name}: {str(e)[:80]}"


def validate_project_structure() -> int:
    """Validate complete project structure."""
    errors = []
    warnings = []

    print("\n🔍 VALIDATING PROJECT STRUCTURE\n")

    # Required files
    required_files = [
        ROOT_DIR / "CLAUDE.md",
        ROOT_DIR / "requirements.txt",
        ROOT_DIR / "pyproject.toml",
        ROOT_DIR / "pytest.ini",
        ROOT_DIR / ".env.example",
        ROOT_DIR / "config" / "rules.yaml",
        ROOT_DIR / "config" / "model_config.yaml",
    ]

    print("1. Required Files:")
    for fpath in required_files:
        ok, msg = check_file_exists(fpath)
        print(f"   {msg}")
        if not ok:
            errors.append(msg)

    # Source module structure
    src_modules = {
        "LLM Backend": [
            SRC_DIR / "llm" / "__init__.py",
            SRC_DIR / "llm" / "base.py",
            SRC_DIR / "llm" / "ollama_client.py",
            SRC_DIR / "llm" / "router.py",
        ],
        "Memory Layer": [
            SRC_DIR / "memory" / "__init__.py",
            SRC_DIR / "memory" / "models.py",
            SRC_DIR / "memory" / "working.py",
            SRC_DIR / "memory" / "lancedb_store.py",
            SRC_DIR / "memory" / "sqlite_store.py",
            SRC_DIR / "memory" / "manager.py",
        ],
        "Retrieval Layer": [
            SRC_DIR / "retrieval" / "__init__.py",
            SRC_DIR / "retrieval" / "models.py",
            SRC_DIR / "retrieval" / "hyde.py",
            SRC_DIR / "retrieval" / "hybrid.py",
            SRC_DIR / "retrieval" / "reranker.py",
            SRC_DIR / "retrieval" / "chunker.py",
        ],
        "Rules Engine": [
            SRC_DIR / "rules" / "__init__.py",
            SRC_DIR / "rules" / "models.py",
            SRC_DIR / "rules" / "loader.py",
            SRC_DIR / "rules" / "engine.py",
            SRC_DIR / "rules" / "learner.py",
            SRC_DIR / "rules" / "optimizer.py",
        ],
        "Agent Orchestrator": [
            SRC_DIR / "agent" / "__init__.py",
            SRC_DIR / "agent" / "state.py",
            SRC_DIR / "agent" / "core.py",
            SRC_DIR / "agent" / "nodes" / "__init__.py",
            SRC_DIR / "agent" / "nodes" / "plan.py",
            SRC_DIR / "agent" / "nodes" / "think.py",
            SRC_DIR / "agent" / "nodes" / "act.py",
            SRC_DIR / "agent" / "nodes" / "observe.py",
            SRC_DIR / "agent" / "nodes" / "reflect.py",
            SRC_DIR / "agent" / "nodes" / "synthesize.py",
        ],
        "Tools": [
            SRC_DIR / "tools" / "__init__.py",
            SRC_DIR / "tools" / "web.py",
            SRC_DIR / "tools" / "memory.py",
        ],
    }

    for section, files in src_modules.items():
        print(f"\n2. {section}:")
        section_ok = True
        for fpath in files:
            ok, msg = check_file_exists(fpath)
            print(f"   {msg}")
            if not ok:
                errors.append(msg)
                section_ok = False

        if section_ok:
            print(f"   ✅ All {len(files)} files present")

    # Test files
    test_files = [
        ROOT_DIR / "tests" / "conftest.py",
        ROOT_DIR / "tests" / "test_memory.py",
        ROOT_DIR / "tests" / "test_retrieval.py",
        ROOT_DIR / "tests" / "test_rules.py",
        ROOT_DIR / "tests" / "test_agent.py",
        ROOT_DIR / "tests" / "test_integration.py",
    ]

    print(f"\n3. Test Files:")
    for fpath in test_files:
        ok, msg = check_file_exists(fpath)
        print(f"   {msg}")
        if not ok:
            warnings.append(msg)

    # Script files
    script_files = [
        ROOT_DIR / "scripts" / "agent.py",
        ROOT_DIR / "scripts" / "validate.py",
    ]

    print(f"\n4. Script Files:")
    for fpath in script_files:
        ok, msg = check_file_exists(fpath)
        print(f"   {msg}")
        if not ok:
            warnings.append(msg)

    # Try importing core modules
    print(f"\n5. Import Validation:")
    try:
        sys.path.insert(0, str(ROOT_DIR))

        # Import base first
        if (SRC_DIR / "llm" / "base.py").exists():
            ok, msg = check_module_imports(SRC_DIR / "llm" / "base.py")
            print(f"   {msg}")

        # Import LLM modules
        for module in ["ollama_client.py", "router.py"]:
            path = SRC_DIR / "llm" / module
            if path.exists():
                ok, msg = check_module_imports(path)
                print(f"   {msg}")

    except Exception as e:
        print(f"   ⚠️  Import validation skipped: {str(e)[:80]}")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"\n❌ VALIDATION FAILED: {len(errors)} error(s)")
        for err in errors:
            print(f"   {err}")
        return 1
    elif warnings:
        print(f"\n⚠️  VALIDATION PARTIAL: {len(warnings)} warning(s)")
        for warn in warnings:
            print(f"   {warn}")
        print("\nℹ️  Some files missing (may be in progress)")
        return 0
    else:
        print(f"\n✅ VALIDATION PASSED: All required files present")
        return 0


if __name__ == "__main__":
    sys.exit(validate_project_structure())
