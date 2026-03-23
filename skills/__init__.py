"""Skill registry with lazy loading. Imports skill classes on demand to avoid
circular imports and gracefully handle missing/broken skills."""

import importlib
from pathlib import Path

SKILL_MODULES = [
    f.stem for f in Path(__file__).parent.glob("*.py")
    if not f.name.startswith("_")
]

_registry: dict = {}


def get_skill(name: str):
    """Lazy-load and cache a skill instance by module name.

    Returns the main class instance, or None if the skill is broken/missing.
    """
    if name in _registry:
        return _registry[name]
    try:
        mod = importlib.import_module(f"skills.{name}")
        # Find the main class (first class defined in the module)
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, type) and obj.__module__ == mod.__name__:
                instance = obj()
                _registry[name] = instance
                return instance
        # No class found — return module itself
        _registry[name] = mod
        return mod
    except Exception:
        _registry[name] = None
        return None


def get_all_skills() -> dict:
    """Load all available skills, skipping broken ones."""
    for name in SKILL_MODULES:
        if name not in _registry:
            get_skill(name)
    return {k: v for k, v in _registry.items() if v is not None}
