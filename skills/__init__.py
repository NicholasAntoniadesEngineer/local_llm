"""Skill loading helpers for runtime integration."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path


class SkillNotFoundError(RuntimeError):
    """Raised when a requested skill cannot be located."""


_SKILL_CACHE: dict[str, object] = {}
_SKILL_DIR = Path(__file__).resolve().parent
if str(_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(_SKILL_DIR))


def _module_name(skill_name: str) -> str:
    normalized_name = (skill_name or "").strip()
    if not normalized_name:
        raise SkillNotFoundError("skill name must be non-empty")
    return normalized_name


def _class_name(skill_name: str) -> str:
    return "".join(part.capitalize() for part in skill_name.split("_"))


def get_skill(name: str):
    """Return the imported module for a skill, cached after first load."""
    module_name = _module_name(name)
    if module_name in _SKILL_CACHE:
        return _SKILL_CACHE[module_name]

    try:
        skill_module = import_module(f".{module_name}", package=__name__)
    except ModuleNotFoundError as error_value:
        raise SkillNotFoundError(f"Skill '{module_name}' not found") from error_value

    _SKILL_CACHE[module_name] = skill_module
    return skill_module


def list_skills() -> list[str]:
    """List skill module names available in the skills package."""
    return sorted(
        path_value.stem
        for path_value in _SKILL_DIR.glob("*.py")
        if path_value.stem != "__init__"
    )


def call_skill(name: str, method: str, *args, **kwargs) -> dict[str, object]:
    """Invoke a method on a skill module or its primary class."""
    try:
        skill_module = get_skill(name)
        if hasattr(skill_module, method):
            result_value = getattr(skill_module, method)(*args, **kwargs)
            return {"ok": True, "result": result_value}

        class_name = _class_name(name)
        if hasattr(skill_module, class_name):
            skill_instance = getattr(skill_module, class_name)()
            if hasattr(skill_instance, method):
                result_value = getattr(skill_instance, method)(*args, **kwargs)
                return {"ok": True, "result": result_value}

        raise AttributeError(f"'{name}' has no callable '{method}'")
    except Exception as error_value:
        return {"ok": False, "error": str(error_value)}
