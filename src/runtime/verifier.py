"""Shared verification logic for agent runs, skill scans, and improve cycles."""

from __future__ import annotations

import ast
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.paths import ROOT, SKILLS_DIR
from src.runtime.policy import ControllerPolicyConfig, POLICY_FILE
from src.runtime.tool_kinds import OBSERVATION_TOOLS


@dataclass
class VerificationResult:
    """Structured verifier outcome."""

    status: str
    accepted: bool
    should_stop: bool
    summary: str
    reward: float = 0.0
    target_path: str = ""
    failure_type: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def _classify_failure(summary: str, tool_name: str = "") -> str:
    """Classify verifier feedback into explicit replanning buckets."""
    lowered_summary = summary.lower()
    if "syntax error" in lowered_summary:
        return "syntax"
    if "tests failed" in lowered_summary or "timed out" in lowered_summary:
        return "test"
    if "missing prereq" in lowered_summary or "bad import" in lowered_summary:
        return "integration"
    if "policy validation failed" in lowered_summary:
        return "policy"
    if "placeholder" in lowered_summary or "too shallow" in lowered_summary or "too few" in lowered_summary:
        return "quality"
    if "traceback" in lowered_summary or tool_name == "run_python":
        return "runtime"
    if lowered_summary.startswith("error"):
        return "tool"
    if "completion signal ignored" in lowered_summary:
        return "completion"
    if tool_name in OBSERVATION_TOOLS:
        return "observation"
    return "verification"


def _annotate_parents(parsed_tree: ast.AST) -> None:
    for node_value in ast.walk(parsed_tree):
        for child_node in ast.iter_child_nodes(node_value):
            child_node.parent = node_value


def _substantive_line_count(function_node: ast.AST, source_text: str) -> int:
    """Count non-empty, non-comment lines inside a function body."""
    if not hasattr(function_node, "lineno") or not hasattr(function_node, "end_lineno"):
        return 0

    source_lines = source_text.splitlines()
    body_line_numbers = set()
    for statement_node in getattr(function_node, "body", []):
        if isinstance(statement_node, ast.Pass):
            continue
        start_line = getattr(statement_node, "lineno", 0)
        end_line = getattr(statement_node, "end_lineno", start_line)
        for line_number in range(start_line, end_line + 1):
            body_line_numbers.add(line_number)

    substantive_count = 0
    for line_number in sorted(body_line_numbers):
        line_text = source_lines[line_number - 1].strip()
        if line_text and not line_text.startswith("#"):
            substantive_count += 1
    return substantive_count


def _prereq_validation(skill_tree, path: Path, parsed_tree: ast.AST, source_text: str) -> tuple[bool, str]:
    """Require prereq imports and visible prereq usage when the skill has dependencies."""
    if skill_tree is None:
        return True, ""

    file_name = path.name
    state = skill_tree.get_state()
    skill_record = next((skill for skill in state["skills"] if skill.get("file") == file_name), None)
    if not skill_record or not skill_record.get("prereqs"):
        return True, ""

    imported_modules = set()
    imported_symbols = set()
    non_import_lines = [
        line_value
        for line_value in source_text.splitlines()
        if not line_value.strip().startswith(("import ", "from "))
    ]
    non_import_source = "\n".join(non_import_lines)

    for node_value in ast.walk(parsed_tree):
        if isinstance(node_value, ast.Import):
            for alias_value in node_value.names:
                imported_modules.add(alias_value.name.split(".")[0])
                imported_symbols.add(alias_value.asname or alias_value.name.split(".")[0])
        elif isinstance(node_value, ast.ImportFrom) and node_value.module:
            module_root = node_value.module.split(".")[0]
            imported_modules.add(module_root)
            for alias_value in node_value.names:
                imported_symbols.add(alias_value.asname or alias_value.name)

    call_targets = set()
    for node_value in ast.walk(parsed_tree):
        if isinstance(node_value, ast.Call):
            if isinstance(node_value.func, ast.Name):
                call_targets.add(node_value.func.id)
            elif isinstance(node_value.func, ast.Attribute):
                call_targets.add(node_value.func.attr)
                if isinstance(node_value.func.value, ast.Name):
                    call_targets.add(node_value.func.value.id)

    for prereq_id in skill_record["prereqs"]:
        prereq_state = next((skill for skill in state["skills"] if skill.get("id") == prereq_id), None)
        if not prereq_state:
            continue
        prereq_module = Path(prereq_state["file"]).stem
        prereq_tokens = {
            prereq_module,
            prereq_state["name"].replace(" ", ""),
            "".join(part.capitalize() for part in prereq_module.split("_")),
        }
        has_import = prereq_module in imported_modules or bool(prereq_tokens & imported_symbols)
        has_usage = bool(prereq_tokens & call_targets) or any(
            token_value in non_import_source for token_value in prereq_tokens
        )
        if not has_import:
            return False, f"Missing prereq import for {prereq_id}"
        if not has_usage:
            return False, f"Missing prereq usage for {prereq_id}"

    return True, ""


def validate_generated_module(filepath: str, skill_tree=None) -> tuple[bool, str]:
    """Validate generated code with structural and integration gates."""
    path = Path(filepath)
    if not path.exists():
        return False, "File not created"

    size = path.stat().st_size
    if size < 200:
        return False, f"Too small ({size}B) - placeholder"

    source = path.read_text()
    try:
        parsed_tree = ast.parse(source)
    except SyntaxError as error_value:
        return False, f"Syntax error: {error_value}"
    _annotate_parents(parsed_tree)

    func_nodes = [node_value for node_value in ast.walk(parsed_tree) if isinstance(node_value, (ast.FunctionDef, ast.AsyncFunctionDef))]
    func_count = len(func_nodes)
    assert_count = source.count("assert ")

    if func_count < 3:
        return False, f"Too few functions ({func_count}). Need ≥3 for production quality."
    if assert_count < 5:
        return False, f"Too few assertions ({assert_count}). Need ≥5 to verify correctness."

    for function_node in func_nodes:
        if function_node.name == "__init__":
            continue
        substantive_count = _substantive_line_count(function_node, source)
        if substantive_count < 8:
            return False, f"Function '{function_node.name}' is too shallow ({substantive_count} substantive lines)"

    if "from src." in source:
        return False, "Bad import: 'from src.*' — skills must be standalone"

    work_marker = "TO" + "DO"
    not_ready_marker = "raise " + "Not" + "Implemented" + "Error"
    if work_marker in source:
        return False, "Placeholder text detected: work marker"
    if not_ready_marker in source:
        return False, "Placeholder text detected: not-implemented marker"

    for node_value in ast.walk(parsed_tree):
        if isinstance(node_value, ast.Pass) and not isinstance(getattr(node_value, "parent", None), ast.ExceptHandler):
            return False, "Placeholder statement detected: pass"

    if len(source.splitlines()) > 100:
        has_try_except = any(isinstance(node_value, ast.Try) for node_value in ast.walk(parsed_tree))
        if not has_try_except:
            return False, "Large module missing try/except error handling"

    prereq_ok, prereq_message = _prereq_validation(skill_tree, path, parsed_tree, source)
    if not prereq_ok:
        return False, prereq_message

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT.resolve()) + ":" + str(SKILLS_DIR.resolve()) + ":" + env.get("PYTHONPATH", "")
        result = subprocess.run(
            ["python3", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd=str(ROOT),
        )
        if (
            result.returncode == 0
            and "ALL TESTS PASSED" in result.stdout
            and "Traceback" not in result.stderr
        ):
            return True, f"Tests passed ({size:,}B, {func_count}fn, {assert_count}asserts)"

        output = result.stdout + result.stderr
        return False, f"Tests failed: {output[:200]}"
    except subprocess.TimeoutExpired:
        return False, "Timed out (30s)"
    except Exception as error_value:
        return False, f"Error: {error_value}"


class RuntimeVerifier:
    """Verifier used by the runtime controller."""

    def __init__(self, skill_tree=None) -> None:
        self.skill_tree = skill_tree

    def evaluate_tool_result(
        self,
        tool_name: str,
        result_text: str,
        written_path: Path | None = None,
    ) -> VerificationResult:
        if tool_name in {"write_file", "edit_file"} and written_path:
            if written_path.name == POLICY_FILE.name:
                try:
                    payload = json.loads(written_path.read_text())
                    ControllerPolicyConfig(**payload)
                    return VerificationResult(
                        status="validated_policy",
                        accepted=True,
                        should_stop=False,
                        summary="Controller policy validated.",
                        reward=0.6,
                        target_path=str(written_path),
                        failure_type="",
                    )
                except Exception as error_value:
                    summary_text = f"Policy validation failed: {error_value}"
                    return VerificationResult(
                        status="rejected_policy",
                        accepted=False,
                        should_stop=False,
                        summary=summary_text,
                        reward=-0.6,
                        target_path=str(written_path),
                        failure_type=_classify_failure(summary_text, tool_name),
                    )
            accepted, summary = validate_generated_module(str(written_path), skill_tree=self.skill_tree)
            return VerificationResult(
                status="validated_write" if accepted else "rejected_write",
                accepted=accepted,
                should_stop=False,
                summary=summary,
                reward=1.0 if accepted else -1.0,
                target_path=str(written_path),
                failure_type="" if accepted else _classify_failure(summary, tool_name),
            )

        if tool_name == "run_python":
            passed = "ALL TESTS PASSED" in result_text and "Traceback" not in result_text
            return VerificationResult(
                status="runtime_check",
                accepted=passed,
                should_stop=False,
                summary=result_text[:200] if result_text.strip() else "(no output)",
                reward=0.3 if passed else -0.3,
                failure_type="" if passed else _classify_failure(result_text, tool_name),
            )

        failure_type = _classify_failure(result_text, tool_name)
        return VerificationResult(
            status="observation",
            accepted=False,
            should_stop=False,
            summary=result_text[:160],
            reward=0.05 if not result_text.startswith("ERROR") else -0.1,
            failure_type="" if failure_type == "observation" else failure_type,
        )

    def evaluate_completion_signal(
        self,
        response_text: str,
        last_verification: VerificationResult | None,
    ) -> VerificationResult:
        if "DONE" in response_text and last_verification and last_verification.accepted:
            return VerificationResult(
                status="accepted_completion",
                accepted=True,
                should_stop=True,
                summary=last_verification.summary,
                reward=last_verification.reward,
                target_path=last_verification.target_path,
                failure_type="",
            )

        if "DONE" in response_text:
            return VerificationResult(
                status="unverified_completion",
                accepted=False,
                should_stop=False,
                summary="Completion signal ignored until verifier accepts an artifact.",
                reward=-0.2,
                failure_type="completion",
            )

        return VerificationResult(
            status="continue",
            accepted=False,
            should_stop=False,
            summary="Continue execution.",
            reward=0.0,
            failure_type="",
        )
