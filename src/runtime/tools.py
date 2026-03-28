"""Tool schemas and execution for the runtime controller."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src.config import CONFIG
from src.paths import ROOT
from src.runtime.patcher import MutationCoordinator
from src.runtime.tool_kinds import EXECUTION_TOOLS, MUTATION_TOOLS, OBSERVATION_TOOLS
from src.write_guard import AtomicWriter


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using DuckDuckGo. Returns titles, URLs, and snippets. "
                "Use for: finding API docs, understanding libraries, researching algorithms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Specific technical search query (5-10 words)"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code in a subprocess and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Complete Python code to execute."}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read file contents from the repo root. For files longer than ~900 lines, omitting "
                "start_line/end_line returns a numbered preview of the first chunk—then re-read with a line range. "
                "Use numbered=true (or any range) to get LINE| prefixes for mapping grep hits to replace_lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path from project root"},
                    "start_line": {
                        "type": "integer",
                        "description": "Optional 1-based first line (inclusive)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional 1-based last line (inclusive); defaults to start+399 or EOF",
                    },
                    "numbered": {
                        "type": "boolean",
                        "description": "If true, prefix lines with N| (recommended for skill edits)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write COMPLETE file content and overwrite the full target file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path"},
                    "content": {"type": "string", "description": "Complete file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace one exact occurrence of old_content with new_content. For functions or blocks, "
                "prefer replace_lines with line numbers from read_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_content": {"type": "string", "description": "Exact string to replace"},
                    "new_content": {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old_content", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_lines",
            "description": (
                "Replace lines start_line..end_line (1-based, inclusive) with the given content. "
                "Use for multi-line refactors; empty content deletes that range. Combine with numbered read_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to project root"},
                    "start_line": {"type": "integer", "description": "First line to replace (1-based, inclusive)"},
                    "end_line": {"type": "integer", "description": "Last line to replace (1-based, inclusive)"},
                    "content": {
                        "type": "string",
                        "description": "New text (may be multiline). Use empty string to delete the line range.",
                    },
                },
                "required": ["path", "start_line", "end_line", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_file",
            "description": "Search for a pattern in files and return matching lines with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "File or directory to search in"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files in a directory with sizes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command and return output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to execute"}
                },
                "required": ["cmd"],
            },
        },
    },
]


@dataclass
class ToolExecutionResult:
    """Structured tool execution result."""

    tool_name: str
    success: bool
    output: str
    summary: str
    result_kind: str
    details: dict[str, str]
    written_path: Optional[Path] = None


class ToolExecutor:
    """Stateful tool execution with caching and safe writes."""

    def __init__(
        self,
        output_dir: Path,
        write_status: Callable[[str, bool], None] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.write_status = write_status
        self.atomic_writer = AtomicWriter()
        self.mutation_coordinator = MutationCoordinator(self.atomic_writer)
        self.search_cache: dict[str, str] = {}
        self.files_written = 0

    def _resolve_path(self, path_value: str) -> Path:
        candidate_path = Path(path_value)
        if candidate_path.is_absolute():
            return candidate_path
        if "skills" in path_value:
            return ROOT / path_value
        return self.output_dir / path_value

    def _result_kind(self, tool_name: str, success: bool, written_path: Optional[Path]) -> str:
        if written_path is not None:
            return "mutation"
        if tool_name in OBSERVATION_TOOLS:
            return "observation" if success else "observation_error"
        if tool_name in EXECUTION_TOOLS:
            return "execution" if success else "execution_error"
        if tool_name in MUTATION_TOOLS:
            return "mutation" if success else "mutation_error"
        return "error" if not success else "result"

    def _summarize_output(self, tool_name: str, output: str) -> str:
        cleaned_output = " ".join(output.strip().split())
        if not cleaned_output:
            return f"{tool_name} returned no output"
        return cleaned_output[:180]

    def execute(self, name: str, args: dict) -> ToolExecutionResult:
        if self.write_status:
            self.write_status(f"TOOL: {name}", False)

        handler_map = {
            "web_search": lambda: self._web_search(args.get("query", "")),
            "run_python": lambda: self._run_python(args.get("code", "")),
            "bash": lambda: self._bash(args.get("cmd", "")),
            "read_file": lambda: self._read_file(
                args.get("path", ""),
                start_line=args.get("start_line"),
                end_line=args.get("end_line"),
                numbered=bool(args.get("numbered", False)),
            ),
            "replace_lines": lambda: self._replace_lines(
                args.get("path", ""),
                args.get("start_line"),
                args.get("end_line"),
                args.get("content", ""),
            ),
            "write_file": lambda: self._write_file(
                args.get("path") or args.get("file_path") or args.get("file_name", ""),
                args.get("content", ""),
            ),
            "edit_file": lambda: self._edit_file(
                args.get("path", ""),
                args.get("old_content", ""),
                args.get("new_content", ""),
            ),
            "grep_file": lambda: self._grep_file(args.get("pattern", ""), args.get("path", "skills/")),
            "list_dir": lambda: self._list_dir(args.get("path", ".")),
        }
        handler = handler_map.get(name)
        if not handler:
            output_text = f"Unknown tool: {name}"
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                output=output_text,
                summary=output_text,
                result_kind="tool_error",
                details={"requested_tool": name},
            )
        try:
            output, written_path = handler()
            success = not output.startswith("ERROR")
            details = {"requested_tool": name}
            if written_path is not None:
                details["written_path"] = str(written_path)
            return ToolExecutionResult(
                tool_name=name,
                success=success,
                output=output,
                summary=self._summarize_output(name, output),
                result_kind=self._result_kind(name, success, written_path),
                details=details,
                written_path=written_path,
            )
        except Exception as error_value:
            output_text = f"ERROR in {name}: {error_value}"
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                output=output_text,
                summary=self._summarize_output(name, output_text),
                result_kind="exception",
                details={"requested_tool": name},
            )

    def _web_search(self, query: str) -> tuple[str, Optional[Path]]:
        if not query:
            return "ERROR: empty query", None
        if query in self.search_cache:
            return f"[CACHED] {self.search_cache[query]}", None
        try:
            import httpx
            from bs4 import BeautifulSoup
            import urllib.parse

            url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            headers = {"User-Agent": "Mozilla/5.0"}
            with httpx.Client(timeout=CONFIG.web_search_timeout) as client:
                response = client.get(url, headers=headers, follow_redirects=True)
                if response.status_code != 200:
                    return f"ERROR: search failed (HTTP {response.status_code})", None
                soup = BeautifulSoup(response.text, "html.parser")
                results = []
                for block in soup.find_all("div", class_="result"):
                    link = block.find("a", href=True)
                    if not link:
                        continue
                    href = link.get("href", "")
                    title = link.get_text(strip=True)
                    snippet_el = block.find("div", class_="result__snippet")
                    snippet = snippet_el.get_text(strip=True)[:150] if snippet_el else ""
                    if not title:
                        continue
                    results.append(f"- {title}\n  {href}\n  {snippet}")
                    if len(results) >= CONFIG.max_search_results:
                        break
                if not results:
                    return f"No results for '{query}'", None
                output = f"Results for '{query}':\n\n" + "\n\n".join(results)
                self.search_cache[query] = output
                return output, None
        except ImportError:
            return "ERROR: pip install httpx beautifulsoup4", None
        except Exception as error_value:
            return f"ERROR: search error: {error_value}", None

    def _run_python(self, code: str) -> tuple[str, Optional[Path]]:
        if not code.strip():
            return "ERROR: empty code", None
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT.resolve()) + ":" + str(self.output_dir.resolve()) + ":" + env.get("PYTHONPATH", "")
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=CONFIG.code_execution_timeout,
                env=env,
                cwd=str(ROOT),
            )
            output = result.stdout + result.stderr
            return (output if output.strip() else "(no output)"), None
        except subprocess.TimeoutExpired:
            return f"ERROR: timed out after {CONFIG.code_execution_timeout}s", None
        except Exception as error_value:
            return f"ERROR: {error_value}", None

    def _bash(self, cmd: str) -> tuple[str, Optional[Path]]:
        if not cmd:
            return "ERROR: empty command", None
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=CONFIG.code_execution_timeout,
                cwd=str(ROOT),
            )
            output = result.stdout + result.stderr
            return (output if output.strip() else "(no output)"), None
        except subprocess.TimeoutExpired:
            return f"ERROR: timed out after {CONFIG.code_execution_timeout}s", None
        except Exception as error_value:
            return f"ERROR: {error_value}", None

    @staticmethod
    def _format_line_slice(
        lines: list[str],
        start_line: int,
        end_line: int,
        *,
        show_line_numbers: bool,
    ) -> str:
        """Return 1-based inclusive slice of lines as text."""
        if start_line < 1:
            start_line = 1
        if end_line > len(lines):
            end_line = len(lines)
        if start_line > end_line or not lines:
            return ""
        chunk = lines[start_line - 1 : end_line]
        if show_line_numbers:
            return "\n".join(f"{start_line + index:5d}| {text}" for index, text in enumerate(chunk))
        return "\n".join(chunk)

    def _read_file(
        self,
        path_value: str,
        start_line: int | None = None,
        end_line: int | None = None,
        numbered: bool = False,
    ) -> tuple[str, Optional[Path]]:
        try:
            candidate_path = Path(path_value)
            if not candidate_path.is_absolute():
                candidate_path = ROOT / path_value
            raw_text = candidate_path.read_text()
            lines = raw_text.splitlines()
            show_numbers = numbered or start_line is not None or end_line is not None

            if len(lines) > 900 and start_line is None and end_line is None:
                preview_end = min(450, len(lines))
                body = self._format_line_slice(lines, 1, preview_end, show_line_numbers=True)
                note = (
                    f"\n\n[Truncated: file has {len(lines)} lines. "
                    "Re-call read_file with start_line and end_line; keep each window under ~400 lines.]"
                )
                return body + note, None

            if start_line is not None:
                try:
                    start_index = int(start_line)
                except (TypeError, ValueError):
                    return "ERROR: start_line must be an integer", None
                if end_line is not None:
                    try:
                        end_index = int(end_line)
                    except (TypeError, ValueError):
                        return "ERROR: end_line must be an integer", None
                else:
                    end_index = min(start_index + 399, len(lines))
                if start_index < 1:
                    return "ERROR: start_line must be >= 1", None
                if start_index > len(lines):
                    return f"ERROR: start_line {start_index} past EOF ({len(lines)} lines)", None
                end_index = max(end_index, start_index)
                end_index = min(end_index, len(lines))
                body = self._format_line_slice(lines, start_index, end_index, show_line_numbers=show_numbers)
                header = f"(lines {start_index}-{end_index} of {len(lines)})\n"
                return header + body, None

            if show_numbers:
                body = self._format_line_slice(lines, 1, len(lines), show_line_numbers=True)
                return body, None
            return raw_text, None
        except FileNotFoundError:
            return f"ERROR: file not found: {path_value}", None
        except Exception as error_value:
            return f"ERROR: {error_value}", None

    def _replace_lines(
        self,
        path_value: str,
        start_line_raw: int | None,
        end_line_raw: int | None,
        new_content: str,
    ) -> tuple[str, Optional[Path]]:
        if not path_value:
            return "ERROR: path required", None
        try:
            start_line = int(start_line_raw)
            end_line = int(end_line_raw)
        except (TypeError, ValueError):
            return "ERROR: start_line and end_line must be integers", None
        if start_line < 1 or end_line < start_line:
            return "ERROR: need 1 <= start_line <= end_line", None
        try:
            target_path = self._resolve_path(path_value)
            if not target_path.exists():
                return f"ERROR: file not found: {path_value}", None
            current_text = target_path.read_text()
            lines = current_text.splitlines()
            if start_line > len(lines):
                return f"ERROR: start_line {start_line} past EOF ({len(lines)} lines)", None
            end_line = min(end_line, len(lines))
            new_lines = new_content.split("\n") if new_content else []
            head_part = lines[: start_line - 1]
            tail_part = lines[end_line:]
            merged_lines = head_part + new_lines + tail_part
            had_trailing = current_text.endswith("\n")
            updated_text = "\n".join(merged_lines)
            if had_trailing and (merged_lines or updated_text):
                if not updated_text.endswith("\n"):
                    updated_text += "\n"
            elif had_trailing and not merged_lines:
                updated_text = ""
            write_result, _ = self.mutation_coordinator.apply_mutation(
                target_path, updated_text, "replace_lines"
            )
            if write_result.success:
                self.files_written += 1
            return write_result.message, write_result.path if write_result.success else None
        except Exception as error_value:
            return f"ERROR: replace_lines failed: {error_value}", None

    def _write_file(self, path_value: str, content_text: str) -> tuple[str, Optional[Path]]:
        if not path_value:
            return "ERROR: path required", None
        if not content_text:
            return "ERROR: content is empty", None
        full_path = self._resolve_path(path_value)
        write_result, _ = self.mutation_coordinator.apply_mutation(full_path, content_text, "write_file")
        if write_result.success:
            self.files_written += 1
        return write_result.message, write_result.path if write_result.success else None

    def _edit_file(self, path_value: str, old_content: str, new_content: str) -> tuple[str, Optional[Path]]:
        try:
            target_path = self._resolve_path(path_value)
            if not target_path.exists():
                return f"ERROR: file not found: {path_value}", None
            current_text = target_path.read_text()
            if old_content not in current_text:
                preview_text = current_text[:300].replace("\n", "\\n")
                return (
                    f"ERROR: old_content not found in {path_value}. File starts with: {preview_text}",
                    None,
                )
            updated_text = current_text.replace(old_content, new_content, 1)
            write_result, _ = self.mutation_coordinator.apply_mutation(target_path, updated_text, "edit_file")
            return write_result.message, write_result.path if write_result.success else None
        except Exception as error_value:
            return f"ERROR: edit failed: {error_value}", None

    def commit_mutation(self, target_path: Path | str) -> str:
        """Commit a previously staged mutation after verifier acceptance."""
        return self.mutation_coordinator.commit_mutation(target_path)

    def rollback_mutation(self, target_path: Path | str) -> str:
        """Rollback a staged mutation after verifier rejection."""
        return self.mutation_coordinator.rollback_mutation(target_path)

    def _grep_file(self, pattern: str, path_value: str = "skills/") -> tuple[str, Optional[Path]]:
        try:
            results = []
            search_path = ROOT / path_value if not Path(path_value).is_absolute() else Path(path_value)
            files = [search_path] if search_path.is_file() else sorted(search_path.rglob("*.py"))
            for file_path in files[:20]:
                try:
                    for line_number, line_text in enumerate(file_path.read_text().splitlines(), 1):
                        if re.search(pattern, line_text):
                            results.append(f"{file_path}:{line_number}: {line_text.strip()}")
                            if len(results) >= 20:
                                break
                except Exception:
                    continue
                if len(results) >= 20:
                    break
            if not results:
                return f"No matches for '{pattern}' in {path_value}", None
            return "\n".join(results), None
        except Exception as error_value:
            return f"ERROR: grep failed: {error_value}", None

    def _list_dir(self, path_value: str = ".") -> tuple[str, Optional[Path]]:
        try:
            target_path = ROOT / path_value if not Path(path_value).is_absolute() else Path(path_value)
            if not target_path.is_dir():
                return f"ERROR: not a directory: {path_value}", None
            entries = []
            for file_path in sorted(target_path.iterdir()):
                if file_path.name.startswith(".") or file_path.name == "__pycache__":
                    continue
                if file_path.is_file():
                    entries.append(f"{file_path.name} ({file_path.stat().st_size:,}B)")
                else:
                    entries.append(f"{file_path.name}/")
            return ("\n".join(entries) if entries else "(empty)"), None
        except Exception as error_value:
            return f"ERROR: list failed: {error_value}", None
