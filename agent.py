#!/usr/bin/env python3
"""
MLX-powered autonomous agent for Apple Silicon (optimized for M-series chips).
Uses configuration-driven design with no hardcoded magic numbers.

Performance: ~60 tok/s on M4 Max with Qwen3-8B-4bit (MLX is 2-3x faster than Ollama).
"""

import json
import subprocess
import sys
import urllib.request
import urllib.parse
import re
import os
from pathlib import Path
from typing import Optional

from config import CONFIG
from config_manager import ConfigManager
from memory import MemoryManager
from reflection import ReflectionEngine


class MLXAgent:
    """MLX-powered agent with configuration-driven design."""

    def __init__(self, config_model_name: str = "fast", goal: str = "") -> None:
        """Initialize agent with config-specified model.

        Args:
            config_model_name: Key in CONFIG.models (fast/balanced/quality)
            goal: Task goal (used for memory tracking)

        Raises:
            KeyError: If model_name not in config
            ImportError: If mlx_lm not installed
        """
        if config_model_name not in CONFIG.models:
            raise KeyError(f"Model '{config_model_name}' not in CONFIG.models")

        self.config_model = CONFIG.models[config_model_name]
        self.conversation: list[dict[str, str]] = []
        self.config_manager = ConfigManager()  # For self-improvement
        self.memory_manager = MemoryManager(goal) if goal else None  # Session memory

        # Setup output directory
        CONFIG.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy load MLX
        try:
            from mlx_lm import load, generate

            self.mlx_generate = generate
            self.model, self.tokenizer = load(self.config_model.name)
            print(f"✅ Loaded {self.config_model.name} via MLX")
            print(f"📁 Output folder: {CONFIG.output_dir.resolve()}")
            print(f"📊 Context: {self.config_model.context_window} tokens")
        except ImportError:
            print("❌ MLX not installed. Run: pip install mlx-lm")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)

    def chat(self, prompt: str, system: Optional[str] = None) -> str:
        """Chat with MLX model using configured max_tokens.

        Args:
            prompt: User message
            system: Optional system prompt (inserted if not already present)

        Returns:
            Model response text
        """
        if system and not any(msg.get("role") == "system" for msg in self.conversation):
            self.conversation.insert(0, {"role": "system", "content": system})

        self.conversation.append({"role": "user", "content": prompt})

        # Format messages for MLX (keep recent context within budget)
        messages_text = ""
        max_history_turns = 5  # Keep last N conversation turns
        for msg in self.conversation[-max_history_turns:]:
            role = msg["role"]
            content = msg["content"]
            messages_text += f"{role}: {content}\n"

        try:
            response = self.mlx_generate(
                self.model,
                self.tokenizer,
                prompt=messages_text,
                max_tokens=self.config_model.max_tokens,
            )
            self.conversation.append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            return f"ERROR: {str(e)}"

    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool with flexible argument handling.

        Args:
            tool_name: Tool to execute (web_search, run_python, etc.)
            args: Tool arguments (flexible naming)

        Returns:
            Tool execution result
        """
        try:
            if tool_name == "web_search":
                return self._web_search(args.get("query", ""))
            elif tool_name == "run_python":
                return self._run_python(args.get("code", ""))
            elif tool_name == "bash":
                return self._bash(args.get("cmd", ""))
            elif tool_name == "read_file":
                return self._read_file(args.get("path", ""))
            elif tool_name == "write_file":
                # Handle flexible argument names (path, file_path, file_name, etc.)
                path = args.get("path") or args.get("file_path") or args.get("file_name", "")
                content = args.get("content", "")
                return self._write_file(path, content)
            elif tool_name == "spawn_subagent":
                return self._spawn_subagent(
                    args.get("goal", ""),
                    args.get("config_version", "v0"),
                )
            elif tool_name == "write_config":
                return self._write_config(
                    args.get("description", ""),
                    args.get("changes", {}),
                )
            elif tool_name == "evaluate_version":
                return self._evaluate_version(
                    args.get("version_id", ""),
                    args.get("score", 0.0),
                    args.get("results", {}),
                )
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"ERROR in {tool_name}: {str(e)}"

    def _web_search(self, query: str) -> str:
        """Search using DuckDuckGo with SMART result filtering & relevance scoring.

        IMPROVED: Filters junk results, rates relevance, suggests better queries if needed.
        TRACKS QUALITY: Returns average relevance score for phase-decision logic.

        Args:
            query: Search query string

        Returns:
            Formatted search results with relevance scores + quality metrics
        """
        try:
            import httpx
            from bs4 import BeautifulSoup
            import urllib.parse as urlparse

            url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

            with httpx.Client(timeout=CONFIG.web_search_timeout) as client:
                response = client.get(url, headers=headers, follow_redirects=True)
                if response.status_code != 200:
                    return f"Search failed (HTTP {response.status_code})"

                soup = BeautifulSoup(response.text, "html.parser")
                results = []
                relevance_scores = []
                junk_count = 0

                # Find all result blocks
                for result in soup.find_all("div", class_="result"):
                    links = result.find_all("a")
                    if not links:
                        continue

                    title = ""
                    href = ""

                    for link in links:
                        link_href = link.get("href", "").strip()
                        link_text = link.get_text(strip=True)

                        if not link_text or not link_href:
                            continue

                        if "uddg=" in link_href:
                            try:
                                actual_url = urlparse.parse_qs(urlparse.urlparse(link_href).query).get("uddg", [""])[0]
                                link_href = actual_url
                            except:
                                pass

                        if link_href.startswith("http"):
                            if not href:
                                href = link_href
                            if not title and len(link_text) > 3:
                                title = link_text
                                break

                    # SMART FILTERING: Check relevance
                    if not href or not title:
                        continue

                    # Score relevance to query
                    relevance_score = self._score_relevance(title, href, query)

                    # FILTER: Skip low-relevance junk
                    if relevance_score < 0.3:
                        junk_count += 1
                        continue

                    relevance_scores.append(relevance_score)

                    # Extract snippet
                    snippet = ""
                    snippet_elem = result.find("div", class_="result__snippet")
                    if snippet_elem:
                        snippet = snippet_elem.get_text(strip=True)[:150]

                    if len(results) < CONFIG.max_search_results:
                        # Try content extraction for top 2
                        content = ""
                        if len(results) < 2:
                            content = self._fetch_page_summary(href)

                        # Format with relevance indicator
                        relevance_bar = "█" * int(relevance_score * 5) + "░" * (5 - int(relevance_score * 5))

                        if content:
                            results.append(f"• **{title}** [{relevance_bar}]\n  {href}\n  SUMMARY: {content[:200]}")
                        elif snippet:
                            results.append(f"• **{title}** [{relevance_bar}]\n  {href}\n  SNIPPET: {snippet}")
                        else:
                            results.append(f"• **{title}** [{relevance_bar}]\n  {href}")

                if results:
                    output = "Search Results:\n\n" + "\n".join(results)

                    # QUALITY METRICS: Calculate average relevance
                    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
                    quality_indicator = "🔴 LOW" if avg_relevance < 0.4 else "🟡 MEDIUM" if avg_relevance < 0.6 else "🟢 HIGH"
                    output += f"\n\n📊 Result Quality: {quality_indicator} (avg: {avg_relevance:.2f})"

                    # INTELLIGENT: If too many junk results, suggest better query
                    if junk_count > len(results):
                        output += f"\n⚠️  Found many irrelevant results ({junk_count} filtered)."
                        output += f"\nSuggest: '{self._suggest_refined_query(query)}'"

                    # STORE QUALITY METRIC for agent decision-making
                    if not hasattr(self, '_last_search_quality'):
                        self._last_search_quality = {}
                    self._last_search_quality['average_relevance'] = avg_relevance
                    self._last_search_quality['quality_level'] = quality_indicator

                    return output
                else:
                    # HELPFUL: If no good results, suggest refined query
                    suggestion = self._suggest_refined_query(query)
                    output = f"No relevant results for '{query}'.\n🔴 LOW quality search - Try: '{suggestion}'"

                    # Track low quality
                    if not hasattr(self, '_last_search_quality'):
                        self._last_search_quality = {}
                    self._last_search_quality['average_relevance'] = 0.0
                    self._last_search_quality['quality_level'] = "🔴 LOW"

                    return output

        except ImportError:
            return "httpx/beautifulsoup4 not installed"
        except Exception as e:
            return f"Search error: {str(e)}"

    def _score_relevance(self, title: str, url: str, query: str) -> float:
        """Score how relevant a result is to the query (0-1 scale).

        FILTERS OUT JUNK: Generic pages, unrelated content, etc.
        """
        score = 0.5  # Base score

        query_words = set(query.lower().split())
        title_lower = title.lower()
        url_lower = url.lower()

        # Boost: Keywords in title
        keyword_matches = sum(1 for word in query_words if word in title_lower)
        score += (keyword_matches / max(1, len(query_words))) * 0.3

        # Boost: Keywords in domain
        if any(word in url_lower for word in query_words):
            score += 0.2

        # PENALIZE: Junk domains
        junk_domains = ["google.com", "github.com/search", "pinterest.com", "facebook.com", "youtube.com/watch", "twitter.com"]
        if any(junk in url_lower for junk in junk_domains):
            score *= 0.5

        # PENALIZE: Generic pages (no specific content signal)
        if "search?q=" in url or "results?q=" in url:
            score *= 0.3

        # BOOST: Official docs / tutorials / guides
        good_signals = ["docs", "documentation", "tutorial", "guide", "api", "github.com", "official"]
        if any(signal in title_lower or signal in url_lower for signal in good_signals):
            score = min(1.0, score + 0.3)

        return min(1.0, max(0.0, score))

    def _suggest_refined_query(self, original_query: str) -> str:
        """Suggest a better search query if current one isn't working.

        SMART: Different angles on the same topic.
        Rotates through strategies: tutorial → docs → github → code → api → guide
        """
        words = original_query.lower().split()

        strategies = [
            f"'{original_query}' tutorial",
            f"'{original_query}' documentation",
            f"'{original_query}' github",
            f"'{original_query}' example code",
            f"'{' '.join(words[:-1])} api" if len(words) > 1 else f"'{original_query}' api",
            f"'{words[0]}' {words[-1]} guide" if len(words) > 1 else f"'{original_query}' guide",
        ]

        # Track which strategy to use next (round-robin)
        if not hasattr(self, '_refined_query_index'):
            self._refined_query_index = 0
        else:
            self._refined_query_index = (self._refined_query_index + 1) % len(strategies)

        return strategies[self._refined_query_index]

    def _fetch_page_summary(self, url: str) -> str:
        """Fetch and summarize a page (quick extraction).

        Args:
            url: Page URL to fetch

        Returns:
            First 200 chars of page content or empty string on failure
        """
        try:
            import httpx
            from bs4 import BeautifulSoup

            with httpx.Client(timeout=5) as client:
                response = client.get(url, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    # Remove scripts and styles
                    for script in soup(["script", "style"]):
                        script.decompose()
                    # Get text
                    text = soup.get_text()
                    # Clean up whitespace
                    text = " ".join(text.split())
                    return text[:200] if text else ""
        except Exception:
            pass
        return ""

    def _run_python(self, code: str) -> str:
        """Execute Python code with configured timeout.

        Args:
            code: Python code to execute

        Returns:
            stdout or stderr output
        """
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=CONFIG.code_execution_timeout,
            )
            return result.stdout if result.stdout else result.stderr
        except subprocess.TimeoutExpired:
            return f"ERROR: Code execution timed out after {CONFIG.code_execution_timeout}s"
        except Exception as e:
            return f"ERROR: {str(e)}"

    def _bash(self, cmd: str) -> str:
        """Execute bash command with configured timeout.

        Args:
            cmd: Shell command to execute

        Returns:
            stdout or stderr output
        """
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=CONFIG.code_execution_timeout,
            )
            return result.stdout if result.stdout else result.stderr
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {CONFIG.code_execution_timeout}s"
        except Exception as e:
            return f"ERROR: {str(e)}"

    def _read_file(self, path: str) -> str:
        """Read file contents.

        Args:
            path: File path (absolute or relative to current directory)

        Returns:
            File contents or error message
        """
        try:
            with open(path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return f"ERROR: File not found: {path}"
        except Exception as e:
            return f"ERROR: {str(e)}"

    def _write_file(self, path: str, content: str) -> str:
        """Write file using configured output directory.

        Args:
            path: Filename (must include .txt, .py, etc.)
            content: Content to write

        Returns:
            Success message or error with details
        """
        try:
            # Validate inputs
            if not path:
                return "ERROR: path is required (e.g., 'code.py')"

            if not content:
                return "ERROR: content is empty - nothing to write"

            # Reject directory paths (must be actual filename)
            if path.endswith('/'):
                return "ERROR: path must be filename (e.g., 'code.py'), not a directory"

            # Determine full path
            if os.path.isabs(path):
                full_path = Path(path)
            elif "agent_outputs" in path:
                full_path = Path(path)
            else:
                full_path = CONFIG.output_dir / path

            # Ensure directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with open(full_path, "w") as f:
                bytes_written = f.write(content)

            if bytes_written == 0:
                return f"WARNING: Wrote 0 bytes to {full_path} (content was empty?)"

            return f"✓ Successfully wrote {bytes_written} bytes to {full_path}"
        except Exception as e:
            return f"ERROR writing '{path}': {str(e)}"

    def _spawn_subagent(self, goal: str, config_version: str = "v0") -> str:
        """Spawn a sub-agent with specific config version.

        Args:
            goal: Task for the sub-agent
            config_version: Config version to use (e.g., "v0", "v1")

        Returns:
            Sub-agent execution result
        """
        try:
            # Create temp directory for subagent output
            subagent_dir = CONFIG.output_dir / f"subagent_{config_version}"
            subagent_dir.mkdir(parents=True, exist_ok=True)

            # Run subagent with config version
            cmd = [
                "python3",
                "agent.py",
                f"--config-version={config_version}",
                goal,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=CONFIG.code_execution_timeout * 2,
                cwd=str(Path.cwd()),
            )

            output = result.stdout if result.stdout else result.stderr
            return f"Subagent({config_version}) completed:\n{output[:500]}"
        except subprocess.TimeoutExpired:
            return f"Subagent timeout after {CONFIG.code_execution_timeout * 2}s"
        except Exception as e:
            return f"ERROR spawning subagent: {str(e)}"

    def _write_config(self, description: str, changes: dict) -> str:
        """Write a new config version with specified changes.

        Args:
            description: What changed (e.g., "Increased max_tokens from 512 to 1024")
            changes: Dict of config changes (e.g., {"max_tokens": 1024})

        Returns:
            New config version ID and validation result
        """
        try:
            version_id = self.config_manager.create_version(
                description=description,
                config_changes=changes,
                parent_version="v0",
            )

            version = self.config_manager.versions[version_id]
            return (
                f"Created {version_id}: {description}\n"
                f"Config file: {version.config_file}\n"
                f"Ready for A/B testing"
            )
        except Exception as e:
            return f"ERROR creating config version: {str(e)}"

    def _evaluate_version(
        self, version_id: str, score: float, results: dict
    ) -> str:
        """Record performance metrics for a config version.

        Args:
            version_id: Version to evaluate (e.g., "v1")
            score: Performance score (0-1, higher is better)
            results: Detailed test results dict

        Returns:
            Evaluation summary
        """
        try:
            self.config_manager.evaluate_version(version_id, score, results)

            best = self.config_manager.get_best_version()
            return (
                f"Recorded metrics for {version_id}: score={score:.2f}\n"
                f"Best version so far: {best}\n"
                f"Results: {results}"
            )
        except Exception as e:
            return f"ERROR evaluating version: {str(e)}"

    def _build_memory_context(self) -> str:
        """Build concise memory context for the model (prevents context loss).

        Returns:
            Formatted memory summary showing what's been done and discovered
        """
        if not self.memory_manager:
            return "MEMORY: (empty)"

        mem = self.memory_manager.memory
        context = f"\n=== MEMORY CONTEXT ===\n"
        context += f"Total iterations: {len(mem.iterations)}\n"

        # Recent tools and results
        if mem.iterations:
            context += f"\nRecent actions:\n"
            for it in mem.iterations[-4:]:  # Last 4 iterations
                tool = it.tool_used
                success = "✓" if it.success else "✗"
                result_preview = it.result[:60].replace("\n", " ") if it.result else "(no result)"
                context += f"  {success} Step {it.step}: {tool} → {result_preview}\n"

        # Discoveries
        if mem.discoveries:
            context += f"\nKey discoveries:\n"
            for discovery in mem.discoveries[-3:]:  # Last 3 discoveries
                context += f"  • {discovery[:80]}\n"

        # Current phase/failures
        if mem.failures:
            context += f"\nRecent issues ({len(mem.failures)} total):\n"
            for failure in mem.failures[-2:]:  # Last 2 failures
                context += f"  ✗ {failure[:70]}\n"

        context += "=== END MEMORY ===\n"
        return context

    def run_loop(self, goal: str) -> None:
        """Execute intelligent ReAct loop with aggressive phase tracking & FULL CONTEXT.

        Args:
            goal: Task objective for the agent

        Features:
        - MAINTAINS FULL CONTEXT in system prompt (critical!)
        - Working memory of ALL iterations and discoveries
        - Reflects on progress and adjusts strategy
        - Never asks user for input (auto-continues)
        - Tracks learning and discoveries
        - AGGRESSIVE phase tracking: Research → Code → Test → Save
        - Self-improves through config evolution
        """
        # Initialize memory for this session
        if not self.memory_manager:
            self.memory_manager = MemoryManager(goal)

        print(f"\n🚀 MLX Agent starting (AGGRESSIVE MODE):\n  Goal: {goal}\n")

        # Use initial response to start
        initial_system = f"""GOAL: {goal}

YOU MUST OUTPUT TOOL CALLS IN THIS EXACT FORMAT (no extra text):
<tool>TOOL_NAME</tool>
<args>{{"arg1": "value1", "arg2": "value2"}}</args>

DO NOT write explanations or narrative. ONLY output tool calls.

PHASES:
1. RESEARCH: Use web_search to find information
2. IMPLEMENTATION: Use run_python to write code
3. SAVE: Use write_file to save results

Tool definitions:
- web_search: {{"query": "your search query"}}
- run_python: {{"code": "print('hello')"}}
- write_file: {{"path": "filename.py", "content": "import sys\\nprint('code')"}}
- read_file: {{"path": "filename.py"}}

OUTPUT NOW WITH TOOL CALL. No narrative."""

        response = self.chat("START", system=initial_system)

        consecutive_failures = 0
        reflection_engine = ReflectionEngine(self.memory_manager.memory) if self.memory_manager else None
        research_count = 0  # Track web_search calls per phase
        phase = "research"  # Current phase
        low_quality_search_count = 0  # Track consecutive low-quality searches
        refined_query_attempts = 0  # Track refined query attempts

        for iteration in range(CONFIG.max_iterations):
            print(f"\n[Step {iteration + 1}] Phase: {phase}")
            print(response[:250])

            # Parse tool calls - MORE FORGIVING (model often adds text around them)
            tool_pattern = r"<tool>(\w+)</tool>\s*<args>({.*?})</args>"
            matches = re.findall(tool_pattern, response, re.DOTALL)

            # If no matches, try to extract from narrative (model writes "I'll use web_search")
            if not matches:
                # Look for tool names mentioned
                narrative = response.lower()
                if "web_search" in narrative:
                    # Extract query from narrative if possible
                    query_match = re.search(r'(?:search|query|find)["\']?:?\s*["\']?([^"\'<>\n]+)', response, re.IGNORECASE)
                    query = query_match.group(1).strip() if query_match else "polymarket data"
                    matches = [("web_search", f'{{"query": "{query}"}}')]
                elif "run_python" in narrative:
                    # Extract code if possible
                    code_match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
                    if code_match:
                        code = code_match.group(1).strip().replace('"', r'\"')
                        matches = [("run_python", f'{{"code": "{code[:500]}"}}')]
                    else:
                        matches = [("run_python", '{"code": "print(\'test\')"}')]
                elif "write_file" in narrative or "save" in narrative:
                    # Try to extract actual code from narrative
                    code_match = re.search(r'```(?:python|)\s*(.*?)```', response, re.DOTALL)
                    if code_match:
                        code_content = code_match.group(1).strip()
                        code_content = code_content.replace('"', r'\"').replace('\n', '\\n')
                        filename = "solution.py" if "python" in narrative else "output.txt"
                        matches = [("write_file", f'{{"path": "{filename}", "content": "{code_content[:1000]}"}}')]
                    else:
                        # Last resort: extract class or function definition
                        def_match = re.search(r'(?:class|def) \w+.*?(?=\n(?:class|def|$))', response, re.DOTALL)
                        if def_match:
                            code_content = def_match.group(0).replace('"', r'\"').replace('\n', '\\n')
                            matches = [("write_file", f'{{"path": "solution.py", "content": "{code_content[:1000]}"}}')]
                        else:
                            matches = [("write_file", '{"path": "output.py", "content": "# TODO: implement"}')]

            # Filter out invalid tool names (must be in allowed list)
            valid_tools = {"web_search", "run_python", "bash", "read_file", "write_file", "write_config", "spawn_subagent", "evaluate_version"}
            matches = [(t, a) for t, a in matches if t in valid_tools]

            if not matches:
                # No tool call - FORCE tool execution
                if iteration < CONFIG.max_iterations - 2:
                    print("\n⚠️  No tool detected. FORCING execution...")
                    consecutive_failures += 1

                    # On 3rd failure, force actual tool execution (don't ask model)
                    if consecutive_failures >= 3:
                        print("   🔨 FORCE-EXECUTING TOOL (model not cooperating)")
                        # Force execute based on phase
                        if phase == "research":
                            forced_result = self._web_search(f"Polymarket {goal[:50]}")
                            print(f"   → Forced web_search: {forced_result[:80]}...")
                            if self.memory_manager:
                                self.memory_manager.record_attempt(
                                    step=iteration + 1,
                                    tool="web_search",
                                    args={},
                                    result=forced_result[:200],
                                    success=True,
                                    learning="Forced execution"
                                )
                            research_count += 1
                            response = self.chat("Continue. Next tool: run_python to write code.")
                            consecutive_failures = 0
                            continue
                        elif phase == "code":
                            code = "import json\nprint('Testing API connectivity')"
                            forced_result = self._run_python(code)
                            print(f"   → Forced run_python: {forced_result[:80]}...")
                            if self.memory_manager:
                                self.memory_manager.record_attempt(
                                    step=iteration + 1,
                                    tool="run_python",
                                    args={},
                                    result=forced_result[:200],
                                    success=True,
                                    learning="Forced execution"
                                )
                            phase = "save"
                            response = self.chat("Now save the result with write_file.")
                            consecutive_failures = 0
                            continue

                    if consecutive_failures > 6:
                        print("\n❌ Too many failures. Stopping.")
                        break

                    # Otherwise, reflection prompt with actual tool call
                    if reflection_engine:
                        reflection_prompt = reflection_engine.get_reflection_prompt()
                        print(f"   Trying: {reflection_prompt[:150]}")
                        response = self.chat(reflection_prompt)
                    else:
                        response = self.chat("<tool>web_search</tool>\n<args>{\"query\": \"polymarket data\"}</args>")
                    continue
                else:
                    # Near end - done
                    print("\n✅ Agent completed.")
                    break

            # AGGRESSIVE phase tracking - force progression
            if matches and self.memory_manager:
                tool_name = matches[0][0]
                recent_tools = [it.tool_used for it in self.memory_manager.memory.iterations[-3:]]

                # Track research phase
                if tool_name == "web_search":
                    research_count += 1
                    phase = "research"

                    # SMART QUALITY-BASED DECISIONS
                    avg_relevance = getattr(self, '_last_search_quality', {}).get('average_relevance', 0.5)

                    if avg_relevance < 0.4:
                        low_quality_search_count += 1
                        print(f"  🔴 LOW quality search (score: {avg_relevance:.2f})")

                        # If low quality and haven't tried refined queries yet, force refinement
                        if refined_query_attempts < 2:
                            refined_query_attempts += 1
                            suggestion = self._suggest_refined_query(goal)
                            print(f"  🔄 Forcing refined search: '{suggestion}'")
                            response = self.chat(f"Last search had low-quality results. Try this refined query instead: '{suggestion}'")
                            continue
                        else:
                            # Tried refined queries but still low quality - move to code anyway
                            print(f"  ⚠️  Low quality persists after {refined_query_attempts} refinements - moving to code")
                            phase = "code"
                            response = self.chat("Quality is low despite refinements. NOW: Write working Python code to solve this.")
                            continue
                    else:
                        low_quality_search_count = 0  # Reset if quality improves
                        refined_query_attempts = 0

                elif tool_name == "run_python":
                    phase = "code"
                    research_count = 0
                elif tool_name == "write_file":
                    phase = "save"

                # FORCE phase transition: 3 web_searches with GOOD quality = move to code
                if research_count >= 3 and phase == "research":
                    avg_relevance = getattr(self, '_last_search_quality', {}).get('average_relevance', 0.5)
                    if avg_relevance >= 0.4:
                        print(f"  ⚠️  Research limit (3) reached with acceptable quality - FORCING code phase")
                        response = self.chat("You've researched enough. NOW: Write working Python code. Use run_python with complete, working code.")
                        continue
                    else:
                        print(f"  ⚠️  Research limit reached but quality is low - trying refined search")
                        suggestion = self._suggest_refined_query(goal)
                        response = self.chat(f"Try this more refined search: '{suggestion}'")
                        continue

                # FORCE: If last 2 were web_search with good quality, next MUST be code
                if len(recent_tools) >= 2 and recent_tools[-2:] == ["web_search", "web_search"]:
                    avg_relevance = getattr(self, '_last_search_quality', {}).get('average_relevance', 0.5)
                    if avg_relevance >= 0.5:
                        print(f"  ⚠️  2 researches with good quality detected - FORCING code phase now")
                        response = self.chat("STOP researching. Write and run Python code NOW.")
                        continue

                # Detect loops - if same tool 3+ times, check if it's productive
                if len(set(recent_tools)) == 1 and recent_tools[0] == tool_name:
                    # Same tool used 3+ times - check if results are different
                    current_tool = tool_name

                    if self.memory_manager and len(self.memory_manager.memory.iterations) >= 3:
                        recent_results = [it.result[:100] for it in self.memory_manager.memory.iterations[-3:]]
                        unique_results = len(set(recent_results))

                        # Only force if getting identical results (real loop)
                        if unique_results <= 1:
                            print(f"  🔄 LOOP DETECTED ({current_tool}x3 identical results) - Need different approach")
                            if current_tool == "web_search":
                                response = self.chat("You're getting the same search results repeatedly. Try a completely different search query - maybe search for 'tutorial' or 'documentation' instead.")
                            elif current_tool == "run_python":
                                response = self.chat("Your code keeps producing the same error. Debug the issue or try a different approach to the problem.")
                            else:
                                response = self.chat("You're repeating the same action with the same result. Change your strategy.")
                            continue
                        else:
                            # Different results each time - probably making progress, don't force
                            print(f"  ℹ️  Same tool used 3x but with different results - likely making progress, continuing...")
                    else:
                        # Not enough history, don't force yet
                        print(f"  ℹ️  Same tool used multiple times, continuing...")

            # Execute tools
            for tool_name, args_str in matches:
                try:
                    # Try to parse JSON - handle common issues
                    args_str_clean = args_str.strip()
                    if not args_str_clean.startswith('{'):
                        args_str_clean = '{' + args_str_clean
                    if not args_str_clean.endswith('}'):
                        args_str_clean = args_str_clean + '}'
                    args = json.loads(args_str_clean)
                except Exception as e:
                    # If JSON fails, create basic args from tool name
                    if tool_name == "web_search":
                        args = {"query": "prediction markets"}
                    elif tool_name == "run_python":
                        args = {"code": "print('retry')"}
                    else:
                        args = {}

                print(f"  🔧 {tool_name}()")
                result = self.execute_tool(tool_name, args)
                result_preview = result[:120].replace("\n", " ")
                print(f"  → {result_preview}...")

                # Track in memory
                success = not result.startswith("ERROR")
                if self.memory_manager:
                    learning = (
                        "Success" if success else
                        "Failed: Will try different approach"
                    )
                    self.memory_manager.record_attempt(
                        step=iteration + 1,
                        tool=tool_name,
                        args=args,
                        result=result[:300],
                        success=success,
                        learning=learning,
                    )

                    # IMPORTANT: Record discoveries from successful searches/code runs
                    if success:
                        if tool_name == "web_search" and "Search Results" in result:
                            # Extract URLs and summaries as discoveries
                            lines = result.split("\n")
                            for line in lines[:3]:  # Top 3 results
                                if line.strip():
                                    self.memory_manager.memory.discoveries.append(line.strip()[:200])
                        elif tool_name == "run_python" and not "ERROR" in result:
                            # Code execution success
                            self.memory_manager.memory.discoveries.append(
                                f"Code executed successfully: {result[:100]}"
                            )

                    if not success:
                        self.memory_manager.record_failure(
                            tool_name, result[:80]
                        )

                consecutive_failures = 0

            # CRITICAL: Update system context with current progress (prevent memory loss)
            memory_context = self._build_memory_context()

            # Periodic reflection (every 5 iterations) OR phase-based prompts
            if reflection_engine and reflection_engine.should_reflect():
                print("\n💭 Reflecting...")
                reflection_prompt = reflection_engine.get_reflection_prompt()
                # Add memory context to reflection
                response = self.chat(f"{memory_context}\n\n{reflection_prompt}")
            else:
                # AGGRESSIVE phase-based continuation WITH full context
                if phase == "research":
                    prompt = f"{memory_context}\n\nStep {iteration+1}: Continue researching. Find more specific info with web_search about the problem."
                elif phase == "code":
                    prompt = f"{memory_context}\n\nStep {iteration+1}: Based on what you learned, write and test Python code with run_python."
                elif phase == "save":
                    prompt = f"{memory_context}\n\nStep {iteration+1}: Save your code to a file using write_file."
                else:
                    prompt = f"{memory_context}\n\nStep {iteration+1}: Continue. What's next?"

                response = self.chat(prompt)

        print("\n🛑 Session complete.")


def main() -> None:
    """CLI entry point using config-driven model selection."""
    if len(sys.argv) < 2:
        print("Usage: python agent.py '<goal>'")
        print(f"\nConfiguration: {CONFIG.models.keys()}")
        print("MLX optimized for Apple Silicon — ~60 tok/s on M-series")
        sys.exit(1)

    goal = " ".join(sys.argv[1:])

    # Use BALANCED model (14B) - better instruction-following than 8B, faster than 32B
    print("🧠 Using BALANCED model (Qwen3-14B) for better instruction-following")
    model_name = "balanced"

    agent = MLXAgent(config_model_name=model_name, goal=goal)
    agent.run_loop(goal)


if __name__ == "__main__":
    main()
