"""
SkillTree v3 – Autonomous Self-Improving Brain.

═══════════════════════════════════════════════════════════════════════════
SQLite persistence + networkx DiGraph + sentence-transformer embeddings.
UCB1 bandit scoring. LLM-driven proposal generation. Fully automated
implement → test → integrate loops. The agent grows its own brain.
═══════════════════════════════════════════════════════════════════════════
"""

import json
import math
import os
import re
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

import networkx as nx

from src.paths import SKILLS_DIR, RUNS_DIR
from src.runtime.llm_text import extract_python_code_block, strip_thinking_tags
from src.runtime.verifier import validate_generated_module


# ═══════════════════════════════════════════════════════════════════════════
# SEED DATA (first-run migration only)
# ═══════════════════════════════════════════════════════════════════════════

SEED_SKILLS = [
    {"id": "search_cache", "name": "Search Result Cache", "file": "search_cache.py", "tier": 1, "impact": 7, "prereqs": [], "domain": "memory",
     "description": "Cache web search results with TTL", "spec": "Class SearchCache with get/set/cleanup/stats.", "test_hint": "Set values, verify hits/misses, verify expiry"},
    {"id": "metrics", "name": "Performance Metrics", "file": "metrics.py", "tier": 1, "impact": 6, "prereqs": [], "domain": "observability",
     "description": "Track tool calls, success rates, timing", "spec": "Class AgentMetrics with record_step/summary/report.", "test_hint": "Simulate 20 steps, verify counts"},
    {"id": "memory_compressor", "name": "Memory Compression", "file": "memory_compressor.py", "tier": 1, "impact": 6, "prereqs": [], "domain": "memory",
     "description": "Compress old session iterations", "spec": "Function compress_session(iterations, keep_recent=10).", "test_hint": "50 iterations compressed to <=10"},
    {"id": "error_recovery", "name": "Error Recovery", "file": "error_recovery.py", "tier": 1, "impact": 8, "prereqs": [], "domain": "routing",
     "description": "Retry with backoff and error classification", "spec": "Class ErrorRecovery with classify/suggest/should_retry/backoff.", "test_hint": "Test each error type and retry logic"},
    {"id": "code_validator", "name": "Code Validator", "file": "code_validator.py", "tier": 1, "impact": 9, "prereqs": [], "domain": "self_improvement",
     "description": "Validate Python: syntax, imports, tests", "spec": "Class CodeValidator with check_syntax/check_imports/run_tests/validate_all.", "test_hint": "Valid code, syntax error, missing import"},
    {"id": "loop_detector", "name": "Loop Detector", "file": "loop_detector.py", "tier": 2, "impact": 8, "prereqs": ["metrics"], "domain": "routing",
     "description": "Detect repeated actions with similar results", "spec": "Class LoopDetector with record/is_stuck/suggest_escape/similarity.", "test_hint": "3 identical=stuck, different=not stuck"},
    {"id": "confidence_scorer", "name": "Confidence Scorer", "file": "confidence_scorer.py", "tier": 2, "impact": 9, "prereqs": ["metrics", "error_recovery"], "domain": "planning",
     "description": "Score agent confidence for decisions", "spec": "Class ConfidenceScorer with knowledge/capability/progress/should_act/overall.", "test_hint": "Low=research, high=save, errors=abort"},
    {"id": "result_evaluator", "name": "Result Evaluator", "file": "result_evaluator.py", "tier": 2, "impact": 7, "prereqs": ["search_cache"], "domain": "routing",
     "description": "Evaluate quality of tool results", "spec": "Class ResultEvaluator with score_search/score_code/is_duplicate/summarize.", "test_hint": "Relevant=high, junk=low, duplicate detection"},
    {"id": "task_planner", "name": "Task Planner", "file": "task_planner.py", "tier": 2, "impact": 8, "prereqs": ["error_recovery"], "domain": "planning",
     "description": "Decompose goals into ordered subtasks", "spec": "Class TaskPlanner with decompose/next_task/is_complete/replan.", "test_hint": "Decompose into 5+ steps, verify ordering"},
    {"id": "smart_router", "name": "Smart Tool Router", "file": "smart_router.py", "tier": 3, "impact": 9, "prereqs": ["confidence_scorer", "loop_detector", "task_planner"], "domain": "routing",
     "description": "Pick optimal tool given current state", "spec": "Class SmartRouter with pick_tool/should_change_phase/format_tool_prompt.", "test_hint": "Low confidence=search, loop=phase change"},
    {"id": "self_evaluator", "name": "Self Evaluator", "file": "self_evaluator.py", "tier": 3, "impact": 10, "prereqs": ["code_validator", "result_evaluator", "metrics"], "domain": "self_improvement",
     "description": "Evaluate own output quality before saving", "spec": "Class SelfEvaluator with evaluate_file returning scores and recommendation.", "test_hint": "Good=keep, error=discard, no tests=fix"},
    {"id": "orchestrator", "name": "Agent Orchestrator", "file": "orchestrator.py", "tier": 4, "impact": 10, "prereqs": ["smart_router", "self_evaluator"], "domain": "planning",
     "description": "Master controller wiring all modules", "spec": "Class Orchestrator with plan_cycle/decide_next/evaluate_progress/post_mortem.", "test_hint": "Empty state=plan, done=save, loop=escape"},
    {"id": "strategy_learner", "name": "Strategy Learner", "file": "strategy_learner.py", "tier": 4, "impact": 10, "prereqs": ["orchestrator", "self_evaluator"], "domain": "self_improvement",
     "description": "Learn what strategies work for what tasks", "spec": "Class StrategyLearner with record_outcome/best_strategy/avoid_strategy/recommend.", "test_hint": "20 outcomes, verify best/worst"},
]

# ═══════════════════════════════════════════════════════════════════════════
# Evolution log
# ═══════════════════════════════════════════════════════════════════════════

EVOLUTION_LOG = RUNS_DIR / "skill_tree_evolution.log"


def _evo_log(msg: str):
    """Append to the evolution log."""
    EVOLUTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EVOLUTION_LOG, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


class SkillTree:
    """Autonomous self-improving brain: UCB1 + embeddings + LLM evolution."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(RUNS_DIR / "skill_tree.db")
        self.graph = nx.DiGraph()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # No external embedding model — use stdlib TF-IDF for zero GPU footprint

        self._init_db()
        self._migrate_v3()
        if self._count() == 0:
            self._seed()
        self._load_graph()
        self._scan_completed()
        # Embeddings computed lazily on first similarity request

    # ── Database ──────────────────────────────────────────────────────────

    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA busy_timeout=5000")
        return c

    def _init_db(self):
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY, name TEXT, file TEXT, tier INTEGER,
                    base_impact REAL, current_impact REAL, level INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'locked', description TEXT, spec TEXT,
                    test_hint TEXT, last_updated TEXT, fail_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0, proposed_by TEXT DEFAULT 'system',
                    embedding BLOB, pull_count INTEGER DEFAULT 0,
                    domain TEXT DEFAULT 'general', version TEXT DEFAULT '1.0'
                );
                CREATE TABLE IF NOT EXISTS prereqs (
                    skill_id TEXT, prereq_id TEXT, PRIMARY KEY (skill_id, prereq_id)
                );
                CREATE TABLE IF NOT EXISTS skill_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, skill_id TEXT,
                    event TEXT, timestamp TEXT, details TEXT
                );
            """)

    def _migrate_v3(self):
        """Add v3 columns if upgrading from v2 DB."""
        with self._conn() as c:
            cols = {row[1] for row in c.execute("PRAGMA table_info(skills)")}
            migrations = {
                "embedding": "ALTER TABLE skills ADD COLUMN embedding BLOB",
                "pull_count": "ALTER TABLE skills ADD COLUMN pull_count INTEGER DEFAULT 0",
                "domain": "ALTER TABLE skills ADD COLUMN domain TEXT DEFAULT 'general'",
                "version": "ALTER TABLE skills ADD COLUMN version TEXT DEFAULT '1.0'",
                "last_fail_reason": "ALTER TABLE skills ADD COLUMN last_fail_reason TEXT DEFAULT ''",
            }
            for col, sql in migrations.items():
                if col not in cols:
                    c.execute(sql)
                    _evo_log(f"MIGRATION: Added column {col}")

    def _count(self):
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM skills").fetchone()[0]

    def _seed(self):
        for s in SEED_SKILLS:
            self.add_skill(s)

    def _load_graph(self):
        self.graph.clear()
        with self._conn() as c:
            for r in c.execute("SELECT * FROM skills"):
                self.graph.add_node(r["id"], **dict(r))
            for r in c.execute("SELECT * FROM prereqs"):
                self.graph.add_edge(r["prereq_id"], r["skill_id"])

    def _scan_completed(self):
        """Detect passing skill files on disk with rigorous validation."""
        for nid in list(self.graph.nodes):
            n = self.graph.nodes[nid]
            if n.get("status") == "completed":
                continue
            fpath = SKILLS_DIR / (self._field(nid, "file") or "")
            if not fpath.exists() or fpath.stat().st_size < 200:
                continue
            try:
                accepted, summary = validate_generated_module(str(fpath), skill_tree=self)
                if accepted:
                    self.mark_completed(nid, summary or "auto-detected")
            except Exception:
                pass

    def _log(self, skill_id, event, details=""):
        with self._conn() as c:
            c.execute("INSERT INTO skill_history (skill_id, event, timestamp, details) VALUES (?,?,?,?)",
                      (skill_id, event, datetime.now().isoformat(), details))

    def _field(self, sid, field):
        with self._conn() as c:
            r = c.execute(f"SELECT [{field}] FROM skills WHERE id=?", (sid,)).fetchone()
            return r[field] if r else None

    def _status(self, sid):
        return self._field(sid, "status") or "locked"

    def _skill_dict(self, sid):
        with self._conn() as c:
            r = c.execute("SELECT * FROM skills WHERE id=?", (sid,)).fetchone()
            if not r:
                return None
            prereqs = [p["prereq_id"] for p in c.execute("SELECT prereq_id FROM prereqs WHERE skill_id=?", (sid,))]
            return {**dict(r), "prereqs": prereqs}

    def _save_node(self, sid):
        """Persist a single graph node's pull_count back to DB."""
        n = self.graph.nodes.get(sid)
        if not n:
            return
        with self._conn() as c:
            c.execute("UPDATE skills SET pull_count=? WHERE id=?", (n.get("pull_count", 0), sid))

    # ── Semantic Layer (pure Python, zero GPU) ──────────────────────────

    def _tokenize(self, text: str) -> set:
        """Simple word tokenization for similarity."""
        return set(re.findall(r'\w+', text.lower()))

    def get_skill_similarity(self, skill1: str, skill2: str) -> float:
        """Jaccard similarity between skill descriptions (no GPU needed)."""
        d1 = self._field(skill1, "description") or ""
        d2 = self._field(skill2, "description") or ""
        s1 = self._field(skill1, "spec") or ""
        s2 = self._field(skill2, "spec") or ""
        t1 = self._tokenize(f"{d1} {s1}")
        t2 = self._tokenize(f"{d2} {s2}")
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)

    # ── Core API ──────────────────────────────────────────────────────────

    def add_skill(self, skill):
        """Add or update a skill with cycle detection + embedding."""
        sid, prereqs = skill["id"], skill.get("prereqs", [])
        g = self.graph.copy()
        g.add_node(sid)
        for p in prereqs:
            g.add_edge(p, sid)
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError(f"Cycle detected adding {sid}")

        domain = skill.get("domain", "general")
        with self._conn() as c:
            c.execute("""INSERT OR REPLACE INTO skills
                (id,name,file,tier,base_impact,current_impact,status,description,spec,test_hint,
                 last_updated,proposed_by,domain,version,pull_count)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sid, skill["name"], skill["file"], skill["tier"], skill["impact"], skill["impact"],
                 "locked", skill.get("description", ""), skill.get("spec", ""), skill.get("test_hint", ""),
                 datetime.now().isoformat(), skill.get("proposed_by", "system"),
                 domain, skill.get("version", "1.0"), 0))
            c.execute("DELETE FROM prereqs WHERE skill_id=?", (sid,))
            for p in prereqs:
                c.execute("INSERT OR IGNORE INTO prereqs VALUES (?,?)", (sid, p))

        self.graph.add_node(sid, **{**skill, "status": "locked", "current_impact": skill["impact"],
                                     "pull_count": 0, "fail_count": 0, "domain": domain})
        for p in prereqs:
            self.graph.add_edge(p, sid)

        # Embedding computed lazily (avoid loading model during seed/init)
        return sid

    def get_weakest_skill(self) -> Optional[dict]:
        """Find the weakest completed skill that needs upgrading."""
        weakest = None
        worst_score = float("inf")
        for nid in self.graph.nodes:
            if self._status(nid) != "completed":
                continue
            fpath = SKILLS_DIR / (self._field(nid, "file") or "")
            if not fpath.exists():
                continue
            source = fpath.read_text()
            lines = len(source.splitlines())
            funcs = source.count("\ndef ") + source.count("\n    def ")
            asserts = source.count("assert ")
            # Quality score: lines + 10*asserts + 5*funcs
            score = lines + 10 * asserts + 5 * funcs
            if score < worst_score:
                worst_score = score
                weakest = {**self.graph.nodes[nid], "id": nid, "quality_score": score}
        return weakest

    def build_upgrade_goal(self, skill) -> str:
        """Build a prompt to upgrade an existing skill to production quality."""
        fpath = SKILLS_DIR / skill["file"]
        current_code = fpath.read_text() if fpath.exists() else "(file missing)"
        prereq_code = self._prereq_code(skill)

        # Analyze current code to give specific feedback
        import ast as _ast
        try:
            tree = _ast.parse(current_code)
            func_count = sum(1 for n in _ast.walk(tree) if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)))
            class_count = sum(1 for n in _ast.walk(tree) if isinstance(n, _ast.ClassDef))
        except Exception:
            func_count = class_count = 0
        assert_count = current_code.count("assert ")
        loc = len(current_code.splitlines())

        specific_issues = []
        if func_count < 3:
            specific_issues.append(f"Only {func_count} functions — need ≥3 to pass validation")
        if assert_count < 5:
            specific_issues.append(f"Only {assert_count} assertions — need ≥5 to pass validation")
        if loc < 80:
            specific_issues.append(f"Only {loc} lines — too shallow for production quality")
        if class_count == 0:
            specific_issues.append("No class defined — should use a class for encapsulation")
        if "from src." in current_code:
            specific_issues.append("Contains 'from src.' imports — skills must be standalone")

        issues_text = "\n".join(f"- {i}" for i in specific_issues) if specific_issues else "- No specific structural issues (focus on algorithmic depth)"

        return f"""You are UPGRADING an existing module to production quality. The current version works but is weak.

=== CURRENT CODE (upgrade this) ===
{current_code}

=== CURRENT METRICS ===
Functions: {func_count} | Assertions: {assert_count} | Lines: {loc} | Classes: {class_count}

=== SPECIFIC ISSUES TO FIX ===
{issues_text}

=== VALIDATION GATE (your code MUST pass these or it gets deleted) ===
1. MINIMUM 3 functions or methods
2. MINIMUM 5 assert statements in the test block
3. No 'from src.' imports
4. Must print 'ALL TESTS PASSED'

=== UPGRADE REQUIREMENTS ===
1. Keep the same class name and method signatures (don't break API)
2. Make every method substantive: real logic, not pass-through
3. Add proper error handling for edge cases (empty input, None, invalid types)
4. Add type hints and docstrings to all public methods
5. Upgrade tests: at least 8 assertions testing real behavior and edge cases
6. Use prereq modules where they add value

=== AVAILABLE PREREQ APIs ===
{prereq_code}

=== IMPORT RULES ===
- from <module_name> import <ClassName> for prereqs
- NEVER import from src.memory, src.config, or src.agent
- stdlib only + prereq imports

Write the UPGRADED version to {skill['file']}. Print 'ALL TESTS PASSED'. Say DONE when passing."""

    def get_next_skill(self) -> Optional[dict]:
        """UCB1 bandit – balances high-impact skills with exploration of under-tested ones."""
        if not self.graph.nodes:
            return None

        N = sum(self.graph.nodes[n].get("pull_count", 0) for n in self.graph.nodes) + 1
        c = 1.41  # standard exploration constant

        best_score = -float("inf")
        best_node = None

        for nid in self.graph.nodes:
            if self._status(nid) == "completed" or not self.is_unlocked(nid):
                continue
            data = self.graph.nodes[nid]
            n = data.get("pull_count", 0) + 1e-6
            r = self._field(nid, "current_impact") or data.get("impact", 5)
            ucb = r + c * math.sqrt(math.log(N) / n)

            # Penalize skills that keep failing (prevents infinite retry loops)
            fail_count = self._field(nid, "fail_count") or 0
            success_count = self._field(nid, "success_count") or 0
            if fail_count > 2:
                failure_rate = fail_count / max(1, fail_count + success_count)
                ucb -= failure_rate * 3.0  # Heavy penalty for repeated failures

            if ucb > best_score:
                best_score = ucb
                best_node = nid

        if best_node:
            self.graph.nodes[best_node]["pull_count"] = self.graph.nodes[best_node].get("pull_count", 0) + 1
            self._save_node(best_node)

        return self._skill_dict(best_node) if best_node else None

    def is_unlocked(self, sid):
        return all(self._status(p) == "completed" for p in self.graph.predecessors(sid))

    def mark_completed(self, sid, output=""):
        with self._conn() as c:
            c.execute("UPDATE skills SET status='completed', success_count=success_count+1, level=level+1, last_updated=? WHERE id=?",
                      (datetime.now().isoformat(), sid))
        if sid in self.graph.nodes:
            self.graph.nodes[sid]["status"] = "completed"
        self._log(sid, "completed", output[:200])
        self._propagate(sid, 0.3)

    def mark_failed(self, sid, error=""):
        with self._conn() as c:
            c.execute("UPDATE skills SET fail_count=fail_count+1, last_fail_reason=?, last_updated=? WHERE id=?",
                      (error[:300], datetime.now().isoformat(), sid))
        self._log(sid, "failed", error[:200])

    def update_impact_from_result(self, sid, gain):
        with self._conn() as c:
            r = c.execute("SELECT current_impact FROM skills WHERE id=?", (sid,)).fetchone()
            if not r:
                return
            new = min(10.0, r["current_impact"] + gain * 3.0)
            c.execute("UPDATE skills SET current_impact=? WHERE id=?", (new, sid))
        self._log(sid, "impact_change", f"gain={gain}")
        self._propagate(sid, gain * 0.3)

    def _propagate(self, sid, gain):
        if abs(gain) < 0.01:
            return
        for s in self.graph.successors(sid):
            with self._conn() as c:
                r = c.execute("SELECT current_impact FROM skills WHERE id=?", (s,)).fetchone()
                if r:
                    c.execute("UPDATE skills SET current_impact=? WHERE id=?", (min(10.0, r["current_impact"] + gain), s))

    # ── Intelligence ──────────────────────────────────────────────────────

    def get_critical_path(self):
        try:
            return nx.dag_longest_path(self.graph)
        except Exception:
            return []

    def get_system_weaknesses(self):
        lines = []
        with self._conn() as c:
            for r in c.execute("SELECT name, fail_count FROM skills WHERE fail_count>0 ORDER BY fail_count DESC LIMIT 3"):
                lines.append(f"Struggling: {r['name']} ({r['fail_count']} fails)")
            path = self.get_critical_path()
            blockers = [n for n in path if self._status(n) != "completed"]
            if blockers:
                lines.append(f"Blockers: {', '.join(blockers)}")
            for r in c.execute("SELECT name, base_impact, current_impact FROM skills WHERE ABS(current_impact-base_impact)>1"):
                d = "more" if r["current_impact"] > r["base_impact"] else "less"
                lines.append(f"Learned: {r['name']} is {d} important ({r['base_impact']:.0f}→{r['current_impact']:.1f})")
        return "\n".join(lines) if lines else "System is healthy"

    # ── Evolution: LLM-driven self-growth ─────────────────────────────────

    def evolve_tree(self):
        """Parse text-file proposals (v2 compat)."""
        f = SKILLS_DIR / "tree_proposals.txt"
        if not f.exists():
            return []
        added = []
        for line in f.read_text().split("\n"):
            if not line.startswith("SKILL:"):
                continue
            parts = {}
            for seg in line.split("|"):
                if ":" in seg:
                    k, v = seg.split(":", 1)
                    parts[k.strip().lower()] = v.strip()
            if "skill" in parts and "file" in parts:
                sid = parts["skill"].lower().replace(" ", "_")
                try:
                    self.add_skill({
                        "id": sid, "name": parts["skill"], "file": parts["file"],
                        "tier": int(parts.get("tier", 2)), "impact": float(parts.get("impact", 7)),
                        "prereqs": [p.strip() for p in parts.get("prereqs", "").split(",") if p.strip()],
                        "description": parts.get("desc", ""), "spec": parts.get("spec", "Build with tests."),
                        "test_hint": "ALL TESTS PASSED", "proposed_by": "agent",
                        "domain": parts.get("domain", "general"),
                    })
                    added.append(sid)
                except ValueError:
                    pass
        if added:
            self._load_graph()
        return added

    def generate_proposals(self, num_proposals: int = 3) -> list:
        """Use MLX model to propose NEW skills based on system state."""
        from src.config import CONFIG
        try:
            from mlx_lm import load, generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError:
            _evo_log("PROPOSAL FAILED: mlx_lm not available")
            return []

        path = self.get_critical_path()
        weak = self.get_system_weaknesses()
        state = self.get_state()

        prompt_text = f"""You are an expert AI systems architect. Here is the CURRENT skill tree state:

Critical path: {path}
Top weaknesses: {weak}
Recent failures: {sum(s.get('fail_count', 0) for s in state['skills'])} total fails
Total skills: {state['total']} | Completed: {state['completed']} | Avg impact: {sum(s.get('current_impact', 0) for s in state['skills']) / max(1, state['total']):.1f}

Propose exactly {num_proposals} NEW skills that would unlock massive capability.
Each proposal must be valid JSON object with:
{{
  "name": "...",
  "description": "...",
  "prereqs": ["existing_id1", "existing_id2"],
  "tier": 2,
  "estimated_impact": 8,
  "test_hint": "...",
  "domain": "routing|memory|planning|self_improvement"
}}

Output ONLY a JSON array. No explanation."""

        try:
            model_cfg = CONFIG.models.get("balanced") or CONFIG.models.get("tool_calling")
            model, tokenizer = load(model_cfg.name)
            sampler = make_sampler(temp=0.3)

            messages = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            response = generate(model, tokenizer, prompt=formatted, max_tokens=2048, sampler=sampler)

            response = strip_thinking_tags(response)

            # Extract JSON array
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if not match:
                _evo_log(f"PROPOSAL FAILED: No JSON array in response")
                return []

            proposals = json.loads(match.group())
            valid = []
            existing_ids = set(self.graph.nodes)

            for p in proposals:
                if not isinstance(p, dict) or "name" not in p:
                    continue
                # Validate prereqs exist
                prereqs = p.get("prereqs", [])
                if not all(pr in existing_ids for pr in prereqs):
                    continue
                sid = p["name"].lower().replace(" ", "_").replace("-", "_")
                if sid in existing_ids:
                    continue
                valid.append({
                    "id": sid,
                    "name": p["name"],
                    "file": f"{sid}.py",
                    "tier": p.get("tier", 2),
                    "impact": p.get("estimated_impact", 7),
                    "prereqs": prereqs,
                    "description": p.get("description", ""),
                    "spec": p.get("description", ""),
                    "test_hint": p.get("test_hint", "ALL TESTS PASSED"),
                    "domain": p.get("domain", "general"),
                    "proposed_by": "agent",
                })

            _evo_log(f"PROPOSALS: Generated {len(valid)} valid from {len(proposals)} raw")
            del model, tokenizer  # Free GPU memory
            return valid

        except Exception as e:
            _evo_log(f"PROPOSAL ERROR: {e}")
            return []

    def auto_implement(self, proposal: dict) -> bool:
        """Generate code for a proposal, test it, integrate if passing."""
        from src.config import CONFIG
        try:
            from mlx_lm import load, generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError:
            return False

        sid = proposal["id"]
        target = SKILLS_DIR / proposal["file"]

        prompt_text = f"""Write a complete Python module for: {proposal['name']}

Description: {proposal.get('description', '')}
Spec: {proposal.get('spec', '')}
Test requirements: {proposal.get('test_hint', '')}

The module MUST:
1. Be standalone (no importing agent.py or loading MLX models)
2. Include 'if __name__ == "__main__":' test block
3. Print 'ALL TESTS PASSED' when all tests pass
4. Use type hints and docstrings

Output ONLY the Python code. No explanation."""

        try:
            model_cfg = CONFIG.models.get("balanced") or CONFIG.models.get("tool_calling")
            model, tokenizer = load(model_cfg.name)
            sampler = make_sampler(temp=0.2)

            messages = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            response = generate(model, tokenizer, prompt=formatted, max_tokens=4096, sampler=sampler)
            response = strip_thinking_tags(response)

            # Extract code from markdown blocks or raw
            code = extract_python_code_block(response) or response

            # Write file
            target.write_text(code)
            _evo_log(f"IMPLEMENT: Wrote {len(code)} bytes to {target}")

            # Test it
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(".").resolve()) + ":" + str(SKILLS_DIR)
            result = subprocess.run(
                ["python3", str(target)],
                capture_output=True, text=True, timeout=30, env=env,
            )

            del model, tokenizer  # Free GPU memory

            if (result.returncode == 0
                    and "ALL TESTS PASSED" in result.stdout
                    and "Traceback" not in result.stderr
                    and len(code) >= 200
                    and ("def " in code or "class " in code)):
                self.add_skill(proposal)
                self.mark_completed(sid, "auto-implemented")
                _evo_log(f"IMPLEMENT SUCCESS: {sid} passes tests")
                return True
            else:
                target.unlink(missing_ok=True)
                _evo_log(f"IMPLEMENT FAILED: {sid} - {(result.stdout + result.stderr)[:200]}")
                return False

        except Exception as e:
            target.unlink(missing_ok=True)
            _evo_log(f"IMPLEMENT ERROR: {sid} - {e}")
            return False

    def evolve(self, iterations: int = 1):
        """Main evolution loop: propose → implement → integrate."""
        _evo_log(f"EVOLVE START: {iterations} iterations")
        total_added = 0

        for i in range(iterations):
            proposals = self.generate_proposals(num_proposals=3)
            if not proposals:
                _evo_log(f"EVOLVE ITER {i+1}: No proposals generated")
                continue

            for p in proposals:
                ok = self.auto_implement(p)
                if ok:
                    total_added += 1
                    _evo_log(f"EVOLVE ITER {i+1}: Added {p['id']}")

        self._load_graph()
        _evo_log(f"EVOLVE END: Added {total_added} skills across {iterations} iterations")
        return total_added

    # ── Goal building ─────────────────────────────────────────────────────

    def build_goal_for_skill(self, skill):
        prereq_code = self._prereq_code(skill)
        code_list = self._codebase()
        tnames = {1: "Foundation", 2: "Intelligence", 3: "Integration", 4: "Optimization"}

        # Build context about what this skill needs to integrate with
        prereq_names = []
        for pid in self.graph.predecessors(skill["id"]):
            pname = self._field(pid, "name")
            pfile = self._field(pid, "file")
            if pname:
                prereq_names.append(f"{pname} ({pfile})")

        # Get downstream skills that depend on this one
        dependents = []
        for sid in self.graph.successors(skill["id"]):
            sname = self._field(sid, "name")
            if sname:
                dependents.append(sname)

        # Get last failure reason if any
        fail_count = self._field(skill["id"], "fail_count") or 0
        last_fail = self._field(skill["id"], "last_fail_reason") or ""
        fail_warning = ""
        if fail_count > 0 and last_fail:
            fail_warning = f"""
=== PREVIOUS FAILURE (attempt #{fail_count}) ===
REASON: {last_fail}
FIX THIS SPECIFIC ISSUE. Do not repeat the same mistake.
"""

        return f"""You are building a production-quality Python module for a self-improving AI agent system.
{fail_warning}
=== TASK ===
Module: {skill['name']} (Tier {skill['tier']}: {tnames.get(skill['tier'],'?')})
File: {skill['file']}
Impact: {skill.get('current_impact', skill.get('impact','?'))}/10

=== WHAT THIS MODULE DOES ===
{skill.get('description','')}

=== DETAILED SPECIFICATION ===
{skill.get('spec','')}

=== WHY IT MATTERS ===
This module is depended on by: {', '.join(dependents) if dependents else 'nothing yet (leaf node)'}
It builds on: {', '.join(prereq_names) if prereq_names else 'nothing (foundation skill)'}

=== VALIDATION GATE (your code MUST pass these or it gets deleted) ===
1. MINIMUM 3 functions or methods (not counting __init__ or test code)
2. MINIMUM 5 assert statements in the test block
3. No 'from src.' imports — skills must be standalone
4. Must print 'ALL TESTS PASSED' when run
5. Each method must have real logic (10+ lines), not single-line returns
6. Handle edge cases: empty inputs, None values, boundary conditions

=== AVAILABLE PREREQ APIs ===
{prereq_code}

=== IMPORT RULES (CRITICAL) ===
- Import prereqs: from <module_name> import <ClassName>
- NEVER import from src.memory, src.config, or src.agent
- Use ONLY stdlib + prereq imports listed above
- If a prereq import fails, catch ImportError and provide a standalone fallback

=== TEST REQUIREMENTS ===
{skill.get('test_hint','')}
- Include 'if __name__ == "__main__":' block
- At least 5 test cases with assertions
- Print 'ALL TESTS PASSED' only if ALL assertions pass
- Test edge cases (empty input, boundary values)

=== EXISTING MODULES ===
{code_list}

Write the complete module to {skill['file']}. Use read_file if you need to understand any prereq. Say DONE when tests pass."""

    def _codebase(self, budget=1500):
        """Minimal codebase context - only prereq files get full content."""
        lines = []
        # Just list what exists, don't dump contents
        for f in sorted(SKILLS_DIR.glob("*.py")):
            if not f.name.startswith("_"):
                loc = len(f.read_text().splitlines())
                lines.append(f"  skills/{f.name} ({loc}L)")
        lines.insert(0, "Existing skills:")
        return "\n".join(lines)

    def _prereq_code(self, skill):
        """Extract only class/function signatures from prereq files (saves tokens)."""
        parts = []
        for pid in self.graph.predecessors(skill["id"]):
            fname = self._field(pid, "file")
            f = SKILLS_DIR / fname
            if not f.exists():
                continue
            # Extract class names and method signatures only
            lines = []
            for line in f.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith(("class ", "def ", "    def ")):
                    lines.append(line)
                elif stripped.startswith(("from ", "import ")):
                    lines.append(line)
            stem = Path(fname).stem
            parts.append(f"# {stem} — import with: from {stem} import *\n" + "\n".join(lines))
        return "\n".join(parts) if parts else "(no prereqs)"

    def _tree_text(self):
        lines = []
        tnames = {1: "FOUNDATION", 2: "INTELLIGENCE", 3: "INTEGRATION", 4: "OPTIMIZATION"}
        with self._conn() as c:
            for t in [1, 2, 3, 4]:
                lines.append(f"\nTIER {t}: {tnames.get(t, '?')}")
                for r in c.execute("SELECT * FROM skills WHERE tier=? ORDER BY current_impact DESC", (t,)):
                    ps = [p["prereq_id"] for p in c.execute("SELECT prereq_id FROM prereqs WHERE skill_id=?", (r["id"],))]
                    deps = f" (needs:{','.join(ps)})" if ps else ""
                    lines.append(f"  [{r['status'].upper()}] {r['name']} -> {r['file']} (impact:{r['current_impact']:.1f} domain:{r['domain']}){deps}")
        return "\n".join(lines)

    # ── Visualization ─────────────────────────────────────────────────────

    def export_to_dot(self) -> str:
        """Export graph as DOT format for Graphviz."""
        lines = ["digraph SkillTree {", '  rankdir=TB;', '  node [shape=box, style=filled];']
        colors = {"completed": "#90EE90", "locked": "#D3D3D3", "failing": "#FFB6C1"}
        with self._conn() as c:
            for r in c.execute("SELECT * FROM skills"):
                color = colors.get(r["status"], "#FFFACD")
                lines.append(f'  {r["id"]} [label="{r["name"]}\\nT{r["tier"]} | {r["current_impact"]:.1f}", fillcolor="{color}"];')
            for r in c.execute("SELECT * FROM prereqs"):
                lines.append(f'  {r["prereq_id"]} -> {r["skill_id"]};')
        lines.append("}")
        return "\n".join(lines)

    def print_tree(self):
        tnames = {1: "FOUNDATION", 2: "INTELLIGENCE", 3: "INTEGRATION", 4: "OPTIMIZATION"}
        with self._conn() as c:
            for t in [1, 2, 3, 4]:
                print(f"\n{'─'*40}\nTIER {t}: {tnames.get(t, '?')}\n{'─'*40}")
                for r in c.execute("SELECT * FROM skills WHERE tier=? ORDER BY current_impact DESC", (t,)):
                    s = r["status"]
                    icon = {"completed": "✅", "failing": "❌"}.get(s, "⬜" if self.is_unlocked(r["id"]) else "🔒")
                    ps = [p["prereq_id"] for p in c.execute("SELECT prereq_id FROM prereqs WHERE skill_id=?", (r["id"],))]
                    deps = f" (needs:{','.join(ps)})" if ps else ""
                    lv = f" Lv{r['level']}" if r['level'] > 1 else ""
                    fl = f" [{r['fail_count']}F]" if r['fail_count'] > 0 else ""
                    dom = f" [{r['domain']}]" if r['domain'] != 'general' else ""
                    pulls = f" pulls:{r['pull_count']}" if r['pull_count'] > 0 else ""
                    print(f"  {icon} [{r['current_impact']:.1f}] {r['name']}{dom}{lv}{fl}{pulls}{deps}")
        nxt = self.get_next_skill()
        print(f"\n→ NEXT: {nxt['name']} ({nxt['file']})" if nxt else "\n→ ALL COMPLETE!")
        w = self.get_system_weaknesses()
        if w != "System is healthy":
            print(f"\n⚠ {w}")

    def get_state(self):
        with self._conn() as c:
            skills = []
            for r in c.execute("SELECT * FROM skills ORDER BY tier, current_impact DESC"):
                ps = [p["prereq_id"] for p in c.execute("SELECT prereq_id FROM prereqs WHERE skill_id=?", (r["id"],))]
                skills.append({**dict(r), "prereqs": ps, "unlocked": self.is_unlocked(r["id"])})
            return {"skills": skills, "total": len(skills),
                    "completed": sum(1 for s in skills if s["status"] == "completed"),
                    "next": self.get_next_skill(), "weaknesses": self.get_system_weaknesses()}


if __name__ == "__main__":
    SkillTree().print_tree()
