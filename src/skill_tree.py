"""
SkillTree v2: Self-learning, self-growing agent brain.

Backed by SQLite for persistence + networkx DiGraph for DAG operations.
The agent can propose new skills, and impact scores evolve based on results.
"""

import json
import os
import sqlite3
import subprocess
import heapq
from pathlib import Path
from datetime import datetime
from typing import Optional

import networkx as nx

from src.paths import SKILLS_DIR, RUNS_DIR


# ═══════════════════════════════════════════════════════════════════════════
# SEED DATA: Initial skills (used for first-run migration only)
# ═══════════════════════════════════════════════════════════════════════════

SEED_SKILLS = [
    {"id": "search_cache", "name": "Search Result Cache", "file": "search_cache.py", "tier": 1, "impact": 7, "prereqs": [],
     "description": "Cache web search results with TTL", "spec": "Class SearchCache with get/set/cleanup/stats.", "test_hint": "Set values, verify hits/misses, verify expiry"},
    {"id": "metrics", "name": "Performance Metrics", "file": "metrics.py", "tier": 1, "impact": 6, "prereqs": [],
     "description": "Track tool calls, success rates, timing", "spec": "Class AgentMetrics with record_step/summary/report.", "test_hint": "Simulate 20 steps, verify counts"},
    {"id": "memory_compressor", "name": "Memory Compression", "file": "memory_compressor.py", "tier": 1, "impact": 6, "prereqs": [],
     "description": "Compress old session iterations", "spec": "Function compress_session(iterations, keep_recent=10).", "test_hint": "50 iterations compressed to <=10"},
    {"id": "error_recovery", "name": "Error Recovery", "file": "error_recovery.py", "tier": 1, "impact": 8, "prereqs": [],
     "description": "Retry with backoff and error classification", "spec": "Class ErrorRecovery with classify/suggest/should_retry/backoff.", "test_hint": "Test each error type and retry logic"},
    {"id": "code_validator", "name": "Code Validator", "file": "code_validator.py", "tier": 1, "impact": 9, "prereqs": [],
     "description": "Validate Python: syntax, imports, tests", "spec": "Class CodeValidator with check_syntax/check_imports/run_tests/validate_all.", "test_hint": "Valid code, syntax error, missing import"},
    {"id": "loop_detector", "name": "Loop Detector", "file": "loop_detector.py", "tier": 2, "impact": 8, "prereqs": ["metrics"],
     "description": "Detect repeated actions with similar results", "spec": "Class LoopDetector with record/is_stuck/suggest_escape/similarity.", "test_hint": "3 identical=stuck, different=not stuck"},
    {"id": "confidence_scorer", "name": "Confidence Scorer", "file": "confidence_scorer.py", "tier": 2, "impact": 9, "prereqs": ["metrics", "error_recovery"],
     "description": "Score agent confidence for decisions", "spec": "Class ConfidenceScorer with knowledge/capability/progress/should_act/overall.", "test_hint": "Low=research, high=save, errors=abort"},
    {"id": "result_evaluator", "name": "Result Evaluator", "file": "result_evaluator.py", "tier": 2, "impact": 7, "prereqs": ["search_cache"],
     "description": "Evaluate quality of tool results", "spec": "Class ResultEvaluator with score_search/score_code/is_duplicate/summarize.", "test_hint": "Relevant=high, junk=low, duplicate detection"},
    {"id": "task_planner", "name": "Task Planner", "file": "task_planner.py", "tier": 2, "impact": 8, "prereqs": ["error_recovery"],
     "description": "Decompose goals into ordered subtasks", "spec": "Class TaskPlanner with decompose/next_task/is_complete/replan.", "test_hint": "Decompose into 5+ steps, verify ordering"},
    {"id": "smart_router", "name": "Smart Tool Router", "file": "smart_router.py", "tier": 3, "impact": 9, "prereqs": ["confidence_scorer", "loop_detector", "task_planner"],
     "description": "Pick optimal tool given current state", "spec": "Class SmartRouter with pick_tool/should_change_phase/format_tool_prompt.", "test_hint": "Low confidence=search, loop=phase change"},
    {"id": "self_evaluator", "name": "Self Evaluator", "file": "self_evaluator.py", "tier": 3, "impact": 10, "prereqs": ["code_validator", "result_evaluator", "metrics"],
     "description": "Evaluate own output quality before saving", "spec": "Class SelfEvaluator with evaluate_file returning scores and recommendation.", "test_hint": "Good=keep, error=discard, no tests=fix"},
    {"id": "orchestrator", "name": "Agent Orchestrator", "file": "orchestrator.py", "tier": 4, "impact": 10, "prereqs": ["smart_router", "self_evaluator"],
     "description": "Master controller wiring all modules", "spec": "Class Orchestrator with plan_cycle/decide_next/evaluate_progress/post_mortem.", "test_hint": "Empty state=plan, done=save, loop=escape"},
    {"id": "strategy_learner", "name": "Strategy Learner", "file": "strategy_learner.py", "tier": 4, "impact": 10, "prereqs": ["orchestrator", "self_evaluator"],
     "description": "Learn what strategies work for what tasks", "spec": "Class StrategyLearner with record_outcome/best_strategy/avoid_strategy/recommend.", "test_hint": "20 outcomes, verify best/worst"},
]


class SkillTree:
    """Self-learning, self-growing skill DAG."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(RUNS_DIR / "skill_tree.db")
        self.graph = nx.DiGraph()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        if self._count() == 0:
            self._seed()
        self._load_graph()
        self._scan_completed()

    def _conn(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _init_db(self):
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY, name TEXT, file TEXT, tier INTEGER,
                    base_impact REAL, current_impact REAL, level INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'locked', description TEXT, spec TEXT,
                    test_hint TEXT, last_updated TEXT, fail_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0, proposed_by TEXT DEFAULT 'system'
                );
                CREATE TABLE IF NOT EXISTS prereqs (
                    skill_id TEXT, prereq_id TEXT, PRIMARY KEY (skill_id, prereq_id)
                );
                CREATE TABLE IF NOT EXISTS skill_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, skill_id TEXT,
                    event TEXT, timestamp TEXT, details TEXT
                );
            """)

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
        for nid in self.graph.nodes:
            n = self.graph.nodes[nid]
            if n.get("status") == "completed":
                continue
            fpath = SKILLS_DIR / (self._field(nid, "file") or "")
            if not fpath.exists():
                continue
            try:
                env = os.environ.copy()
                env["PYTHONPATH"] = str(Path(".").resolve()) + ":" + str(SKILLS_DIR)
                r = subprocess.run(["python3", str(fpath)], capture_output=True, text=True, timeout=10, env=env)
                if "PASSED" in r.stdout:
                    self.mark_completed(nid, "auto-detected")
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
            if not r: return None
            prereqs = [p["prereq_id"] for p in c.execute("SELECT prereq_id FROM prereqs WHERE skill_id=?", (sid,))]
            return {**dict(r), "prereqs": prereqs}

    # ── Core API ──────────────────────────────────────────────────────────

    def add_skill(self, skill):
        sid, prereqs = skill["id"], skill.get("prereqs", [])
        g = self.graph.copy()
        g.add_node(sid)
        for p in prereqs:
            g.add_edge(p, sid)
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError(f"Cycle detected adding {sid}")

        with self._conn() as c:
            c.execute("""INSERT OR REPLACE INTO skills
                (id,name,file,tier,base_impact,current_impact,status,description,spec,test_hint,last_updated,proposed_by)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sid, skill["name"], skill["file"], skill["tier"], skill["impact"], skill["impact"],
                 "locked", skill.get("description",""), skill.get("spec",""), skill.get("test_hint",""),
                 datetime.now().isoformat(), skill.get("proposed_by","system")))
            c.execute("DELETE FROM prereqs WHERE skill_id=?", (sid,))
            for p in prereqs:
                c.execute("INSERT OR IGNORE INTO prereqs VALUES (?,?)", (sid, p))

        self.graph.add_node(sid, **skill)
        for p in prereqs:
            self.graph.add_edge(p, sid)
        return sid

    def get_next_skill(self):
        candidates = []
        max_d = max((len(nx.ancestors(self.graph, n)) for n in self.graph.nodes), default=1) or 1

        for nid in self.graph.nodes:
            if self._status(nid) == "completed" or not self.is_unlocked(nid):
                continue
            impact = self._field(nid, "current_impact") or 5
            tier = self._field(nid, "tier") or 1
            depth = len(nx.ancestors(self.graph, nid))
            fails = self._field(nid, "fail_count") or 0
            proposed = self._field(nid, "proposed_by") or "system"

            score = impact * (1.6 ** (tier - 1)) * (1 - depth / (max_d + 1))
            score += 0.5 if proposed == "agent" else 0
            score -= fails * 0.3
            heapq.heappush(candidates, (-score, nid))

        if not candidates:
            return None
        _, best = heapq.heappop(candidates)
        return self._skill_dict(best)

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
            c.execute("UPDATE skills SET fail_count=fail_count+1, last_updated=? WHERE id=?",
                      (datetime.now().isoformat(), sid))
        self._log(sid, "failed", error[:200])

    def update_impact_from_result(self, sid, gain):
        with self._conn() as c:
            r = c.execute("SELECT current_impact FROM skills WHERE id=?", (sid,)).fetchone()
            if not r: return
            new = min(10.0, r["current_impact"] + gain * 3.0)
            c.execute("UPDATE skills SET current_impact=? WHERE id=?", (new, sid))
        self._log(sid, "impact_change", f"gain={gain}")
        self._propagate(sid, gain * 0.3)

    def _propagate(self, sid, gain):
        if abs(gain) < 0.01: return
        for s in self.graph.successors(sid):
            with self._conn() as c:
                r = c.execute("SELECT current_impact FROM skills WHERE id=?", (s,)).fetchone()
                if r:
                    c.execute("UPDATE skills SET current_impact=? WHERE id=?", (min(10.0, r["current_impact"] + gain), s))

    # ── Intelligence ──────────────────────────────────────────────────────

    def get_critical_path(self):
        try: return nx.dag_longest_path(self.graph)
        except: return []

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

    def evolve_tree(self):
        f = SKILLS_DIR / "tree_proposals.txt"
        if not f.exists(): return []
        added = []
        for line in f.read_text().split("\n"):
            if not line.startswith("SKILL:"): continue
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
                    })
                    added.append(sid)
                except ValueError: pass
        if added: self._load_graph()
        return added

    # ── Goal building ─────────────────────────────────────────────────────

    def build_goal_for_skill(self, skill):
        tree = self._tree_text()
        weak = self.get_system_weaknesses()
        code = self._codebase(4000)
        tnames = {1: "Foundation", 2: "Intelligence", 3: "Integration", 4: "Optimization"}
        pfiles = [self._field(p, "file") for p in self.graph.predecessors(skill["id"])]
        pfiles = [f for f in pfiles if f]

        return f"""You are a self-improving AI agent.

=== CODEBASE ===
{code}

=== SKILL TREE ===
{tree}

=== WEAKNESSES ===
{weak}

=== TASK ===
BUILD: {skill['name']} (Tier {skill['tier']}: {tnames.get(skill['tier'],'?')}, Impact: {skill.get('current_impact', skill.get('impact','?'))}/10)
FILE: {skill['file']}
WHY: {skill.get('description','')}

SPEC: {skill.get('spec','')}
TESTS: {skill.get('test_hint','')}

IMPORTS: {chr(10).join(f'from {Path(f).stem} import *' for f in pfiles) if pfiles else '(none)'}

RULES: Write complete code in {skill['file']}. Tests print 'ALL TESTS PASSED'. Don't import agent.py. Say DONE when passing.

PROPOSE NEW SKILLS to skills/tree_proposals.txt: SKILL: name | FILE: x.py | TIER: N | PREREQS: a,b | DESC: ... | SPEC: ..."""

    def _codebase(self, budget=4000):
        lines, used = [], 0
        def add(p, full=True):
            nonlocal used
            f = Path(p)
            if not f.exists(): return
            c = f.read_text()
            est = len(c) // 4
            if not full or used + est > budget:
                loc = len(c.splitlines())
                cls = [l.split("(")[0].replace("class ","").strip() for l in c.splitlines() if l.strip().startswith("class ")]
                fns = [l.split("(")[0].replace("def ","").strip() for l in c.splitlines() if l.strip().startswith("def ")]
                s = f"\n{p} ({loc}L)" + (f" Classes:{','.join(cls[:6])}" if cls else "") + (f" Funcs:{','.join(fns[:8])}" if fns else "")
                lines.append(s); used += len(s)//4
            else:
                lines.append(f"\n--- {p} ---\n{c}"); used += est
        add("src/config.py")
        add("src/memory.py")
        for f in sorted(SKILLS_DIR.glob("*.py")):
            if not f.name.startswith("_"): add(f"skills/{f.name}")
        add("src/agent.py", full=False)
        return "\n".join(lines)

    def _tree_text(self):
        lines = []
        tnames = {1:"FOUNDATION",2:"INTELLIGENCE",3:"INTEGRATION",4:"OPTIMIZATION"}
        with self._conn() as c:
            for t in [1,2,3,4]:
                lines.append(f"\nTIER {t}: {tnames.get(t,'?')}")
                for r in c.execute("SELECT * FROM skills WHERE tier=? ORDER BY current_impact DESC",(t,)):
                    ps = [p["prereq_id"] for p in c.execute("SELECT prereq_id FROM prereqs WHERE skill_id=?",(r["id"],))]
                    deps = f" (needs:{','.join(ps)})" if ps else ""
                    lines.append(f"  [{r['status'].upper()}] {r['name']} -> {r['file']} (impact:{r['current_impact']:.1f}){deps}")
        return "\n".join(lines)

    # ── Display ───────────────────────────────────────────────────────────

    def print_tree(self):
        tnames = {1:"FOUNDATION",2:"INTELLIGENCE",3:"INTEGRATION",4:"OPTIMIZATION"}
        with self._conn() as c:
            for t in [1,2,3,4]:
                print(f"\n{'─'*40}\nTIER {t}: {tnames.get(t,'?')}\n{'─'*40}")
                for r in c.execute("SELECT * FROM skills WHERE tier=? ORDER BY current_impact DESC",(t,)):
                    s = r["status"]
                    icon = {"completed":"✅","failing":"❌"}.get(s, "⬜" if self.is_unlocked(r["id"]) else "🔒")
                    ps = [p["prereq_id"] for p in c.execute("SELECT prereq_id FROM prereqs WHERE skill_id=?",(r["id"],))]
                    deps = f" (needs:{','.join(ps)})" if ps else ""
                    lv = f" Lv{r['level']}" if r['level']>1 else ""
                    fl = f" [{r['fail_count']}F]" if r['fail_count']>0 else ""
                    print(f"  {icon} [{r['current_impact']:.1f}] {r['name']} -> {r['file']}{lv}{fl}{deps}")
        nxt = self.get_next_skill()
        print(f"\n→ NEXT: {nxt['name']} ({nxt['file']})" if nxt else "\n→ ALL COMPLETE!")
        w = self.get_system_weaknesses()
        if w != "System is healthy": print(f"\n⚠ {w}")

    def get_state(self):
        with self._conn() as c:
            skills = []
            for r in c.execute("SELECT * FROM skills ORDER BY tier, current_impact DESC"):
                ps = [p["prereq_id"] for p in c.execute("SELECT prereq_id FROM prereqs WHERE skill_id=?",(r["id"],))]
                skills.append({**dict(r), "prereqs": ps, "unlocked": self.is_unlocked(r["id"])})
            return {"skills": skills, "total": len(skills),
                    "completed": sum(1 for s in skills if s["status"]=="completed"),
                    "next": self.get_next_skill(), "weaknesses": self.get_system_weaknesses()}


if __name__ == "__main__":
    SkillTree().print_tree()
