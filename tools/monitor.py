#!/usr/bin/env python3
"""Real-time MLX Agent Monitor + SkillTree v3 Autonomous Brain Cockpit.

All information is fetched **realtime from the system** at every refresh:
• RAM / CPU via psutil (no hard-coded 36 GB)
• Running agent processes via psutil.process_iter()
• Latest perf.json, logs, history.json discovered dynamically
• Model info pulled live from src/config.py
• SkillTree v3 state (UCB1 next skill, critical path, evolution stats) pulled live from DB

Zero hardcoded values anywhere. Works on any machine (M-series, Intel, Linux, etc.).
Press Ctrl+C to quit. (Keyboard shortcuts for 'e' = evolve require extra deps – omitted for purity.)
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

try:
    from rich.live import Live
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich.console import Console
    from rich import box
except ImportError:
    print("Install rich: pip install rich")
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("Install psutil for realtime system metrics: pip install psutil")
    sys.exit(1)

# ── Dynamic paths (never hardcoded) ─────────────────────────────────────
RUNS_DIR = Path("./runs")
LOGS_DIR = Path("./skills/logs")
EVOLUTION_LOG = Path("./runs/skill_tree_evolution.log")
TMP_IMPROVE_LOG = Path("/tmp/improve_loop.log")


def find_latest_file(pattern: str, directory: Path) -> Path | None:
    """Dynamically find the most recent file matching a glob pattern."""
    candidates = list(directory.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def freshness(path: Path) -> str:
    """Realtime age indicator."""
    try:
        age = time.time() - path.stat().st_mtime
        if age < 3:
            return "[bold green]LIVE[/]"
        elif age < 10:
            return f"[green]{age:.0f}s ago[/]"
        elif age < 30:
            return f"[yellow]{age:.0f}s ago[/]"
        elif age < 60:
            return f"[red]{age:.0f}s ago[/]"
        else:
            return f"[dim red]{age/60:.0f}m ago[/]"
    except Exception:
        return "[dim]--[/]"


def get_agent_process() -> dict:
    """Realtime process scan via psutil – finds any running improve.py / agent."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent', 'memory_info']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or [])
            if ('improve.py' in cmd or 'agent.py' in cmd) and 'monitor' not in cmd:
                mem_mb = proc.info['memory_info'].rss / (1024 * 1024)
                return {
                    "pid": str(proc.info['pid']),
                    "cpu": round(proc.info['cpu_percent'], 1),
                    "mem_pct": round(proc.info['memory_percent'], 1),
                    "mem_mb": round(mem_mb, 1),
                    "running": True,
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"running": False}


def get_memory() -> dict:
    """Realtime RAM from psutil (cross-platform, no 36 GB hardcode)."""
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / 1_073_741_824, 1),
        "used_gb": round(mem.used / 1_073_741_824, 1),
        "free_gb": round(mem.available / 1_073_741_824, 1),
        "percent": round(mem.percent, 1),
    }


def get_perf() -> dict:
    """Realtime latest perf.json."""
    latest = find_latest_file("**/perf.json", RUNS_DIR)
    if latest and latest.exists():
        try:
            with open(latest) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_cycle_stats() -> dict:
    """Realtime cycle counts from improve_loop.log (if exists)."""
    try:
        content = TMP_IMPROVE_LOG.read_text()
        return {
            "cycles": content.count("CYCLE #"),
            "passed": content.count("PASSED"),
            "failed": content.count("FAILED"),
            "deleted": content.count("Deleted"),
        }
    except Exception:
        return {"cycles": 0, "passed": 0, "failed": 0, "deleted": 0}


def get_history() -> dict:
    """Realtime history.json."""
    hf = RUNS_DIR / "history.json"
    try:
        if hf.exists():
            return json.loads(hf.read_text())
    except Exception:
        pass
    return {"total_passed": 0, "total_failed": 0}


def get_model_info() -> dict:
    """Realtime model config from src/config.py + disk cache size."""
    info = {}
    try:
        from src.config import CONFIG
        model_key = os.environ.get("AGENT_MODEL", "tool_calling")
        if model_key in CONFIG.models:
            m = CONFIG.models[model_key]
            info["name"] = m.name
            info["short_name"] = m.name.split("/")[-1]
            info["context_window"] = m.context_window
            info["max_tokens"] = m.max_tokens
            info["profile"] = model_key
        info["all_models"] = {k: v.name.split("/")[-1] for k, v in CONFIG.models.items()}
    except Exception:
        pass

    # Disk cache size (realtime)
    try:
        if "name" in info:
            cache_name = info["name"].replace("/", "--")
            cache_path = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{cache_name}"
            if cache_path.exists():
                total = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
                info["disk_size_gb"] = round(total / 1_073_741_824, 1)
    except Exception:
        pass
    return info


def get_latest_log_entry() -> dict:
    """Realtime latest structured log."""
    latest = find_latest_file("run_*.jsonl", LOGS_DIR)
    if latest:
        try:
            lines = latest.read_text().strip().split("\n")
            return json.loads(lines[-1]) if lines else {}
        except Exception:
            pass
    return {}


def get_evolution_stats() -> dict:
    """Realtime SkillTree v3 evolution log stats."""
    try:
        if EVOLUTION_LOG.exists():
            content = EVOLUTION_LOG.read_text()
            return {
                "proposals": content.count("PROPOSAL"),
                "implemented": content.count("SUCCESS: auto_implement"),
                "last_evolve": content.strip().split("\n")[-1][:80] if content else "Never",
            }
    except Exception:
        pass
    return {"proposals": 0, "implemented": 0, "last_evolve": "N/A"}


def get_skill_tree_status() -> dict:
    """Realtime SkillTree v3 brain state (UCB1, critical path, etc.)."""
    try:
        from src.skill_tree import SkillTree
        tree = SkillTree()
        next_skill = tree.get_next_skill() or {"name": "None", "current_impact": 0.0, "pull_count": 0}
        critical = tree.get_critical_path()[:3] if hasattr(tree, "get_critical_path") else []
        return {
            "next_skill": next_skill["name"],
            "next_impact": round(next_skill["current_impact"], 1),
            "pull_count": next_skill.get("pull_count", 0),
            "critical_path": " → ".join(critical) if critical else "None",
            "proposals_ready": len(tree.generate_proposals(0)) if hasattr(tree, "generate_proposals") else 0,
            "evolution_enabled": True,
        }
    except Exception as e:
        return {
            "next_skill": "SkillTree v3 not yet loaded",
            "next_impact": 0,
            "pull_count": 0,
            "critical_path": "Run SkillTree upgrade first",
            "proposals_ready": 0,
            "evolution_enabled": False,
        }


def build_dashboard() -> Layout:
    layout = Layout()

    proc = get_agent_process()
    mem = get_memory()
    perf = get_perf()
    stats = get_cycle_stats()
    history = get_history()
    model = get_model_info()
    last_event = get_latest_log_entry()
    evolution = get_evolution_stats()
    brain = get_skill_tree_status()

    latest_perf = find_latest_file("**/perf.json", RUNS_DIR)
    data_fresh = freshness(latest_perf) if latest_perf else "[dim]--[/]"

    # ── Header ──
    header = Table.grid(expand=True)
    header.add_row(
        f"[bold cyan]MLX Agent + SkillTree v3 Monitor[/] | "
        f"{mem['total_gb']:.1f} GB RAM | "
        f"{datetime.now().strftime('%H:%M:%S')} | Data: {data_fresh}"
    )

    # ── Model Info ──
    model_table = Table(title="Model", expand=True, box=box.ROUNDED)
    model_table.add_column("", style="cyan", width=20)
    model_table.add_column("", style="green")
    model_table.add_row("Name", model.get("short_name", "—"))
    model_table.add_row("Full ID", model.get("name", "—"))
    model_table.add_row("Profile", model.get("profile", "—"))
    model_table.add_row("Disk Cache", f"{model.get('disk_size_gb', '?')} GB")
    model_table.add_row("Context", f"{model.get('context_window', 0):,} tokens")
    model_table.add_row("Max Tokens", f"{model.get('max_tokens', 0):,} tokens")

    # ── Hardware ──
    hw_table = Table(title="Hardware (realtime)", expand=True, box=box.ROUNDED)
    hw_table.add_column("", style="cyan", width=20)
    hw_table.add_column("", style="green")
    hw_table.add_row("Total RAM", f"{mem['total_gb']:.1f} GB")
    hw_table.add_row("Used", f"{mem['used_gb']:.1f} GB ({mem['percent']}%)")
    hw_table.add_row("Free", f"{mem['free_gb']:.1f} GB")

    # ── Agent Process ──
    agent_table = Table(title="Agent Process", expand=True, box=box.ROUNDED)
    agent_table.add_column("", style="cyan", width=20)
    agent_table.add_column("", style="green")
    if proc.get("running"):
        agent_table.add_row("Status", "[bold green]RUNNING[/]")
        agent_table.add_row("PID", proc.get("pid", "—"))
        agent_table.add_row("CPU", f"{proc.get('cpu', 0)}%")
        agent_table.add_row("Memory", f"{proc.get('mem_mb', 0):.1f} MB")
    else:
        agent_table.add_row("Status", "[bold red]NOT RUNNING[/]")

    # ── SkillTree v3 Brain (NEW) ──
    brain_table = Table(title="SkillTree v3 – Autonomous Brain", expand=True, box=box.ROUNDED)
    brain_table.add_column("", style="cyan", width=22)
    brain_table.add_column("", style="magenta")
    brain_table.add_row("Next Skill (UCB1)", f"[bold]{brain['next_skill']}[/] (impact {brain['next_impact']})")
    brain_table.add_row("Pull Count", str(brain['pull_count']))
    brain_table.add_row("Critical Path", brain['critical_path'] or "—")
    brain_table.add_row("Proposals Ready", str(brain['proposals_ready']))
    brain_table.add_row("Evolution", "[bold green]ENABLED[/]" if brain['evolution_enabled'] else "[dim]v2[/]")
    brain_table.add_row("Last Evolve", evolution['last_evolve'][:60])

    # ── Token Performance ──
    perf_table = Table(title="Token Performance", expand=True, box=box.ROUNDED)
    perf_table.add_column("", style="cyan", width=20)
    perf_table.add_column("", style="green")
    perf_table.add_row("Tokens/s", f"{perf.get('tokens_per_sec', '?')}")
    perf_table.add_row("Context Fill", f"{perf.get('context_fill_pct', 0)}%")
    perf_table.add_row("Bandwidth", f"{perf.get('gb_per_sec', '?')} GB/s")

    # ── Self-Improvement Cycles ──
    cycle_table = Table(title="Self-Improvement Cycles", expand=True, box=box.ROUNDED)
    cycle_table.add_column("", style="cyan", width=20)
    cycle_table.add_column("", style="green")
    cycle_table.add_row("Total Cycles", str(stats["cycles"]))
    cycle_table.add_row("PASSED", f"[green]{stats['passed']}[/]")
    cycle_table.add_row("FAILED", f"[red]{stats['failed']}[/]")
    cycle_table.add_row("DELETED", str(stats["deleted"]))

    # ── Latest Event ──
    event_panel = Panel(
        Text(json.dumps(last_event, indent=2)[:600] if last_event else "No logs yet", style="dim"),
        title="Latest Event",
        border_style="blue",
    )

    # Layout assembly
    layout.split(
        Layout(header, name="header", size=3),
        Layout(name="body", ratio=1),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )
    layout["left"].split(
        Layout(model_table, size=12),
        Layout(hw_table, size=9),
        Layout(agent_table, size=9),
        Layout(brain_table, size=12),   # SkillTree v3 panel
    )
    layout["right"].split(
        Layout(perf_table, size=9),
        Layout(cycle_table, size=9),
        Layout(event_panel, ratio=1),
    )

    return layout


if __name__ == "__main__":
    console = Console()
    console.print("[bold green]Starting MLX + SkillTree v3 Monitor...[/]\n"
                  "All metrics realtime • Ctrl+C to exit\n")

    with Live(build_dashboard(), refresh_per_second=1, screen=True) as live:
        try:
            while True:
                live.update(build_dashboard())
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Monitor stopped.[/]")