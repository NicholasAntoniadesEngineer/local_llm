#!/usr/bin/env python3
"""Real-time agent performance monitor for M4 Max."""

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
except ImportError:
    print("Install rich: pip install rich")
    sys.exit(1)


PERF_FILE = Path("./agent_outputs/.perf_stats.json")


def freshness(path: Path) -> str:
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
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=3)
        for line in result.stdout.split("\n"):
            if "improve.py" in line and "grep" not in line and "tail" not in line:
                parts = line.split()
                return {
                    "pid": parts[1], "cpu": float(parts[2]),
                    "mem_pct": float(parts[3]),
                    "mem_mb": float(parts[5]) / 1024 if len(parts) > 5 else 0,
                    "running": True,
                }
        return {"running": False}
    except Exception:
        return {"running": False}


def get_memory() -> dict:
    try:
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3)
        stats = {"total_gb": 36.0}
        for line in result.stdout.split("\n"):
            if "Pages active" in line:
                stats["active_gb"] = int(line.split(":")[1].strip().rstrip(".")) * 16384 / 1073741824
            elif "Pages wired" in line:
                stats["wired_gb"] = int(line.split(":")[1].strip().rstrip(".")) * 16384 / 1073741824
        stats["used_gb"] = stats.get("active_gb", 0) + stats.get("wired_gb", 0)
        stats["free_gb"] = stats["total_gb"] - stats["used_gb"]
        return stats
    except Exception:
        return {"total_gb": 36.0, "used_gb": 0, "free_gb": 36.0}


def get_perf() -> dict:
    try:
        if PERF_FILE.exists():
            with open(PERF_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def get_cycle_stats() -> dict:
    try:
        with open("/tmp/improve_loop.log") as f:
            content = f.read()
        return {
            "cycles": content.count("CYCLE #"),
            "passed": content.count("PASSED"),
            "failed": content.count("FAILED"),
            "deleted": content.count("Deleted"),
        }
    except Exception:
        return {"cycles": 0, "passed": 0, "failed": 0, "deleted": 0}


def get_history() -> dict:
    try:
        hf = Path("./agent_outputs/.history.json")
        if hf.exists():
            with open(hf) as f:
                return json.load(f)
    except Exception:
        pass
    return {"cycles": [], "total_passed": 0, "total_failed": 0}


def get_model_info() -> dict:
    """Get model info from config + perf stats."""
    info = {}
    try:
        from config import CONFIG
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

    # Model file size from cache
    try:
        if "name" in info:
            cache_name = info["name"].replace("/", "--")
            cache_path = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{cache_name}" / "blobs"
            if cache_path.exists():
                total = sum(f.stat().st_size for f in cache_path.iterdir() if f.is_file())
                info["disk_size_gb"] = round(total / 1073741824, 1)
    except Exception:
        pass

    return info


def get_latest_log_entry() -> dict:
    """Get the most recent structured log entry."""
    try:
        log_dir = Path("./agent_outputs/logs")
        logs = sorted(log_dir.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not logs:
            return {}
        with open(logs[0]) as f:
            lines = f.readlines()
        if lines:
            return json.loads(lines[-1])
    except Exception:
        pass
    return {}


def build_dashboard() -> Layout:
    layout = Layout()

    proc = get_agent_process()
    mem = get_memory()
    perf = get_perf()
    stats = get_cycle_stats()
    history = get_history()
    model = get_model_info()
    last_event = get_latest_log_entry()

    data_fresh = freshness(PERF_FILE)

    # ── Header ──
    header = Table.grid(expand=True)
    header.add_row(
        f"[bold cyan]MLX Agent Monitor[/] | M4 Max 36GB | "
        f"{datetime.now().strftime('%H:%M:%S')} | Data: {data_fresh}"
    )

    # ── Model Info ──
    model_table = Table(title="Model", expand=True)
    model_table.add_column("", style="cyan", width=18)
    model_table.add_column("", style="green")

    model_table.add_row("Name", model.get("short_name", "loading..."))
    model_table.add_row("Full ID", model.get("name", ""))
    model_table.add_row("Profile", model.get("profile", ""))
    model_table.add_row("Weights on Disk", f"{model.get('disk_size_gb', '?')} GB")
    model_table.add_row("Context Window", f"{model.get('context_window', 0):,} tokens")
    model_table.add_row("Max Generation", f"{model.get('max_tokens', 0):,} tokens")
    model_table.add_row("Quantization", "4-bit")
    model_table.add_row("Runtime", "MLX (Metal GPU)")

    avail = model.get("all_models", {})
    if avail:
        model_table.add_row("", "")
        model_table.add_row("[dim]Available:", "")
        for k, v in avail.items():
            marker = " [bold green]<< active[/]" if k == model.get("profile") else ""
            model_table.add_row(f"  {k}", f"{v}{marker}")

    # ── Hardware ──
    hw_table = Table(title="Hardware", expand=True)
    hw_table.add_column("", style="cyan", width=18)
    hw_table.add_column("", style="green")

    hw_table.add_row("Chip", "Apple M4 Max")
    hw_table.add_row("Memory", f"{mem['used_gb']:.1f} / 36 GB")
    hw_table.add_row("Free", f"{mem['free_gb']:.1f} GB")
    hw_table.add_row("Bandwidth", "546 GB/s")

    if proc.get("running"):
        hw_table.add_row("", "")
        hw_table.add_row("Agent Status", "[bold green]RUNNING[/]")
        hw_table.add_row("PID", str(proc["pid"]))
        hw_table.add_row("CPU", f"{proc['cpu']:.1f}%")
        hw_table.add_row("Agent RAM", f"{proc['mem_mb']:.0f} MB ({proc['mem_pct']:.1f}%)")
    else:
        hw_table.add_row("", "")
        hw_table.add_row("Agent Status", "[bold red]STOPPED[/]")

    # ── Token Performance ──
    tok_table = Table(title="Token Performance", expand=True)
    tok_table.add_column("", style="cyan", width=18)
    tok_table.add_column("", style="green")

    if perf:
        gen_s = perf.get("gen_tok_s", 0)
        if gen_s >= 20:
            speed_str = f"[bold green]{gen_s} tok/s[/]"
        elif gen_s >= 10:
            speed_str = f"[green]{gen_s} tok/s[/]"
        elif gen_s >= 5:
            speed_str = f"[yellow]{gen_s} tok/s[/]"
        else:
            speed_str = f"[red]{gen_s} tok/s[/]"

        tok_table.add_row("Gen Speed", speed_str)
        tok_table.add_row("Average", f"{perf.get('avg_tok_s', 0)} tok/s")
        tok_table.add_row("Peak", f"{perf.get('peak_tok_s', 0)} tok/s")
        bw = perf.get("bandwidth_used_gbs", 0)
        bw_pct = round(bw / 546 * 100, 1) if bw else 0
        tok_table.add_row("Bandwidth", f"{bw} GB/s ({bw_pct}% of 546)")
        tok_table.add_row("", "")
        tok_table.add_row("Prompt (last)", f"{perf.get('prompt_tokens', 0):,} tokens")
        tok_table.add_row("Generated (last)", f"{perf.get('gen_tokens', 0):,} tokens")
        tok_table.add_row("Total Processed", f"{perf.get('total_all_tokens', 0):,} tokens")
        tok_table.add_row("", "")
        ctx_pct = perf.get("context_pct", 0)
        ctx_bar = "█" * int(ctx_pct / 5) + "░" * (20 - int(ctx_pct / 5))
        tok_table.add_row("Context Fill", f"{ctx_bar} {ctx_pct}%")
        tok_table.add_row("Step Time", f"{perf.get('elapsed', 0)}s")
        tok_table.add_row("Avg Step", f"{perf.get('avg_step_time', 0)}s")
        tok_table.add_row("Step #", str(perf.get("step", 0)))
        tok_table.add_row("Tool Success", f"{perf.get('tool_success_rate', 0):.0f}%")
    else:
        tok_table.add_row("Status", "[yellow]Waiting for first generation...[/]")

    # ── Cycles ──
    cycle_table = Table(title="Improvement Cycles", expand=True)
    cycle_table.add_column("", style="cyan", width=18)
    cycle_table.add_column("", style="green")

    h = history
    total_c = h["total_passed"] + h["total_failed"]
    rate = h["total_passed"] / max(1, total_c)
    cycle_table.add_row("Total Cycles", str(total_c))
    cycle_table.add_row("Passed", f"[green]{h['total_passed']}[/]")
    cycle_table.add_row("Failed", f"[red]{h['total_failed']}[/]")
    cycle_table.add_row("Success Rate", f"{rate:.0%}")
    cycle_table.add_row("", "")

    # Recent history
    for c in h.get("cycles", [])[-5:]:
        st = "[green]PASS[/]" if c["passed"] else "[red]FAIL[/]"
        cycle_table.add_row(f"  {st} {c['target']}", c.get("reason", "")[:30])

    # ── Skill Tree ──
    tree_table = Table(title="Skill Tree", expand=True)
    tree_table.add_column("Skill", style="cyan")
    tree_table.add_column("Status", width=8)
    tree_table.add_column("Size", width=8)

    try:
        from skill_tree import SKILLS, get_system_state as _gs
        _st = _gs()
        for sid, sk in sorted(SKILLS.items(), key=lambda x: (x[1].tier, -x[1].impact)):
            fpath = Path(f"./agent_outputs/{sk.file}")
            if sid in _st["passing"]:
                st = "[green]DONE[/]"
                sz = f"{fpath.stat().st_size:,}B" if fpath.exists() else ""
            elif sid in _st["failing"]:
                st = "[red]FAIL[/]"
                sz = f"{fpath.stat().st_size:,}B" if fpath.exists() else ""
            else:
                prereqs_met = all(p in _st["passing"] for p in sk.prereqs)
                st = "[yellow]NEXT[/]" if prereqs_met else "[dim]LOCK[/]"
                sz = ""
            tree_table.add_row(f"T{sk.tier} {sk.name}", st, sz)
    except Exception as e:
        tree_table.add_row(f"Error: {e}", "", "")

    # ── Model Output (streaming) ──
    stream_text = ""
    try:
        sf = Path("./agent_outputs/.stream.txt")
        if sf.exists() and (time.time() - sf.stat().st_mtime) < 30:
            stream_text = sf.read_text()[-2000:]
        else:
            stream_text = "(idle - waiting for generation)"
    except Exception:
        stream_text = "(error)"
    stream_text = stream_text.replace("[", "\\[")
    stream_display = Text(stream_text)

    # ── Agent Log ──
    try:
        with open("/tmp/improve_loop.log") as f:
            log_lines = f.readlines()
        log_raw = "".join(log_lines[-25:])[-2000:]
    except Exception:
        log_raw = "(no log)"
    log_raw = log_raw.replace("[", "\\[")
    log_text = Text(log_raw)

    # ═══ BUILD LAYOUT ═══
    layout.split_column(
        Layout(Panel(header, style="bold"), size=3),
        Layout(name="top", size=18),
        Layout(name="mid", size=14),
        Layout(name="bottom"),
    )

    layout["top"].split_row(
        Layout(Panel(model_table), ratio=1),
        Layout(Panel(hw_table), ratio=1),
        Layout(Panel(tok_table), ratio=1),
    )

    layout["mid"].split_row(
        Layout(Panel(tree_table), ratio=1),
        Layout(Panel(cycle_table), ratio=1),
    )

    layout["bottom"].split_column(
        Layout(Panel(stream_display, title="Model Output (real-time)"), ratio=1),
        Layout(Panel(log_text, title="Agent Log"), ratio=1),
    )

    return layout


def main():
    console = Console()
    console.clear()
    try:
        with Live(build_dashboard(), refresh_per_second=4, console=console) as live:
            while True:
                live.update(build_dashboard())
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
