"""Pulsar Real-Time Terminal CLI Monitor.

Provides a modern, high-tech terminal TUI monitoring Pulsar training telemetry:
- Step, Progress %, Tokens Seen, Step Time, Elapsed & ETA
- Train/Val Loss, PPL, Sparkline Loss Descent Trend
- Multi-Pulse recurrent losses (L1, L2, L3) and hidden state deltas (Δh1, Δh2, Δh3)
- MoE Router Confidence, Across-Pulse Expert Load Balance, Bias Magnitude
- Dual-Optimizer LRs (Muon, Adam) and Pre-Clip Gradient Norms
- Hardware Saturation: tok/s, MFU %, and NVIDIA RTX 4060 VRAM Gauges
- Live Canary Generations Inspection Bay

Usage:
  uv run python monitor/cli_monitor.py
"""

import json
import math
import os
import sys
import time
from datetime import datetime

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

LIVE_STATUS_FILE = "checkpoints/live_status.json"
METRICS_LOG_FILE = "checkpoints/metrics.jsonl"


def format_tok_rate(rate: float) -> str:
    if rate >= 1e6:
        return f"{rate / 1e6:.2f}M tok/s"
    if rate >= 1e3:
        return f"{rate / 1e3:.1f}k tok/s"
    return f"{rate:.0f} tok/s"


def format_seconds(seconds: float) -> str:
    if seconds <= 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def make_gauge(pct: float, width: int = 12, fill_char: str = "█", empty_char: str = "░", color: str = "cyan") -> Text:
    clamped = max(0.0, min(100.0, pct))
    filled_len = int(width * (clamped / 100.0))
    empty_len = width - filled_len
    t = Text()
    t.append(fill_char * filled_len, style=f"bold {color}")
    t.append(empty_char * empty_len, style="dim white")
    return t


def make_sparkline(values: list[float], length: int = 16) -> str:
    if not values:
        return "·" * length
    sub = values[-length:]
    if len(sub) < length:
        sub = [sub[0]] * (length - len(sub)) + sub
    min_v, max_v = min(sub), max(sub)
    if max_v - min_v < 1e-6:
        return "▄" * length
    chars = " ▂▃▄▅▆▇█"
    res = []
    for v in sub:
        norm = (v - min_v) / (max_v - min_v)
        idx = int(norm * (len(chars) - 1))
        res.append(chars[max(0, min(len(chars) - 1, idx))])
    return "".join(res)


def read_recent_losses(limit: int = 24) -> list[float]:
    if not os.path.exists(METRICS_LOG_FILE):
        return []
    losses = []
    try:
        with open(METRICS_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "train_loss" in rec:
                        losses.append(float(rec["train_loss"]))
                except Exception:
                    continue
    except Exception:
        pass
    return losses[-limit:]


def make_layout() -> Layout:
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="progress", size=4),
        Layout(name="body", size=14),
        Layout(name="canaries", size=9),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )
    return layout


def build_ui(data: dict | None, recent_losses: list[float]) -> Layout:
    layout = make_layout()
    now_str = datetime.now().strftime("%H:%M:%S")

    # 1. Header
    is_active = data and data.get("status") != "idle" and (time.time() - data.get("timestamp", 0) < 60)
    status_pill = (
        Text.assemble((" ● ", "bold bright_green"), ("RUNNING ", "bold white on green"))
        if is_active
        else Text.assemble((" ○ ", "bold yellow"), ("STANDBY ", "bold white on rgb(60,50,20)"))
    )

    header_text = Text.assemble(
        (" PULSAR ", "bold black on bright_cyan"),
        ("  Recurrent MoE (K=3, 8L→24L) + 32k Hybrid Attention  ", "bold white on rgb(15,23,42)"),
        ("  201.8M Params (107M Active)  ", "bold bright_yellow on rgb(30,41,59)"),
        ("  ", ""),
        status_pill,
        (f"  [{now_str}]", "dim bright_cyan"),
    )
    layout["header"].update(Panel(Align.center(header_text), box=box.ROUNDED, border_style="bright_cyan", padding=(0, 0)))

    # Empty State
    if not data or data.get("status") == "idle":
        layout["progress"].update(
            Panel(
                Align.center(Text("Waiting for training run to initialize checkpoints/live_status.json...", style="bold yellow")),
                box=box.ROUNDED,
                border_style="yellow",
            )
        )
        layout["left"].update(Panel(Text("No active recurrent telemetry received yet.", style="dim italic"), box=box.ROUNDED, border_style="dim"))
        layout["right"].update(Panel(Text("No optimizer telemetry received yet.", style="dim italic"), box=box.ROUNDED, border_style="dim"))
        layout["canaries"].update(Panel(Text("Canaries pending first evaluation interval.", style="dim italic"), box=box.ROUNDED, border_style="dim"))
        return layout

    step = data.get("step", 0)
    max_steps = data.get("max_steps", 1000)
    pct = data.get("progress_pct", 0.0)
    toks = data.get("tokens_seen", 0)
    step_time = data.get("step_time_s", 0.0)
    speed = data.get("speed_tok_s", 0.0)

    # Remaining ETA
    remaining_steps = max(0, max_steps - step)
    eta_sec = remaining_steps * step_time if step_time > 0 else 0
    eta_str = format_seconds(eta_sec)

    # 2. Progress Bar
    bar_gauge = make_gauge(pct, width=32, fill_char="█", empty_char="░", color="bright_cyan")
    prog_line1 = Text.assemble(
        (" Step ", "bold bright_white"),
        (f"{step:,}", "bold bright_cyan"),
        (f" / {max_steps:,} ", "dim white"),
        ("[", "bold bright_white"),
        bar_gauge,
        (f"] {pct:5.1f}%  ", "bold bright_green"),
        (f"Tokens: {toks / 1e6:6.2f}M  ", "bold bright_yellow"),
        (f"ETA: {eta_str} ", "bold bright_magenta"),
        (f"({step_time:.2f}s/step)", "dim white"),
    )
    layout["progress"].update(Panel(Align.center(prog_line1), box=box.ROUNDED, border_style="bright_blue", padding=(0, 1)))

    # 3. Left Panel: Recurrent Engine & Engram Subsystem
    table_left = Table(show_header=True, header_style="bold bright_cyan", box=None, padding=(0, 1), expand=True)
    table_left.add_column("Metric", style="bold white", no_wrap=True)
    table_left.add_column("Dynamics & Memory", style="bright_cyan")

    train_loss = data.get("train_loss", 0.0)
    train_ppl = data.get("train_ppl", 0.0)
    sparkline_str = make_sparkline(recent_losses, length=8)

    table_left.add_row(
        "Train Loss",
        Text.assemble(
            (f"{train_loss:.4f} ", "bold bright_cyan"),
            (f"(PPL {train_ppl:4.1f}) ", "dim white"),
            (f"[{sparkline_str}]", "bold bright_green"),
        ),
    )

    val_loss = data.get("val_loss")
    val_ppl = data.get("val_ppl")
    best_val = data.get("best_val_loss")
    if val_loss is not None:
        table_left.add_row("Val Loss", f"{val_loss:.4f} (PPL {val_ppl:4.1f}) [best: {best_val:.4f}]")
    else:
        table_left.add_row("Val Loss", f"Evaluating (Best: {best_val if best_val else '--'})")

    p_losses = data.get("pulse_losses", [0.0, 0.0, 0.0])
    table_left.add_row(
        "Pulse Loss",
        f"L1:{p_losses[0]:.2f} L2:{p_losses[1]:.2f} L3:{p_losses[2]:.2f}",
    )

    p_deltas = data.get("pulse_deltas", [0.0, 0.0, 0.0])
    ratio = (p_deltas[2] / (p_deltas[0] + 1e-6)) if len(p_deltas) >= 3 else 1.0
    table_left.add_row(
        "Recurrent Δh",
        f"[{p_deltas[0]:.1f}, {p_deltas[1]:.1f}, {p_deltas[2]:.1f}] ({ratio:.2f}x)",
    )

    gate_mean = data.get("engram_gate_mean", 0.0)
    gate_gauge = make_gauge(gate_mean * 100.0, width=5, color="bright_magenta")
    mix_lambda = data.get("mix_lambda", 0.0)
    mix_gauge = make_gauge(mix_lambda * 100.0, width=5, color="bright_yellow")

    table_left.add_row(
        "Engram Gate",
        Text.assemble((f"{gate_mean:.3f} ", "bold bright_magenta"), gate_gauge, (f" λ:{mix_lambda:.3f} ", "dim white"), mix_gauge),
    )

    engram_norm = data.get("engram_norm", 0.0)
    alpha_abs = data.get("engram_alpha_abs", 0.0)
    table_left.add_row(
        "Engram State",
        f"|α|: {alpha_abs:.3f}  Norm: {engram_norm:.3f}",
    )

    layout["left"].update(
        Panel(table_left, title="[bold bright_cyan]Recurrent Core & Engram[/]", box=box.ROUNDED, border_style="bright_cyan")
    )

    # 4. Right Panel: Hardware Saturation & Dual Optimizer
    table_right = Table(show_header=True, header_style="bold bright_green", box=None, padding=(0, 1), expand=True)
    table_right.add_column("System", style="bold white", no_wrap=True)
    table_right.add_column("Telemetry & Gauges", style="bright_green")

    mfu = data.get("mfu_pct", 0.0)
    mfu_gauge = make_gauge(mfu * 2.0, width=6, color="bright_green")
    table_right.add_row(
        "Throughput",
        Text.assemble((format_tok_rate(speed) + " ", "bold bright_green"), mfu_gauge, (f" {mfu:4.1f}% MFU", "dim white")),
    )

    vram_peak = data.get("vram_peak_gb", 0.0)
    vram_free = data.get("vram_free_gb", 0.0)
    vram_pct = (vram_peak / 8.0) * 100.0
    vram_gauge = make_gauge(vram_pct, width=6, color="bright_cyan" if vram_pct < 85 else "bright_red")
    table_right.add_row(
        "VRAM (RTX4060)",
        Text.assemble((f"{vram_peak:.2f}G ", "bold bright_cyan"), vram_gauge, (f" Free:{vram_free:.1f}G", "dim white")),
    )

    total_grad = data.get("grad_norm_total", 0.0)
    m_grad = data.get("grad_norm_muon", 0.0)
    a_grad = data.get("grad_norm_adam", 0.0)
    e_grad = data.get("grad_norm_engram", 0.0)
    table_right.add_row(
        "Grad Norms",
        f"{total_grad:.2f} (M:{m_grad:.1f} A:{a_grad:.1f} E:{e_grad:.1f})",
    )

    lr_m = data.get("lr_muon", 0.0)
    table_right.add_row(
        "Muon Matrix LR",
        f"{lr_m:.4f} (2D weights)",
    )
    lr_a = data.get("lr_adam", 0.0)
    lr_e = data.get("lr_engram", 0.0)
    table_right.add_row(
        "Adam / Engram",
        f"A:{lr_a:.1e} | E:{lr_e:.1e}",
    )

    layout["right"].update(
        Panel(table_right, title="[bold bright_green]Hardware Saturation & Optimizers[/]", box=box.ROUNDED, border_style="bright_green")
    )

    # 5. Bottom Panel: Canaries Inspection Bay
    canaries = data.get("canaries", [])
    if canaries:
        canary_table = Table(show_header=True, header_style="bold bright_magenta", box=None, padding=(0, 1), expand=True)
        canary_table.add_column("Domain Tag", style="bold bright_magenta", width=14, no_wrap=True)
        canary_table.add_column("Prompt", style="dim white", width=26, no_wrap=True)
        canary_table.add_column("Live Model Generation", style="bright_white")

        for c in canaries:
            tag = c.get("tag", "Canary")
            p_text = c.get("prompt", "")[:25]
            comp = c.get("completion", "").strip().replace("\n", " \\n ")
            if len(comp) > 95:
                comp = comp[:95] + "..."
            canary_table.add_row(f"[{tag}]", p_text, comp)
        layout["canaries"].update(
            Panel(canary_table, title="[bold bright_magenta]Live Canary Evaluation Bay[/]", box=box.ROUNDED, border_style="bright_magenta")
        )
    else:
        layout["canaries"].update(
            Panel(
                Align.center(Text("Canary generation tests trigger every validation interval (25 steps).", style="dim italic")),
                title="[bold bright_magenta]╭─ Live Canary Evaluation Bay ─╮[/]",
                box=box.ROUNDED,
                border_style="bright_magenta",
            )
        )

    return layout


def main():
    console = Console()
    with Live(console=console, screen=True, refresh_per_second=4) as live:
        while True:
            data = None
            if os.path.exists(LIVE_STATUS_FILE):
                try:
                    with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    pass

            recent_losses = read_recent_losses(limit=24)
            live.update(build_ui(data, recent_losses))
            time.sleep(0.4)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
