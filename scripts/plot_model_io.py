#!/usr/bin/env python
"""Plot input frames + full model output for a single action.

Creates two separate PNG figures:
  - fig_8b_sv_action<N>.png   : 8B single-view reasoning
  - fig_2b_mv_action<N>.png   : 2B multi-view reasoning

Layout per figure:
  ┌──────────────────────────────────────────┐
  │  title / gold / prediction verdict       │
  ├──────────────────────────────────────────┤
  │  [frame 1] [frame 2] … [frame 6]  clip_0│
  │  [frame 1] [frame 2] … [frame 6]  clip_1│  ← 2B only
  ├──────────────────────────────────────────┤
  │  <think>                                 │
  │    ... full think text (all lines) ...   │
  │  </think>                                │
  │  <answer>...</answer>                    │
  └──────────────────────────────────────────┘

Usage (from repo root):
    python scripts/plot_model_io.py --action-id 17
    python scripts/plot_model_io.py --action-id 0 --out-dir outputs/plots
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train.frame_utils import foul_anchored_timestamps, T_FOUL, T_START, T_END

# ── default paths ────────────────────────────────────────────────────────────

ROWS_8B_PATH = "outputs/reason_eval/reason_valid321_rows.json"
ROWS_2B_PATH = "outputs/qlora_cosmos2b_multiview_reason_1/best_checkpoint/val_rows.json"
DATA_ROOT    = Path("data/SoccerNet/mvfouls")
N_FRAMES     = 6   # frames shown per clip

CORRECT_COLOUR   = "#4CAF50"
INCORRECT_COLOUR = "#F44336"
BG_DARK          = "#1C1C1E"
BG_TEXT          = "#111115"


# ── frame extraction ─────────────────────────────────────────────────────────

def extract_frames(video_path: str, n: int = 6,
                   t_start: float = T_START, t_end: float = T_END,
                   t_foul: float = T_FOUL) -> list[tuple]:
    """Extract n frames via ffmpeg, always including one at t_foul.

    Returns list of (ndarray, timestamp_sec) tuples in chronological order.
    VARS spec: foul at t≈3.0 s in the 5-second clip.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", video_path],
            capture_output=True, text=True,
        )
        duration = 5.0
        try:
            for s in json.loads(probe.stdout).get("streams", []):
                if s.get("codec_type") == "video":
                    duration = float(s.get("duration", 5.0))
                    break
        except Exception:
            pass

        t0 = max(0.0, t_start)
        t1 = min(duration, t_end)
        timestamps = foul_anchored_timestamps(n, t0, t1, t_foul)

        frames = []
        for i, t in enumerate(timestamps):
            out = tmp_path / f"f{i:03d}.jpg"
            subprocess.run(
                ["ffmpeg", "-ss", f"{t:.3f}", "-i", video_path,
                 "-frames:v", "1", "-q:v", "2", str(out),
                 "-y", "-loglevel", "quiet"],
                check=False,
            )
            if out.exists():
                frames.append((np.array(Image.open(out).convert("RGB")), t))
            else:
                frames.append((None, t))   # placeholder

        # Fill any failed extractions with a blank
        h, w = next(
            (f[0].shape[:2] for f in frames if f[0] is not None),
            (180, 320)
        )
        blank = np.full((h, w, 3), 30, dtype=np.uint8)
        frames = [(img if img is not None else blank, t) for img, t in frames]
        return frames


# ── text helpers ─────────────────────────────────────────────────────────────

def parse_raw(raw: str) -> tuple[str, str]:
    """Split raw model output into (think_body, answer_body)."""
    import re
    think_m = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
    ans_m   = re.search(r"<answer>(.*?)</answer>", raw, flags=re.DOTALL)
    # also catch <antwort> used by 2B model
    if ans_m is None:
        ans_m = re.search(r"<antwort>(.*?)</\s*antwort\s*>", raw,
                          flags=re.DOTALL | re.IGNORECASE)
    think  = think_m.group(1).strip() if think_m else raw.strip()
    answer = ans_m.group(1).strip()   if ans_m   else "(no <answer> tag found)"
    return think, answer


def count_display_lines(text: str, wrap_width: int = 100) -> int:
    total = 0
    for para in text.split("\n"):
        if para.strip() == "":
            total += 1
        else:
            total += max(1, len(para) // wrap_width + 1)
    return total


def format_full_output(raw: str, wrap_width: int = 100) -> str:
    """Return the full raw output with line-wrapped paragraphs."""
    import re

    # Replace <think> tags on their own line
    out = raw.strip()
    out = re.sub(r"<think>\s*", "<think>\n", out)
    out = re.sub(r"\s*</think>", "\n</think>", out)
    out = re.sub(r"<answer>", "\n<answer>", out)
    out = re.sub(r"</answer>", "</answer>\n", out)

    # Wrap long lines (but preserve short ones)
    lines = []
    for line in out.split("\n"):
        if len(line) <= wrap_width:
            lines.append(line)
        else:
            lines.extend(textwrap.wrap(line, wrap_width) or [""])
    return "\n".join(lines)


# ── figure builder ────────────────────────────────────────────────────────────

def _add_frame_row(fig, gs, row_idx: int, frames: list[tuple],
                   label: str, border_colour: str, n: int = N_FRAMES,
                   foul_t: float = 3.0):
    """Add one row of video frames to the GridSpec.

    frames: list of (ndarray, timestamp_sec) tuples.
    Frames closest to foul_t get a highlight marker.
    """
    for i, (frame, t) in enumerate(frames):
        ax = fig.add_subplot(gs[row_idx, i])
        ax.imshow(frame)
        ax.set_xticks([])
        ax.set_yticks([])

        # Highlight the frame that is exactly the foul anchor (closest to foul_t)
        is_foul_frame = abs(t - foul_t) < 0.01
        edge_col = "#FFD700" if is_foul_frame else border_colour
        lw = 2.5 if is_foul_frame else 1.5
        for spine in ax.spines.values():
            spine.set_edgecolor(edge_col)
            spine.set_linewidth(lw)

        label_t = f"t={t:.2f}s"
        if is_foul_frame:
            label_t += " ★"
        ax.set_xlabel(label_t, color="#FFD700" if is_foul_frame else "#999999",
                      fontsize=7.5, labelpad=2,
                      fontweight="bold" if is_foul_frame else "normal")
        if i == 0:
            ax.set_title(label, color=border_colour, fontsize=8.5,
                         pad=3, loc="left", fontweight="bold")


def _add_text_panel(fig, gs, row_idx: int, col_span: int,
                    raw: str, pred: dict, gold: dict,
                    view_legend: bool = False):
    """Add the full model output text panel."""
    think, answer = parse_raw(raw)

    gold_ac = gold.get("action_class", "?")
    pred_ac = pred.get("action_class", "?")
    gold_os = gold.get("offence_severity", "?")
    pred_os = pred.get("offence_severity", "?")
    ac_ok   = pred_ac.lower() == gold_ac.lower()
    os_ok   = pred_os.lower() == gold_os.lower()

    ax = fig.add_subplot(gs[row_idx, :col_span])
    ax.set_facecolor(BG_TEXT)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    # ── think block ──────────────────────────────────────────────────────────
    full_output = format_full_output(raw, wrap_width=102)
    lines = full_output.split("\n")

    # Colour each line based on content
    y = 0.99
    lh = 1.0 / (len(lines) + 3)   # line height as fraction of axis

    for line in lines:
        stripped = line.strip()
        if stripped in ("<think>", "</think>"):
            colour = "#666666"
            fs = 8.5
            fw = "normal"
        elif stripped.startswith("<answer>") or stripped.startswith("<antwort>"):
            ac_col = CORRECT_COLOUR if ac_ok else INCORRECT_COLOUR
            colour = ac_col
            fs = 9.0
            fw = "bold"
        elif stripped.startswith('"action_class"') or stripped.startswith('"action_CLASS"'):
            colour = CORRECT_COLOUR if ac_ok else INCORRECT_COLOUR
            fs = 8.5
            fw = "bold"
        elif (stripped.startswith('"offence_severity"') or
              stripped.startswith('"OFFENCE_SEVERITY"') or
              stripped.startswith('"offense_safety"')):
            colour = CORRECT_COLOUR if os_ok else INCORRECT_COLOUR
            fs = 8.5
            fw = "bold"
        elif "Analyzing" in stripped or "Contact:" in stripped or \
             "Body part:" in stripped or "Ball interaction:" in stripped:
            colour = "#88BBFF"
            fs = 8.5
            fw = "normal"
        else:
            colour = "#CCCCCC"
            fs = 8.5
            fw = "normal"

        ax.text(0.008, y, line,
                transform=ax.transAxes,
                fontsize=fs, color=colour,
                fontfamily="monospace", fontweight=fw,
                verticalalignment="top", linespacing=1.0,
                clip_on=True)
        y -= lh

    # Title + prediction verdict
    ax.set_title("Model output", color="#888888", fontsize=8.5,
                 loc="left", pad=3)

    # Right-side verdict box
    verdict_str = f"action: {'✓' if ac_ok else '✗'}  |  offence_sev: {'✓' if os_ok else '✗'}"
    v_col = CORRECT_COLOUR if (ac_ok and os_ok) else (
        "#FFAA00" if (ac_ok or os_ok) else INCORRECT_COLOUR)
    ax.text(0.99, 1.01, verdict_str,
            transform=ax.transAxes,
            fontsize=8.5, color=v_col, fontweight="bold",
            ha="right", va="bottom")

    if view_legend:
        p1 = mpatches.Patch(color="#3A86FF", label="View 1: clip_0")
        p2 = mpatches.Patch(color="#FF6B6B", label="View 2: clip_1")
        ax.legend(handles=[p1, p2], loc="upper right",
                  facecolor=BG_DARK, edgecolor="#444444",
                  labelcolor="white", fontsize=7.5, framealpha=0.9)


def _fig_header(fig, model_name: str, action_id: str, split: str,
                pred: dict, gold: dict):
    """Title + gold/pred line at the top."""
    gold_ac = gold.get("action_class", "?")
    gold_os = gold.get("offence_severity", "?")
    pred_ac = pred.get("action_class", "?")
    pred_os = pred.get("offence_severity", "?")
    correct = pred_ac.lower() == gold_ac.lower()
    verdict = "CORRECT ✓" if correct else "INCORRECT ✗"
    v_col   = CORRECT_COLOUR if correct else INCORRECT_COLOUR

    fig.text(0.01, 0.975,
             f"Model: {model_name}   |   action_{action_id}   |   {split}",
             color="white", fontsize=12, fontweight="bold", va="top")
    fig.text(0.01, 0.945,
             f"GOLD:  action_class = {gold_ac}   /   offence_severity = {gold_os}",
             color="#BBBBBB", fontsize=10, va="top")
    fig.text(0.01, 0.918,
             f"PRED:  action_class = {pred_ac}   /   offence_severity = {pred_os}"
             f"    [{verdict}]",
             color=v_col, fontsize=10, fontweight="bold", va="top")


# ── per-model figure constructors ─────────────────────────────────────────────

def figure_8b(row: dict, action_id: str, split: str) -> plt.Figure:
    clip_path = row.get("clip_path") or \
        str(DATA_ROOT / split / f"action_{action_id}" / "clip_0.mp4")
    pred = row.get("prediction", {})
    gold = row.get("gold", {})
    raw  = row.get("raw_output", "")

    frames   = extract_frames(clip_path, N_FRAMES)
    raw_fmt  = format_full_output(raw, 102)
    n_lines  = len(raw_fmt.split("\n"))

    # Figure height: 3 in header + 1.6 in frames + (n_lines * 0.185) in text
    text_h_in = max(4.0, n_lines * 0.185)
    fig_h     = 1.8 + 1.9 + text_h_in
    frame_frac = 1.9 / fig_h
    text_frac  = text_h_in / fig_h

    fig = plt.figure(figsize=(18, fig_h), facecolor=BG_DARK)
    gs  = GridSpec(
        2, N_FRAMES, figure=fig,
        height_ratios=[frame_frac, text_frac],
        hspace=0.08, wspace=0.04,
        top=0.90, bottom=0.01, left=0.005, right=0.995,
    )

    _fig_header(fig, "Cosmos-Reason2-8B  (Single-View Reasoning FT)",
                action_id, split, pred, gold)
    _add_frame_row(fig, gs, 0, frames, "clip_0  [Main camera]",
                   border_colour="#3A86FF")
    _add_text_panel(fig, gs, 1, N_FRAMES, raw, pred, gold, view_legend=False)
    return fig


def figure_2b(row: dict, action_id: str, split: str) -> plt.Figure:
    clip0 = DATA_ROOT / split / f"action_{action_id}" / "clip_0.mp4"
    clip1 = DATA_ROOT / split / f"action_{action_id}" / "clip_1.mp4"
    has_clip1 = clip1.exists()

    pred = row.get("prediction", {})
    gold = row.get("gold", {})
    raw  = row.get("raw_output", "")

    frames0 = extract_frames(str(clip0), N_FRAMES)
    frames1 = extract_frames(str(clip1), N_FRAMES) if has_clip1 else []

    raw_fmt = format_full_output(raw, 102)
    n_lines = len(raw_fmt.split("\n"))

    n_frame_rows = 2 if has_clip1 else 1
    frame_h_in   = 1.9 * n_frame_rows
    text_h_in    = max(4.0, n_lines * 0.185)
    fig_h        = 1.8 + frame_h_in + text_h_in

    frame_frac   = [1.9 / fig_h] * n_frame_rows
    text_frac    = [text_h_in / fig_h]

    fig = plt.figure(figsize=(18, fig_h), facecolor=BG_DARK)
    gs  = GridSpec(
        n_frame_rows + 1, N_FRAMES, figure=fig,
        height_ratios=frame_frac + text_frac,
        hspace=0.08, wspace=0.04,
        top=0.90, bottom=0.01, left=0.005, right=0.995,
    )

    _fig_header(fig, "Cosmos-Reason2-2B  (Multi-View Reasoning FT, 2 views)",
                action_id, split, pred, gold)
    _add_frame_row(fig, gs, 0, frames0, "clip_0  [Main camera]",
                   border_colour="#3A86FF")
    if has_clip1:
        _add_frame_row(fig, gs, 1, frames1, "clip_1  [Close-up / replay]",
                       border_colour="#FF6B6B")
    _add_text_panel(fig, gs, n_frame_rows, N_FRAMES, raw, pred, gold,
                    view_legend=has_clip1)
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--action-id", default="17")
    p.add_argument("--split", default="Valid", choices=["Train", "Valid", "Test"])
    p.add_argument("--out-dir", default="outputs/plots")
    p.add_argument("--rows-8b", default=ROWS_8B_PATH)
    p.add_argument("--rows-2b", default=ROWS_2B_PATH)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_8b = {r["action_id"]: r
               for r in json.loads(Path(args.rows_8b).read_text())}
    rows_2b = {r["action_id"]: r
               for r in json.loads(Path(args.rows_2b).read_text())}

    aid = str(args.action_id)
    if aid not in rows_8b:
        raise SystemExit(f"action_id={aid} not found in 8B rows")
    if aid not in rows_2b:
        raise SystemExit(f"action_id={aid} not found in 2B rows")

    print(f"action_{aid}:")
    print(f"  gold    : {rows_8b[aid]['gold']}")
    print(f"  8B pred : {rows_8b[aid]['prediction']}")
    print(f"  2B pred : {rows_2b[aid]['prediction']}")
    print()

    print("Rendering 8B figure …")
    fig = figure_8b(rows_8b[aid], aid, args.split)
    p = out_dir / f"fig_8b_sv_action{aid}.png"
    fig.savefig(p, dpi=args.dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {p}")

    print("Rendering 2B figure …")
    fig = figure_2b(rows_2b[aid], aid, args.split)
    p = out_dir / f"fig_2b_mv_action{aid}.png"
    fig.savefig(p, dpi=args.dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
