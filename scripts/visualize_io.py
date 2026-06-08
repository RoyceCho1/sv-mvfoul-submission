#!/usr/bin/env python
"""Visualize input/output structure for 8B single-view and 2B multi-view reasoning.

Shows prompt layout, synthesized think/answer, and token statistics
WITHOUT loading the model (fast, CPU-only).

Usage (from repo root):
    python scripts/visualize_io.py [--action-id 3] [--split Valid]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from zero_shot.zero_shot_eval import (
    ACTION_CLASSES,
    OFFENCE_SEVERITY_CLASSES,
    official_target,
)
from train.train_reason import synthesize_think, build_reasoning_prompt
from train.train_multiview_reason import camera_tag

# ── terminal colours ────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
MAGENTA= "\033[95m"
BLUE   = "\033[94m"
DIM    = "\033[2m"
RED    = "\033[91m"

def h1(s): print(f"\n{BOLD}{CYAN}{'═'*72}{RESET}\n{BOLD}{CYAN}  {s}{RESET}\n{BOLD}{CYAN}{'═'*72}{RESET}")
def h2(s): print(f"\n{BOLD}{YELLOW}{'─'*72}{RESET}\n{BOLD}{YELLOW}  {s}{RESET}\n{BOLD}{YELLOW}{'─'*72}{RESET}")
def h3(s): print(f"\n{BOLD}{BLUE}  ▶ {s}{RESET}")
def label(k, v): print(f"  {BOLD}{k:<22}{RESET} {v}")
def code(s, colour=GREEN):
    for line in s.splitlines():
        print(f"  {colour}{line}{RESET}")
def note(s): print(f"  {DIM}# {s}{RESET}")


# ── helpers ──────────────────────────────────────────────────────────────────

def available_clips(data_root: Path, split: str, action_id: str) -> list[int]:
    return [i for i in range(10)
            if (data_root / split / f"action_{action_id}" / f"clip_{i}.mp4").exists()]


def count_chars(text: str) -> dict:
    import re
    think_m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    ans_m   = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return {
        "total_chars": len(text),
        "think_chars": len(think_m.group(1)) if think_m else 0,
        "answer_chars": len(ans_m.group(1)) if ans_m else 0,
    }


def rough_token_estimate(text: str) -> int:
    """Rough estimate: ~3 chars per token for English text."""
    return max(1, len(text) // 3)


def get_sample(data_root: Path, split: str, action_id: str | None) -> tuple[str, dict, list[int]]:
    ann = json.loads((data_root / split / "annotations.json").read_text())
    if action_id:
        record = ann["Actions"].get(action_id)
        if record is None:
            raise ValueError(f"action_id={action_id!r} not found in {split}")
        clips = available_clips(data_root, split, action_id)
        return action_id, record, clips
    # pick first usable sample (≥2 clips, valid target)
    for aid, rec in ann["Actions"].items():
        if official_target(rec) is None:
            continue
        clips = available_clips(data_root, split, aid)
        if len(clips) >= 2:
            return aid, rec, clips
    raise RuntimeError("No suitable sample found")


# ── section renderers ────────────────────────────────────────────────────────

def render_annotation(action_id, record, clips):
    h3("ANNOTATION (ground truth)")
    target = official_target(record)
    action_class, offence_severity, offence, severity = target
    label("action_id:",       f"action_{action_id}")
    label("Action class:",    f"{BOLD}{action_class}{RESET}")
    label("Offence:",         record.get("Offence", "?"))
    label("Severity:",        record.get("Severity", "?"))
    label("offence_severity:",f"{BOLD}{offence_severity}{RESET}")
    label("Contact:",         record.get("Contact", "?"))
    label("Bodypart:",        record.get("Bodypart", "?"))
    label("Upper body part:", record.get("Upper body part", "?"))
    label("Try to play:",     record.get("Try to play", "?"))
    label("Touch ball:",      record.get("Touch ball", "?"))
    label("Handball:",        record.get("Handball", "?"))
    label("Available clips:", f"clip_{', clip_'.join(str(c) for c in clips)} → {len(clips)} views")


def render_8b_sv(action_id, record, clips, data_root, split):
    h2("MODEL 1 — 8B Single-View Reasoning  (train_reason.py)")

    target = official_target(record)
    action_class, offence_severity, _, _ = target
    clip_idx = clips[0]
    clip_info = record["Clips"][clip_idx]
    vid_path = str(data_root / split / f"action_{action_id}" / f"clip_{clip_idx}.mp4")

    # ── INPUT ────────────────────────────────────────────────────────────────
    h3("INPUT  (what goes into the model)")
    prompt = build_reasoning_prompt()

    note("Chat template wraps this as: <|im_start|>user\\n ... <|im_end|>")
    print()
    note("content list structure:")
    code(f'[', CYAN)
    code(f'  {{"type": "video", "video": "{vid_path}", "nframes": 16}},', CYAN)
    code(f'  {{"type": "text",  "text": <prompt>}}', CYAN)
    code(f']', CYAN)
    print()
    note("prompt text:")
    code(prompt, GREEN)
    print()
    note(f"video: {vid_path}")
    print()
    note("sequence layout after processor:")
    print(f"  {DIM}[SYS tokens] [VIDEO tokens: ~{16*128} tokens (16 frames × ~128 tok/frame)] [PROMPT tokens: ~{rough_token_estimate(prompt)}] [ASSISTANT header]{RESET}")
    print(f"  {DIM}total seq_len ≈ 8,000–9,000 tokens (8B has no pixel budget cap){RESET}")

    # ── LOSS MASK ────────────────────────────────────────────────────────────
    h3("LOSS MASK")
    note("find <|im_start|>assistant\\n header  →  mask everything BEFORE it")
    print(f"  {DIM}[        -100 (masked)        ] [  loss computed here  ]{RESET}")
    print(f"  {DIM}└── system + user turn ────────┘└── assistant turn ─────┘{RESET}")
    print(f"  {DIM}unmasked ≈ 129 tokens (think + answer){RESET}")

    # ── TARGET OUTPUT ────────────────────────────────────────────────────────
    h3("TARGET OUTPUT  (synthesized — full think+answer in loss)")
    answer_str = synthesize_think(record, action_class, offence_severity)
    stats = count_chars(answer_str)
    code(answer_str, GREEN)
    print()
    think_end = answer_str.find("</think>") + 8
    ans_sample = '{"action_class":"' + action_class + '"}'
    note(f"think block: ~{rough_token_estimate(answer_str[:think_end])} tokens  |  "
         f"answer block: ~{rough_token_estimate(ans_sample)+10} tokens")
    note(f"total output chars: {stats['total_chars']}  (think: {stats['think_chars']}, answer: {stats['answer_chars']})")

    # ── INFERENCE OUTPUT FORMAT ───────────────────────────────────────────────
    h3("INFERENCE OUTPUT  (model generates, parse with extract_answer_json)")
    note("expected format:")
    code("<think>", MAGENTA)
    code("Analyzing the video clip:", MAGENTA)
    code("  Contact: ...", MAGENTA)
    code("  Body part: ...", MAGENTA)
    code("  Ball interaction: ...", MAGENTA)
    code("  [reasoning sentences E + F]", MAGENTA)
    code("</think>", MAGENTA)
    code(f'<answer>{{"action_class": "{action_class}", "offence_severity": "{offence_severity}"}}</answer>', MAGENTA)
    print()
    note("extract_answer_json() scans for <answer>...</answer>, falls back to raw JSON")


def render_2b_mv(action_id, record, clips, data_root, split):
    h2("MODEL 2 — 2B Multi-View Reasoning  (train_multiview_reason.py)")

    target = official_target(record)
    action_class, offence_severity, _, _ = target
    eval_clips = clips[:2]   # fixed clip_0 + clip_1

    # ── INPUT ────────────────────────────────────────────────────────────────
    h3("INPUT  (what goes into the model)")
    prompt = build_reasoning_prompt()

    note("Chat template: <|im_start|>user\\n ... <|im_end|>")
    print()
    note("content list structure (2 views):")
    code('[', CYAN)
    for clip_idx in eval_clips:
        clip_info = record["Clips"][clip_idx] if clip_idx < len(record.get("Clips", [])) else {}
        tag = camera_tag(clip_info)
        vid_path = str(data_root / split / f"action_{action_id}" / f"clip_{clip_idx}.mp4")
        code(f'  {{"type": "text",  "text": "{tag}"}},', CYAN)
        code(f'  {{"type": "video", "video": "{vid_path}", "nframes": 16}},', CYAN)
    code('  {"type": "text",  "text": <prompt>}', CYAN)
    code(']', CYAN)
    print()
    note("prompt text (identical to 8B):")
    code(prompt, GREEN)
    print()
    note("pixel budget: longest_edge=4,500,000  → resize each frame to ~544×992")
    note("sequence layout after processor (longest_edge=4.5M, 2 views × 16 frames):")
    print(f"  {DIM}[SYS] [tag1] [VIDEO1: ~{16*128} tok] [tag2] [VIDEO2: ~{16*128} tok] [PROMPT: ~{rough_token_estimate(prompt)}] [ASST header]{RESET}")
    print(f"  {DIM}total seq_len ≈ 4,385 tokens  (pixel budget halves tokens vs uncapped){RESET}")

    # ── VIEW SELECTION ────────────────────────────────────────────────────────
    h3("VIEW SELECTION")
    print(f"  {BOLD}Training{RESET}  : random 2 from available {clips} each step (augmentation)")
    print(f"  {BOLD}Eval    {RESET}  : fixed clip_0 + clip_1 = {eval_clips} (deterministic)")
    note("clips are sorted by ascending index to keep temporal order")

    # ── LOSS MASK ────────────────────────────────────────────────────────────
    h3("LOSS MASK")
    note("same as 8B: scan for <|im_start|>assistant\\n header → mask before it")
    print(f"  {DIM}[───────────────── -100 (masked) ─────────────────] [ loss computed ]{RESET}")
    print(f"  {DIM} system + user (2×video + 2×tag + prompt)           assistant output {RESET}")
    print(f"  {DIM}unmasked ≈ 129 tokens (think + answer){RESET}")

    # ── TARGET OUTPUT ────────────────────────────────────────────────────────
    h3("TARGET OUTPUT  (identical synthesis as 8B — same think+answer format)")
    answer_str = synthesize_think(record, action_class, offence_severity)
    stats = count_chars(answer_str)
    code(answer_str, GREEN)
    print()
    think_end2 = answer_str.find("</think>") + 8
    ans_sample2 = '{"action_class":"' + action_class + '"}'
    note(f"think block: ~{rough_token_estimate(answer_str[:think_end2])} tokens  |  "
         f"answer block: ~{rough_token_estimate(ans_sample2)+10} tokens")

    # ── INFERENCE ────────────────────────────────────────────────────────────
    h3("INFERENCE OUTPUT  (same extraction logic as 8B)")
    note("repetition_penalty=1.5 applied to prevent <think>\\n...<think>\\n... loops in 2B")
    note("expected format:")
    code("<think>", MAGENTA)
    code("Analyzing the video clip:", MAGENTA)
    code("  [observations from both views synthesized into one reasoning block]", MAGENTA)
    code("</think>", MAGENTA)
    code(f'<answer>{{"action_class": "{action_class}", "offence_severity": "{offence_severity}"}}</answer>', MAGENTA)


def render_comparison(action_id, record, clips, data_root, split):
    h1("COMPARISON: 8B Single-View vs 2B Multi-View")
    target = official_target(record)
    action_class, offence_severity, _, _ = target

    W = 70
    print(f"\n  {'Attribute':<30} {'8B Single-View':>18}  {'2B Multi-View':>18}")
    print(f"  {'─'*30} {'─'*18}  {'─'*18}")
    rows = [
        ("Model",              "Cosmos-Reason2-8B",    "Cosmos-Reason2-2B"),
        ("Parameters",         "8 billion",            "2 billion"),
        ("LoRA trainable",     "~60M (0.75%)",         "~17M (0.71%)"),
        ("Input views",        "clip_0 only (1 view)", f"clip_0+1 (2 views)"),
        ("Frames per view",    "16",                   "16"),
        ("Total frames",       "16",                   "32 (2×16)"),
        ("Pixel budget",       "uncapped (~9k tok)",   "4.5M → ~4.4k tok"),
        ("Seq len (approx)",   "~8,000–9,000",         "~4,385"),
        ("Loss target",        "full think+answer",    "full think+answer"),
        ("View augmentation",  "none (fixed clip_0)",  "random 2 of N (train)"),
        ("Eval view strategy", "fixed clip_0",         "fixed clip_0+clip_1"),
        ("rep. penalty (gen)", "1.5",                  "1.5"),
        ("Training time",      "~12h (3 epochs)",      "~4–4.5h (3 epochs)"),
        ("Step time",          "~6.2 s/step",          "~2.3 s/step"),
    ]
    for attr, sv, mv in rows:
        print(f"  {BOLD}{attr:<30}{RESET} {sv:>18}  {mv:>18}")

    h3("Input/Output flow diagram")
    print()
    # 8B
    print(f"  {BOLD}{CYAN}8B Single-View:{RESET}")
    print(f"  {CYAN}┌─────────────────────────────────────────────────────┐{RESET}")
    print(f"  {CYAN}│ USER: [{GREEN}clip_0.mp4{CYAN}] + [prompt]                         │{RESET}")
    print(f"  {CYAN}│       ↓ (seq_len ~8,000 tokens)                     │{RESET}")
    print(f"  {CYAN}│ MODEL: 8B LLM (QLoRA, 36 layers)                   │{RESET}")
    print(f"  {CYAN}│       ↓                                             │{RESET}")
    print(f"  {CYAN}│ ASST: <think>reasoning</think>                      │{RESET}")
    print(f"  {CYAN}│       <answer>{{action_class, offence_severity}}</answer> │{RESET}")
    print(f"  {CYAN}└─────────────────────────────────────────────────────┘{RESET}")
    print()
    # 2B
    print(f"  {BOLD}{MAGENTA}2B Multi-View:{RESET}")
    print(f"  {MAGENTA}┌─────────────────────────────────────────────────────┐{RESET}")
    print(f"  {MAGENTA}│ USER: [Main Camera] [{GREEN}clip_0.mp4{MAGENTA}]                    │{RESET}")
    print(f"  {MAGENTA}│       [Close-up]   [{GREEN}clip_1.mp4{MAGENTA}]                    │{RESET}")
    print(f"  {MAGENTA}│       [prompt]                                      │{RESET}")
    print(f"  {MAGENTA}│       ↓ (seq_len ~4,385 tokens, pixel budget=4.5M) │{RESET}")
    print(f"  {MAGENTA}│ MODEL: 2B LLM (QLoRA, 28 layers)                   │{RESET}")
    print(f"  {MAGENTA}│       ↓                                             │{RESET}")
    print(f"  {MAGENTA}│ ASST: <think>reasoning</think>                      │{RESET}")
    print(f"  {MAGENTA}│       <answer>{{action_class, offence_severity}}</answer> │{RESET}")
    print(f"  {MAGENTA}└─────────────────────────────────────────────────────┘{RESET}")

    # Gold answer
    h3(f"Gold answer for this sample (action_{action_id})")
    import json as _json
    gold = _json.dumps({"action_class": action_class, "offence_severity": offence_severity}, indent=2)
    code(gold, YELLOW)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/SoccerNet/mvfouls")
    p.add_argument("--split", default="Valid", choices=["Train", "Valid", "Test"])
    p.add_argument("--action-id", default=None,
                   help="Specific action ID to visualize (default: auto-pick first with ≥2 clips)")
    p.add_argument("--no-colour", action="store_true", help="Disable ANSI colour codes")
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_colour:
        global RESET, BOLD, CYAN, GREEN, YELLOW, MAGENTA, BLUE, DIM, RED
        RESET = BOLD = CYAN = GREEN = YELLOW = MAGENTA = BLUE = DIM = RED = ""

    data_root = Path(args.data_root)
    action_id, record, clips = get_sample(data_root, args.split, args.action_id)

    h1(f"SoccerNet-MVFoul  —  I/O Visualization  (split={args.split})")

    render_annotation(action_id, record, clips)
    render_8b_sv(action_id, record, clips, data_root, args.split)
    render_2b_mv(action_id, record, clips, data_root, args.split)
    render_comparison(action_id, record, clips, data_root, args.split)

    print(f"\n{DIM}Tip: --action-id <id> to visualize a different sample{RESET}\n")


if __name__ == "__main__":
    main()
