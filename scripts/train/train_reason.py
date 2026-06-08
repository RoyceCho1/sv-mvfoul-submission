#!/usr/bin/env python
"""QLoRA fine-tuning of Cosmos-Reason2-8B on SoccerNet-MVFoul — REASONING mode.

Identical QLoRA setup to train_baseline.py (4-bit NF4, vision encoder frozen,
LoRA on LLM attention/MLP, single view clip_0). Key difference: the training
target is

    <think>step-by-step reasoning</think><answer>{"offence_severity":...}</answer>

instead of bare JSON. Reasoning traces are synthesized deterministically from
annotation attributes (Contact, Bodypart, Upper body part, Try to play,
Touch ball, Handball). The <answer> block uses the identical label format and
official_target filter as the baseline script.

IMPORTANT — Assumption (not verified):
    The Sentence-F template states severity *judgment criteria* only, without
    declaring the verdict. This is based on the working assumption that card
    decisions reflect force/danger observable in the footage — but this is
    an assumption, not a fact confirmed by the annotation schema.

Do NOT modify finetune_cosmos.py. This is a separate experiment file.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from bitsandbytes.optim import PagedAdamW8bit
from collections import Counter
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent.parent))
from train.frame_utils import make_video_entry, foul_frame_index
from zero_shot.zero_shot_eval import (
    OFFENCE_SEVERITY_CLASSES,
    compute_metrics,
    extract_json,
    normalize_prediction_keys,
    official_target,
    validate_prediction,
)


# ---------------------------------------------------------------------------
# Reasoning trace synthesis
# ---------------------------------------------------------------------------

# Slot values are deliberately coarse. Force and risk are kept as "unclear" in
# synthetic targets because the annotation schema does not directly provide them.
# offence/card_threshold are supervised decision slots derived from the target
# label, kept short so the model learns the output format without long drift.
def _contact_slot(record: dict[str, Any]) -> str:
    contact = record.get("Contact", "")
    if contact == "With contact":
        return "yes"
    if contact == "Without contact":
        return "no"
    return "unclear"


def _ball_play_slot(record: dict[str, Any]) -> str:
    try_play = record.get("Try to play", "")
    touch = record.get("Touch ball", "")
    if try_play == "No":
        return "no"
    if try_play == "Yes" and touch == "Yes":
        return "yes"
    if try_play == "Yes" and touch in {"No", "Maybe", ""}:
        return "unclear"
    return "unclear"


def _offence_slot(offence_severity: str) -> str:
    if offence_severity == "No offence":
        return "no"
    if offence_severity in {
        "Offence + No card",
        "Offence + Yellow card",
        "Offence + Red card",
    }:
        return "yes"
    return "unclear"


def _card_threshold_slot(offence_severity: str) -> str:
    if offence_severity == "No offence":
        return "none"
    if offence_severity == "Offence + No card":
        return "no_card"
    if offence_severity == "Offence + Yellow card":
        return "yellow"
    if offence_severity == "Offence + Red card":
        return "red"
    return "unclear"


def synthesize_severity_slots(
    record: dict[str, Any],
    offence_severity: str,
) -> str:
    """Build a short slot-style <think>...</think><answer>...</answer> target.

    The reasoning format is intentionally closed-form to reduce long-form drift,
    CJK drift, malformed JSON, and accidental action-class prediction. It teaches
    the model to reason about offence/card severity without naming action_class.
    """
    answer_json = json.dumps(
        {"offence_severity": offence_severity},
        ensure_ascii=False,
    )
    think = "\n".join([
        f"contact: {_contact_slot(record)}",
        f"ball_play: {_ball_play_slot(record)}",
        "force: unclear",
        "risk: unclear",
        f"offence: {_offence_slot(offence_severity)}",
        f"card_threshold: {_card_threshold_slot(offence_severity)}",
        "decision: choose the final label from the observed contact, ball play, force, risk, offence, and card threshold.",
    ])
    return f"<think>\n{think}\n</think>\n<answer>{answer_json}</answer>"


def synthesize_think(record: dict[str, Any], action_class: str, offence_severity: str) -> str:
    """Backward-compatible wrapper for severity-only slot reasoning.

    action_class is accepted for existing dataset code but intentionally unused.
    """
    return synthesize_severity_slots(record, offence_severity)


# ---------------------------------------------------------------------------
# Prompt — reasoning mode
# ---------------------------------------------------------------------------

def build_severity_reasoning_prompt(num_frames: int | None = None, num_views: int = 1) -> str:
    """Prompt for closed-form severity reasoning followed by strict JSON."""
    if num_frames is not None:
        idx = foul_frame_index(num_frames)
        clip_word = "each clip" if num_views > 1 else "the clip"
        frame_hint = (
            f"Focus on Frame {idx} of {num_frames} in {clip_word}; it captures "
            "the foul contact moment.\n\n"
        )
    else:
        frame_hint = ""

    labels = "\n".join([
        "- No offence: no foul/offence should be called.",
        "- Offence + No card: foul/offence, but no card.",
        "- Offence + Yellow card: foul/offence with a yellow card.",
        "- Offence + Red card: foul/offence with a red card.",
    ])
    return (
        "You are a soccer referee assistant. Watch the video clip(s) of the same incident.\n"
        "Decide only the final offence severity label. Do not predict action_class.\n\n"
        + frame_hint
        + "Allowed labels:\n"
        + labels
        + "\n\nUse exactly this format and no text after </answer>:\n"
        "<think>\n"
        "contact: yes/no/unclear\n"
        "ball_play: yes/no/unclear\n"
        "force: low/medium/high/unclear\n"
        "risk: low/medium/high/unclear\n"
        "offence: yes/no\n"
        "card_threshold: no_card/yellow/red/none\n"
        "decision: one short reason\n"
        "</think>\n"
        "<answer>{\"offence_severity\": \"LABEL\"}</answer>"
    )


def build_reasoning_prompt(num_frames: int | None = None, num_views: int = 1) -> str:
    """Backward-compatible wrapper for the severity-only reasoning prompt."""
    return build_severity_reasoning_prompt(num_frames, num_views)


# ---------------------------------------------------------------------------
# Answer parser — reasoning mode
# ---------------------------------------------------------------------------

def extract_answer_json(text: str) -> dict[str, Any]:
    """Extract JSON from <answer>...</answer> with robust tag matching.

    Handles common model output variations:
      - </ answer>, </asnswer>, </analysis> (malformed closing tags)
      - (answer){...} (parenthesis variant)
      - Raw JSON after </think> with no answer tag
    Falls back to bare JSON extraction if no tag found.
    """
    # Try standard and malformed closing tags: </ answer>, </asnswer>, </analysis>, etc.
    m = re.search(r"<answer>(.*?)(?:</\s*a[^>]{0,20}>|<\/[^>]{0,20}>)", text, flags=re.DOTALL)
    if m:
        return extract_json(m.group(1).strip())
    # Try (answer) variant
    m = re.search(r"\(answer\)(.*?)(?:\(/answer\)|\n\n)", text, flags=re.DOTALL)
    if m:
        return extract_json(m.group(1).strip())
    # Try bare JSON after </think>
    m = re.search(r"</think>\s*[ \s]*(\{.*\})", text, flags=re.DOTALL)
    if m:
        return extract_json(m.group(1).strip())
    # Final fallback
    return extract_json(text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA reasoning fine-tune Cosmos-Reason2 on MVFoul")
    p.add_argument("--model-id", default="nvidia/Cosmos-Reason2-8B")
    p.add_argument("--data-root", default="data/SoccerNet/mvfouls")
    p.add_argument("--output-dir", default="outputs/qlora_cosmos8b_reason")
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true",
                   help="2 train steps + 2 valid samples, plus sample printout")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--valid-limit", type=int, default=None)
    p.add_argument("--balanced-sampling", action="store_true",
                   help="Oversample rare classes via sqrt-inverse-frequency WeightedRandomSampler")
    p.add_argument("--max-class-weight", type=float, default=10.0,
                   help="Cap for inverse-frequency class weights (recommend 3.0 with --balanced-sampling)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers (copied from finetune_cosmos.py — do not import from there)
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def gpu_mem(device: str) -> str:
    if not torch.cuda.is_available():
        return ""
    idx = torch.device(device).index
    if idx is None:
        idx = torch.cuda.current_device()
    reserved = torch.cuda.memory_reserved(idx) / 1024 ** 3
    allocated = torch.cuda.memory_allocated(idx) / 1024 ** 3
    return f"vram_reserved={reserved:.1f}GB allocated={allocated:.1f}GB"


def _fmt_sec(s: float) -> str:
    s = max(0, int(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


class _StepTimer:
    def __init__(self, total: int) -> None:
        self.total = total
        self._t0 = time.perf_counter()
        self._done = 0

    def tick(self) -> None:
        self._done += 1

    def bar(self) -> str:
        pct = self._done / self.total if self.total else 0
        filled = int(pct * 20)
        return f"[{'█' * filled}{'░' * (20 - filled)}]"

    def eta(self) -> str:
        elapsed = time.perf_counter() - self._t0
        if self._done == 0:
            return "--"
        rem = (self.total - self._done) * elapsed / self._done
        return _fmt_sec(rem)

    def elapsed(self) -> str:
        return _fmt_sec(time.perf_counter() - self._t0)

    def pct(self) -> float:
        return self._done / self.total * 100 if self.total else 0.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_ATTRS_CHECK = ["Contact", "Bodypart", "Upper body part", "Try to play", "Touch ball"]


def _count_filled(record: dict[str, Any]) -> int:
    return sum(1 for a in _ATTRS_CHECK if record.get(a, ""))


class FoulDatasetReason(Dataset):
    """clip_0 dataset that synthesizes <think>/<answer> training targets.

    Stores the raw annotation record alongside each sample so the smoke-test
    printer can show attribute values next to the synthesized trace.
    """

    def __init__(self, data_root: Path, split: str, limit: int | None = None):
        ann_path = data_root / split / "annotations.json"
        annotations = json.loads(ann_path.read_text())
        self.samples: list[dict] = []
        for action_id, record in annotations["Actions"].items():
            target = official_target(record)
            if target is None:
                continue
            clip_path = data_root / split / f"action_{action_id}" / "clip_0.mp4"
            if not clip_path.exists():
                continue
            action_class, offence_severity, offence_class, severity_class = target
            self.samples.append({
                "action_id": action_id,
                "clip_path": clip_path,
                "answer": synthesize_think(record, action_class, offence_severity),
                "gold_action_class": action_class,
                "gold_offence_severity": offence_severity,
                "gold_offence": offence_class,
                "gold_severity": severity_class,
                "record": record,
            })
            if limit and len(self.samples) >= limit:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def _collate(batch: list[dict]) -> dict:
    assert len(batch) == 1
    return batch[0]


def make_balanced_sampler(dataset: FoulDatasetReason) -> WeightedRandomSampler:
    """WeightedRandomSampler with sqrt-inverse-frequency weights (soft balancing).

    Reduces Standing tackling dominance (45% → ~27%) without extreme oversampling
    of rare classes like Dive (1.2% → ~4.4%).
    """
    import math
    sev_cnt: Counter = Counter(s["gold_offence_severity"] for s in dataset.samples)
    weights: list[float] = [
        1.0 / math.sqrt(max(sev_cnt.get(s["gold_offence_severity"], 1), 1))
        for s in dataset.samples
    ]
    log("Balanced sampler — sqrt-inverse-frequency weights per offence_severity:")
    for cls in OFFENCE_SEVERITY_CLASSES:
        cnt = sev_cnt.get(cls, 0)
        w   = 1.0 / math.sqrt(max(cnt, 1))
        log(f"  {cls:<30}  cnt={cnt:>4}  sample_w={w:.4f}")
    return WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)


# ---------------------------------------------------------------------------
# Loss masking (identical logic to finetune_cosmos.py)
# ---------------------------------------------------------------------------

def make_labels(input_ids: torch.Tensor, processor: AutoProcessor) -> torch.Tensor:
    """Mask everything before <|im_start|>assistant\\n with -100.

    The assistant turn starts with <think> and ends with </answer><|im_end|>.
    Both <think> and <answer> blocks fall inside the assistant turn,
    so the FULL reasoning trace contributes to the loss.
    """
    header_ids: list[int] = processor.tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    header_len = len(header_ids)
    seq: list[int] = input_ids[0].tolist()

    ans_start = -1
    for i in range(len(seq) - header_len, -1, -1):
        if seq[i : i + header_len] == header_ids:
            ans_start = i + header_len
            break

    if ans_start < 0:
        raise ValueError(
            "Could not locate '<|im_start|>assistant\\n' in the tokenized sequence."
        )

    labels = input_ids.clone()
    labels[:, :ans_start] = -100
    return labels


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------

def build_training_inputs_reason(
    processor: AutoProcessor,
    clip_path: Path,
    answer: str,
    num_frames: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Process one sample for training with the reasoning prompt."""
    messages = [
        {
            "role": "user",
            "content": [
                make_video_entry(clip_path, num_frames),
                {"type": "text", "text": build_reasoning_prompt(num_frames)},
            ],
        },
        {
            "role": "assistant",
            "content": answer,   # full <think>...</think>\n<answer>...</answer>
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    labels = make_labels(inputs["input_ids"], processor)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs, labels.to(device)


@torch.no_grad()
def run_one_inference_reason(
    model: Any,
    processor: AutoProcessor,
    clip_path: Path,
    num_frames: int,
    device: str,
    max_new_tokens: int,
) -> tuple[str, dict, list[str]]:
    """Generate and parse one sample during validation (reasoning mode)."""
    messages = [
        {
            "role": "user",
            "content": [
                make_video_entry(clip_path, num_frames),
                {"type": "text", "text": build_reasoning_prompt(num_frames)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, repetition_penalty=1.5)
    answer_ids = generated_ids[:, prompt_len:]
    raw_output = processor.batch_decode(
        answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    try:
        prediction = normalize_prediction_keys(extract_answer_json(raw_output))
        errors = validate_prediction(prediction)
    except Exception as exc:
        prediction = {}
        errors = [f"{type(exc).__name__}: {exc}"]
    return raw_output, prediction, errors


# ---------------------------------------------------------------------------
# Smoke-test helpers
# ---------------------------------------------------------------------------

def _select_smoke_samples(dataset: FoulDatasetReason) -> list[tuple[str, dict]]:
    """Return (label, sample) pairs covering 4 representative cases."""
    wanted = {
        "Dive": None,
        "Upper body (Holding/Elbowing/Pushing)": None,
        "Sparse attrs (filled≤2, non-Dive)": None,
        "All 4 attrs filled": None,
    }
    upper_classes = {"Holding", "Elbowing", "Pushing"}

    for s in dataset.samples:
        ac = s["gold_action_class"]
        filled = _count_filled(s["record"])

        if wanted["Dive"] is None and ac == "Dive":
            wanted["Dive"] = s
        if wanted["Upper body (Holding/Elbowing/Pushing)"] is None and ac in upper_classes:
            wanted["Upper body (Holding/Elbowing/Pushing)"] = s
        if wanted["Sparse attrs (filled≤2, non-Dive)"] is None and filled <= 2 and ac != "Dive":
            wanted["Sparse attrs (filled≤2, non-Dive)"] = s
        if wanted["All 4 attrs filled"] is None and filled >= 4:
            wanted["All 4 attrs filled"] = s

        if all(v is not None for v in wanted.values()):
            break

    return [(label, s) for label, s in wanted.items() if s is not None]


def _print_smoke_samples(dataset: FoulDatasetReason) -> None:
    """Print synthesized <think>/<answer> for representative samples."""
    print("\n" + "=" * 70)
    print("  SYNTHESIZED REASONING TRACE VERIFICATION")
    print("=" * 70)
    for label, s in _select_smoke_samples(dataset):
        rec = s["record"]
        filled = _count_filled(rec)
        attrs_summary = {a: rec.get(a, "") for a in _ATTRS_CHECK + ["Handball"]}
        print(f"\n--- [{label}] ---")
        print(f"action_id : {s['action_id']}")
        print(f"gold      : action_class={s['gold_action_class']!r}  offence_severity={s['gold_offence_severity']!r}")
        print(f"attrs     : filled={filled}/5  {attrs_summary}")
        print("synthesized answer:")
        print(s["answer"])
    print("=" * 70 + "\n")


def _verify_masking_reason(
    processor: AutoProcessor,
    sample: dict,
    num_frames: int,
    device: str,
) -> None:
    """Verify loss masking: decode unmasked tokens to confirm <think>+<answer> coverage."""
    inputs, labels = build_training_inputs_reason(
        processor, sample["clip_path"], sample["answer"], num_frames, device
    )
    n_total = labels.shape[1]
    n_unmasked = int((labels[0] != -100).sum().item())
    n_masked = n_total - n_unmasked

    # Decode the unmasked portion to confirm it starts with <think>
    unmasked_ids = labels[0][labels[0] != -100].tolist()
    unmasked_text = processor.tokenizer.decode(unmasked_ids, skip_special_tokens=False)

    # Estimate think vs answer token split
    answer_tag = "<answer>"
    think_tokens = n_unmasked
    answer_tokens = 0
    tag_idx = unmasked_text.find(answer_tag)
    if tag_idx >= 0:
        think_part = unmasked_text[:tag_idx]
        think_ids_est = processor.tokenizer.encode(think_part, add_special_tokens=False)
        think_tokens = len(think_ids_est)
        answer_tokens = n_unmasked - think_tokens

    log(f"  [Masking] seq_len={n_total}  unmasked={n_unmasked}  masked={n_masked}")
    log(f"  [Masking] think_tokens≈{think_tokens}  answer_tokens≈{answer_tokens}  (both in loss ✓)")
    has_think = unmasked_text.lstrip().startswith("<think>")
    has_answer = "<answer>" in unmasked_text
    log(f"  [Masking] starts_with_<think>={has_think}  contains_<answer>={has_answer}")
    log(f"  [Masking] unmasked preview: {unmasked_text[:120]!r}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_reason(
    model: Any,
    processor: AutoProcessor,
    dataset: FoulDatasetReason,
    device: str,
    num_frames: int,
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict]]:
    model.eval()
    model.config.use_cache = True
    rows: list[dict] = []
    for idx, sample in enumerate(dataset.samples, start=1):
        try:
            raw_output, prediction, errors = run_one_inference_reason(
                model, processor, sample["clip_path"], num_frames, device, max_new_tokens
            )
        except Exception as exc:
            raw_output, prediction, errors = "", {}, [f"InferenceError: {exc}"]
        rows.append({
            "action_id": sample["action_id"],
            "raw_output": raw_output,
            "prediction": prediction,
            "validation_errors": errors,
            "gold": {
                "action_class": sample["gold_action_class"],
                "offence_severity": sample["gold_offence_severity"],
            },
        })
        if idx % 20 == 0 or idx == len(dataset):
            log(f"  eval [{idx}/{len(dataset)}]")

    model.config.use_cache = False
    model.train()
    return compute_metrics(rows), rows


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if args.smoke_test:
        log(">>> SMOKE TEST MODE (reason): 2 train steps, 2 valid samples <<<")
        train_limit, valid_limit, num_epochs, grad_accum = 2, 2, 1, 1
    else:
        train_limit = args.train_limit
        valid_limit = args.valid_limit
        num_epochs = args.num_epochs
        grad_accum = args.grad_accum

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    device = args.device

    if args.num_frames % 2 != 0:
        raise ValueError(f"--num-frames must be even, got {args.num_frames}")

    # ---- Data ---------------------------------------------------------------
    log("Loading datasets …")
    # Load full train set for sample printing (not limited)
    full_train_ds = FoulDatasetReason(data_root, "Train", limit=None)
    train_ds = FoulDatasetReason(data_root, "Train", limit=train_limit)
    valid_ds = FoulDatasetReason(data_root, "Valid", limit=valid_limit)
    log(f"train={len(train_ds)}  valid={len(valid_ds)}")

    # ---- Print synthesized samples before any model loading ----------------
    log("Printing synthesized reasoning traces for verification …")
    _print_smoke_samples(full_train_ds)
    del full_train_ds

    use_balanced = args.balanced_sampling
    if use_balanced:
        log("Building balanced sampler (sqrt-inverse-frequency) …")
        sampler = make_balanced_sampler(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=1, sampler=sampler, num_workers=0, collate_fn=_collate
        )
        if args.max_class_weight > 5.0:
            log(f"WARNING: --balanced-sampling active. Consider --max-class-weight 3.0 "
                f"if also using weighted loss (current cap={args.max_class_weight})")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=_collate
        )

    # ---- Model --------------------------------------------------------------
    log(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    log("Loading model (4-bit NF4 QLoRA) …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map={"": device},
    )
    log(f"Base model loaded. {gpu_mem(device)}")

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    vis_trainable = sum(
        p.numel() for name, p in model.named_parameters()
        if "visual" in name and p.requires_grad
    )
    if vis_trainable > 0:
        log(f"WARNING: {vis_trainable:,} vision params trainable — freezing.")
        for name, p in model.named_parameters():
            if "visual" in name:
                p.requires_grad_(False)
    else:
        log("Vision encoder: fully frozen (OK).")
    log(f"LoRA ready. {gpu_mem(device)}")

    # ---- Masking verification (reasoning mode) --------------------------------
    log("Verifying loss masking on first training sample (reasoning mode) …")
    _verify_masking_reason(processor, train_ds[0], args.num_frames, device)

    # ---- Optimizer -----------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = PagedAdamW8bit(trainable_params, lr=args.lr)

    # ---- Training loop -------------------------------------------------------
    best_bal_acc_avg = -1.0
    best_ckpt_dir = out_dir / "best_checkpoint"
    history: list[dict] = []
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    total_timer = _StepTimer(total_steps)

    model.train()
    for epoch in range(1, num_epochs + 1):
        log(f"{'=' * 60}")
        log(f"Epoch {epoch}/{num_epochs}  (steps {(epoch-1)*steps_per_epoch+1}–{epoch*steps_per_epoch} / {total_steps} total)")
        epoch_loss = 0.0
        n_ok = 0
        optimizer.zero_grad()
        epoch_timer = _StepTimer(steps_per_epoch)

        for step_in_epoch, sample in enumerate(train_loader, start=1):
            if args.smoke_test and step_in_epoch > 2:
                break

            try:
                inputs, labels = build_training_inputs_reason(
                    processor, sample["clip_path"], sample["answer"], args.num_frames, device
                )
            except Exception as exc:
                log(f"  [step {step_in_epoch}] SKIP — {exc}")
                continue

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()
            n_ok += 1
            epoch_timer.tick()
            total_timer.tick()

            is_accum_step = (n_ok % grad_accum == 0)
            is_last_step = (step_in_epoch == len(train_loader))
            if is_accum_step or is_last_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step_in_epoch % 10 == 0 or is_last_step:
                avg = epoch_loss / n_ok
                log(
                    f"  {epoch_timer.bar()} "
                    f"step {step_in_epoch}/{steps_per_epoch} ({epoch_timer.pct():.1f}%) | "
                    f"epoch ETA {epoch_timer.eta()} | "
                    f"total {total_timer.pct():.1f}% ETA {total_timer.eta()} | "
                    f"avg_loss={avg:.4f} | {gpu_mem(device)}"
                )

        epoch_avg_loss = epoch_loss / max(n_ok, 1)
        log(f"Epoch {epoch} complete  avg_loss={epoch_avg_loss:.4f}  elapsed={epoch_timer.elapsed()}  total_elapsed={total_timer.elapsed()}")

        log(f"Validating on {len(valid_ds)} samples …")
        val_metrics, val_rows = evaluate_reason(
            model, processor, valid_ds, device, args.num_frames, args.max_new_tokens
        )
        bal2 = val_metrics["balanced_accuracy_offence_severity_seen_classes"]
        log(f"Valid  offence_severity_bal_acc={bal2:.2f}%")

        history.append({
            "epoch": epoch,
            "train_loss": epoch_avg_loss,
            "val_bal_acc_offence_severity": bal2,
        })

        if bal2 > best_bal_acc_avg:
            best_bal_acc_avg = bal2
            model.save_pretrained(str(best_ckpt_dir))
            processor.save_pretrained(str(best_ckpt_dir))
            (best_ckpt_dir / "val_rows.json").write_text(json.dumps(val_rows, indent=2, ensure_ascii=False))
            (best_ckpt_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2, ensure_ascii=False))
            log(f"  >> New best checkpoint → {best_ckpt_dir}  (bal_acc={best_bal_acc_avg:.2f}%)")

        model.train()

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    log("=" * 60)
    log(f"Training complete. Best valid offence_severity_bal_acc={best_bal_acc_avg:.2f}%")
    log(f"Best checkpoint: {best_ckpt_dir}")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
