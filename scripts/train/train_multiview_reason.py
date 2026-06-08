#!/usr/bin/env python
"""QLoRA fine-tuning — Cosmos-Reason2-2B, multi-view (2 views), reasoning mode.

Extends train_reason.py (single-view 8B) to 2-view input:
  - Training : random 2 clips per action each step  (augmentation)
  - Evaluation: clip_0 + clip_1 fixed               (deterministic)

Everything else is identical to the single-view reasoning script:
  synthesize_think / build_reasoning_prompt / extract_answer_json /
  make_labels / compute_metrics are imported directly — no duplication.

─── Frame-sampling notes (VARS spec) ────────────────────────────────────────
  VARS spec: each 5-second clip is cut so the foul occurs at approximately
  the 3-second mark (60% through the clip, ≈frame 75 at 25 fps), NOT at the
  center. Uniform nframes-based sampling is used here because:
    1. nframes covers the full clip including the foul region.
    2. We do not have per-clip start offsets in the annotation, so we cannot
       crop around frame 75 without additional metadata.
  If a tighter crop is needed later, target [frame ~50 – ~100] per clip.

  Assumption (not verified): replay clips (Replay speed > 1.0) show the same
  incident in slow motion; the same nframes uniform sampling is applied.
  High-speed replays (e.g., 10×) cover a much shorter real-time window than
  the main camera clip, so temporal ranges across views may not align — this
  is an accepted limitation of uniform sampling without per-clip timestamps.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from collections import Counter
from bitsandbytes.optim import PagedAdamW8bit
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent.parent))
from zero_shot.zero_shot_eval import (
    OFFENCE_SEVERITY_CLASSES,
    compute_metrics,
    official_target,
    normalize_prediction_keys,
    validate_prediction,
)
from train.train_reason import (
    synthesize_think,
    build_reasoning_prompt,
    extract_answer_json,
    make_labels,
    set_seed,
    log,
    gpu_mem,
    _fmt_sec,
    _StepTimer,
)
from train.frame_utils import make_video_entry




# ---------------------------------------------------------------------------
# Pixel budget control (transformers 5.9.0+ Qwen3VL)
# ---------------------------------------------------------------------------

def apply_pixel_budget(processor: AutoProcessor, longest_edge: int) -> None:
    """Set the 3-D pixel budget on processor.video_processor.

    In transformers 5.9.0+, the Qwen3VL video_processor uses
    processor.video_processor.size["longest_edge"] as the TOTAL pixel
    constraint (T × H × W). Content-dict max_pixels is silently ignored.

    longest_edge examples (1280×720, nframes=8, 2 views):
      3_000_000 → seq_len ≈ 5,700  VRAM ≈ 19 GB
      4_500_000 → seq_len ≈ 7,000  VRAM ≈ 22 GB
      5_000_000 → seq_len ≈ 7,400  VRAM ≈ 23 GB
    """
    import math
    if not hasattr(processor, "video_processor"):
        log("WARNING: no video_processor — pixel budget not applied")
        return

    vp = processor.video_processor
    # SizeDict (UserDict subclass) supports dict-style and attr-style access
    try:
        old_le = vp.size["longest_edge"]
        vp.size["longest_edge"] = longest_edge
    except (TypeError, AttributeError, KeyError):
        old_le = getattr(vp.size, "longest_edge", "?")
        vp.size.longest_edge = longest_edge

    # Estimate effect on 1280×720 clips
    t, h, w, factor = 8, 720, 1280, 32      # typical 2B config
    total = t * h * w
    if total > longest_edge:
        beta = math.sqrt(total / longest_edge)
        h_bar = math.floor(h / (beta * factor)) * factor
        w_bar = math.floor(w / (beta * factor)) * factor
        tok_pf = (h_bar // factor) * (w_bar // factor)
        seq_est = 2 * t * tok_pf + 200       # 2 views + text
        log(f"Pixel budget: longest_edge {old_le:,} → {longest_edge:,}")
        log(f"  1280×720 × {t}f per view, 2 views: resize {h_bar}×{w_bar}  "
            f"tok/frame≈{tok_pf}  seq_len≈{seq_est}")
    else:
        log(f"Pixel budget: longest_edge {old_le:,} → {longest_edge:,} (no downscale for this clip size)")

# ---------------------------------------------------------------------------
# View selection helpers
# ---------------------------------------------------------------------------

def _available_clips(
    data_root: Path, split: str, action_id: str, record: dict
) -> list[tuple[int, dict]]:
    """Return [(clip_idx, clip_info), ...] for existing .mp4 files."""
    return [
        (i, c)
        for i, c in enumerate(record.get("Clips", []))
        if (data_root / split / f"action_{action_id}" / f"clip_{i}.mp4").exists()
    ]


def select_train_clips(
    data_root: Path, split: str, action_id: str, record: dict, rng: random.Random
) -> list[tuple[int, dict]]:
    """Random 2 clips for training (augmentation). Falls back to 1 if only 1 available."""
    avail = _available_clips(data_root, split, action_id, record)
    if len(avail) <= 2:
        return avail
    chosen = rng.sample(avail, 2)
    return sorted(chosen, key=lambda x: x[0])   # keep ascending clip index


def select_eval_clips(
    data_root: Path, split: str, action_id: str, record: dict
) -> list[tuple[int, dict]]:
    """Fixed clip_0 + clip_1 for evaluation (deterministic)."""
    avail = _available_clips(data_root, split, action_id, record)
    # Take the two lowest-indexed clips (always clip_0 and clip_1 when both exist)
    return avail[:2]


def camera_tag(clip_info: dict) -> str:
    """Convert annotation Camera type to a bracketed prompt tag."""
    return f"[{clip_info.get('Camera type', 'Camera')}]"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FoulDatasetMultiview(Dataset):
    """Multi-view foul dataset.

    Stores the annotation record and all available clip paths per sample.
    View selection (random/fixed) happens in the collate / input-building step,
    not here, so the dataset itself is view-agnostic.
    """

    def __init__(
        self,
        data_root: Path,
        split: str,
        limit: int | None = None,
    ) -> None:
        ann_path = data_root / split / "annotations.json"
        annotations = json.loads(ann_path.read_text())
        self.data_root = data_root
        self.split = split
        self.samples: list[dict] = []

        for action_id, record in annotations["Actions"].items():
            target = official_target(record)
            if target is None:
                continue
            avail = _available_clips(data_root, split, action_id, record)
            if not avail:
                continue
            action_class, offence_severity, offence_class, severity_class = target
            self.samples.append({
                "action_id": action_id,
                "record": record,
                "available_clips": avail,   # [(idx, clip_info), ...]
                "answer": synthesize_think(record, action_class, offence_severity),
                "gold_action_class": action_class,
                "gold_offence_severity": offence_severity,
                "gold_offence": offence_class,
                "gold_severity": severity_class,
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


# ---------------------------------------------------------------------------
# Balanced sampler
# ---------------------------------------------------------------------------

def make_balanced_sampler(dataset: FoulDatasetMultiview) -> WeightedRandomSampler:
    """WeightedRandomSampler with sqrt-inverse-frequency weights (soft balancing).

    Hard weighting (1/count) would give Dive 37× more draws than Standing
    tackling, risking severe overfitting on ~28 Dive examples.
    Sqrt weighting reduces this to ~6×, preserving diversity while still
    improving minority-class coverage.
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
# Multi-view input builder
# ---------------------------------------------------------------------------

def build_multiview_content(
    clips: list[tuple[int, dict]],
    data_root: Path,
    split: str,
    action_id: str,
    num_frames: int,
    max_pixels: int | None = None,
) -> list[dict]:
    """Build the content list: [tag, video, tag, video, prompt] for N clips.

    max_pixels: per-frame pixel budget (controls tokens/frame via smart_resize).
    VARS spec: foul occurs ~3 s into the 5-s clip (≈frame 75 at 25 fps), not at
    center. Uniform nframes sampling is used because clip-start offsets are not
    in the annotation. If max_pixels is None, qwen_vl_utils uses its default
    (VIDEO_FRAME_MAX_PIXELS = 768 * image_factor²).
    Assumption (not verified): high-speed replay clips cover a shorter real-time
    window, so cross-view temporal ranges may not align precisely.
    """
    content: list[dict] = []
    for clip_idx, clip_info in clips:
        tag = camera_tag(clip_info)
        vid_path = str(data_root / split / f"action_{action_id}" / f"clip_{clip_idx}.mp4")
        content.append({"type": "text", "text": tag})
        content.append(make_video_entry(vid_path, num_frames))
    content.append({"type": "text", "text": build_reasoning_prompt(num_frames, num_views=len(clips))})
    return content


def build_training_inputs_multiview(
    processor: AutoProcessor,
    sample: dict,
    clips: list[tuple[int, dict]],
    num_frames: int,
    device: str,
    max_pixels: int | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    data_root = Path(sample["_data_root"])
    split = sample["_split"]
    content = build_multiview_content(
        clips, data_root, split, sample["action_id"], num_frames, max_pixels
    )
    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": sample["answer"]},
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
def run_one_inference_multiview(
    model: Any,
    processor: AutoProcessor,
    sample: dict,
    clips: list[tuple[int, dict]],
    num_frames: int,
    device: str,
    max_new_tokens: int,
    max_pixels: int | None = None,
) -> tuple[str, dict, list[str]]:
    data_root = Path(sample["_data_root"])
    split = sample["_split"]
    content = build_multiview_content(
        clips, data_root, split, sample["action_id"], num_frames, max_pixels
    )
    messages = [{"role": "user", "content": content}]
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
    raw_output = processor.batch_decode(
        generated_ids[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
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

def _print_view_composition(
    sample: dict,
    clips: list[tuple[int, dict]],
    num_frames: int,
    label: str = "",
    max_pixels: int | None = None,
) -> None:
    """Print which clips and Camera types were selected for a sample."""
    print(f"  [{label}] action_{sample['action_id']}  "
          f"gold={sample['gold_action_class']!r} / {sample['gold_offence_severity']!r}")
    print(f"    available clips: {len(sample['available_clips'])}  nframes/view={num_frames}")
    for clip_idx, clip_info in clips:
        speed = clip_info.get("Replay speed", "?")
        cam = clip_info.get("Camera type", "?")
        clip_path = (Path(sample["_data_root"]) / sample["_split"]
                / f"action_{sample['action_id']}" / f"clip_{clip_idx}.mp4")
        print(f"    clip_{clip_idx}: cam={cam!r}  speed={speed}  "
              f"nframes={num_frames}  exists={clip_path.exists()}")


def _verify_masking_multiview(
    processor: AutoProcessor,
    sample: dict,
    clips: list[tuple[int, dict]],
    num_frames: int,
    device: str,
    max_pixels: int | None = None,
) -> None:
    """Verify <think>+<answer> are unmasked; print token counts."""
    inputs, labels = build_training_inputs_multiview(
        processor, sample, clips, num_frames, device, max_pixels
    )
    n_total = labels.shape[1]
    n_unmasked = int((labels[0] != -100).sum().item())
    n_masked = n_total - n_unmasked

    unmasked_ids = labels[0][labels[0] != -100].tolist()
    unmasked_text = processor.tokenizer.decode(unmasked_ids, skip_special_tokens=False)

    tag_idx = unmasked_text.find("<answer>")
    if tag_idx >= 0:
        n_think_est = len(processor.tokenizer.encode(
            unmasked_text[:tag_idx], add_special_tokens=False))
        n_answer_est = n_unmasked - n_think_est
    else:
        n_think_est, n_answer_est = n_unmasked, 0

    n_views = len(clips)
    video_tokens = n_masked  # rough proxy
    tok_per_frame_est = (n_total - 165 - n_unmasked) / (n_views * num_frames) if n_views * num_frames else 0
    log(f"  [masking] seq_len={n_total}  unmasked={n_unmasked}  masked={n_masked}")
    log(f"  [masking] video_tokens≈{n_total - 165 - n_unmasked}  "
        f"tok/frame≈{tok_per_frame_est:.0f}  ({n_views} views × {num_frames} frames)")
    log(f"  [masking] think≈{n_think_est}  answer≈{n_answer_est}  (both in loss)")
    log(f"  [masking] starts_with_<think>={unmasked_text.lstrip().startswith('<think>')}"
        f"  has_<answer>={'<answer>' in unmasked_text}")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_multiview(
    model: Any,
    processor: AutoProcessor,
    dataset: FoulDatasetMultiview,
    num_frames: int,
    device: str,
    max_new_tokens: int,
    max_pixels: int | None = None,
) -> tuple[dict[str, Any], list[dict]]:
    model.eval()
    model.config.use_cache = True
    rows: list[dict] = []
    for idx, sample in enumerate(dataset.samples, start=1):
        clips = select_eval_clips(
            Path(sample["_data_root"]), sample["_split"],
            sample["action_id"], sample["record"]
        )
        try:
            raw_output, prediction, errors = run_one_inference_multiview(
                model, processor, sample, clips, num_frames, device, max_new_tokens, max_pixels
            )
        except Exception as exc:
            raw_output, prediction, errors = "", {}, [f"InferenceError: {exc}"]
        rows.append({
            "action_id": sample["action_id"],
            "n_views": len(clips),
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
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-view reasoning QLoRA fine-tune — Cosmos-Reason2-2B"
    )
    p.add_argument("--model-id", default="nvidia/Cosmos-Reason2-2B")
    p.add_argument("--data-root", default="data/SoccerNet/mvfouls")
    p.add_argument("--output-dir", default="outputs/qlora_cosmos2b_multiview_reason")
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--num-frames", type=int, default=8,
                   help="Frames per view (must be even). Total tokens ≈ 2× single-view.")
    p.add_argument(
        "--max-pixels", type=int, default=3_000_000,
        help=(
            "processor.video_processor.size['longest_edge'] — 3-D total pixel budget (T×H×W). "
            "3_000_000 (default): 1280×720×8 frames → seq_len≈5700, VRAM≈19 GB. "
            "4_500_000: seq_len≈7000, VRAM≈22 GB. "
            "0 or negative: skip (use preprocessor_config default 16,777,216 → OOM)."
        ),
    )
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true",
                   help="2 train steps + 2 valid samples + view/masking printout")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--valid-limit", type=int, default=None)
    p.add_argument("--balanced-sampling", action="store_true",
                   help="Oversample rare classes via sqrt-inverse-frequency WeightedRandomSampler")
    p.add_argument("--max-class-weight", type=float, default=10.0,
                   help="Cap for inverse-frequency class weights (recommend 3.0 with --balanced-sampling)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    rng = random.Random(args.seed)   # separate RNG for view selection

    if args.smoke_test:
        log(">>> SMOKE TEST MODE (multiview-reason): 2 steps, 2 valid <<<")
        train_limit, valid_limit, num_epochs, grad_accum = 4, 4, 1, 1
    else:
        train_limit = args.train_limit
        valid_limit = args.valid_limit
        num_epochs = args.num_epochs
        grad_accum = args.grad_accum

    if args.num_frames % 2 != 0:
        raise ValueError(f"--num-frames must be even, got {args.num_frames}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    device = args.device

    # ---- Data ---------------------------------------------------------------
    log("Loading datasets …")
    train_ds = FoulDatasetMultiview(data_root, "Train", limit=train_limit)
    valid_ds = FoulDatasetMultiview(data_root, "Valid", limit=valid_limit)

    # Inject data_root/split so build_* helpers can construct paths
    for s in train_ds.samples + valid_ds.samples:
        s["_data_root"] = str(data_root)
        s["_split"] = "Train" if s in train_ds.samples else "Valid"

    # Fix split label (simpler)
    for s in train_ds.samples:
        s["_split"] = "Train"
    for s in valid_ds.samples:
        s["_split"] = "Valid"

    log(f"train={len(train_ds)}  valid={len(valid_ds)}")

    # ---- Smoke: view composition printout (before model load) ---------------
    if args.smoke_test:
        log("=== View composition verification ===")
        for i, sample in enumerate(train_ds.samples[:2]):
            clips = select_train_clips(
                data_root, "Train", sample["action_id"],
                sample["record"], rng
            )
            _print_view_composition(sample, clips, args.num_frames, f"train sample {i}", args.max_pixels)
        for i, sample in enumerate(valid_ds.samples[:2]):
            clips = select_eval_clips(
                data_root, "Valid", sample["action_id"], sample["record"]
            )
            _print_view_composition(sample, clips, args.num_frames, f"valid sample {i}", args.max_pixels)
        log("=====================================")

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
    if args.max_pixels and args.max_pixels > 0:
        apply_pixel_budget(processor, args.max_pixels)

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

    # ---- Masking verification -----------------------------------------------
    log("Verifying masking on first training sample (multi-view) …")
    _s0 = train_ds.samples[0]
    _clips0 = select_train_clips(
        data_root, "Train", _s0["action_id"], _s0["record"], rng
    )
    _verify_masking_multiview(processor, _s0, _clips0, args.num_frames, device, args.max_pixels)
    torch.cuda.empty_cache()
    log(f'GPU after masking check: {gpu_mem(device)}')

    # ---- Optimizer ----------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = PagedAdamW8bit(trainable_params, lr=args.lr)

    # ---- Training loop ------------------------------------------------------
    best_bal_acc_avg = -1.0
    best_ckpt_dir = out_dir / "best_checkpoint"
    history: list[dict] = []
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    total_timer = _StepTimer(total_steps)
    smoke_steps_done = 0

    model.train()
    for epoch in range(1, num_epochs + 1):
        log(f"{'=' * 60}")
        log(f"Epoch {epoch}/{num_epochs}  "
            f"(steps {(epoch-1)*steps_per_epoch+1}–{epoch*steps_per_epoch} / {total_steps} total)")
        epoch_loss = 0.0
        n_ok = 0
        optimizer.zero_grad()
        epoch_timer = _StepTimer(steps_per_epoch)

        for step_in_epoch, sample in enumerate(train_loader, start=1):
            if args.smoke_test and smoke_steps_done >= 2:
                break

            clips = select_train_clips(
                data_root, "Train", sample["action_id"],
                sample["record"], rng
            )
            try:
                inputs, labels = build_training_inputs_multiview(
                    processor, sample, clips, args.num_frames, device, args.max_pixels
                )
            except Exception as exc:
                log(f"  [step {step_in_epoch}] SKIP — {exc}")
                continue

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()
            n_ok += 1
            smoke_steps_done += 1
            epoch_timer.tick()
            total_timer.tick()

            is_accum_step = (n_ok % grad_accum == 0)
            is_last_step = (step_in_epoch == len(train_loader)) or \
                           (args.smoke_test and smoke_steps_done >= 2)
            if is_accum_step or is_last_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step_in_epoch % 10 == 0 or is_last_step:
                avg = epoch_loss / n_ok
                log(
                    f"  {epoch_timer.bar()} "
                    f"step {step_in_epoch}/{steps_per_epoch} ({epoch_timer.pct():.1f}%) | "
                    f"epoch ETA {epoch_timer.eta()} | total ETA {total_timer.eta()} | "
                    f"avg_loss={avg:.4f} | n_views={len(clips)} | {gpu_mem(device)}"
                )
            if args.smoke_test and smoke_steps_done >= 2:
                break

        epoch_avg_loss = epoch_loss / max(n_ok, 1)
        log(f"Epoch {epoch} complete  avg_loss={epoch_avg_loss:.4f}  "
            f"elapsed={epoch_timer.elapsed()}  total_elapsed={total_timer.elapsed()}")

        # ---- Validation -----------------------------------------------------
        log(f"Validating on {len(valid_ds)} samples …")
        val_metrics, val_rows = evaluate_multiview(
            model, processor, valid_ds, args.num_frames, device, args.max_new_tokens, args.max_pixels
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
            (best_ckpt_dir / "val_rows.json").write_text(
                json.dumps(val_rows, indent=2, ensure_ascii=False))
            (best_ckpt_dir / "val_metrics.json").write_text(
                json.dumps(val_metrics, indent=2, ensure_ascii=False))
            log(f"  >> Best checkpoint → {best_ckpt_dir}  (bal_acc={best_bal_acc_avg:.2f}%)")

        model.train()
        if args.smoke_test:
            break   # single epoch in smoke mode

    (out_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False))
    log("=" * 60)
    log(f"Training complete. Best valid offence_severity_bal_acc={best_bal_acc_avg:.2f}%")
    log(f"Best checkpoint: {best_ckpt_dir}")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
