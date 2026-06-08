#!/usr/bin/env python
"""QLoRA fine-tuning — Cosmos-Reason2-8B, reasoning mode, answer-only loss + weighted loss.

Three improvements over train_reason.py:

1. Answer-only loss (--answer-only, default ON):
   Loss computed ONLY on <answer>JSON</answer> tokens, NOT on <think>...</think>.
   Motivation: <think> blocks are annotation-derived templates.  Training on them
   teaches format memorization, not video-based reasoning.  The <answer> JSON is
   what actually gets evaluated, so focusing the loss there improves classification.

   Implementation: find </think> (token 151668) in the sequence; mask everything
   up to and including </think>\n, leaving only <answer>...</answer> in the loss.

2. Weighted loss (--weighted-loss, default ON):
   Each training sample is scaled by the inverse-frequency weight of its class.
   Caps at max_weight=10 to avoid extreme values for very rare classes.
   Combined weight = mean(action_weight, severity_weight).
   Motivation: Standing tackling dominates (44.8% of train); Dive has 1.2%.
   Balanced accuracy penalises minority-class failures equally.

3. Balanced sampling (--balanced-sampling, default OFF):
   Rare-class samples are drawn more frequently via WeightedRandomSampler.
   Uses sqrt-inverse-frequency weighting (soft balancing) to avoid extreme
   repetition of very rare classes (e.g., Dive: 28 samples).
   Formula: sample_weight = 1 / sqrt(class_count)
   Effect: reduces Standing-tackling:Dive ratio from 37× → ~6× per epoch.
   When combined with weighted loss, use --max-class-weight 3.0 to avoid
   double-correcting for class imbalance.

Everything else is identical to train_reason.py — imports, prompts, eval,
checkpoint saving, LoRA config, training loop structure.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from bitsandbytes.optim import PagedAdamW8bit
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent.parent))
from zero_shot.zero_shot_eval import (
    ACTION_CLASSES,
    OFFENCE_SEVERITY_CLASSES,
    compute_metrics,
    extract_json,
    normalize_prediction_keys,
    official_target,
    validate_prediction,
)
from train.train_reason import (
    FoulDatasetReason,
    synthesize_think,
    build_reasoning_prompt,
    extract_answer_json,
    run_one_inference_reason,
    evaluate_reason,
    set_seed,
    log,
    gpu_mem,
    _fmt_sec,
    _StepTimer,
)
from train.frame_utils import make_video_entry

# ── Token IDs (verified from Cosmos-Reason2-8B tokenizer) ──────────────────
_THINK_END_ID = 151668   # </think>
_NEWLINE_ID   = 198      # \n


# ---------------------------------------------------------------------------
# Answer-only loss masking
# ---------------------------------------------------------------------------

def make_labels_answer_only(
    input_ids: torch.Tensor,
    processor: AutoProcessor,
) -> torch.Tensor:
    """Mask everything except <answer>JSON</answer> tokens.

    Sequence structure inside the assistant turn:
      <think>\n  ...reasoning...  </think>\n  <answer>JSON</answer>  <|im_end|>

    We locate </think> (token 151668) and mask everything up to and including
    the \n that follows it.  Only <answer>...</answer> contributes to the loss.

    Falls back to full-answer masking if </think> is not found (e.g. bad sample).
    """
    header_ids: list[int] = processor.tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    header_len = len(header_ids)
    seq: list[int] = input_ids[0].tolist()

    # Find last occurrence of the assistant header
    ans_start = -1
    for i in range(len(seq) - header_len, -1, -1):
        if seq[i : i + header_len] == header_ids:
            ans_start = i + header_len
            break
    if ans_start < 0:
        raise ValueError("Could not locate <|im_start|>assistant\\n in sequence.")

    # Find </think> after ans_start and skip the trailing \n
    answer_start = ans_start  # fallback: full assistant turn
    for i in range(ans_start, len(seq)):
        if seq[i] == _THINK_END_ID:
            j = i + 1
            # skip one newline after </think>
            if j < len(seq) and seq[j] == _NEWLINE_ID:
                j += 1
            answer_start = j
            break

    labels = input_ids.clone()
    labels[:, :answer_start] = -100
    return labels


def make_labels_full(
    input_ids: torch.Tensor,
    processor: AutoProcessor,
) -> torch.Tensor:
    """Full assistant turn loss (think + answer) — same as train_reason.py."""
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
        raise ValueError("Could not locate <|im_start|>assistant\\n in sequence.")
    labels = input_ids.clone()
    labels[:, :ans_start] = -100
    return labels


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(
    dataset: FoulDatasetReason,
    max_weight: float = 10.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Inverse-frequency weights, capped at max_weight.

    Returns (action_weights, severity_weights) dicts mapping class name → weight.
    Weight formula: n_samples / (n_classes × class_count), capped at max_weight.
    """
    action_cnt: Counter = Counter(s["gold_action_class"] for s in dataset.samples)
    sev_cnt: Counter    = Counter(s["gold_offence_severity"] for s in dataset.samples)
    n = len(dataset.samples)

    action_w = {
        cls: min(n / (len(ACTION_CLASSES) * max(action_cnt.get(cls, 1), 1)), max_weight)
        for cls in ACTION_CLASSES
    }
    sev_w = {
        cls: min(n / (len(OFFENCE_SEVERITY_CLASSES) * max(sev_cnt.get(cls, 1), 1)), max_weight)
        for cls in OFFENCE_SEVERITY_CLASSES
    }

    log("Class weights (action):")
    for cls, w in action_w.items():
        log(f"  {cls:<25}  cnt={action_cnt.get(cls,0):>4}  w={w:.2f}")
    log("Class weights (severity):")
    for cls, w in sev_w.items():
        log(f"  {cls:<25}  cnt={sev_cnt.get(cls,0):>4}  w={w:.2f}")

    return action_w, sev_w


def sample_weight(
    sample: dict,
    action_w: dict[str, float],
    sev_w: dict[str, float],
) -> float:
    """Combined weight for one sample = mean(action_w, severity_w)."""
    aw = action_w.get(sample["gold_action_class"], 1.0)
    sw = sev_w.get(sample["gold_offence_severity"], 1.0)
    return (aw + sw) / 2.0


# ---------------------------------------------------------------------------
# Balanced sampler
# ---------------------------------------------------------------------------

def make_balanced_sampler(dataset: FoulDatasetReason) -> WeightedRandomSampler:
    """WeightedRandomSampler with sqrt-inverse-frequency weights (soft balancing).

    Hard weighting (1/count) would give Dive 37× more draws than Standing
    tackling, risking severe overfitting on ~28 Dive examples.
    Sqrt weighting reduces this to ~6×, preserving diversity while still
    improving minority-class coverage.

    Samples with replacement so every class gets proportional draws even
    when rare classes have far fewer unique examples.
    """
    import math
    action_cnt: Counter = Counter(s["gold_action_class"] for s in dataset.samples)

    weights: list[float] = [
        1.0 / math.sqrt(max(action_cnt.get(s["gold_action_class"], 1), 1))
        for s in dataset.samples
    ]

    log("Balanced sampler — sqrt-inverse-frequency weights per action class:")
    for cls in ACTION_CLASSES:
        cnt = action_cnt.get(cls, 0)
        w   = 1.0 / math.sqrt(max(cnt, 1))
        log(f"  {cls:<25}  cnt={cnt:>4}  sample_w={w:.4f}")

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Training input builder
# ---------------------------------------------------------------------------

def build_training_inputs(
    processor: AutoProcessor,
    sample: dict,
    num_frames: int,
    device: str,
    answer_only: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Process one sample; use answer-only or full-answer loss masking."""
    messages = [
        {
            "role": "user",
            "content": [
                make_video_entry(sample["clip_path"], num_frames),
                {"type": "text", "text": build_reasoning_prompt()},
            ],
        },
        {"role": "assistant", "content": sample["answer"]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    label_fn = make_labels_answer_only if answer_only else make_labels_full
    labels = label_fn(inputs["input_ids"], processor)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs, labels.to(device)


def _collate(batch):
    assert len(batch) == 1
    return batch[0]


# ---------------------------------------------------------------------------
# Sanity check helper
# ---------------------------------------------------------------------------

def _verify_masking(
    processor: AutoProcessor,
    sample: dict,
    num_frames: int,
    device: str,
    answer_only: bool,
) -> None:
    inputs, labels = build_training_inputs(
        processor, sample, num_frames, device, answer_only
    )
    n_total   = labels.shape[1]
    n_unmasked = int((labels[0] != -100).sum().item())
    n_masked   = n_total - n_unmasked

    unmasked_ids  = labels[0][labels[0] != -100].tolist()
    unmasked_text = processor.tokenizer.decode(unmasked_ids, skip_special_tokens=False)

    has_think  = "<think>"  in unmasked_text
    has_answer = "<answer>" in unmasked_text

    log(f"  [masking] seq_len={n_total}  unmasked={n_unmasked}  masked={n_masked}")
    log(f"  [masking] answer_only={answer_only}  "
        f"has_<think>_in_loss={has_think}  has_<answer>_in_loss={has_answer}")
    if answer_only:
        assert not has_think, "FATAL: <think> tokens in loss — answer-only masking failed"
        assert has_answer,    "FATAL: <answer> not in loss — masking too aggressive"
    log(f"  [masking] preview: {unmasked_text[:80]!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QLoRA reasoning fine-tune — answer-only loss + weighted loss"
    )
    p.add_argument("--model-id",     default="nvidia/Cosmos-Reason2-8B")
    p.add_argument("--data-root",    default="data/SoccerNet/mvfouls")
    p.add_argument("--output-dir",   default="outputs/qlora_cosmos8b_reason_answer")
    p.add_argument("--num-epochs",   type=int,   default=3)
    p.add_argument("--num-frames",   type=int,   default=16)
    p.add_argument("--grad-accum",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--smoke-test",   action="store_true")
    p.add_argument("--train-limit",  type=int,   default=None)
    p.add_argument("--valid-limit",  type=int,   default=None)
    # Ablation flags
    p.add_argument("--no-answer-only",    action="store_true",
                   help="Use full think+answer loss instead of answer-only")
    p.add_argument("--no-weighted-loss",  action="store_true",
                   help="Disable class-frequency weighting")
    p.add_argument("--max-class-weight",  type=float, default=10.0,
                   help="Cap for inverse-frequency class weights (recommend 3.0 with --balanced-sampling)")
    p.add_argument("--balanced-sampling", action="store_true",
                   help="Oversample rare classes via sqrt-inverse-frequency WeightedRandomSampler")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    answer_only    = not args.no_answer_only
    use_weights    = not args.no_weighted_loss
    use_balanced   = args.balanced_sampling

    if args.smoke_test:
        log(f">>> SMOKE TEST MODE (answer_only={answer_only}, weighted={use_weights}, "
            f"balanced_sampling={use_balanced}) <<<")
        train_limit, valid_limit, num_epochs, grad_accum = 2, 2, 1, 1
    else:
        train_limit  = args.train_limit
        valid_limit  = args.valid_limit
        num_epochs   = args.num_epochs
        grad_accum   = args.grad_accum

    if args.num_frames % 2 != 0:
        raise ValueError(f"--num-frames must be even, got {args.num_frames}")

    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    device    = args.device

    # ── Data ─────────────────────────────────────────────────────────────────
    log("Loading datasets …")
    train_ds = FoulDatasetReason(data_root, "Train", limit=train_limit)
    valid_ds = FoulDatasetReason(data_root, "Valid", limit=valid_limit)
    log(f"train={len(train_ds)}  valid={len(valid_ds)}")

    # ── Class weights ─────────────────────────────────────────────────────────
    if use_weights:
        log("Computing class weights …")
        action_w, sev_w = compute_class_weights(train_ds, max_weight=args.max_class_weight)
    else:
        action_w, sev_w = {c: 1.0 for c in ACTION_CLASSES}, {c: 1.0 for c in OFFENCE_SEVERITY_CLASSES}
        log("Class weighting disabled.")

    if use_balanced:
        log("Building balanced sampler (sqrt-inverse-frequency) …")
        sampler = make_balanced_sampler(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=1, sampler=sampler, num_workers=0, collate_fn=_collate
        )
        if use_weights and args.max_class_weight > 5.0:
            log(f"WARNING: balanced_sampling + weighted_loss both active. "
                f"Consider --max-class-weight 3.0 to avoid double-correcting "
                f"(current cap={args.max_class_weight})")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=_collate
        )

    # ── Model ─────────────────────────────────────────────────────────────────
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

    # ── Masking sanity check ──────────────────────────────────────────────────
    log(f"Sanity-checking loss masking (answer_only={answer_only}) …")
    _s0 = train_ds[0]
    _verify_masking(processor, _s0, args.num_frames, device, answer_only)
    torch.cuda.empty_cache()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = PagedAdamW8bit(trainable_params, lr=args.lr)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_bal_acc_avg = -1.0
    best_ckpt_dir    = out_dir / "best_checkpoint"
    history: list[dict] = []
    steps_per_epoch = len(train_loader)
    total_steps     = num_epochs * steps_per_epoch
    total_timer     = _StepTimer(total_steps)

    model.train()
    for epoch in range(1, num_epochs + 1):
        log(f"{'='*60}")
        log(f"Epoch {epoch}/{num_epochs}  "
            f"(steps {(epoch-1)*steps_per_epoch+1}–{epoch*steps_per_epoch} / {total_steps})")
        epoch_loss = 0.0
        n_ok       = 0
        optimizer.zero_grad()
        epoch_timer = _StepTimer(steps_per_epoch)

        for step_in_epoch, sample in enumerate(train_loader, start=1):
            if args.smoke_test and n_ok >= 2:
                break

            try:
                inputs, labels = build_training_inputs(
                    processor, sample, args.num_frames, device, answer_only
                )
            except Exception as exc:
                log(f"  [step {step_in_epoch}] SKIP — {exc}")
                continue

            outputs = model(**inputs, labels=labels)

            # Weighted loss: scale by combined class weight
            sw = sample_weight(sample, action_w, sev_w)
            loss = (outputs.loss / grad_accum) * sw
            loss.backward()

            epoch_loss += outputs.loss.item()   # unweighted for logging
            n_ok       += 1
            epoch_timer.tick()
            total_timer.tick()

            is_accum_step = (n_ok % grad_accum == 0)
            is_last_step  = (step_in_epoch == len(train_loader)) or \
                            (args.smoke_test and n_ok >= 2)
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
                    f"avg_loss={avg:.4f} | w={sw:.2f} | {gpu_mem(device)}"
                )
            if args.smoke_test and n_ok >= 2:
                break

        epoch_avg_loss = epoch_loss / max(n_ok, 1)
        log(f"Epoch {epoch} complete  avg_loss={epoch_avg_loss:.4f}  "
            f"elapsed={epoch_timer.elapsed()}  total={total_timer.elapsed()}")

        # ── Validation ────────────────────────────────────────────────────────
        log(f"Validating on {len(valid_ds)} samples …")
        val_metrics, val_rows = evaluate_reason(
            model, processor, valid_ds, device, args.num_frames, args.max_new_tokens
        )
        bal1 = val_metrics["balanced_accuracy_action_seen_classes"]
        bal2 = val_metrics["balanced_accuracy_offence_severity_seen_classes"]
        bal_avg = (bal1 + bal2) / 2.0
        log(f"Valid  task1={bal1:.2f}%  task2={bal2:.2f}%  avg={bal_avg:.2f}%")

        history.append({
            "epoch": epoch,
            "train_loss": epoch_avg_loss,
            "val_bal_acc_task1": bal1,
            "val_bal_acc_task2": bal2,
            "val_bal_acc_avg": bal_avg,
        })

        if bal_avg > best_bal_acc_avg:
            best_bal_acc_avg = bal_avg
            model.save_pretrained(str(best_ckpt_dir))
            processor.save_pretrained(str(best_ckpt_dir))
            (best_ckpt_dir / "val_rows.json").write_text(
                json.dumps(val_rows, indent=2, ensure_ascii=False))
            (best_ckpt_dir / "val_metrics.json").write_text(
                json.dumps(val_metrics, indent=2, ensure_ascii=False))
            log(f"  >> Best checkpoint → {best_ckpt_dir}  "
                f"(avg={best_bal_acc_avg:.2f}%)")

        model.train()
        if args.smoke_test:
            break

    (out_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False))
    log("=" * 60)
    log(f"Training complete.  Best valid bal_acc_avg={best_bal_acc_avg:.2f}%")
    log(f"Best checkpoint: {best_ckpt_dir}")
    log(f"Config: answer_only={answer_only}  weighted_loss={use_weights}")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
