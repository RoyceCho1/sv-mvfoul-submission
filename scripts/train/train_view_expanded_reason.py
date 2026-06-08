#!/usr/bin/env python
"""View-expanded single-view QLoRA training for late-fusion MV inference.

Training sample unit:
    one existing action_i/clip_j.mp4 -> action-level offence_severity label

Validation unit:
    action-level late fusion over clip_j predictions, using the same SV prompt.
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.eval_late_fusion_reason import ActionViewDataset, evaluate_late_fusion, make_jsonable  # noqa: E402
from train.train_reason import (  # noqa: E402
    _StepTimer,
    _fmt_sec,
    build_training_inputs_reason,
    gpu_mem,
    log,
    set_seed,
    synthesize_think,
)
from zero_shot.zero_shot_eval import OFFENCE_SEVERITY_CLASSES, compute_metrics, official_target  # noqa: E402


class ViewExpandedReasonDataset(Dataset):
    """Every existing clip_i becomes a single-view training sample."""

    def __init__(self, data_root: Path, split: str, limit: int | None = None, max_views: int = 0) -> None:
        ann_path = data_root / split / "annotations.json"
        annotations = json.loads(ann_path.read_text())
        self.samples: list[dict[str, Any]] = []
        self.num_actions = 0
        for action_id, record in annotations["Actions"].items():
            target = official_target(record)
            if target is None:
                continue
            action_class, offence_severity, offence_class, severity_class = target
            clips_for_action = []
            for clip_idx, clip_info in enumerate(record.get("Clips", [])):
                clip_path = data_root / split / f"action_{action_id}" / f"clip_{clip_idx}.mp4"
                if clip_path.exists():
                    clips_for_action.append((clip_idx, clip_info, clip_path))
            if max_views > 0:
                clips_for_action = clips_for_action[:max_views]
            if not clips_for_action:
                continue
            self.num_actions += 1
            for clip_idx, clip_info, clip_path in clips_for_action:
                self.samples.append({
                    "action_id": action_id,
                    "clip_idx": clip_idx,
                    "clip_path": clip_path,
                    "camera_type": clip_info.get("Camera type", ""),
                    "replay_speed": clip_info.get("Replay speed", ""),
                    "answer": synthesize_think(record, action_class, offence_severity),
                    "gold_action_class": action_class,
                    "gold_offence_severity": offence_severity,
                    "gold_offence": offence_class,
                    "gold_severity": severity_class,
                    "record": record,
                })
                if limit and len(self.samples) >= limit:
                    return

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(batch) == 1
    return batch[0]


def make_balanced_sampler(dataset: ViewExpandedReasonDataset) -> WeightedRandomSampler:
    import math
    sev_cnt: Counter = Counter(s["gold_offence_severity"] for s in dataset.samples)
    weights = [
        1.0 / math.sqrt(max(sev_cnt.get(s["gold_offence_severity"], 1), 1))
        for s in dataset.samples
    ]
    log("Balanced sampler — view-expanded sqrt-inverse-frequency weights per offence_severity:")
    for cls in OFFENCE_SEVERITY_CLASSES:
        cnt = sev_cnt.get(cls, 0)
        w = 1.0 / math.sqrt(max(cnt, 1))
        log(f"  {cls:<30}  view_cnt={cnt:>5}  sample_w={w:.4f}")
    return WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="View-expanded SV reasoning QLoRA for late fusion")
    p.add_argument("--model-id", default="nvidia/Cosmos-Reason2-8B")
    p.add_argument("--data-root", default="data/SoccerNet/mvfouls")
    p.add_argument("--output-dir", default="outputs/qlora_cosmos8b_view_expanded_reason")
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-train-views", type=int, default=0, help="0 = all clips per Train action")
    p.add_argument("--max-eval-views", type=int, default=0, help="0 = all clips per Valid action")
    p.add_argument("--fusion-rule", choices=["main_first", "majority_vote"], default="main_first")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--train-limit", type=int, default=None, help="Limit view-level train samples")
    p.add_argument("--valid-limit", type=int, default=None, help="Limit valid actions")
    p.add_argument("--balanced-sampling", action="store_true")
    return p.parse_args()


def _print_dataset_summary(name: str, dataset: ViewExpandedReasonDataset) -> None:
    sev = Counter(s["gold_offence_severity"] for s in dataset.samples)
    views = Counter(s["clip_idx"] for s in dataset.samples)
    log(f"{name}: view_samples={len(dataset)} actions={dataset.num_actions}")
    log(f"{name}: severity view-counts={dict(sev)}")
    log(f"{name}: clip_idx counts={dict(sorted(views.items()))}")
    if dataset.samples:
        s = dataset.samples[0]
        log(f"{name}: first sample action_{s['action_id']} clip_{s['clip_idx']} cam={s['camera_type']!r} gold={s['gold_offence_severity']!r}")
        print(s["answer"], flush=True)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.num_frames % 2 != 0:
        raise ValueError(f"--num-frames must be even, got {args.num_frames}")

    if args.smoke_test:
        log(">>> SMOKE TEST MODE (view-expanded): 2 train steps + 2 valid actions <<<")
        train_limit, valid_limit, num_epochs, grad_accum = 2, 2, 1, 1
    else:
        train_limit, valid_limit = args.train_limit, args.valid_limit
        num_epochs, grad_accum = args.num_epochs, args.grad_accum

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    device = args.device

    log("Loading view-expanded train dataset …")
    train_ds = ViewExpandedReasonDataset(data_root, "Train", limit=train_limit, max_views=args.max_train_views)
    valid_actions = ActionViewDataset(data_root, "Valid", limit=valid_limit or 0, max_views=args.max_eval_views)
    _print_dataset_summary("train", train_ds)
    log(f"valid: actions={len(valid_actions)} max_eval_views={args.max_eval_views or 'all'}")

    if args.balanced_sampling:
        train_loader = DataLoader(
            train_ds, batch_size=1, sampler=make_balanced_sampler(train_ds),
            num_workers=0, collate_fn=_collate,
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=_collate)

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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    vis_trainable = sum(p.numel() for name, p in model.named_parameters() if "visual" in name and p.requires_grad)
    if vis_trainable > 0:
        log(f"WARNING: {vis_trainable:,} vision params trainable — freezing.")
        for name, p in model.named_parameters():
            if "visual" in name:
                p.requires_grad_(False)
    log(f"LoRA ready. {gpu_mem(device)}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = PagedAdamW8bit(trainable_params, lr=args.lr)

    best_bal_acc = -1.0
    best_ckpt_dir = out_dir / "best_checkpoint"
    history: list[dict[str, Any]] = []
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    total_timer = _StepTimer(total_steps)

    model.train()
    for epoch in range(1, num_epochs + 1):
        log("=" * 60)
        log(f"Epoch {epoch}/{num_epochs}  steps={steps_per_epoch}  total_steps={total_steps}")
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

            is_accum_step = n_ok % grad_accum == 0
            is_last_step = step_in_epoch == len(train_loader)
            if is_accum_step or is_last_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step_in_epoch % 10 == 0 or is_last_step or args.smoke_test:
                avg = epoch_loss / max(n_ok, 1)
                log(
                    f"  {epoch_timer.bar()} step {step_in_epoch}/{steps_per_epoch} "
                    f"({epoch_timer.pct():.1f}%) | epoch ETA {epoch_timer.eta()} | "
                    f"total {total_timer.pct():.1f}% ETA {total_timer.eta()} | "
                    f"avg_loss={avg:.4f} | {gpu_mem(device)}"
                )

        epoch_avg_loss = epoch_loss / max(n_ok, 1)
        log(f"Epoch {epoch} complete avg_loss={epoch_avg_loss:.4f} elapsed={epoch_timer.elapsed()}")

        log(f"Validating late fusion on {len(valid_actions)} actions …")
        val_rows = evaluate_late_fusion(
            model, processor, valid_actions, args.num_frames, device,
            args.max_new_tokens, args.fusion_rule, log_every=20,
        )
        val_metrics = compute_metrics(val_rows)
        bal = val_metrics["balanced_accuracy_offence_severity_seen_classes"]
        log(f"Valid late-fusion offence_severity_bal_acc={bal:.2f}%")
        history.append({"epoch": epoch, "train_loss": epoch_avg_loss, "val_late_fusion_bal_acc_offence_severity": bal})

        if bal > best_bal_acc:
            best_bal_acc = bal
            model.save_pretrained(str(best_ckpt_dir))
            processor.save_pretrained(str(best_ckpt_dir))
            (best_ckpt_dir / "val_rows.json").write_text(json.dumps(make_jsonable(val_rows), indent=2, ensure_ascii=False))
            (best_ckpt_dir / "val_metrics.json").write_text(json.dumps(make_jsonable(val_metrics), indent=2, ensure_ascii=False))
            log(f"  >> New best checkpoint -> {best_ckpt_dir} (bal_acc={best_bal_acc:.2f}%)")
        model.train()
        model.config.use_cache = False

    (out_dir / "training_history.json").write_text(json.dumps(make_jsonable(history), indent=2, ensure_ascii=False))
    log("=" * 60)
    log(f"Training complete. Best valid late-fusion BA={best_bal_acc:.2f}%")
    log(f"Best checkpoint: {best_ckpt_dir}")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
