#!/usr/bin/env python
"""View-expanded QLoRA training aligned with label-likelihood evaluation.

Training target is a short fixed answer candidate, without free-form <think>.
Validation uses 4-label likelihood scoring and score-level late fusion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from bitsandbytes.optim import PagedAdamW8bit
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.eval_late_fusion_likelihood import (  # noqa: E402
    build_scoring_prompt,
    candidate_text,
    compute_priors,
    parse_csv_floats,
    parse_csv_methods,
    rows_for_setting,
    score_one_view,
    view_level_summary,
)
from eval.eval_late_fusion_reason import ActionViewDataset, make_jsonable  # noqa: E402
from train.train_reason import _StepTimer, gpu_mem, log, make_labels, set_seed  # noqa: E402
from train.frame_utils import make_video_entry  # noqa: E402
from zero_shot.zero_shot_eval import OFFENCE_SEVERITY_CLASSES, compute_metrics, make_prediction_json, official_target  # noqa: E402


class ViewExpandedLikelihoodDataset(Dataset):
    """Every existing clip_i becomes a single-view answer-only sample."""

    def __init__(
        self,
        data_root: Path,
        split: str,
        target_format: str,
        limit: int | None = None,
        max_views: int = 0,
    ) -> None:
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
            answer = candidate_text(offence_severity, target_format)
            for clip_idx, clip_info, clip_path in clips_for_action:
                self.samples.append({
                    "action_id": action_id,
                    "clip_idx": clip_idx,
                    "clip_path": clip_path,
                    "camera_type": clip_info.get("Camera type", ""),
                    "replay_speed": clip_info.get("Replay speed", ""),
                    "answer": answer,
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


def collate_one(batch: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(batch) == 1
    return batch[0]


def make_balanced_sampler(dataset: ViewExpandedLikelihoodDataset) -> WeightedRandomSampler:
    import math
    sev_cnt: Counter = Counter(s["gold_offence_severity"] for s in dataset.samples)
    weights = [1.0 / math.sqrt(max(sev_cnt.get(s["gold_offence_severity"], 1), 1)) for s in dataset.samples]
    log("Balanced sampler — view-expanded likelihood sqrt-inverse-frequency weights:")
    for cls in OFFENCE_SEVERITY_CLASSES:
        cnt = sev_cnt.get(cls, 0)
        w = 1.0 / math.sqrt(max(cnt, 1))
        log(f"  {cls:<30}  view_cnt={cnt:>5}  sample_w={w:.4f}")
    return WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)


def build_training_inputs_likelihood(
    processor: AutoProcessor,
    clip_path: Path,
    answer: str,
    num_frames: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                make_video_entry(clip_path, num_frames),
                {"type": "text", "text": build_scoring_prompt(num_frames)},
            ],
        },
        {"role": "assistant", "content": answer},
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="View-expanded answer-only QLoRA for likelihood scoring")
    p.add_argument("--model-id", default="nvidia/Cosmos-Reason2-8B")
    p.add_argument("--data-root", default="data/SoccerNet/mvfouls")
    p.add_argument("--output-dir", default="outputs/qlora_cosmos8b_view_expanded_likelihood")
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-format", choices=["json", "answer_tag"], default="json")
    p.add_argument("--score-reduction", choices=["mean", "sum"], default="mean")
    p.add_argument("--fusion-methods", default="score_mean,score_sum,clip0,clip1,weighted_clip1")
    p.add_argument("--prior-alphas", default="0,0.1,0.25,0.5,0.75,1.0")
    p.add_argument("--max-train-views", type=int, default=0)
    p.add_argument("--max-eval-views", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--valid-limit", type=int, default=None)
    p.add_argument("--balanced-sampling", action="store_true")
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--no-epoch-eval", action="store_true")
    return p.parse_args()


def print_dataset_summary(name: str, dataset: ViewExpandedLikelihoodDataset) -> None:
    sev = Counter(s["gold_offence_severity"] for s in dataset.samples)
    views = Counter(s["clip_idx"] for s in dataset.samples)
    log(f"{name}: view_samples={len(dataset)} actions={dataset.num_actions}")
    log(f"{name}: severity view-counts={dict(sev)}")
    log(f"{name}: clip_idx counts={dict(sorted(views.items()))}")
    if dataset.samples:
        sample = dataset.samples[0]
        log(
            f"{name}: first sample action_{sample['action_id']} clip_{sample['clip_idx']} "
            f"gold={sample['gold_offence_severity']!r}"
        )
        print(sample["answer"], flush=True)


@torch.inference_mode()
def evaluate_likelihood(
    model: Any,
    processor: AutoProcessor,
    valid_actions: ActionViewDataset,
    num_frames: int,
    device: str,
    target_format: str,
    score_reduction: str,
    methods: list[str],
    alphas: list[float],
    priors: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    was_training = model.training
    old_use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    if old_use_cache is not None:
        model.config.use_cache = True

    base_rows = []
    for i, action in enumerate(valid_actions.actions, start=1):
        if i == 1 or i == len(valid_actions.actions) or i % 20 == 0:
            log(f"  likelihood valid [{i}/{len(valid_actions.actions)}] action_{action['action_id']} views={len(action['clips'])}")
        view_scores = []
        for clip in action["clips"]:
            try:
                scored = score_one_view(
                    model, processor, clip["clip_path"], num_frames, device,
                    target_format, score_reduction,
                )
                view_scores.append({
                    "clip_idx": clip["clip_idx"],
                    "clip_path": str(clip["clip_path"]),
                    "camera_type": clip.get("camera_type", ""),
                    "replay_speed": clip.get("replay_speed", ""),
                    **scored,
                })
            except Exception as exc:
                log(f"    view error clip_{clip['clip_idx']}: {type(exc).__name__}: {exc}")
        base_rows.append({
            "action_id": action["action_id"],
            "n_views": len(view_scores),
            "view_scores": view_scores,
            "gold": {
                "action_class": action["gold_action_class"],
                "offence_severity": action["gold_offence_severity"],
                "Offence": action["gold_offence"],
                "Severity": action["gold_severity"],
            },
        })

    grid = []
    best = None
    best_rows = None
    for method in methods:
        for alpha in alphas:
            rows = rows_for_setting(base_rows, method, priors, alpha, clip1_weight=1.2, other_view_weight=0.8)
            metrics = compute_metrics(rows)
            metrics["view_level"] = view_level_summary(rows)
            item = {"fusion_method": method, "prior_alpha": alpha, "metrics": metrics}
            grid.append(item)
            ba = metrics["balanced_accuracy_offence_severity_seen_classes"]
            if best is None or ba > best["metrics"]["balanced_accuracy_offence_severity_seen_classes"]:
                best = item
                best_rows = rows

    if old_use_cache is not None:
        model.config.use_cache = old_use_cache
    if was_training:
        model.train()
    assert best is not None and best_rows is not None
    return base_rows, best, best_rows


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.smoke_test:
        log(">>> SMOKE TEST MODE (view-expanded likelihood): 2 train steps + 2 valid actions <<<")
        train_limit, valid_limit, num_epochs, grad_accum = 2, 2, 1, 1
    else:
        train_limit, valid_limit = args.train_limit, args.valid_limit
        num_epochs, grad_accum = args.num_epochs, args.grad_accum

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    device = args.device
    methods = parse_csv_methods(args.fusion_methods)
    alphas = parse_csv_floats(args.prior_alphas)
    priors = compute_priors(data_root, "Train")

    log("Loading view-expanded likelihood train dataset …")
    train_ds = ViewExpandedLikelihoodDataset(
        data_root, "Train", args.target_format, limit=train_limit, max_views=args.max_train_views
    )
    valid_actions = ActionViewDataset(data_root, "Valid", limit=valid_limit or 0, max_views=args.max_eval_views)
    print_dataset_summary("train", train_ds)
    log(f"valid: actions={len(valid_actions)} max_eval_views={args.max_eval_views or 'all'}")
    log(f"priors={priors}")

    if args.balanced_sampling:
        train_loader = DataLoader(train_ds, batch_size=1, sampler=make_balanced_sampler(train_ds), num_workers=0, collate_fn=collate_one)
    else:
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_one)

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
    best_ba = -1.0
    best_ckpt_dir = out_dir / "best_checkpoint"
    history = []
    steps_per_epoch = len(train_loader)
    total_timer = _StepTimer(num_epochs * steps_per_epoch)

    model.train()
    for epoch in range(1, num_epochs + 1):
        log("=" * 60)
        log(f"Epoch {epoch}/{num_epochs} steps={steps_per_epoch}")
        epoch_timer = _StepTimer(steps_per_epoch)
        epoch_loss = 0.0
        n_ok = 0
        optimizer.zero_grad()
        for step, sample in enumerate(train_loader, start=1):
            if args.smoke_test and step > 2:
                break
            try:
                inputs, labels = build_training_inputs_likelihood(
                    processor, sample["clip_path"], sample["answer"], args.num_frames, device
                )
            except Exception as exc:
                log(f"  [step {step}] SKIP — {exc}")
                continue
            outputs = model(**inputs, labels=labels)
            (outputs.loss / grad_accum).backward()
            epoch_loss += outputs.loss.item()
            n_ok += 1
            epoch_timer.tick()
            total_timer.tick()
            if n_ok % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            if step % 10 == 0 or step == len(train_loader) or args.smoke_test:
                avg = epoch_loss / max(n_ok, 1)
                log(
                    f"  {epoch_timer.bar()} step {step}/{steps_per_epoch} ({epoch_timer.pct():.1f}%) | "
                    f"epoch ETA {epoch_timer.eta()} | total {total_timer.pct():.1f}% ETA {total_timer.eta()} | "
                    f"avg_loss={avg:.4f} | {gpu_mem(device)}"
                )

        epoch_avg = epoch_loss / max(n_ok, 1)
        log(f"Epoch {epoch} complete avg_loss={epoch_avg:.4f} elapsed={epoch_timer.elapsed()}")
        epoch_ckpt_dir = out_dir / f"epoch_{epoch}"
        if args.save_every_epoch or args.no_epoch_eval:
            model.save_pretrained(str(epoch_ckpt_dir))
            processor.save_pretrained(str(epoch_ckpt_dir))
            log(f"  >> Saved epoch checkpoint -> {epoch_ckpt_dir}")

        history_item = {
            "epoch": epoch,
            "train_loss": epoch_avg,
            "epoch_checkpoint": str(epoch_ckpt_dir) if (args.save_every_epoch or args.no_epoch_eval) else None,
        }
        if args.no_epoch_eval:
            history.append(history_item)
            model.train()
            model.config.use_cache = False
            continue

        log(f"Validating likelihood scoring on {len(valid_actions)} actions …")
        base_rows, best, best_rows = evaluate_likelihood(
            model, processor, valid_actions, args.num_frames, device, args.target_format,
            args.score_reduction, methods, alphas, priors,
        )
        ba = best["metrics"]["balanced_accuracy_offence_severity_seen_classes"]
        acc = best["metrics"]["accuracy_offence_severity"]
        log(f"Valid likelihood best acc={acc:.2f}% ba={ba:.2f}% method={best['fusion_method']} alpha={best['prior_alpha']}")
        history_item.update({
            "best_likelihood_acc": acc,
            "best_likelihood_bal_acc": ba,
            "best_likelihood_method": best["fusion_method"],
            "best_likelihood_alpha": best["prior_alpha"],
        })
        history.append(history_item)
        if ba > best_ba:
            best_ba = ba
            model.save_pretrained(str(best_ckpt_dir))
            processor.save_pretrained(str(best_ckpt_dir))
            (best_ckpt_dir / "val_score_rows.json").write_text(json.dumps(make_jsonable(base_rows), indent=2, ensure_ascii=False))
            (best_ckpt_dir / "val_best_rows.json").write_text(json.dumps(make_jsonable(best_rows), indent=2, ensure_ascii=False))
            (best_ckpt_dir / "val_best_metrics.json").write_text(json.dumps(make_jsonable(best["metrics"]), indent=2, ensure_ascii=False))
            predictions = make_prediction_json(valid_actions.annotations, best_rows)
            (best_ckpt_dir / "val_best_predictions.json").write_text(json.dumps(make_jsonable(predictions), indent=2, ensure_ascii=False))
            (best_ckpt_dir / "val_best_setting.json").write_text(json.dumps(make_jsonable(best), indent=2, ensure_ascii=False))
            log(f"  >> New best checkpoint -> {best_ckpt_dir} (ba={best_ba:.2f}%)")
        model.train()
        model.config.use_cache = False

    (out_dir / "training_history.json").write_text(json.dumps(make_jsonable(history), indent=2, ensure_ascii=False))
    log("=" * 60)
    if args.no_epoch_eval:
        log("Training complete. Epoch validation was skipped.")
        log(f"Epoch checkpoints: {out_dir}/epoch_1 ... {out_dir}/epoch_{num_epochs}")
    else:
        log(f"Training complete. Best valid likelihood BA={best_ba:.2f}%")
        log(f"Best checkpoint: {best_ckpt_dir}")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
