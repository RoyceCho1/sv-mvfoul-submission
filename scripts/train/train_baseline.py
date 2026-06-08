#!/usr/bin/env python
"""QLoRA fine-tuning of Cosmos-Reason2-8B on SoccerNet-MVFoul (single-view, clip_0).

Architecture notes:
  - Qwen3VLForConditionalGeneration (verified from config.json)
  - Vision encoder uses: qkv, linear_fc1, linear_fc2  → NOT targeted by LoRA
  - LLM uses: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj → LoRA targets
  - Loss is computed ONLY on answer tokens; prompt/video tokens are masked with -100.
  - Official Train/Valid/Test splits are action-based → no clip-level leakage.
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
from bitsandbytes.optim import PagedAdamW8bit
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

# Reuse shared utilities — no duplication
sys.path.insert(0, str(Path(__file__).parent.parent))
from zero_shot.zero_shot_eval import (
    ACTION_CLASSES,
    ACTION_TO_INDEX,
    OFFENCE_SEVERITY_CLASSES,
    OFFENCE_SEVERITY_TO_INDEX,
    build_prompt,
    compute_metrics,
    extract_json,
    normalize_prediction_keys,
    official_target,
    validate_prediction,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QLoRA fine-tune Cosmos-Reason2 on SoccerNet-MVFoul"
    )
    p.add_argument("--model-id", default="nvidia/Cosmos-Reason2-8B")
    p.add_argument("--data-root", default="data/SoccerNet/mvfouls")
    p.add_argument("--output-dir", default="outputs/qlora_cosmos8b")
    p.add_argument("--num-epochs", type=int, default=3)
    # nframes must be even (FRAME_FACTOR=2 in qwen_vl_utils)
    p.add_argument("--num-frames", type=int, default=8, help="Frames per clip for training AND eval (must be even)")
    p.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens during validation generation")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 2 train steps + 2 valid samples to verify end-to-end pipeline (OOM check)",
    )
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--valid-limit", type=int, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
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
    """Tracks elapsed time and computes ETA for a fixed total number of steps."""

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

class FoulDataset(Dataset):
    """Single-view (clip_0) foul dataset for one official split.

    Uses the same official_target filter as VARS evaluation — identical
    filtering guarantees no label leakage and fair comparison.
    Each action maps to exactly one clip_0.mp4, so cross-split contamination
    is impossible given the official action-based split.
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
            self.samples.append(
                {
                    "action_id": action_id,
                    "clip_path": clip_path,
                    # Canonical answer: JSON string the model must reproduce
                    "answer": json.dumps(
                        {"action_class": action_class, "offence_severity": offence_severity},
                        ensure_ascii=False,
                    ),
                    "gold_action_class": action_class,
                    "gold_offence_severity": offence_severity,
                    "gold_offence": offence_class,
                    "gold_severity": severity_class,
                }
            )
            if limit and len(self.samples) >= limit:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def _collate(batch: list[dict]) -> dict:
    """DataLoader collate for batch_size=1 — just returns the single sample dict."""
    assert len(batch) == 1
    return batch[0]


# ---------------------------------------------------------------------------
# Loss masking
# ---------------------------------------------------------------------------

def make_labels(input_ids: torch.Tensor, processor: AutoProcessor) -> torch.Tensor:
    """Return labels tensor where all tokens before the assistant answer are -100.

    Strategy: tokenize the exact header string '<|im_start|>assistant\\n' and
    scan input_ids from the right for its last occurrence. Everything up to and
    including the header is masked; only the answer (and trailing <|im_end|>)
    contributes to the loss.

    Single processor call suffices — no redundant video processing.
    """
    header_ids: list[int] = processor.tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    header_len = len(header_ids)
    seq: list[int] = input_ids[0].tolist()  # batch_size == 1

    # Scan from the right: the LAST occurrence is the assistant turn we target.
    ans_start = -1
    for i in range(len(seq) - header_len, -1, -1):
        if seq[i : i + header_len] == header_ids:
            ans_start = i + header_len
            break

    if ans_start < 0:
        raise ValueError(
            "Could not locate '<|im_start|>assistant\\n' in the tokenized sequence. "
            "Check that the chat template produces the expected header."
        )

    labels = input_ids.clone()
    labels[:, :ans_start] = -100
    return labels


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------

def build_training_inputs(
    processor: AutoProcessor,
    clip_path: Path,
    answer: str,
    num_frames: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Process one sample for training; returns model inputs and loss-masked labels."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(clip_path), "nframes": num_frames},
                {"type": "text", "text": build_prompt()},
            ],
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,  # full sequence including answer
        return_dict=True,
        return_tensors="pt",
    )
    labels = make_labels(inputs["input_ids"], processor)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs, labels.to(device)


@torch.no_grad()
def run_one_inference(
    model: Any,
    processor: AutoProcessor,
    clip_path: Path,
    num_frames: int,
    device: str,
    max_new_tokens: int,
) -> tuple[str, dict, list[str]]:
    """Generate and parse one sample during validation."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(clip_path), "nframes": num_frames},
                {"type": "text", "text": build_prompt()},
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

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer_ids = generated_ids[:, prompt_len:]
    raw_output = processor.batch_decode(
        answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    try:
        prediction = normalize_prediction_keys(extract_json(raw_output))
        errors = validate_prediction(prediction)
    except Exception as exc:
        prediction = {}
        errors = [f"{type(exc).__name__}: {exc}"]
    return raw_output, prediction, errors


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: Any,
    processor: AutoProcessor,
    dataset: FoulDataset,
    device: str,
    num_frames: int,
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict]]:
    """Run inference on every sample in dataset; return (metrics_dict, rows)."""
    model.eval()
    model.config.use_cache = True

    rows: list[dict] = []
    for idx, sample in enumerate(dataset.samples, start=1):
        try:
            raw_output, prediction, errors = run_one_inference(
                model, processor, sample["clip_path"], num_frames, device, max_new_tokens
            )
        except Exception as exc:
            raw_output, prediction, errors = "", {}, [f"InferenceError: {exc}"]
        rows.append(
            {
                "action_id": sample["action_id"],
                "raw_output": raw_output,
                "prediction": prediction,
                "validation_errors": errors,
                "gold": {
                    "action_class": sample["gold_action_class"],
                    "offence_severity": sample["gold_offence_severity"],
                },
            }
        )
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

    # Smoke-test overrides
    if args.smoke_test:
        log(">>> SMOKE TEST MODE: 2 train steps, 2 valid samples, 1 epoch <<<")
        train_limit = 2
        valid_limit = 2
        num_epochs = 1
        grad_accum = 1
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
        raise ValueError(f"--num-frames must be even (FRAME_FACTOR=2), got {args.num_frames}")

    # ---- Data ---------------------------------------------------------------
    log("Loading datasets …")
    train_ds = FoulDataset(data_root, "Train", limit=train_limit)
    valid_ds = FoulDataset(data_root, "Valid", limit=valid_limit)
    log(f"train={len(train_ds)}  valid={len(valid_ds)}")

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

    # ---- k-bit training prep + gradient checkpointing -----------------------
    # use_reentrant=False avoids known reentrant-checkpoint issues with PEFT
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.config.use_cache = False  # required when gradient_checkpointing=True

    # ---- LoRA ---------------------------------------------------------------
    # Vision encoder modules: qkv, linear_fc1, linear_fc2  → NOT in target list
    # LLM modules:           q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",       # MLP
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Paranoia check: verify vision encoder has no trainable params
    vis_trainable = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "visual" in name and p.requires_grad
    )
    if vis_trainable > 0:
        log(f"WARNING: {vis_trainable:,} vision params are trainable — explicitly freezing.")
        for name, p in model.named_parameters():
            if "visual" in name:
                p.requires_grad_(False)
    else:
        log("Vision encoder: fully frozen (OK).")

    log(f"LoRA ready. {gpu_mem(device)}")

    # ---- Sanity check: loss masking on first sample -------------------------
    log("Sanity-checking loss masking on first training sample …")
    _s = train_ds[0]
    _inputs, _labels = build_training_inputs(
        processor, _s["clip_path"], _s["answer"], args.num_frames, device
    )
    n_total = _labels.shape[1]
    n_answer = int((_labels[0] != -100).sum().item())
    n_masked = n_total - n_answer
    log(f"  seq_len={n_total}  answer_tokens={n_answer}  masked(prompt+video)={n_masked}")
    assert n_answer > 0, "FATAL: all labels masked — loss masking is broken"
    assert n_masked > 0, "FATAL: no tokens masked — prompt/video tokens included in loss"
    del _inputs, _labels

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

    model.train()
    for epoch in range(1, num_epochs + 1):
        log(f"{'='*60}")
        log(f"Epoch {epoch}/{num_epochs}  (steps {(epoch-1)*steps_per_epoch+1}–{epoch*steps_per_epoch} / {total_steps} total)")
        epoch_loss = 0.0
        n_ok = 0
        optimizer.zero_grad()
        epoch_timer = _StepTimer(steps_per_epoch)

        for step_in_epoch, sample in enumerate(train_loader, start=1):
            if args.smoke_test and step_in_epoch > 2:
                break

            try:
                inputs, labels = build_training_inputs(
                    processor,
                    sample["clip_path"],
                    sample["answer"],
                    args.num_frames,
                    device,
                )
            except Exception as exc:
                log(f"  [step {step_in_epoch}] SKIP — processing error: {exc}")
                continue

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()
            n_ok += 1
            epoch_timer.tick()
            total_timer.tick()

            # Optimizer step after accumulating grad_accum gradients (or at end of epoch)
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

        # ---- Validation -----------------------------------------------------
        log(f"Validating on {len(valid_ds)} samples …")
        val_metrics, val_rows = evaluate(
            model, processor, valid_ds, device, args.num_frames, args.max_new_tokens
        )
        bal_acc_task1 = val_metrics["balanced_accuracy_action_seen_classes"]
        bal_acc_task2 = val_metrics["balanced_accuracy_offence_severity_seen_classes"]
        bal_acc_avg = (bal_acc_task1 + bal_acc_task2) / 2.0

        log(
            f"Valid  task1_bal_acc={bal_acc_task1:.2f}%  "
            f"task2_bal_acc={bal_acc_task2:.2f}%  avg={bal_acc_avg:.2f}%"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_avg_loss,
                "val_bal_acc_task1": bal_acc_task1,
                "val_bal_acc_task2": bal_acc_task2,
                "val_bal_acc_avg": bal_acc_avg,
            }
        )

        # ---- Save best checkpoint -------------------------------------------
        if bal_acc_avg > best_bal_acc_avg:
            best_bal_acc_avg = bal_acc_avg
            model.save_pretrained(str(best_ckpt_dir))
            processor.save_pretrained(str(best_ckpt_dir))
            # Save val rows for inspection
            (best_ckpt_dir / "val_rows.json").write_text(
                json.dumps(val_rows, indent=2, ensure_ascii=False)
            )
            (best_ckpt_dir / "val_metrics.json").write_text(
                json.dumps(val_metrics, indent=2, ensure_ascii=False)
            )
            log(
                f"  >> New best checkpoint saved → {best_ckpt_dir}  "
                f"(bal_acc_avg={best_bal_acc_avg:.2f}%)"
            )

        model.train()

    # ---- Final summary -------------------------------------------------------
    (out_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False)
    )
    log("=" * 60)
    log(f"Training complete.  Best valid bal_acc_avg={best_bal_acc_avg:.2f}%")
    log(f"Best checkpoint: {best_ckpt_dir}")
    log(f"History: {out_dir / 'training_history.json'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
