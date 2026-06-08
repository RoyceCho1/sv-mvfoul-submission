#!/usr/bin/env python
"""Evaluate a QLoRA fine-tuned Cosmos-Reason2 checkpoint on SoccerNet-MVFoul.

Loads base model + LoRA adapter, runs inference, and prints a comparison table
against the VARS baseline.  Reuses all shared utilities from zero_shot_eval.py.

Usage:
    python scripts/eval_finetuned.py \
        --adapter-path outputs/qlora_cosmos32b/best_checkpoint \
        --split Valid \
        --out-dir outputs/finetuned_eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent.parent))
from zero_shot.zero_shot_eval import (
    build_prompt,
    compute_metrics,
    extract_json,
    load_samples,
    make_prediction_json,
    normalize_prediction_keys,
    prediction_to_offence_severity,
    validate_prediction,
)

# VARS paper / leaderboard reference numbers (balanced accuracy %)
VARS_REFERENCE = {
    "bal_acc_task1_action": 47.0,
    "bal_acc_task2_offence_severity": 43.0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fine-tuned Cosmos-Reason2 on SoccerNet-MVFoul")
    p.add_argument("--model-id", default="nvidia/Cosmos-Reason2-8B", help="Base model id")
    p.add_argument(
        "--adapter-path",
        required=True,
        help="Path to saved LoRA adapter (output of finetune_cosmos.py best_checkpoint)",
    )
    p.add_argument("--data-root", default="data/SoccerNet/mvfouls")
    p.add_argument("--split", default="Valid", choices=["Train", "Valid", "Test"])
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples (None = all)")
    p.add_argument("--num-frames", type=int, default=8, help="Frames per clip (must be even)")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default="outputs/finetuned_eval")
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def gpu_mem(device: str) -> str:
    if not torch.cuda.is_available():
        return ""
    idx = torch.device(device).index
    if idx is None:
        idx = torch.cuda.current_device()
    return f"vram={torch.cuda.memory_reserved(idx)/1024**3:.1f}GB"


def load_model(
    model_id: str,
    adapter_path: str,
    device: str,
) -> tuple[Any, AutoProcessor]:
    """Load base model in 4-bit and attach the fine-tuned LoRA adapter."""
    log(f"Loading processor from adapter path: {adapter_path}")
    # Prefer processor saved alongside the adapter; fall back to base model id
    proc_path = adapter_path if Path(adapter_path, "tokenizer_config.json").exists() else model_id
    processor = AutoProcessor.from_pretrained(proc_path)

    log(f"Loading base model {model_id} (4-bit NF4) …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map={"": device},
    )
    log(f"Base model loaded. {gpu_mem(device)}")

    log(f"Loading LoRA adapter from {adapter_path} …")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    log(f"Fine-tuned model ready. {gpu_mem(device)}")
    return model, processor


@torch.inference_mode()
def run_one(
    model: Any,
    processor: AutoProcessor,
    clip_path: Path,
    num_frames: int,
    device: str,
    max_new_tokens: int,
) -> tuple[str, dict, list[str], float]:
    """Generate and parse prediction for one clip. Returns (raw_output, prediction, errors, elapsed_s)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(clip_path), "nframes": num_frames},
                {"type": "text", "text": build_prompt()},
            ],
        }
    ]
    t0 = time.perf_counter()
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
    elapsed = time.perf_counter() - t0

    try:
        prediction = normalize_prediction_keys(extract_json(raw_output))
        errors = validate_prediction(prediction)
    except Exception as exc:
        prediction = {}
        errors = [f"{type(exc).__name__}: {exc}"]
    return raw_output, prediction, errors, elapsed


def print_comparison_table(
    finetuned_metrics: dict[str, Any],
    zeroshot_metrics: dict[str, Any] | None,
    vars_reference: dict[str, float],
    split: str,
    model_id: str = "nvidia/Cosmos-Reason2-8B",
) -> None:
    """Print a side-by-side comparison table."""
    model_name = model_id.split("/")[-1]  # e.g. "Cosmos-Reason2-8B"

    bal1_ft = finetuned_metrics["balanced_accuracy_action_seen_classes"]
    bal2_ft = finetuned_metrics["balanced_accuracy_offence_severity_seen_classes"]
    avg_ft = (bal1_ft + bal2_ft) / 2.0

    bal1_vars = vars_reference["bal_acc_task1_action"]
    bal2_vars = vars_reference["bal_acc_task2_offence_severity"]
    avg_vars = (bal1_vars + bal2_vars) / 2.0

    print("\n" + "=" * 70)
    print(f"  COMPARISON TABLE  —  {split} set")
    print("=" * 70)
    print(f"{'Model':<30}  {'Task1 Bal.Acc':>14}  {'Task2 Bal.Acc':>14}  {'Avg':>8}")
    print("-" * 70)
    print(f"{'VARS (paper baseline)':<30}  {bal1_vars:>13.2f}%  {bal2_vars:>13.2f}%  {avg_vars:>7.2f}%")
    if zeroshot_metrics is not None:
        bal1_zs = zeroshot_metrics["balanced_accuracy_action_seen_classes"]
        bal2_zs = zeroshot_metrics["balanced_accuracy_offence_severity_seen_classes"]
        avg_zs = (bal1_zs + bal2_zs) / 2.0
        n_zs = zeroshot_metrics.get("num_samples", "?")
        print(f"{f'{model_name} zero-shot':<30}  {bal1_zs:>13.2f}%  {bal2_zs:>13.2f}%  {avg_zs:>7.2f}%  (n={n_zs})")
    print(f"{f'{model_name} QLoRA FT':<30}  {bal1_ft:>13.2f}%  {bal2_ft:>13.2f}%  {avg_ft:>7.2f}%  (n={finetuned_metrics['num_samples']})")
    print("=" * 70)
    print(f"  Delta vs VARS:   Task1 {bal1_ft - bal1_vars:+.2f}%   Task2 {bal2_ft - bal2_vars:+.2f}%   Avg {avg_ft - avg_vars:+.2f}%")
    print("=" * 70 + "\n")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    if args.num_frames % 2 != 0:
        raise ValueError(f"--num-frames must be even, got {args.num_frames}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    # Load samples using the same official_target filter as VARS
    base_annotations, samples = load_samples(data_root, args.split, args.limit or 0)
    if not samples:
        raise RuntimeError(f"No valid samples found for split={args.split}")
    log(f"split={args.split}  samples={len(samples)}")

    model, processor = load_model(args.model_id, args.adapter_path, args.device)

    rows: list[dict] = []
    running_action_ok = 0
    running_offsev_ok = 0

    for idx, sample in enumerate(samples, start=1):
        should_log = args.log_every > 0 and (
            idx == 1 or idx == len(samples) or idx % args.log_every == 0
        )
        if should_log:
            log(f"[{idx}/{len(samples)}] action_{sample['action_id']}  {sample['clip_path']}")

        raw_output, prediction, errors, elapsed = run_one(
            model, processor, sample["clip_path"],
            args.num_frames, args.device, args.max_new_tokens,
        )
        row = {
            "action_id": sample["action_id"],
            "clip_path": str(sample["clip_path"]),
            "raw_output": raw_output,
            "prediction": prediction,
            "validation_errors": errors,
            "elapsed_sec": elapsed,
            "gold": {
                "action_class": sample["gold_action_class"],
                "offence_severity": sample["gold_offence_severity"],
                "Offence": sample["gold_offence"],
                "Severity": sample["gold_severity"],
            },
        }
        rows.append(row)
        if prediction.get("action_class") == sample["gold_action_class"]:
            running_action_ok += 1
        if prediction.get("offence_severity") == sample["gold_offence_severity"]:
            running_offsev_ok += 1

        if should_log:
            status = "ok" if not errors else "parse_error"
            log(
                f"  {status} | pred={prediction} | gold={row['gold']} | "
                f"running_acc action={running_action_ok/idx*100:.1f}% "
                f"offence_sev={running_offsev_ok/idx*100:.1f}% | {elapsed:.1f}s"
            )

    metrics = compute_metrics(rows)
    predictions_json = make_prediction_json(base_annotations, rows)

    prefix = args.output_prefix or f"finetuned_{args.split.lower()}{len(samples)}"
    rows_path = out_dir / f"{prefix}_rows.json"
    predictions_path = out_dir / f"{prefix}_predictions.json"
    metrics_path = out_dir / f"{prefix}_metrics.json"

    rows_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    predictions_path.write_text(json.dumps(predictions_json, indent=2, ensure_ascii=False))
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    log("Metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # Load zero-shot metrics for comparison if available
    zs_metrics = None
    zs_candidates = list(Path("outputs/zero_shot_32b").glob("valid*_metrics.json")) + \
                    list(Path("outputs/zero_shot").glob("zero_shot_valid*_metrics.json"))
    if zs_candidates:
        # Pick the one with the most samples
        zs_candidates.sort(key=lambda p: json.loads(p.read_text()).get("num_samples", 0), reverse=True)
        zs_metrics = json.loads(zs_candidates[0].read_text())
        log(f"Zero-shot reference: {zs_candidates[0]} (n={zs_metrics.get('num_samples')})")

    print_comparison_table(metrics, zs_metrics, VARS_REFERENCE, args.split, args.model_id)

    log("Written:")
    print(rows_path)
    print(predictions_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
