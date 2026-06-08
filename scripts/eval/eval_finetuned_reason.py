#!/usr/bin/env python
"""Evaluate a reasoning QLoRA checkpoint on SoccerNet-MVFoul.

Prints a 4-row comparison table:
    VARS | zero-shot | non-reason QLoRA FT | reason QLoRA FT

Usage:
    python scripts/eval_finetuned_reason.py \
        --adapter-path outputs/qlora_cosmos8b_reason/best_checkpoint \
        --split Valid \
        --out-dir outputs/reason_eval
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
    compute_metrics,
    load_samples,
    make_prediction_json,
    normalize_prediction_keys,
    validate_prediction,
)
from train.train_reason import (
    build_reasoning_prompt,
    extract_answer_json,
)

VARS_REFERENCE = {
    "bal_acc_task2_offence_severity": 43.0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="nvidia/Cosmos-Reason2-8B")
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--data-root", default="data/SoccerNet/mvfouls")
    p.add_argument("--split", default="Valid", choices=["Train", "Valid", "Test"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default="outputs/reason_eval")
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--baseline-metrics",
        default=None,
        help="Path to non-reason QLoRA metrics JSON (auto-discovered if omitted)",
    )
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


def load_model(model_id: str, adapter_path: str, device: str):
    log(f"Loading processor …")
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
    log(f"Base loaded. {gpu_mem(device)}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    log(f"Adapter loaded. {gpu_mem(device)}")
    return model, processor


@torch.inference_mode()
def run_one(model, processor, clip_path, num_frames, device, max_new_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(clip_path), "nframes": num_frames},
                {"type": "text", "text": build_reasoning_prompt(num_frames)},
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
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, repetition_penalty=1.5)
    raw_output = processor.batch_decode(
        generated_ids[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    elapsed = time.perf_counter() - t0

    try:
        prediction = normalize_prediction_keys(extract_answer_json(raw_output))
        errors = validate_prediction(prediction)
    except Exception as exc:
        prediction = {}
        errors = [f"{type(exc).__name__}: {exc}"]
    return raw_output, prediction, errors, elapsed


def _load_json_if_exists(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _auto_discover(pattern_dirs: list[Path], glob: str) -> dict | None:
    """Find the metrics JSON with the most samples among candidates."""
    candidates = []
    for d in pattern_dirs:
        candidates.extend(d.glob(glob))
    if not candidates:
        return None
    candidates.sort(
        key=lambda p: json.loads(p.read_text()).get("num_samples", 0), reverse=True
    )
    return json.loads(candidates[0].read_text())


def print_comparison_table(
    reason_metrics: dict,
    baseline_metrics: dict | None,
    zeroshot_metrics: dict | None,
    vars_ref: dict,
    split: str,
    model_id: str,
) -> None:
    model_name = model_id.split("/")[-1]
    W = 72
    print("\n" + "=" * W)
    print(f"  COMPARISON TABLE  —  {split} set")
    print("=" * W)
    print(f"{'Model':<34}  {'Sev Bal.Acc':>12}")
    print("-" * W)

    def _row(label, m, n=None):
        b2 = m["balanced_accuracy_offence_severity_seen_classes"]
        suffix = f"  (n={n})" if n is not None else ""
        print(f"{label:<34}  {b2:>11.2f}%{suffix}")
        return b2

    bv2 = VARS_REFERENCE["bal_acc_task2_offence_severity"]
    print(f"{'VARS (paper baseline)':<34}  {bv2:>11.2f}%")

    if zeroshot_metrics is not None:
        _row(f"{model_name} zero-shot", zeroshot_metrics, zeroshot_metrics.get("num_samples"))

    if baseline_metrics is not None:
        _row(f"{model_name} QLoRA (no reason)", baseline_metrics, baseline_metrics.get("num_samples"))

    br2 = _row(f"{model_name} QLoRA + reason", reason_metrics, reason_metrics.get("num_samples"))

    print("=" * W)
    print(f"  Delta vs VARS:  Sev {br2 - bv2:+.2f}%")
    if baseline_metrics is not None:
        bb2 = baseline_metrics["balanced_accuracy_offence_severity_seen_classes"]
        print(f"  Delta vs no-reason FT: Sev {br2 - bb2:+.2f}%")
    print("=" * W + "\n")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    if args.num_frames % 2 != 0:
        raise ValueError(f"--num-frames must be even, got {args.num_frames}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_annotations, samples = load_samples(Path(args.data_root), args.split, args.limit or 0)
    if not samples:
        raise RuntimeError(f"No samples for split={args.split}")
    log(f"split={args.split}  samples={len(samples)}")

    model, processor = load_model(args.model_id, args.adapter_path, args.device)

    rows: list[dict] = []
    run_ok = run_offsev = 0

    prefix = args.output_prefix or f"reason_{args.split.lower()}{len(samples)}"
    jsonl_path = out_dir / f"{prefix}_rows.jsonl"
    jsonl_file = jsonl_path.open("w", encoding="utf-8")

    for idx, sample in enumerate(samples, start=1):
        should_log = idx == 1 or idx == len(samples) or (args.log_every > 0 and idx % args.log_every == 0)
        if should_log:
            log(f"[{idx}/{len(samples)}] action_{sample['action_id']}")

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
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        jsonl_file.flush()
        run_ok += 1
        if prediction.get("offence_severity") == sample["gold_offence_severity"]:
            run_offsev += 1

        if should_log:
            log(f"  {'ok' if not errors else 'parse_err'} | pred={prediction} | gold={row['gold']} | "
                f"acc offence_sev={run_offsev/run_ok*100:.1f}% | {elapsed:.1f}s")

    jsonl_file.close()

    metrics = compute_metrics(rows)
    predictions_json = make_prediction_json(base_annotations, rows)

    (out_dir / f"{prefix}_rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    (out_dir / f"{prefix}_predictions.json").write_text(json.dumps(predictions_json, indent=2, ensure_ascii=False))
    metrics_path = out_dir / f"{prefix}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    log("Metrics:")
    print(json.dumps(metrics, indent=2))

    # Auto-discover comparison metrics
    zs_metrics = _auto_discover(
        [Path("outputs/zero_shot_32b"), Path("outputs/zero_shot")],
        "*valid*_metrics.json",
    )
    if zs_metrics:
        log(f"Zero-shot reference found (n={zs_metrics.get('num_samples')})")

    baseline_metrics = None
    if args.baseline_metrics:
        baseline_metrics = _load_json_if_exists(Path(args.baseline_metrics))
    if baseline_metrics is None:
        # Auto-discover: look in finetuned_eval dir
        split_lower = args.split.lower()
        baseline_metrics = _auto_discover(
            [Path("outputs/finetuned_eval")],
            f"finetuned_{split_lower}*_metrics.json",
        )
    if baseline_metrics:
        log(f"Baseline (no-reason) FT metrics found (n={baseline_metrics.get('num_samples')})")

    print_comparison_table(metrics, baseline_metrics, zs_metrics, VARS_REFERENCE, args.split, args.model_id)
    log(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
