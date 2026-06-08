#!/usr/bin/env python
"""Zero-shot evaluation of Cosmos-Reason2 on SoccerNet-MVFoul — REASONING mode.

Uses the <think>...</think><answer>{...}</answer> prompt format.
No fine-tuned adapter; evaluates the base model's zero-shot reasoning ability.

Usage:
    python scripts/eval_zeroshot_reason.py \
        --model-id nvidia/Cosmos-Reason2-8B \
        --split Valid --limit 50
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
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent.parent))
from zero_shot.zero_shot_eval import (
    compute_metrics,
    load_samples,
    log,
    make_prediction_json,
    normalize_prediction_keys,
    gpu_memory_summary,
    validate_prediction,
)
from train.train_reason import build_reasoning_prompt, extract_answer_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot reasoning eval on SoccerNet-MVFoul")
    p.add_argument("--model-id", default="nvidia/Cosmos-Reason2-8B")
    p.add_argument("--data-root", default="data/SoccerNet/mvfouls")
    p.add_argument("--split", default="Valid", choices=["Train", "Valid", "Test"])
    p.add_argument("--limit", type=int, default=5, help="0 = all samples")
    p.add_argument("--num-frames", type=int, default=8, help="Must be even")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--out-dir", default="outputs/zero_shot_reason")
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


def run_one(
    model: Any,
    processor: AutoProcessor,
    clip_path: Path,
    num_frames: int,
    device: str,
    max_new_tokens: int,
) -> tuple[str, dict, list[str], float]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(clip_path), "nframes": num_frames},
                {"type": "text", "text": build_reasoning_prompt()},
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
    inputs = inputs.to(device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    raw_output = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    elapsed = time.perf_counter() - t0

    try:
        prediction = normalize_prediction_keys(extract_answer_json(raw_output))
        errors = validate_prediction(prediction)
    except Exception as exc:
        prediction = {}
        errors = [f"{type(exc).__name__}: {exc}"]
    return raw_output, prediction, errors, elapsed


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    if args.num_frames % 2 != 0:
        raise ValueError(f"--num-frames must be even, got {args.num_frames}")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_annotations, samples = load_samples(data_root, args.split, args.limit)
    if not samples:
        raise RuntimeError(f"No valid samples for split={args.split}")

    log(f"model_id   : {args.model_id}")
    log(f"split      : {args.split}")
    log(f"samples    : {len(samples)}")
    log(f"num_frames : {args.num_frames}")
    log(f"4bit       : {not args.no_4bit}")
    log(f"prompt     : reasoning (<think>/<answer>)")

    load_start = time.perf_counter()
    log("Loading processor …")
    processor = AutoProcessor.from_pretrained(args.model_id)

    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    log("Loading model …")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map={"": args.device},
    )
    model.eval()
    log(f"Model loaded in {time.perf_counter() - load_start:.1f}s  {gpu_memory_summary(args.device)}")

    rows: list[dict] = []
    run_action_ok = run_offsev_ok = 0
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
            run_action_ok += 1
        if prediction.get("offence_severity") == sample["gold_offence_severity"]:
            run_offsev_ok += 1

        if should_log:
            status = "ok" if not errors else "parse_err"
            log(
                f"  {status} | pred={prediction} | gold={row['gold']} | "
                f"running_acc action={run_action_ok/idx*100:.1f}% "
                f"offence_sev={run_offsev_ok/idx*100:.1f}% | {elapsed:.1f}s | "
                f"{gpu_memory_summary(args.device)}"
            )

    prefix = args.output_prefix or f"zeroshot_reason_{args.split.lower()}{len(samples)}"
    rows_path = out_dir / f"{prefix}_rows.json"
    predictions_path = out_dir / f"{prefix}_predictions.json"
    metrics_path = out_dir / f"{prefix}_metrics.json"

    metrics = compute_metrics(rows)
    predictions = make_prediction_json(base_annotations, rows)

    rows_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    predictions_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False))
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    log("Metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    log("Written:")
    print(rows_path)
    print(predictions_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
