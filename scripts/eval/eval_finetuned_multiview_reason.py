#!/usr/bin/env python
"""Evaluate multi-view reasoning QLoRA checkpoint on SoccerNet-MVFoul.

Uses clip_0 + clip_1 (fixed, deterministic) for evaluation.
Prints a comparison table: VARS | zero-shot-sv | single-view FT | multi-view FT.

Usage:
    python scripts/eval/eval_finetuned_multiview_reason.py \
        --adapter-path outputs/qlora_cosmos2b_multiview_reason/best_checkpoint \
        --split Valid
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
    log,
    make_prediction_json,
    normalize_prediction_keys,
    validate_prediction,
)
from train.train_reason import build_reasoning_prompt, extract_answer_json
from train.train_multiview_reason import (
    FoulDatasetMultiview,
    select_eval_clips,
    build_multiview_content,
    apply_pixel_budget,
)

VARS_REFERENCE = {
    "bal_acc_task2_offence_severity": 43.0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="nvidia/Cosmos-Reason2-2B")
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--data-root", default="data/SoccerNet/mvfouls")
    p.add_argument("--split", default="Valid", choices=["Train", "Valid", "Test"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default="outputs/multiview_reason_eval")
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-pixels", type=int, default=3_000_000,
                   help="processor.video_processor.size['longest_edge']. Must match training value.")
    p.add_argument("--sv-reason-metrics", default=None,
                   help="Path to single-view reason metrics JSON for comparison")
    return p.parse_args()


def gpu_mem(device: str) -> str:
    if not torch.cuda.is_available():
        return ""
    idx = torch.device(device).index or torch.cuda.current_device()
    return f"vram={torch.cuda.memory_reserved(idx)/1024**3:.1f}GB"


def load_model(model_id: str, adapter_path: str, device: str):
    log("Loading processor …")
    proc_path = adapter_path if Path(adapter_path, "tokenizer_config.json").exists() \
        else model_id
    processor = AutoProcessor.from_pretrained(proc_path)
    log(f"Loading base model {model_id} (4-bit NF4) …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        quantization_config=bnb_config, device_map={"": device},
    )
    log(f"Base loaded. {gpu_mem(device)}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    log(f"Adapter loaded. {gpu_mem(device)}")
    return model, processor  # caller must call apply_pixel_budget(processor, longest_edge)


@torch.inference_mode()
def run_one(model, processor, sample, clips, num_frames, device, max_new_tokens, max_pixels=None):
    data_root = Path(sample["_data_root"])
    split = sample["_split"]
    content = build_multiview_content(
        clips, data_root, split, sample["action_id"], num_frames, max_pixels
    )
    messages = [{"role": "user", "content": content}]
    t0 = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, repetition_penalty=1.5)
    raw_output = processor.batch_decode(
        generated_ids[:, prompt_len:],
        skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    elapsed = time.perf_counter() - t0
    try:
        prediction = normalize_prediction_keys(extract_answer_json(raw_output))
        errors = validate_prediction(prediction)
    except Exception as exc:
        prediction = {}
        errors = [f"{type(exc).__name__}: {exc}"]
    return raw_output, prediction, errors, elapsed


def _load_json(path: Path | None) -> dict | None:
    return json.loads(path.read_text()) if path and path.exists() else None


def _auto_discover(dirs: list[Path], glob: str) -> dict | None:
    candidates = [p for d in dirs for p in d.glob(glob) if d.exists()]
    if not candidates:
        return None
    candidates.sort(
        key=lambda p: json.loads(p.read_text()).get("num_samples", 0), reverse=True
    )
    return json.loads(candidates[0].read_text())


def print_comparison_table(mv_metrics, sv_metrics, zs_metrics, vars_ref, split, model_id):
    W = 76
    model_name = model_id.split("/")[-1]
    print("\n" + "=" * W)
    print(f"  COMPARISON TABLE — {split} set")
    print("=" * W)
    print(f"{'Model':<36}  {'Sev Bal.Acc':>12}")
    print("-" * W)

    def row(label, m, n=None):
        b2 = m["balanced_accuracy_offence_severity_seen_classes"]
        suffix = f"  n={n}" if n else ""
        print(f"{label:<36}  {b2:>11.2f}%{suffix}")
        return b2

    bv2 = vars_ref["bal_acc_task2_offence_severity"]
    print(f"{'VARS (paper baseline)':<36}  {bv2:>11.2f}%")

    if zs_metrics:
        row("zero-shot (single-view)", zs_metrics, zs_metrics.get("num_samples"))
    if sv_metrics:
        row(f"{model_name[:12]} reason SV FT", sv_metrics, sv_metrics.get("num_samples"))

    bm2 = row(
        f"{model_name[:12]} reason MV FT",
        mv_metrics, mv_metrics.get("num_samples")
    )
    print("=" * W)
    print(f"  Delta vs VARS: Sev {bm2-bv2:+.2f}%")
    if sv_metrics:
        bs2 = sv_metrics["balanced_accuracy_offence_severity_seen_classes"]
        print(f"  Delta vs SV FT: Sev {bm2-bs2:+.2f}%")
    print("=" * W + "\n")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    if args.num_frames % 2 != 0:
        raise ValueError(f"--num-frames must be even, got {args.num_frames}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    # load using FoulDatasetMultiview to access clip metadata
    dataset = FoulDatasetMultiview(data_root, args.split, limit=args.limit)
    for s in dataset.samples:
        s["_data_root"] = str(data_root)
        s["_split"] = args.split

    # also load base_annotations for make_prediction_json
    base_annotations, _ = load_samples(data_root, args.split, 0)

    if not dataset.samples:
        raise RuntimeError(f"No samples for split={args.split}")
    log(f"split={args.split}  samples={len(dataset)}")

    model, processor = load_model(args.model_id, args.adapter_path, args.device)
    if args.max_pixels and args.max_pixels > 0:
        apply_pixel_budget(processor, args.max_pixels)

    rows: list[dict] = []
    run_ok = run_offsev = 0

    prefix = args.output_prefix or f"mv_reason_{args.split.lower()}{len(dataset)}"
    jsonl_path = out_dir / f"{prefix}_rows.jsonl"
    jsonl_file = jsonl_path.open("w", encoding="utf-8")

    for idx, sample in enumerate(dataset.samples, start=1):
        clips = select_eval_clips(
            data_root, args.split, sample["action_id"], sample["record"]
        )
        should_log = idx == 1 or idx == len(dataset) or \
                     (args.log_every > 0 and idx % args.log_every == 0)
        if should_log:
            cam_types = [c.get("Camera type", "?") for _, c in clips]
            log(f"[{idx}/{len(dataset)}] action_{sample['action_id']}  "
                f"views={len(clips)}  cams={cam_types}")

        raw_output, prediction, errors, elapsed = run_one(
            model, processor, sample, clips,
            args.num_frames, args.device, args.max_new_tokens, args.max_pixels,
        )
        row = {
            "action_id": sample["action_id"],
            "n_views": len(clips),
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
            log(f"  {'ok' if not errors else 'parse_err'} | pred={prediction} | "
                f"gold={row['gold']} | "
                f"acc offence_sev={run_offsev/run_ok*100:.1f}% | {elapsed:.1f}s")

    jsonl_file.close()

    metrics = compute_metrics(rows)
    predictions_json = make_prediction_json(base_annotations, rows)

    (out_dir / f"{prefix}_rows.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False))
    (out_dir / f"{prefix}_predictions.json").write_text(
        json.dumps(predictions_json, indent=2, ensure_ascii=False))
    metrics_path = out_dir / f"{prefix}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    log("Metrics:")
    print(json.dumps(metrics, indent=2))

    zs_metrics = _auto_discover(
        [Path("outputs/zero_shot_32b"), Path("outputs/zero_shot_baseline")],
        "*valid*_metrics.json",
    )
    sv_metrics = _load_json(Path(args.sv_reason_metrics)) if args.sv_reason_metrics else None
    if sv_metrics is None:
        sv_metrics = _auto_discover(
            [Path("outputs/reason_eval")], f"reason_{args.split.lower()}*_metrics.json"
        )

    print_comparison_table(
        metrics, sv_metrics, zs_metrics, VARS_REFERENCE, args.split, args.model_id
    )
    log(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
