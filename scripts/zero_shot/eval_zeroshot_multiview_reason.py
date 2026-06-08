#!/usr/bin/env python
"""Zero-shot multi-view reasoning eval for Cosmos-Reason2 on SoccerNet-MVFoul.

Uses the base model with the same fixed evaluation view policy as MV fine-tune:
clip_0 + clip_1 when available, otherwise the lowest-indexed available clip(s).
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
from zero_shot.zero_shot_eval import (  # noqa: E402
    compute_metrics,
    gpu_memory_summary,
    log,
    make_prediction_json,
)
from train.train_multiview_reason import (  # noqa: E402
    FoulDatasetMultiview,
    apply_pixel_budget,
    run_one_inference_multiview,
    select_eval_clips,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot multi-view reasoning eval on SoccerNet-MVFoul")
    p.add_argument("--model-id", default="nvidia/Cosmos-Reason2-8B")
    p.add_argument("--data-root", default="data/SoccerNet/mvfouls")
    p.add_argument("--split", default="Valid", choices=["Train", "Valid", "Test"])
    p.add_argument("--limit", type=int, default=5, help="0 = all official-target samples")
    p.add_argument("--num-frames", type=int, default=16, help="Frames per view; must be even")
    p.add_argument("--max-pixels", type=int, default=10_000_000)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--out-dir", default="outputs/zero_shot_multiview_reason_severity")
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    if args.num_frames % 2 != 0:
        raise ValueError(f"--num-frames must be even, got {args.num_frames}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu only for debugging.")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    limit = None if args.limit == 0 else args.limit
    dataset = FoulDatasetMultiview(data_root, args.split, limit=limit)
    for sample in dataset.samples:
        sample["_data_root"] = str(data_root)
        sample["_split"] = args.split

    if not dataset.samples:
        raise RuntimeError(f"No valid samples for split={args.split}")

    ann_path = data_root / args.split / "annotations.json"
    base_annotations = json.loads(ann_path.read_text())

    log(f"model_id      : {args.model_id}")
    log(f"split         : {args.split}")
    log(f"samples       : {len(dataset.samples)}")
    log(f"num_frames    : {args.num_frames} per view")
    log(f"max_pixels    : {args.max_pixels}")
    log(f"4bit          : {not args.no_4bit}")
    log("prompt        : multi-view reasoning (<think>/<answer>)")

    load_start = time.perf_counter()
    log("Loading processor …")
    processor = AutoProcessor.from_pretrained(args.model_id)
    apply_pixel_budget(processor, args.max_pixels)

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

    rows: list[dict[str, Any]] = []
    running_offsev_ok = 0
    for idx, sample in enumerate(dataset.samples, start=1):
        clips = select_eval_clips(data_root, args.split, sample["action_id"], sample["record"])
        clip_ids = [clip_idx for clip_idx, _ in clips]
        should_log = args.log_every > 0 and (idx == 1 or idx == len(dataset.samples) or idx % args.log_every == 0)
        if should_log:
            log(f"[{idx}/{len(dataset.samples)}] action_{sample['action_id']} clips={clip_ids}")

        t0 = time.perf_counter()
        try:
            raw_output, prediction, errors = run_one_inference_multiview(
                model=model,
                processor=processor,
                sample=sample,
                clips=clips,
                num_frames=args.num_frames,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                max_pixels=args.max_pixels,
            )
        except Exception as exc:
            raw_output, prediction, errors = "", {}, [f"InferenceError: {type(exc).__name__}: {exc}"]
        elapsed = time.perf_counter() - t0

        row = {
            "action_id": sample["action_id"],
            "n_views": len(clips),
            "clip_indices": clip_ids,
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
        if prediction.get("offence_severity") == sample["gold_offence_severity"]:
            running_offsev_ok += 1

        if should_log:
            status = "ok" if not errors else "parse/label_error"
            log(
                f"  {status} | pred={prediction} | gold={row['gold']} | "
                f"running_acc offence_severity={running_offsev_ok / idx * 100:.1f}% | "
                f"{elapsed:.1f}s | {gpu_memory_summary(args.device)}"
            )

    prefix = args.output_prefix or f"zeroshot_mv_reason_{args.split.lower()}{len(rows)}"
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
    print(rows_path, flush=True)
    print(predictions_path, flush=True)
    print(metrics_path, flush=True)


if __name__ == "__main__":
    main()
