#!/usr/bin/env python
"""Option 2: Evaluate VLM adapter with skeleton text injected into the prompt.

Same setup as eval_finetuned_reason.py, but for each sample we load the
skeleton NPZ and prepend a body-pose description to the prompt. No retraining
required — inference-only change.

Usage:
    python scripts/eval/eval_skeleton_hint.py \
        --adapter-path outputs/qlora_cosmos8b_reason_2/best_checkpoint \
        --split Valid \
        --out-dir outputs/skeleton_hint_eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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
from train.train_reason import build_reasoning_prompt, extract_answer_json
from train.frame_utils import make_video_entry
from train.skeleton_utils import SK_ROOT_DEFAULT, load_qc_ok_set, load_skeleton, skeleton_to_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id",      default="nvidia/Cosmos-Reason2-8B")
    p.add_argument("--adapter-path",  required=True)
    p.add_argument("--data-root",     default="data/SoccerNet/mvfouls")
    p.add_argument("--split",         default="Valid", choices=["Train", "Valid", "Test"])
    p.add_argument("--limit",         type=int, default=None)
    p.add_argument("--num-frames",    type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--device",        default="cuda:0")
    p.add_argument("--skeleton-root", default=str(SK_ROOT_DEFAULT))
    p.add_argument("--out-dir",       default="outputs/skeleton_hint_eval")
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--log-every",     type=int, default=10)
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def gpu_mem(device: str) -> str:
    if not torch.cuda.is_available():
        return ""
    idx = torch.device(device).index or torch.cuda.current_device()
    return f"vram={torch.cuda.memory_reserved(idx)/1024**3:.1f}GB"


def load_model(model_id: str, adapter_path: str, device: str):
    proc_path = adapter_path if Path(adapter_path, "tokenizer_config.json").exists() else model_id
    processor = AutoProcessor.from_pretrained(proc_path)
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
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    log(f"Model loaded. {gpu_mem(device)}")
    return model, processor


def build_prompt_with_skeleton(
    num_frames: int,
    clip_path: Path,
    sk_root: str,
    ok_set: set | None,
) -> str:
    """Build reasoning prompt, optionally prepending skeleton description."""
    base_prompt = build_reasoning_prompt(num_frames)
    sk = load_skeleton(clip_path, sk_root=sk_root, ok_set=ok_set)
    if sk is None:
        return base_prompt
    sk_text = skeleton_to_text(sk)
    # Insert skeleton block between frame hint and label instructions
    # The base_prompt starts with "You are a soccer referee assistant..."
    # We inject skeleton text right before the label list
    marker = "Use only these action_class labels:"
    if marker in base_prompt:
        parts = base_prompt.split(marker, 1)
        return parts[0] + sk_text + "\n\n" + marker + parts[1]
    return base_prompt + "\n\n" + sk_text


@torch.inference_mode()
def run_one(model, processor, clip_path: Path, num_frames: int, device: str, max_new_tokens: int,
            sk_root: str = "", ok_set: set | None = None):
    prompt_text = build_prompt_with_skeleton(num_frames, clip_path, sk_root, ok_set)
    messages = [
        {
            "role": "user",
            "content": [
                make_video_entry(clip_path, num_frames),
                {"type": "text", "text": prompt_text},
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

    ok_set = load_qc_ok_set(args.skeleton_root, args.split)
    sk_count = sum(
        1 for s in samples
        if (Path(s["clip_path"]).parent.parent.name.lower() == args.split.lower()
            and (Path(args.skeleton_root) / args.split.lower()
                 / Path(s["clip_path"]).parent.name
                 / f"{Path(s['clip_path']).stem}.npz").exists()
            and (Path(s["clip_path"]).parent.name, Path(s["clip_path"]).stem) in ok_set)
    )
    log(f"split={args.split}  samples={len(samples)}  skeleton_ok={sk_count}")

    model, processor = load_model(args.model_id, args.adapter_path, args.device)

    rows: list[dict] = []
    run_ok = run_action = run_offsev = 0

    prefix = args.output_prefix or f"sk_hint_{args.split.lower()}{len(samples)}"
    jsonl_path = out_dir / f"{prefix}_rows.jsonl"
    jsonl_file = jsonl_path.open("w", encoding="utf-8")

    for idx, sample in enumerate(samples, start=1):
        should_log = idx == 1 or idx == len(samples) or (args.log_every > 0 and idx % args.log_every == 0)
        if should_log:
            log(f"[{idx}/{len(samples)}] action_{sample['action_id']}")

        raw_output, prediction, errors, elapsed = run_one(
            model, processor, sample["clip_path"],
            args.num_frames, args.device, args.max_new_tokens,
            sk_root=args.skeleton_root, ok_set=ok_set,
        )
        row = {
            "action_id":        sample["action_id"],
            "clip_path":        str(sample["clip_path"]),
            "has_skeleton":     (Path(args.skeleton_root) / args.split.lower()
                                 / Path(sample["clip_path"]).parent.name
                                 / f"{Path(sample['clip_path']).stem}.npz").exists(),
            "raw_output":       raw_output,
            "prediction":       prediction,
            "validation_errors": errors,
            "elapsed_sec":      elapsed,
            "gold": {
                "action_class":    sample["gold_action_class"],
                "offence_severity": sample["gold_offence_severity"],
                "Offence":         sample["gold_offence"],
                "Severity":        sample["gold_severity"],
            },
        }
        rows.append(row)
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        jsonl_file.flush()

        run_ok += 1
        if prediction.get("action_class") == sample["gold_action_class"]:
            run_action += 1
        if prediction.get("offence_severity") == sample["gold_offence_severity"]:
            run_offsev += 1

        if should_log:
            log(f"  {'ok' if not errors else 'parse_err'} | "
                f"pred={prediction} | gold={row['gold']} | "
                f"acc action={run_action/run_ok*100:.1f}% "
                f"offence_sev={run_offsev/run_ok*100:.1f}% | {elapsed:.1f}s")

    jsonl_file.close()

    metrics = compute_metrics(rows)
    predictions_json = make_prediction_json(base_annotations, rows)

    (out_dir / f"{prefix}_rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    (out_dir / f"{prefix}_predictions.json").write_text(json.dumps(predictions_json, indent=2, ensure_ascii=False))
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    log("Metrics:")
    print(json.dumps(metrics, indent=2))
    log(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
