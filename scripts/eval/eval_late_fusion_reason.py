#!/usr/bin/env python
"""Action-level late-fusion evaluation for SoccerNet-MVFoul reasoning models.

Each clip_i of an action is evaluated independently with the single-view prompt.
The final action prediction is produced by fusing per-view predictions.

If --adapter-path is omitted, this is zero-shot late fusion. If provided, the
same code evaluates an SV-trained QLoRA adapter with multi-view inference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent.parent))
from train.frame_utils import make_video_entry  # noqa: E402
from train.train_reason import build_reasoning_prompt, extract_answer_json  # noqa: E402
from zero_shot.zero_shot_eval import (  # noqa: E402
    OFFENCE_SEVERITY_CLASSES,
    compute_metrics,
    make_prediction_json,
    normalize_prediction_keys,
    official_target,
    validate_prediction,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Late-fusion eval for SV reasoning VLM on MVFoul")
    p.add_argument("--model-id", default="nvidia/Cosmos-Reason2-8B")
    p.add_argument("--adapter-path", default=None, help="Optional QLoRA adapter checkpoint")
    p.add_argument("--data-root", default="data/SoccerNet/mvfouls")
    p.add_argument("--split", default="Valid", choices=["Train", "Valid", "Test"])
    p.add_argument("--limit", type=int, default=0, help="0 = all official-target actions")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--max-views", type=int, default=0, help="0 = all available views; otherwise first N clips")
    p.add_argument(
        "--fusion-rule",
        choices=["main_first", "clip1_first", "majority_vote", "majority_clip1_tiebreak", "conservative_card"],
        default="main_first",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--out-dir", default="outputs/late_fusion_reason")
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=20, help="Write partial JSON outputs every N newly processed actions; 0 disables")
    p.add_argument("--resume", action="store_true", help="Resume from existing *_rows.jsonl or *_rows.json for this output prefix")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def make_jsonable(obj: Any) -> Any:
    """Convert nested rows/metrics into values accepted by json.dumps."""
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(make_jsonable(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_rows_for_resume(rows_path: Path, rows_jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if rows_jsonl_path.exists():
        for line in rows_jsonl_path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    elif rows_path.exists():
        rows = json.loads(rows_path.read_text())

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        action_id = str(row.get("action_id", ""))
        if not action_id or action_id in seen:
            continue
        seen.add(action_id)
        deduped.append(row)
    return deduped


def append_row_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(make_jsonable(row), ensure_ascii=False) + "\n")
        f.flush()


def write_eval_outputs(
    rows: list[dict[str, Any]],
    annotations: dict[str, Any],
    rows_path: Path,
    predictions_path: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    metrics = compute_metrics(rows)
    metrics["view_level"] = view_level_summary(rows)
    predictions = make_prediction_json(annotations, rows)
    rows_path.write_text(json.dumps(make_jsonable(rows), indent=2, ensure_ascii=False))
    predictions_path.write_text(json.dumps(make_jsonable(predictions), indent=2, ensure_ascii=False))
    metrics_path.write_text(json.dumps(make_jsonable(metrics), indent=2, ensure_ascii=False))
    return metrics


def gpu_mem(device: str) -> str:
    if not torch.cuda.is_available():
        return ""
    idx = torch.device(device).index
    if idx is None:
        idx = torch.cuda.current_device()
    return f"vram={torch.cuda.memory_reserved(idx) / 1024**3:.1f}GB"


class ActionViewDataset:
    """Action-level dataset with all existing clip_i paths per action."""

    def __init__(self, data_root: Path, split: str, limit: int = 0, max_views: int = 0) -> None:
        self.data_root = data_root
        self.split = split
        ann_path = data_root / split / "annotations.json"
        self.annotations = json.loads(ann_path.read_text())
        self.actions: list[dict[str, Any]] = []
        for action_id, record in self.annotations["Actions"].items():
            target = official_target(record)
            if target is None:
                continue
            clips = []
            for idx, clip_info in enumerate(record.get("Clips", [])):
                clip_path = data_root / split / f"action_{action_id}" / f"clip_{idx}.mp4"
                if clip_path.exists():
                    clips.append({
                        "clip_idx": idx,
                        "clip_path": clip_path,
                        "camera_type": clip_info.get("Camera type", ""),
                        "replay_speed": clip_info.get("Replay speed", ""),
                    })
            if not clips:
                continue
            if max_views > 0:
                clips = clips[:max_views]
            action_class, offence_severity, offence_class, severity_class = target
            self.actions.append({
                "action_id": action_id,
                "record": record,
                "clips": clips,
                "gold_action_class": action_class,
                "gold_offence_severity": offence_severity,
                "gold_offence": offence_class,
                "gold_severity": severity_class,
            })
            if limit and len(self.actions) >= limit:
                break

    def __len__(self) -> int:
        return len(self.actions)


def load_model_and_processor(args: argparse.Namespace):
    proc_path = args.adapter_path if args.adapter_path and Path(args.adapter_path, "tokenizer_config.json").exists() else args.model_id
    log(f"Loading processor: {proc_path}")
    processor = AutoProcessor.from_pretrained(proc_path)

    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    log(f"Loading base model: {args.model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map={"": args.device},
    )
    if args.adapter_path:
        log(f"Loading adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    log(f"Model ready. {gpu_mem(args.device)}")
    return model, processor


@torch.inference_mode()
def run_one_view(
    model: Any,
    processor: AutoProcessor,
    clip_path: Path,
    num_frames: int,
    device: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any], list[str], float]:
    messages = [{
        "role": "user",
        "content": [
            make_video_entry(clip_path, num_frames),
            {"type": "text", "text": build_reasoning_prompt(num_frames)},
        ],
    }]
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


def _valid_label(prediction: dict[str, Any]) -> str | None:
    label = prediction.get("offence_severity")
    return label if label in OFFENCE_SEVERITY_CLASSES else None


def _choose_first_valid(valid: list[tuple[int, str]], preferred: list[int]) -> tuple[int, str]:
    for wanted_idx in preferred:
        for idx, label in valid:
            if idx == wanted_idx:
                return idx, label
    return valid[0]


def _majority_with_tiebreak(
    valid: list[tuple[int, str]],
    preferred_tiebreak: list[int],
    rule_name: str,
) -> tuple[dict[str, str], str]:
    counts = Counter(label for _, label in valid)
    best_count = max(counts.values())
    tied = {label for label, count in counts.items() if count == best_count}
    if len(tied) == 1:
        label = next(iter(tied))
        return {"offence_severity": label}, f"{rule_name}:majority:{label}:{best_count}votes"

    tied_valid = [(idx, label) for idx, label in valid if label in tied]
    idx, label = _choose_first_valid(tied_valid, preferred_tiebreak)
    return {"offence_severity": label}, f"{rule_name}:tie:{label}:clip_{idx}"


def fuse_predictions(view_rows: list[dict[str, Any]], rule: str) -> tuple[dict[str, str], str]:
    valid = [(v["clip_idx"], _valid_label(v.get("prediction") or {})) for v in view_rows]
    valid = [(idx, label) for idx, label in valid if label is not None]
    if not valid:
        return {}, "no_valid_view"

    if rule == "main_first":
        idx, label = _choose_first_valid(valid, [0])
        detail = "main_first:clip_0" if idx == 0 else f"main_first:fallback_clip_{idx}"
        return {"offence_severity": label}, detail

    if rule == "clip1_first":
        idx, label = _choose_first_valid(valid, [1, 0])
        detail = "clip1_first:clip_1" if idx == 1 else f"clip1_first:fallback_clip_{idx}"
        return {"offence_severity": label}, detail

    if rule == "majority_vote":
        return _majority_with_tiebreak(valid, [0], "majority_vote")

    if rule == "majority_clip1_tiebreak":
        return _majority_with_tiebreak(valid, [1, 0], "majority_clip1_tiebreak")

    if rule == "conservative_card":
        counts = Counter(label for _, label in valid)
        if counts["Offence + Red card"] >= 2:
            return {"offence_severity": "Offence + Red card"}, "conservative_card:red_confirmed_2plus"
        if counts["Offence + Red card"] == 1:
            non_red = [(idx, label) for idx, label in valid if label != "Offence + Red card"]
            if non_red:
                prediction, detail = _majority_with_tiebreak(non_red, [1, 0], "conservative_card:no_red_majority")
                return prediction, detail + ":red_downgraded"
            return {"offence_severity": "Offence + Yellow card"}, "conservative_card:red_singleton_downgrade_yellow"
        return _majority_with_tiebreak(valid, [1, 0], "conservative_card")

    raise ValueError(f"Unknown fusion rule: {rule}")


def evaluate_late_fusion(
    model: Any,
    processor: AutoProcessor,
    dataset: ActionViewDataset,
    num_frames: int,
    device: str,
    max_new_tokens: int,
    fusion_rule: str,
    log_every: int = 10,
    existing_rows: list[dict[str, Any]] | None = None,
    rows_jsonl_path: Path | None = None,
    save_every: int = 0,
    save_callback: Any | None = None,
) -> list[dict[str, Any]]:
    was_training = model.training
    old_use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    if old_use_cache is not None:
        model.config.use_cache = True

    rows: list[dict[str, Any]] = list(existing_rows or [])
    done_action_ids = {str(row.get("action_id")) for row in rows}
    running_ok = sum(
        1 for row in rows
        if (row.get("prediction") or {}).get("offence_severity") == (row.get("gold") or {}).get("offence_severity")
    )
    processed_new = 0
    if rows:
        log(f"Resuming with {len(rows)} completed actions")
    for i, action in enumerate(dataset.actions, start=1):
        if str(action["action_id"]) in done_action_ids:
            continue
        should_log = i == 1 or i == len(dataset.actions) or (log_every > 0 and i % log_every == 0)
        if should_log:
            log(f"[{i}/{len(dataset.actions)}] action_{action['action_id']} views={len(action['clips'])}")
        view_rows = []
        for clip in action["clips"]:
            try:
                raw_output, prediction, errors, elapsed = run_one_view(
                    model, processor, clip["clip_path"], num_frames, device, max_new_tokens
                )
            except Exception as exc:
                raw_output, prediction, errors, elapsed = "", {}, [f"InferenceError: {type(exc).__name__}: {exc}"], 0.0
            view_rows.append({
                "clip_idx": clip["clip_idx"],
                "clip_path": str(clip["clip_path"]),
                "camera_type": clip["camera_type"],
                "replay_speed": clip["replay_speed"],
                "raw_output": raw_output,
                "prediction": prediction,
                "validation_errors": errors,
                "elapsed_sec": elapsed,
            })
        final_prediction, fusion_detail = fuse_predictions(view_rows, fusion_rule)
        row = {
            "action_id": action["action_id"],
            "n_views": len(view_rows),
            "fusion_rule": fusion_rule,
            "fusion_detail": fusion_detail,
            "view_predictions": view_rows,
            "prediction": final_prediction,
            "validation_errors": [] if final_prediction else ["no valid fused prediction"],
            "gold": {
                "action_class": action["gold_action_class"],
                "offence_severity": action["gold_offence_severity"],
                "Offence": action["gold_offence"],
                "Severity": action["gold_severity"],
            },
        }
        rows.append(row)
        done_action_ids.add(str(action["action_id"]))
        processed_new += 1
        if rows_jsonl_path is not None:
            append_row_jsonl(rows_jsonl_path, row)
        if final_prediction.get("offence_severity") == action["gold_offence_severity"]:
            running_ok += 1
        if save_every > 0 and save_callback is not None and processed_new % save_every == 0:
            save_callback(rows)
            log(f"  partial saved: {len(rows)}/{len(dataset.actions)} actions")
        if should_log:
            log(f"  pred={final_prediction} gold={row['gold']} acc={running_ok / len(rows) * 100:.1f}% {gpu_mem(device)}")
    if old_use_cache is not None:
        model.config.use_cache = old_use_cache
    if was_training:
        model.train()
    return rows


def view_level_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_idx: dict[int, list[dict[str, Any]]] = defaultdict(list)
    parse_errors = 0
    total_views = 0
    for row in rows:
        gold = row["gold"]["offence_severity"]
        for view in row["view_predictions"]:
            total_views += 1
            if view.get("validation_errors"):
                parse_errors += 1
            by_idx[int(view["clip_idx"])].append({
                "prediction": view.get("prediction") or {},
                "gold": {"offence_severity": gold},
                "validation_errors": view.get("validation_errors") or [],
            })
    out: dict[str, Any] = {"total_views": total_views, "view_parse_errors": parse_errors, "by_clip_idx": {}}
    for idx, view_rows in sorted(by_idx.items()):
        out["by_clip_idx"][str(idx)] = compute_metrics(view_rows)
    return out


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.num_frames % 2 != 0:
        raise ValueError(f"--num-frames must be even, got {args.num_frames}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu only for debugging.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = ActionViewDataset(Path(args.data_root), args.split, limit=args.limit, max_views=args.max_views)
    if not dataset.actions:
        raise RuntimeError(f"No valid actions for split={args.split}")
    log(f"split={args.split} actions={len(dataset.actions)} max_views={args.max_views or 'all'} fusion={args.fusion_rule}")

    adapter_tag = "ft" if args.adapter_path else "zs"
    views_tag = f"views{args.max_views}" if args.max_views else "viewsall"
    prefix = args.output_prefix or f"latefusion_{adapter_tag}_{args.fusion_rule}_{views_tag}_{args.split.lower()}{len(dataset.actions)}"
    rows_path = out_dir / f"{prefix}_rows.json"
    rows_jsonl_path = out_dir / f"{prefix}_rows.jsonl"
    predictions_path = out_dir / f"{prefix}_predictions.json"
    metrics_path = out_dir / f"{prefix}_metrics.json"

    existing_rows: list[dict[str, Any]] = []
    if args.resume:
        existing_rows = load_rows_for_resume(rows_path, rows_jsonl_path)
        log(f"resume={args.resume} loaded_rows={len(existing_rows)} from {rows_jsonl_path if rows_jsonl_path.exists() else rows_path}")
    else:
        for stale_path in (rows_path, rows_jsonl_path, predictions_path, metrics_path):
            if stale_path.exists():
                stale_path.unlink()

    def save_partial(current_rows: list[dict[str, Any]]) -> None:
        write_eval_outputs(current_rows, dataset.annotations, rows_path, predictions_path, metrics_path)

    if len(existing_rows) >= len(dataset.actions):
        log("All actions already completed; writing final outputs without loading model")
        rows = existing_rows
    else:
        model, processor = load_model_and_processor(args)
        rows = evaluate_late_fusion(
            model, processor, dataset, args.num_frames, args.device,
            args.max_new_tokens, args.fusion_rule, args.log_every,
            existing_rows=existing_rows,
            rows_jsonl_path=rows_jsonl_path,
            save_every=args.save_every,
            save_callback=save_partial,
        )
    metrics = write_eval_outputs(rows, dataset.annotations, rows_path, predictions_path, metrics_path)

    log("Metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    log("Written:")
    print(rows_path, flush=True)
    print(predictions_path, flush=True)
    print(metrics_path, flush=True)


if __name__ == "__main__":
    main()
