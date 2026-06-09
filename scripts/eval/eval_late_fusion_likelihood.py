#!/usr/bin/env python
"""Late-fusion evaluation with 4-label likelihood scoring.

Instead of generating free-form <think>/<answer> text, this script scores the
four allowed offence_severity labels as fixed candidate assistant responses.
This removes JSON parsing failures, CJK drift, and max_new_tokens effects from
classification.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.eval_late_fusion_reason import ActionViewDataset, gpu_mem, make_jsonable  # noqa: E402
from train.frame_utils import foul_frame_index, make_video_entry  # noqa: E402
from zero_shot.zero_shot_eval import (  # noqa: E402
    OFFENCE_SEVERITY_CLASSES,
    compute_metrics,
    make_prediction_json,
)

FUSION_METHODS = ["score_mean", "score_sum", "clip0", "clip1", "weighted_clip1"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Likelihood-scoring late fusion for MVFoul severity classification")
    p.add_argument("--model-id", default="nvidia/Cosmos-Reason2-8B")
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--data-root", default="data/SoccerNet/mvfouls")
    p.add_argument("--split", default="Valid", choices=["Train", "Valid", "Test"])
    p.add_argument("--limit", type=int, default=0, help="0 = all official-target actions")
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--max-views", type=int, default=0, help="0 = all available views")
    p.add_argument("--candidate-format", choices=["json", "answer_tag"], default="json")
    p.add_argument("--score-reduction", choices=["mean", "sum"], default="mean")
    p.add_argument("--fusion-methods", default=",".join(FUSION_METHODS))
    p.add_argument("--prior-alphas", default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--prior-split", default="Train", choices=["Train", "Valid", "Test"])
    p.add_argument("--clip1-weight", type=float, default=1.2)
    p.add_argument("--other-view-weight", type=float, default=0.8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--out-dir", default="outputs/late_fusion_likelihood")
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_methods(text: str) -> list[str]:
    methods = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [m for m in methods if m not in FUSION_METHODS]
    if unknown:
        raise ValueError(f"Unknown fusion methods: {unknown}; allowed={FUSION_METHODS}")
    return methods


def build_scoring_prompt(num_frames: int, num_views: int = 1) -> str:
    idx = foul_frame_index(num_frames)
    clip_word = "each clip" if num_views > 1 else "the clip"
    labels = "\n".join(f"- {label}" for label in OFFENCE_SEVERITY_CLASSES)
    return (
        "You are a soccer referee assistant. Watch the video clip of a football incident.\n"
        "Choose exactly one offence severity label. Do not predict action_class.\n"
        f"Focus on Frame {idx} of {num_frames} in {clip_word}; it captures the foul contact moment.\n\n"
        "Allowed labels:\n"
        f"{labels}\n\n"
        "Respond only with valid JSON using this exact key: offence_severity."
    )


def candidate_text(label: str, fmt: str) -> str:
    js = json.dumps({"offence_severity": label}, ensure_ascii=False)
    if fmt == "json":
        return js
    if fmt == "answer_tag":
        return f"<answer>{js}</answer>"
    raise ValueError(fmt)


def to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    moved = {}
    for k, v in inputs.items():
        moved[k] = v.to(device) if hasattr(v, "to") else v
    return moved


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


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
def score_candidate(
    model: Any,
    processor: AutoProcessor,
    video_entry: dict[str, Any],
    prompt_text: str,
    answer_text: str,
    device: str,
) -> tuple[float, float, int]:
    user_msg = [{
        "role": "user",
        "content": [video_entry, {"type": "text", "text": prompt_text}],
    }]
    full_msg = [{
        "role": "user",
        "content": [video_entry, {"type": "text", "text": prompt_text}],
    }, {
        "role": "assistant",
        "content": [{"type": "text", "text": answer_text}],
    }]

    prompt_inputs = processor.apply_chat_template(
        user_msg,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_inputs = processor.apply_chat_template(
        full_msg,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )

    prompt_ids = prompt_inputs["input_ids"][0].tolist()
    full_ids = full_inputs["input_ids"][0].tolist()
    prefix_len = common_prefix_len(prompt_ids, full_ids)
    if prefix_len >= len(full_ids):
        prefix_len = max(0, min(len(prompt_ids), len(full_ids) - 1))

    labels = full_inputs["input_ids"].clone()
    labels[:, :prefix_len] = -100
    token_count = int((labels != -100).sum().item())
    if token_count <= 0:
        return float("-inf"), float("inf"), 0

    full_inputs = to_device(full_inputs, device)
    labels = labels.to(device)
    outputs = model(**full_inputs, labels=labels)
    mean_nll = float(outputs.loss.item())
    sum_nll = mean_nll * token_count
    return -mean_nll, sum_nll, token_count


def score_one_view(
    model: Any,
    processor: AutoProcessor,
    clip_path: Path,
    num_frames: int,
    device: str,
    candidate_format: str,
    score_reduction: str,
) -> dict[str, Any]:
    video_entry = make_video_entry(clip_path, num_frames)
    prompt = build_scoring_prompt(num_frames)
    scores: dict[str, float] = {}
    nlls: dict[str, float] = {}
    token_counts: dict[str, int] = {}
    t0 = time.perf_counter()
    for label in OFFENCE_SEVERITY_CLASSES:
        mean_score, sum_nll, tok_count = score_candidate(
            model, processor, video_entry, prompt, candidate_text(label, candidate_format), device
        )
        token_counts[label] = tok_count
        nlls[label] = sum_nll
        scores[label] = mean_score if score_reduction == "mean" else -sum_nll
    elapsed = time.perf_counter() - t0
    pred = max(scores, key=scores.get)
    return {
        "scores": scores,
        "nlls": nlls,
        "token_counts": token_counts,
        "prediction": {"offence_severity": pred},
        "elapsed_sec": elapsed,
    }


def compute_priors(data_root: Path, split: str) -> dict[str, float]:
    dataset = ActionViewDataset(data_root, split, limit=0, max_views=0)
    counts = Counter(action["gold_offence_severity"] for action in dataset.actions)
    total = sum(counts.values())
    # Add a tiny smoothing count so Red never becomes undefined.
    smooth = 1.0
    denom = total + smooth * len(OFFENCE_SEVERITY_CLASSES)
    return {label: (counts.get(label, 0) + smooth) / denom for label in OFFENCE_SEVERITY_CLASSES}


def adjust_scores(scores: dict[str, float], priors: dict[str, float], alpha: float) -> dict[str, float]:
    return {label: score - alpha * math.log(max(priors[label], 1e-12)) for label, score in scores.items()}


def fuse_score_rows(
    view_rows: list[dict[str, Any]],
    method: str,
    priors: dict[str, float],
    alpha: float,
    clip1_weight: float,
    other_view_weight: float,
) -> tuple[dict[str, str], dict[str, float], str]:
    valid = [v for v in view_rows if v.get("scores")]
    if not valid:
        return {}, {}, "no_valid_view"

    selected: list[tuple[dict[str, Any], float]] = []
    if method == "clip0":
        chosen = next((v for v in valid if int(v["clip_idx"]) == 0), valid[0])
        selected = [(chosen, 1.0)]
    elif method == "clip1":
        chosen = next((v for v in valid if int(v["clip_idx"]) == 1), valid[0])
        selected = [(chosen, 1.0)]
    elif method in {"score_mean", "score_sum", "weighted_clip1"}:
        for v in valid:
            idx = int(v["clip_idx"])
            weight = 1.0
            if method == "weighted_clip1":
                weight = clip1_weight if idx == 1 else (1.0 if idx == 0 else other_view_weight)
            selected.append((v, weight))
    else:
        raise ValueError(method)

    fused = {label: 0.0 for label in OFFENCE_SEVERITY_CLASSES}
    total_weight = 0.0
    for v, weight in selected:
        adjusted = adjust_scores(v["scores"], priors, alpha)
        for label in OFFENCE_SEVERITY_CLASSES:
            fused[label] += adjusted[label] * weight
        total_weight += weight
    if method in {"score_mean", "weighted_clip1", "clip0", "clip1"} and total_weight > 0:
        fused = {label: value / total_weight for label, value in fused.items()}
    label = max(fused, key=fused.get)
    detail = f"{method}:alpha={alpha}:views=" + ",".join(f"clip_{v['clip_idx']}x{w:g}" for v, w in selected)
    return {"offence_severity": label}, fused, detail


def rows_for_setting(
    base_rows: list[dict[str, Any]],
    method: str,
    priors: dict[str, float],
    alpha: float,
    clip1_weight: float,
    other_view_weight: float,
) -> list[dict[str, Any]]:
    rows = []
    for row in base_rows:
        pred, fused_scores, detail = fuse_score_rows(
            row.get("view_scores") or [], method, priors, alpha, clip1_weight, other_view_weight
        )
        new_row = {
            "action_id": row["action_id"],
            "n_views": row["n_views"],
            "fusion_method": method,
            "prior_alpha": alpha,
            "fusion_detail": detail,
            "fused_scores": fused_scores,
            "view_scores": row.get("view_scores") or [],
            "prediction": pred,
            "validation_errors": [] if pred else ["no valid fused prediction"],
            "gold": row["gold"],
        }
        rows.append(new_row)
    return rows


def view_level_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_idx: dict[int, list[dict[str, Any]]] = {}
    total = 0
    for row in rows:
        gold = row["gold"]["offence_severity"]
        for view in row.get("view_scores") or []:
            total += 1
            idx = int(view["clip_idx"])
            by_idx.setdefault(idx, []).append({
                "prediction": view.get("prediction") or {},
                "gold": {"offence_severity": gold},
                "validation_errors": [],
            })
    return {
        "total_views": total,
        "view_parse_errors": 0,
        "by_clip_idx": {str(idx): compute_metrics(vrows) for idx, vrows in sorted(by_idx.items())},
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    seen = set()
    out = []
    for row in rows:
        aid = str(row.get("action_id", ""))
        if aid and aid not in seen:
            out.append(row)
            seen.add(aid)
    return out


def save_base_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(make_jsonable(rows), indent=2, ensure_ascii=False))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(make_jsonable(row), ensure_ascii=False) + "\n")
        f.flush()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    methods = parse_csv_methods(args.fusion_methods)
    alphas = parse_csv_floats(args.prior_alphas)
    dataset = ActionViewDataset(data_root, args.split, limit=args.limit, max_views=args.max_views)
    priors = compute_priors(data_root, args.prior_split)

    adapter_tag = "ft" if args.adapter_path else "zs"
    views_tag = f"views{args.max_views}" if args.max_views else "viewsall"
    prefix = args.output_prefix or f"likelihood_{adapter_tag}_{views_tag}_{args.split.lower()}{len(dataset.actions)}"
    base_rows_path = out_dir / f"{prefix}_score_rows.json"
    base_jsonl_path = out_dir / f"{prefix}_score_rows.jsonl"
    summary_path = out_dir / f"{prefix}_metrics_grid.json"
    best_rows_path = out_dir / f"{prefix}_best_rows.json"
    best_metrics_path = out_dir / f"{prefix}_best_metrics.json"
    best_predictions_path = out_dir / f"{prefix}_best_predictions.json"

    base_rows: list[dict[str, Any]] = []
    if args.resume:
        base_rows = load_jsonl(base_jsonl_path)
        if not base_rows and base_rows_path.exists():
            base_rows = json.loads(base_rows_path.read_text())
        log(f"resume loaded_rows={len(base_rows)}")
    else:
        for path in [base_rows_path, base_jsonl_path, summary_path, best_rows_path, best_metrics_path, best_predictions_path]:
            if path.exists():
                path.unlink()

    done = {str(r.get("action_id")) for r in base_rows}
    log(f"split={args.split} actions={len(dataset.actions)} max_views={args.max_views or 'all'}")
    log(f"candidate_format={args.candidate_format} score_reduction={args.score_reduction}")
    log(f"priors({args.prior_split})={priors}")

    if len(base_rows) < len(dataset.actions):
        model, processor = load_model_and_processor(args)
        for i, action in enumerate(dataset.actions, start=1):
            if str(action["action_id"]) in done:
                continue
            should_log = i == 1 or i == len(dataset.actions) or (args.log_every > 0 and i % args.log_every == 0)
            if should_log:
                log(f"[{i}/{len(dataset.actions)}] action_{action['action_id']} views={len(action['clips'])}")
            view_scores = []
            for clip in action["clips"]:
                try:
                    scored = score_one_view(
                        model, processor, clip["clip_path"], args.num_frames, args.device,
                        args.candidate_format, args.score_reduction,
                    )
                    view_scores.append({
                        "clip_idx": clip["clip_idx"],
                        "clip_path": str(clip["clip_path"]),
                        "camera_type": clip.get("camera_type", ""),
                        "replay_speed": clip.get("replay_speed", ""),
                        **scored,
                    })
                except Exception as exc:
                    log(f"  view error clip_{clip['clip_idx']}: {type(exc).__name__}: {exc}")
            row = {
                "action_id": action["action_id"],
                "n_views": len(view_scores),
                "view_scores": view_scores,
                "gold": {
                    "action_class": action["gold_action_class"],
                    "offence_severity": action["gold_offence_severity"],
                    "Offence": action["gold_offence"],
                    "Severity": action["gold_severity"],
                },
            }
            base_rows.append(row)
            append_jsonl(base_jsonl_path, row)
            if args.save_every > 0 and len(base_rows) % args.save_every == 0:
                save_base_rows(base_rows_path, base_rows)
                log(f"  partial saved: {len(base_rows)}/{len(dataset.actions)} {gpu_mem(args.device)}")
            if should_log:
                tmp_rows = rows_for_setting(base_rows, methods[0], priors, alphas[0], args.clip1_weight, args.other_view_weight)
                tmp_metrics = compute_metrics(tmp_rows)
                pred = tmp_rows[-1]["prediction"]
                log(f"  pred={pred} gold={row['gold']} acc={tmp_metrics['accuracy_offence_severity']:.1f}% {gpu_mem(args.device)}")

    save_base_rows(base_rows_path, base_rows)

    grid = []
    best = None
    best_rows = None
    for method in methods:
        for alpha in alphas:
            rows = rows_for_setting(base_rows, method, priors, alpha, args.clip1_weight, args.other_view_weight)
            metrics = compute_metrics(rows)
            metrics["view_level"] = view_level_summary(rows)
            item = {"fusion_method": method, "prior_alpha": alpha, "metrics": metrics}
            grid.append(item)
            score = metrics["balanced_accuracy_offence_severity_seen_classes"]
            if best is None or score > best["metrics"]["balanced_accuracy_offence_severity_seen_classes"]:
                best = item
                best_rows = rows

    summary = {
        "model_id": args.model_id,
        "adapter_path": args.adapter_path,
        "split": args.split,
        "num_actions": len(base_rows),
        "num_frames": args.num_frames,
        "candidate_format": args.candidate_format,
        "score_reduction": args.score_reduction,
        "priors": priors,
        "grid": grid,
        "best": best,
    }
    summary_path.write_text(json.dumps(make_jsonable(summary), indent=2, ensure_ascii=False))
    assert best is not None and best_rows is not None
    best_rows_path.write_text(json.dumps(make_jsonable(best_rows), indent=2, ensure_ascii=False))
    best_metrics_path.write_text(json.dumps(make_jsonable(best["metrics"]), indent=2, ensure_ascii=False))
    preds = make_prediction_json(dataset.annotations, best_rows)
    best_predictions_path.write_text(json.dumps(make_jsonable(preds), indent=2, ensure_ascii=False))

    log("Grid summary:")
    for item in sorted(grid, key=lambda x: x["metrics"]["balanced_accuracy_offence_severity_seen_classes"], reverse=True):
        m = item["metrics"]
        print(
            f"{item['fusion_method']} alpha={item['prior_alpha']}: "
            f"acc={m['accuracy_offence_severity']:.2f} "
            f"ba={m['balanced_accuracy_offence_severity_seen_classes']:.2f}",
            flush=True,
        )
    log("Best:")
    print(json.dumps(make_jsonable(best), indent=2, ensure_ascii=False), flush=True)
    log("Written:")
    print(base_rows_path, flush=True)
    print(summary_path, flush=True)
    print(best_rows_path, flush=True)
    print(best_metrics_path, flush=True)
    print(best_predictions_path, flush=True)


if __name__ == "__main__":
    main()
