#!/usr/bin/env python
"""Run Cosmos Reason2 zero-shot evaluation on SoccerNet-MVFoul clip_0 videos."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration


ACTION_CLASSES = [
    "Tackling",
    "Standing tackling",
    "High leg",
    "Holding",
    "Pushing",
    "Elbowing",
    "Challenge",
    "Dive",
]

ACTION_TO_INDEX = {label: i for i, label in enumerate(ACTION_CLASSES)}

OFFENCE_SEVERITY_CLASSES = [
    "No offence",
    "Offence + No card",
    "Offence + Yellow card",
    "Offence + Red card",
]

OFFENCE_SEVERITY_TO_INDEX = {label: i for i, label in enumerate(OFFENCE_SEVERITY_CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="nvidia/Cosmos-Reason2-8B")
    parser.add_argument("--data-root", default="data/SoccerNet/mvfouls")
    parser.add_argument("--split", default="Valid", choices=["Train", "Valid", "Test"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--out-dir", default="outputs/zero_shot")
    parser.add_argument("--output-prefix", default=None, help="Output filename prefix. Defaults to zero_shot_<split><num_samples>.")
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def gpu_memory_summary(device: str) -> str:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return "gpu_mem=n/a"
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    allocated_gb = torch.cuda.memory_allocated(device_index) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(device_index) / 1024**3
    max_allocated_gb = torch.cuda.max_memory_allocated(device_index) / 1024**3
    return (
        f"gpu_mem allocated={allocated_gb:.2f}GB "
        f"reserved={reserved_gb:.2f}GB max_allocated={max_allocated_gb:.2f}GB"
    )



def build_prompt() -> str:
    return (
        "You are a soccer referee assistant. Watch the video clip and assess the offence severity.\n"
        "Use only these offence_severity labels:\n"
        f"{', '.join(OFFENCE_SEVERITY_CLASSES)}.\n\n"
        "Return only valid JSON with exactly this key:\n"
        '{"offence_severity": "..."}'
    )


def official_target(record: dict[str, Any]) -> tuple[str, str, str, str] | None:
    action_class = record.get("Action class", "")
    offence_class = record.get("Offence", "")
    severity_class = record.get("Severity", "")

    if action_class == "" or action_class == "Dont know":
        return None
    if (offence_class == "" or offence_class == "Between") and action_class != "Dive":
        return None
    if (
        (severity_class == "" or severity_class == "2.0" or severity_class == "4.0")
        and action_class != "Dive"
        and offence_class not in {"No offence", "No Offence"}
    ):
        return None

    if offence_class == "" or offence_class == "Between":
        offence_class = "Offence"
    if severity_class == "" or severity_class == "2.0" or severity_class == "4.0":
        severity_class = "1.0"
    if offence_class in {"No Offence", "No offence"}:
        offence_class = "No offence"
        severity_class = ""

    if offence_class == "No offence":
        offence_severity = "No offence"
    elif offence_class == "Offence" and severity_class == "1.0":
        offence_severity = "Offence + No card"
    elif offence_class == "Offence" and severity_class == "3.0":
        offence_severity = "Offence + Yellow card"
    elif offence_class == "Offence" and severity_class == "5.0":
        offence_severity = "Offence + Red card"
    else:
        return None

    if action_class == "Dive":
        severity_class = "1.0"

    return action_class, offence_severity, offence_class, severity_class


def prediction_to_offence_severity(prediction: dict[str, Any]) -> tuple[str, str]:
    label = prediction.get("offence_severity")
    if label == "No offence":
        return "No offence", ""
    if label == "Offence + No card":
        return "Offence", "1.0"
    if label == "Offence + Yellow card":
        return "Offence", "3.0"
    if label == "Offence + Red card":
        return "Offence", "5.0"
    return "", ""


def load_samples(data_root: Path, split: str, limit: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    annotation_path = data_root / split / "annotations.json"
    annotations = json.loads(annotation_path.read_text())
    samples = []
    for action_id, record in annotations["Actions"].items():
        target = official_target(record)
        if target is None:
            continue
        clip_path = data_root / split / f"action_{action_id}" / "clip_0.mp4"
        if not clip_path.exists():
            continue
        action_class, offence_severity, offence_class, severity_class = target
        samples.append(
            {
                "action_id": action_id,
                "clip_path": clip_path,
                "gold_action_class": action_class,
                "gold_offence_severity": offence_severity,
                "gold_offence": offence_class,
                "gold_severity": severity_class,
            }
        )
        if limit and len(samples) >= limit:
            break
    return annotations, samples


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        recovered = recover_offence_severity_from_text(cleaned)
        if recovered:
            return recovered
        raise ValueError(f"No JSON object found in model output: {text!r}")
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Single-quote JSON fallback (model sometimes uses Python dict syntax)
    try:
        import ast
        val = ast.literal_eval(candidate)
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    recovered = recover_offence_severity_from_text(candidate)
    if recovered:
        return recovered
    raise ValueError(f"Could not parse JSON from: {candidate!r}")


def _fuzzy_match_label(value: str, valid_labels: list[str]) -> str:
    """Map a model-generated label string to the nearest valid label.

    Handles common variations:
      - Offense vs Offence (US/UK spelling)
      - Standing tackle vs Standing tackling
      - case / spacing / punctuation differences
    Returns the original value unchanged if no close match found.
    """
    import difflib
    if not isinstance(value, str) or not value.strip():
        return value
    v = value.strip()
    # Exact match (fast path)
    if v in valid_labels:
        return v
    # Known aliases (model generates these variants consistently)
    _ALIASES: dict[str, str] = {
        "Diving": "Dive",
        "Dived": "Dive",
        "Tacking": "Tackling",
        "Standing tackle": "Standing tackling",
        "Standing Tackle": "Standing tackling",
        "Standingtacking": "Standing tackling",
        "High Leg": "High leg",
        "Handball": "Challenge",
    }
    if v in _ALIASES and _ALIASES[v] in valid_labels:
        return _ALIASES[v]
    # Normalize for comparison: lowercase, collapse spaces, unify Offense→Offence
    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"\s+", " ", s)
        s = s.replace("offense", "offence").replace("offenc e", "offence")
        s = s.replace(" + ", "+").replace("+ ", "+").replace(" +", "+")
        s = re.sub(r"[^a-z0-9+]", "", s)
        return s
    v_norm = _norm(v)
    for label in valid_labels:
        if _norm(label) == v_norm:
            return label
    # Fuzzy fallback (difflib)
    matches = difflib.get_close_matches(v_norm, [_norm(l) for l in valid_labels], n=1, cutoff=0.7)
    if matches:
        idx = [_norm(l) for l in valid_labels].index(matches[0])
        return valid_labels[idx]
    return v


def recover_offence_severity_from_text(text: str) -> dict[str, str] | None:
    """Recover the severity label from malformed JSON or free text.

    This fallback is limited to the four official offence_severity labels and
    runs only after stricter JSON/Python-literal parsing has failed.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    def _norm(s: str) -> str:
        s = s.lower()
        s = s.replace("offense", "offence")
        s = s.replace("yellowcard", "yellow card").replace("redcard", "red card")
        return re.sub(r"[^a-z0-9]+", "", s)

    norm = _norm(text)
    alias_to_label = {
        "nooffence": "No offence",
        "nooffense": "No offence",
        "notanoffence": "No offence",
        "nofoul": "No offence",
        "offencenocard": "Offence + No card",
        "offensenocard": "Offence + No card",
        "offencenocaution": "Offence + No card",
        "offensenocaution": "Offence + No card",
        "nocard": "Offence + No card",
        "noyellow": "Offence + No card",
        "offenceyellowcard": "Offence + Yellow card",
        "offenseyellowcard": "Offence + Yellow card",
        "offenceyellow": "Offence + Yellow card",
        "offenseyellow": "Offence + Yellow card",
        "yellowcard": "Offence + Yellow card",
        "yellow": "Offence + Yellow card",
        "offenceredcard": "Offence + Red card",
        "offenseredcard": "Offence + Red card",
        "offencered": "Offence + Red card",
        "offensered": "Offence + Red card",
        "redcard": "Offence + Red card",
        "red": "Offence + Red card",
    }
    for alias, label in sorted(alias_to_label.items(), key=lambda kv: len(kv[0]), reverse=True):
        if alias in norm:
            return {"offence_severity": label}
    return None


def normalize_prediction_keys(prediction: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(prediction)
    for key, value in prediction.items():
        canonical_key = re.sub(r"[^a-z]", "", key.lower())
        if canonical_key == "actionclass" and "action_class" not in normalized:
            normalized["action_class"] = value
        if canonical_key == "offenceseverity" and "offence_severity" not in normalized:
            normalized["offence_severity"] = value
        # Also catch "offenseseverity" (US spelling key)
        if canonical_key == "offenseseverity" and "offence_severity" not in normalized:
            normalized["offence_severity"] = value
        if (
            canonical_key in {"severity", "card", "cardseverity", "label", "finallabel", "decision"}
            and "offence_severity" not in normalized
        ):
            normalized["offence_severity"] = value

    # Normalize label values to valid class strings
    if "action_class" in normalized:
        normalized["action_class"] = _fuzzy_match_label(
            normalized["action_class"], ACTION_CLASSES
        )
    if "offence_severity" in normalized:
        normalized["offence_severity"] = _fuzzy_match_label(
            normalized["offence_severity"], OFFENCE_SEVERITY_CLASSES
        )
        if normalized["offence_severity"] not in OFFENCE_SEVERITY_CLASSES:
            recovered = recover_offence_severity_from_text(str(normalized["offence_severity"]))
            if recovered:
                normalized["offence_severity"] = recovered["offence_severity"]
    elif isinstance(prediction, dict):
        recovered = recover_offence_severity_from_text(json.dumps(prediction, ensure_ascii=False))
        if recovered:
            normalized["offence_severity"] = recovered["offence_severity"]
    return normalized


def validate_prediction(prediction: dict[str, Any]) -> list[str]:
    errors = []
    if prediction.get("offence_severity") not in OFFENCE_SEVERITY_CLASSES:
        errors.append(f"invalid offence_severity: {prediction.get('offence_severity')!r}")
    return errors


def run_one(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    clip_path: Path,
    fps: float,
    device: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any], list[str], float]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(clip_path), "fps": fps},
                {"type": "text", "text": build_prompt()},
            ],
        }
    ]
    start = time.perf_counter()
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
    generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    raw_output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    elapsed = time.perf_counter() - start

    try:
        prediction = normalize_prediction_keys(extract_json(raw_output))
        errors = validate_prediction(prediction)
    except Exception as exc:
        prediction = {}
        errors = [f"{type(exc).__name__}: {exc}"]
    return raw_output, prediction, errors, elapsed


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    off_gt = np.zeros(4, dtype=np.float64)
    off_ok = np.zeros(4, dtype=np.float64)

    for row in rows:
        gold_offsev = row["gold"]["offence_severity"]
        off_idx = OFFENCE_SEVERITY_TO_INDEX[gold_offsev]
        off_gt[off_idx] += 1

        pred = row.get("prediction") or {}
        if pred.get("offence_severity") == gold_offsev:
            off_ok[off_idx] += 1

    off_seen = off_gt > 0
    accuracy_offence_severity = float(off_ok.sum() / off_gt.sum()) if off_gt.sum() else 0.0
    balanced_accuracy_offence_severity = float(np.mean(off_ok[off_seen] / off_gt[off_seen])) if off_seen.any() else 0.0

    return {
        "num_samples": len(rows),
        "accuracy_offence_severity": accuracy_offence_severity * 100,
        "balanced_accuracy_offence_severity_seen_classes": balanced_accuracy_offence_severity * 100,
        "offence_severity_support": {
            label: int(off_gt[i]) for i, label in enumerate(OFFENCE_SEVERITY_CLASSES) if off_gt[i] > 0
        },
    }


def make_prediction_json(base_annotations: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build SoccerNet-style predictions for the severity-only task.

    The model no longer predicts Action class. Preserve the source annotation's
    Action class when available so generated prediction files do not contain
    misleading empty action labels. Metrics only use Offence/Severity.
    """
    predictions = {"Set": base_annotations.get("Set"), "Actions": {}}
    base_actions = base_annotations.get("Actions", {})
    for row in rows:
        prediction = row.get("prediction") or {}
        offence, severity = prediction_to_offence_severity(prediction)
        source_record = base_actions.get(str(row["action_id"]), {})
        predictions["Actions"][row["action_id"]] = {
            "Action class": source_record.get("Action class", ""),
            "Offence": offence,
            "Severity": severity,
        }
    return predictions


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu only for debugging.")

    base_annotations, samples = load_samples(data_root, args.split, args.limit)
    if not samples:
        raise RuntimeError(f"No official-eval samples found for split={args.split}")

    log(f"model_id: {args.model_id}")
    log(f"split: {args.split}")
    log(f"samples: {len(samples)}")
    log(f"fps: {args.fps}")
    log(f"device: {args.device}")
    log(f"4bit: {not args.no_4bit}")

    load_start = time.perf_counter()
    log("loading processor")
    processor = AutoProcessor.from_pretrained(args.model_id)
    log("processor loaded")
    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    log("loading model")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map={"": args.device},
    )
    model.eval()
    log(f"model loaded in {time.perf_counter() - load_start:.1f}s; {gpu_memory_summary(args.device)}")

    rows = []
    running_offsev_ok = 0
    for index, sample in enumerate(samples, start=1):
        should_log = args.log_every > 0 and (index == 1 or index == len(samples) or index % args.log_every == 0)
        if should_log:
            log(f"[{index}/{len(samples)}] start action_{sample['action_id']} {sample['clip_path']}")
        raw_output, prediction, errors, elapsed = run_one(
            model=model,
            processor=processor,
            clip_path=sample["clip_path"],
            fps=args.fps,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
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
        if prediction.get("offence_severity") == sample["gold_offence_severity"]:
            running_offsev_ok += 1
        status = "ok" if not errors else "parse/label_error"
        if should_log:
            offsev_acc = running_offsev_ok / index * 100
            log(
                f"[{index}/{len(samples)}] {status} "
                f"pred={prediction} gold={row['gold']} elapsed={elapsed:.1f}s "
                f"running_acc offence_severity={offsev_acc:.1f}% "
                f"{gpu_memory_summary(args.device)}"
            )

    prefix = args.output_prefix or f"zero_shot_{args.split.lower()}{len(samples)}"
    rows_path = out_dir / f"{prefix}_rows.json"
    predictions_path = out_dir / f"{prefix}_predictions.json"
    metrics_path = out_dir / f"{prefix}_metrics.json"

    metrics = compute_metrics(rows)
    predictions = make_prediction_json(base_annotations, rows)

    rows_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    predictions_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False))
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    log("metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    log("written:")
    print(rows_path, flush=True)
    print(predictions_path, flush=True)
    print(metrics_path, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
