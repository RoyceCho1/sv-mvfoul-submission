#!/usr/bin/env python
"""Run Cosmos Reason2 zero-shot inference on one SoccerNet-MVFoul clip."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from zero_shot_eval import (
    OFFENCE_SEVERITY_CLASSES,
    extract_json,
    normalize_prediction_keys,
    validate_prediction,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="nvidia/Cosmos-Reason2-8B")
    parser.add_argument(
        "--clip",
        default="data/SoccerNet/mvfouls/Train/action_0/clip_0.mp4",
        help="Path to one SoccerNet-MVFoul MP4 clip.",
    )
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--annotation-root", default="data/SoccerNet/mvfouls")
    return parser.parse_args()


def build_prompt() -> str:
    return (
        "You are a soccer referee assistant. Watch the video clip and assess the offence severity.\n"
        "Do not predict action_class. Use only these offence_severity labels:\n"
        f"{', '.join(OFFENCE_SEVERITY_CLASSES)}.\n\n"
        "Return only valid JSON with exactly this key:\n"
        '{"offence_severity": "..."}'
    )


def load_gold(clip_path: Path, annotation_root: Path) -> dict[str, str] | None:
    parts = clip_path.resolve().parts
    split = action_id = None
    for i, part in enumerate(parts):
        if part in {"Train", "Valid", "Test"} and i + 1 < len(parts):
            split = part
            maybe_action = parts[i + 1]
            if maybe_action.startswith("action_"):
                action_id = maybe_action.removeprefix("action_")
            break
    if split is None or action_id is None:
        return None

    ann_path = annotation_root / split / "annotations.json"
    if not ann_path.exists():
        return None

    annotations = json.loads(ann_path.read_text())
    record = annotations.get("Actions", {}).get(action_id)
    if record is None:
        return None

    offence = record.get("Offence", "")
    severity = record.get("Severity", "")
    if offence in {"No offence", "No Offence"}:
        offence_severity = "No offence"
    elif offence == "Offence" and severity == "1.0":
        offence_severity = "Offence + No card"
    elif offence == "Offence" and severity == "3.0":
        offence_severity = "Offence + Yellow card"
    elif offence == "Offence" and severity == "5.0":
        offence_severity = "Offence + Red card"
    else:
        offence_severity = f"{offence} + severity={severity}"

    return {
        "split": split,
        "action_id": action_id,
        "action_class": record.get("Action class", ""),
        "offence_severity": offence_severity,
    }


def main() -> None:
    args = parse_args()
    clip_path = Path(args.clip).resolve()
    if not clip_path.exists():
        raise FileNotFoundError(clip_path)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu only for debugging.")

    print(f"model_id: {args.model_id}")
    print(f"clip: {clip_path}")
    print(f"fps: {args.fps}")

    processor = AutoProcessor.from_pretrained(args.model_id)

    quantization_config = None
    torch_dtype = torch.bfloat16
    device_map: str | dict[str, str] = {"": args.device}
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map=device_map,
    )
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(clip_path),
                    "fps": args.fps,
                },
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
    inputs = inputs.to(args.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    raw_output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("\nraw_output:")
    print(raw_output)

    prediction = normalize_prediction_keys(extract_json(raw_output))
    errors = validate_prediction(prediction)

    print("\nprediction:")
    print(json.dumps(prediction, indent=2, ensure_ascii=False))
    if errors:
        print("\nvalidation_errors:")
        print(json.dumps(errors, indent=2, ensure_ascii=False))

    gold = load_gold(clip_path, Path(args.annotation_root))
    if gold is not None:
        print("\ngold:")
        print(json.dumps(gold, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
