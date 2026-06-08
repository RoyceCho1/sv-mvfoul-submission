#!/usr/bin/env python
"""Re-apply a late-fusion rule to an existing *_rows.json file.

This compares fusion rules fairly because it reuses the same per-view model
outputs instead of re-running generation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.eval_late_fusion_reason import fuse_predictions, make_jsonable, view_level_summary  # noqa: E402
from zero_shot.zero_shot_eval import compute_metrics, make_prediction_json  # noqa: E402

RULES = ["main_first", "clip1_first", "majority_vote", "majority_clip1_tiebreak", "conservative_card"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-fuse existing late-fusion rows with a different rule")
    p.add_argument("--rows", required=True, help="Existing *_rows.json with view_predictions")
    p.add_argument("--fusion-rule", choices=RULES, required=True)
    p.add_argument("--annotations", default=None, help="Optional annotations.json for predictions export")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--output-prefix", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows_path = Path(args.rows)
    rows = json.loads(rows_path.read_text())
    refused = []
    for row in rows:
        new_row = dict(row)
        pred, detail = fuse_predictions(row.get("view_predictions") or [], args.fusion_rule)
        new_row["fusion_rule"] = args.fusion_rule
        new_row["fusion_detail"] = detail
        new_row["prediction"] = pred
        new_row["validation_errors"] = [] if pred else ["no valid fused prediction"]
        refused.append(new_row)

    metrics = compute_metrics(refused)
    metrics["view_level"] = view_level_summary(refused)

    out_dir = Path(args.out_dir) if args.out_dir else rows_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"{rows_path.stem}_refused_{args.fusion_rule}"
    out_rows = out_dir / f"{prefix}_rows.json"
    out_metrics = out_dir / f"{prefix}_metrics.json"
    out_rows.write_text(json.dumps(make_jsonable(refused), indent=2, ensure_ascii=False))
    out_metrics.write_text(json.dumps(make_jsonable(metrics), indent=2, ensure_ascii=False))

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(out_rows)
    print(out_metrics)

    if args.annotations:
        annotations = json.loads(Path(args.annotations).read_text())
        predictions = make_prediction_json(annotations, refused)
        out_predictions = out_dir / f"{prefix}_predictions.json"
        out_predictions.write_text(json.dumps(make_jsonable(predictions), indent=2, ensure_ascii=False))
        print(out_predictions)


if __name__ == "__main__":
    main()
