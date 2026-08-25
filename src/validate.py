"""Evaluate a trained Ultralytics YOLO model on the validation split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--data", default="data/dataset.yaml", help="Dataset YAML path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="runs/evaluation", help="Directory for JSON and CSV reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.model, args.data):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    metrics = YOLO(args.model).val(data=args.data, imgsz=args.imgsz, device=args.device, plots=True)
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    names = metrics.names
    class_positions = {int(class_id): index for index, class_id in enumerate(metrics.box.ap_class_index)}
    precision = metrics.box.p.tolist()
    recall = metrics.box.r.tolist()
    ap50 = metrics.box.ap50.tolist()
    maps = metrics.box.maps.tolist()
    rows = []
    for class_id, class_name in names.items():
        position = class_positions.get(int(class_id))
        rows.append(
            {
                "class_id": class_id,
                "class": class_name,
                "precision": round(float(precision[position]), 6) if position is not None else 0.0,
                "recall": round(float(recall[position]), 6) if position is not None else 0.0,
                "map50": round(float(ap50[position]), 6) if position is not None else 0.0,
                "map50_95": round(float(maps[int(class_id)]), 6),
            }
        )
    with (output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": args.model,
        "data": args.data,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "speed_ms": metrics.speed,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved evaluation reports to {output}")


if __name__ == "__main__":
    main()
