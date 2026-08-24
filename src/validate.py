"""Evaluate a trained Ultralytics YOLO model on the validation split."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--data", default="data/dataset.yaml", help="Dataset YAML path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.model, args.data):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    metrics = YOLO(args.model).val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
