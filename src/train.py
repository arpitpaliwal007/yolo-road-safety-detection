"""Fine-tune an Ultralytics YOLO detector on a custom dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dataset.yaml", help="Dataset YAML path")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained model or model YAML")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="CUDA device (for example 0) or cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default=None, help="Optional output directory")
    parser.add_argument("--name", default="custom_detector")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.data).is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {args.data}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
    )


if __name__ == "__main__":
    main()
