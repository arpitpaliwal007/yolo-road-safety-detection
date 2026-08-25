"""Train and compare multiple YOLO checkpoints on the same dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"])
    parser.add_argument("--data", default="data/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="runs/model_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.data).is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {args.data}")

    rows: list[dict[str, object]] = []
    for model_spec in args.models:
        run_name = f"compare_{Path(model_spec).stem}"
        model = YOLO(model_spec)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            name=run_name,
            seed=42,
            exist_ok=True,
        )
        best_path = Path(model.trainer.best)
        metrics = YOLO(best_path).val(data=args.data, imgsz=args.imgsz, device=args.device, plots=False)
        rows.append(
            {
                "model": Path(model_spec).stem,
                "parameters": int(sum(parameter.numel() for parameter in model.model.parameters())),
                "checkpoint_mb": round(best_path.stat().st_size / (1024 * 1024), 2),
                "precision": round(float(metrics.box.mp), 5),
                "recall": round(float(metrics.box.mr), 5),
                "map50": round(float(metrics.box.map50), 5),
                "map50_95": round(float(metrics.box.map), 5),
                "inference_ms": round(float(metrics.speed.get("inference", 0.0)), 3),
                "checkpoint": str(best_path),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved model comparison to {output}")


if __name__ == "__main__":
    main()
