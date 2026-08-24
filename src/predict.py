"""Run YOLO inference on an image, video, directory, URL, or webcam."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_source(value: str) -> str | int:
    """Convert numeric webcam sources while preserving file paths and URLs."""
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--source", required=True, help="Image/video path, URL, folder, or webcam index")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--show", action="store_true", help="Display predictions during inference")
    parser.add_argument("--project", default=None, help="Optional output directory")
    parser.add_argument("--name", default="predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.model).is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    YOLO(args.model).predict(
        source=parse_source(args.source),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        show=args.show,
        save=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
