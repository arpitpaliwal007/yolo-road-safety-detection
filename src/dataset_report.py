"""Report image, label, and class-balance statistics for a YOLO dataset."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import yaml


def load_names(config: dict) -> dict[int, str]:
    names = config["names"]
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(key): str(value) for key, value in names.items()}


def scan_split(root: Path, relative_images: str, names: dict[int, str]) -> tuple[int, int, Counter[str]]:
    image_dir = root / relative_images
    relative_parts = Path(relative_images).parts
    if not relative_parts or relative_parts[0] != "images":
        raise ValueError(f"Expected an images/... path in dataset YAML, got: {relative_images}")
    label_dir = root / "labels" / Path(*relative_parts[1:])
    image_files = [path for path in image_dir.glob("*.*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    labelled_images = 0
    instances: Counter[str] = Counter()
    for image in image_files:
        label = label_dir / f"{image.stem}.txt"
        lines = label.read_text().splitlines() if label.exists() else []
        if lines:
            labelled_images += 1
        for line in lines:
            values = line.split()
            if values:
                class_id = int(values[0])
                instances[names.get(class_id, f"unknown_{class_id}")] += 1
    return len(image_files), labelled_images, instances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dataset.yaml")
    parser.add_argument("--output", default="runs/dataset_report.csv")
    args = parser.parse_args()

    config_path = Path(args.data)
    config = yaml.safe_load(config_path.read_text())
    root = Path(config.get("path", config_path.parent))
    names = load_names(config)
    rows: list[dict[str, object]] = []
    for split in ("train", "val"):
        total, labelled, instances = scan_split(root, config[split], names)
        print(f"{split}: {total} images ({labelled} with labels), {sum(instances.values())} objects")
        for name in names.values():
            rows.append(
                {
                    "split": split,
                    "class": name,
                    "instances": instances[name],
                    "total_images": total,
                    "labelled_images": labelled,
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved dataset report to {output}")


if __name__ == "__main__":
    main()
