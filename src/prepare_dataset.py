"""Download COCO128 and create a compact road-safety YOLO dataset.

The source dataset uses COCO class IDs. This script preserves person, bicycle,
car, motorcycle, bus, and truck labels and maps them to IDs 0–5.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
ROOT = Path("data")
ARCHIVE = ROOT / "raw" / "coco128.zip"
EXTRACTED = ROOT / "raw" / "coco128"
CLASS_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 7: 5}


def download_coco128() -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading COCO128 from {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, ARCHIVE)
    if not EXTRACTED.exists():
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extractall(ARCHIVE.parent)


def convert_split(split: str, image_paths: list[Path]) -> int:
    source_labels = EXTRACTED / "labels" / "train2017"
    target_images = ROOT / "images" / split
    target_labels = ROOT / "labels" / split
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    kept_images = 0
    for image_path in image_paths:
        source_label = source_labels / f"{image_path.stem}.txt"
        lines: list[str] = []
        if source_label.exists():
            for line in source_label.read_text().splitlines():
                values = line.split()
                if values and int(values[0]) in CLASS_MAP:
                    values[0] = str(CLASS_MAP[int(values[0])])
                    lines.append(" ".join(values))
        # Preserve empty-label images as negative examples for a realistic detector.
        shutil.copy2(image_path, target_images / image_path.name)
        (target_labels / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if lines:
            kept_images += 1
    return kept_images


def main() -> None:
    download_coco128()
    # COCO128 ships as one split. Use a deterministic 80/20 split for this demo.
    source_images = sorted((EXTRACTED / "images" / "train2017").glob("*.jpg"))
    for directory in (ROOT / "images" / "train", ROOT / "images" / "val", ROOT / "labels" / "train", ROOT / "labels" / "val"):
        shutil.rmtree(directory, ignore_errors=True)
    train_images = [image for index, image in enumerate(source_images) if index % 5]
    val_images = [image for index, image in enumerate(source_images) if not index % 5]
    counts = {
        "train": convert_split("train", train_images),
        "val": convert_split("val", val_images),
    }
    print(f"Prepared road-safety labels in: {counts['train']} train / {counts['val']} val images")


if __name__ == "__main__":
    main()
