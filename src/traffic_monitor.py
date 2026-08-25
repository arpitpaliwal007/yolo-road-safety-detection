"""Track, count, and measure traffic density in a video or webcam stream."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class LineCrossingCounter:
    """Count tracked objects that cross a horizontal line."""

    line_y: int
    previous_y: dict[int, float] = field(default_factory=dict)
    counts: dict[str, Counter[str]] = field(
        default_factory=lambda: {"up": Counter(), "down": Counter()}
    )

    def update(self, track_id: int, class_name: str, center_y: float) -> str | None:
        previous = self.previous_y.get(track_id)
        self.previous_y[track_id] = center_y
        if previous is None:
            return None
        if previous < self.line_y <= center_y:
            self.counts["down"][class_name] += 1
            return "down"
        if previous > self.line_y >= center_y:
            self.counts["up"][class_name] += 1
            return "up"
        return None


def density_level(occupancy: float, medium_threshold: float, high_threshold: float) -> str:
    """Convert frame occupancy into a human-readable traffic level."""
    if occupancy >= high_threshold:
        return "HIGH"
    if occupancy >= medium_threshold:
        return "MODERATE"
    return "LOW"


def calculate_occupancy(boxes: list[tuple[int, int, int, int]], frame_shape: tuple[int, ...]) -> float:
    """Return the non-overlapping fraction of the frame covered by detections."""
    height, width = frame_shape[:2]
    if not boxes or height <= 0 or width <= 0:
        return 0.0
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        mask[y1:y2, x1:x2] = 1
    return float(mask.mean())


def draw_dashboard(
    frame: np.ndarray,
    current_counts: Counter[str],
    crossing_counter: LineCrossingCounter,
    occupancy: float,
    level: str,
) -> None:
    """Draw traffic statistics directly onto a video frame."""
    cv2.line(frame, (0, crossing_counter.line_y), (frame.shape[1], crossing_counter.line_y), (0, 255, 255), 2)
    panel_height = 100 + 22 * min(len(current_counts), 6)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (370, panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    color = {"LOW": (70, 210, 70), "MODERATE": (0, 190, 255), "HIGH": (50, 50, 255)}[level]
    cv2.putText(frame, f"Traffic: {level} ({occupancy:.1%})", (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    total_up = sum(crossing_counter.counts["up"].values())
    total_down = sum(crossing_counter.counts["down"].values())
    cv2.putText(frame, f"Crossings  Up: {total_up}  Down: {total_down}", (22, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
    cv2.putText(frame, "Visible objects:", (22, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    for index, (name, count) in enumerate(current_counts.most_common(6)):
        cv2.putText(frame, f"{name}: {count}", (36, 116 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1)


def monitor_video(
    model_path: str,
    source: str | int,
    output_path: str | Path,
    *,
    confidence: float = 0.25,
    device: str | None = None,
    tracker: str = "bytetrack.yaml",
    line_position: float = 0.60,
    medium_threshold: float = 0.08,
    high_threshold: float = 0.20,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Process a source and return aggregate monitoring statistics."""
    if not 0.0 < line_position < 1.0:
        raise ValueError("line_position must be between 0 and 1")
    if not 0.0 <= medium_threshold < high_threshold <= 1.0:
        raise ValueError("density thresholds must satisfy 0 <= medium < high <= 1")

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video source: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create output video: {output_path}")

    model = YOLO(model_path)
    crossing_counter = LineCrossingCounter(line_y=int(height * line_position))
    events: list[dict[str, object]] = []
    occupancy_sum = 0.0
    peak_occupancy = 0.0
    seen_tracks: dict[str, set[int]] = defaultdict(set)
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.track(
                frame,
                persist=True,
                tracker=tracker,
                conf=confidence,
                device=device,
                verbose=False,
            )[0]
            annotated = result.plot()
            current_counts: Counter[str] = Counter()
            density_boxes: list[tuple[int, int, int, int]] = []

            if result.boxes is not None:
                xyxy = result.boxes.xyxy.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()
                track_ids = (
                    result.boxes.id.int().cpu().tolist()
                    if result.boxes.id is not None
                    else [None] * len(xyxy)
                )
                for box, class_id, track_id in zip(xyxy, class_ids, track_ids):
                    class_name = str(result.names[class_id])
                    current_counts[class_name] += 1
                    density_boxes.append(tuple(box))
                    if track_id is None:
                        continue
                    seen_tracks[class_name].add(track_id)
                    center_y = (box[1] + box[3]) / 2
                    direction = crossing_counter.update(track_id, class_name, center_y)
                    if direction:
                        events.append(
                            {
                                "frame": frame_index,
                                "time_seconds": round(frame_index / fps, 3),
                                "track_id": track_id,
                                "class": class_name,
                                "direction": direction,
                            }
                        )

            occupancy = calculate_occupancy(density_boxes, frame.shape)
            occupancy_sum += occupancy
            peak_occupancy = max(peak_occupancy, occupancy)
            level = density_level(occupancy, medium_threshold, high_threshold)
            draw_dashboard(annotated, current_counts, crossing_counter, occupancy, level)
            writer.write(annotated)
            frame_index += 1
            if progress_callback:
                progress_callback(frame_index, total_frames)
    finally:
        capture.release()
        writer.release()

    event_path = output_path.with_suffix(".csv")
    with event_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=["frame", "time_seconds", "track_id", "class", "direction"])
        writer_csv.writeheader()
        writer_csv.writerows(events)

    return {
        "frames": frame_index,
        "average_occupancy": occupancy_sum / frame_index if frame_index else 0.0,
        "peak_occupancy": peak_occupancy,
        "unique_tracks": {name: len(ids) for name, ids in sorted(seen_tracks.items())},
        "crossings_up": dict(crossing_counter.counts["up"]),
        "crossings_down": dict(crossing_counter.counts["down"]),
        "events_csv": str(event_path),
        "output_video": str(output_path),
    }


def parse_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trained .pt checkpoint")
    parser.add_argument("--source", required=True, help="Video path or webcam index")
    parser.add_argument("--output", default="runs/traffic_monitor/annotated.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tracker", default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"])
    parser.add_argument("--line-position", type=float, default=0.60, help="Counting line as a fraction of frame height")
    parser.add_argument("--medium-density", type=float, default=0.08)
    parser.add_argument("--high-density", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.model).is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    stats = monitor_video(
        args.model,
        parse_source(args.source),
        args.output,
        confidence=args.conf,
        device=args.device,
        tracker=args.tracker,
        line_position=args.line_position,
        medium_threshold=args.medium_density,
        high_threshold=args.high_density,
    )
    print("Traffic monitoring complete")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
