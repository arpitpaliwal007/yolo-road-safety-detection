"""Streamlit interface for road-safety detection and traffic monitoring."""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

from src.traffic_monitor import monitor_video


st.set_page_config(page_title="YOLO Traffic Monitor", page_icon="🚦", layout="wide")
st.title("YOLO Road-Safety Traffic Monitor")
st.caption("Detect objects in images or track, count, and measure traffic density in videos.")

with st.sidebar:
    st.header("Configuration")
    model_path = st.text_input("Model checkpoint", "runs/detect/colab_road_safety/weights/best.pt")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    mode = st.radio("Input type", ["Image", "Video"])
    if mode == "Video":
        tracker = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"])
        line_position = st.slider("Counting-line position", 0.1, 0.9, 0.6, 0.05)

extensions = ["jpg", "jpeg", "png", "webp"] if mode == "Image" else ["mp4", "mov", "avi", "mkv"]
uploaded = st.file_uploader(f"Upload a {mode.lower()}", type=extensions)

if st.button("Run analysis", type="primary", disabled=uploaded is None):
    if not Path(model_path).is_file():
        st.error(f"Model checkpoint not found: {model_path}")
        st.stop()

    suffix = Path(uploaded.name).suffix
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        input_path.write_bytes(uploaded.getvalue())

        if mode == "Image":
            result = YOLO(model_path).predict(str(input_path), conf=confidence, verbose=False)[0]
            annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
            classes = result.boxes.cls.int().cpu().tolist() if result.boxes is not None else []
            counts = Counter(str(result.names[class_id]) for class_id in classes)
            left, right = st.columns([2, 1])
            left.image(annotated, caption="Detected road users and vehicles", use_container_width=True)
            right.subheader("Visible objects")
            if counts:
                right.table({"Class": list(counts), "Count": list(counts.values())})
            else:
                right.info("No objects exceeded the selected confidence threshold.")
        else:
            output_path = Path(temp_dir) / "traffic_monitor.mp4"
            progress = st.progress(0.0, text="Processing video...")

            def update_progress(frame: int, total: int) -> None:
                if total > 0:
                    progress.progress(min(frame / total, 1.0), text=f"Processing frame {frame}/{total}")

            stats = monitor_video(
                model_path,
                str(input_path),
                output_path,
                confidence=confidence,
                tracker=tracker,
                line_position=line_position,
                progress_callback=update_progress,
            )
            progress.empty()
            st.video(output_path.read_bytes())
            metric_columns = st.columns(4)
            metric_columns[0].metric("Frames", stats["frames"])
            metric_columns[1].metric("Average occupancy", f"{stats['average_occupancy']:.1%}")
            metric_columns[2].metric("Peak occupancy", f"{stats['peak_occupancy']:.1%}")
            metric_columns[3].metric("Unique tracks", sum(stats["unique_tracks"].values()))
            st.subheader("Class-wise statistics")
            st.json(
                {
                    "unique_tracks": stats["unique_tracks"],
                    "crossings_up": stats["crossings_up"],
                    "crossings_down": stats["crossings_down"],
                }
            )
            st.download_button(
                "Download annotated video",
                data=output_path.read_bytes(),
                file_name="traffic_monitor.mp4",
                mime="video/mp4",
            )
            event_path = Path(stats["events_csv"])
            st.download_button(
                "Download crossing events (CSV)",
                data=event_path.read_bytes(),
                file_name="crossing_events.csv",
                mime="text/csv",
            )
