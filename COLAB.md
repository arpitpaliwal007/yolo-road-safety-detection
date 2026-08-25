# Google Colab workflow

Select a T4 GPU under **Runtime → Change runtime type**, then run:

```python
!git clone https://github.com/arpitpaliwal007/yolo-road-safety-detection.git
%cd yolo-road-safety-detection
!pip install -q -r requirements.txt
```

Verify CUDA and prepare the demonstration dataset:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
!python src/prepare_dataset.py
!python src/dataset_report.py
```

Train one model:

```python
!python src/train.py --data data/dataset.yaml --model yolo11s.pt --epochs 100 --imgsz 640 --batch 16 --device 0 --name road_safety_yolo11s
```

Validate it and write per-class reports:

```python
!python src/validate.py --model runs/detect/road_safety_yolo11s/weights/best.pt --data data/dataset.yaml --device 0
```

Compare multiple model sizes (reduce the batch size if a larger checkpoint runs out of GPU memory):

```python
!python src/compare_models.py --models yolo11n.pt yolo11s.pt yolo11m.pt --data data/dataset.yaml --epochs 50 --device 0
```

The recorded comparison selected YOLO11m as the strongest checkpoint. Process an uploaded traffic video with it:

```python
from google.colab import files
uploaded = files.upload()
video_name = next(iter(uploaded))

!python src/traffic_monitor.py \
  --model runs/detect/compare_yolo11m/weights/best.pt \
  --source "$video_name" \
  --output runs/traffic_monitor/annotated.mp4 \
  --device 0
```

For a reproducible demonstration without a local upload, download the public sample clip from the [TSEC traffic-scenario repository](https://github.com/sduhaoph/TSEC-Dataset) and run the same monitor:

```python
sample_url = "https://raw.githubusercontent.com/sduhaoph/TSEC-Dataset/main/video_caption.mp4"
!wget -q -O traffic_sample.mp4 "$sample_url"

!python src/traffic_monitor.py \
  --model runs/detect/compare_yolo11m/weights/best.pt \
  --source traffic_sample.mp4 \
  --output runs/traffic_monitor/annotated.mp4 \
  --device 0
```

Convert the OpenCV output to browser-compatible H.264 and display it in Colab:

```python
!ffmpeg -y -i runs/traffic_monitor/annotated.mp4 \
  -vcodec libx264 -pix_fmt yuv420p -an \
  runs/traffic_monitor/annotated_h264.mp4

from IPython.display import Video, display
display(Video("runs/traffic_monitor/annotated_h264.mp4", embed=True, width=900))
```

Colab storage is temporary. Run this final cell after training, monitoring, and H.264 conversion so all artifacts are copied to Google Drive before disconnecting:

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p "/content/drive/MyDrive/yolo-road-safety/runs"
!cp -r runs/. "/content/drive/MyDrive/yolo-road-safety/runs/"
```

Expected video paths in Drive:

```text
MyDrive/yolo-road-safety/runs/traffic_monitor/annotated.mp4
MyDrive/yolo-road-safety/runs/traffic_monitor/annotated_h264.mp4
```

If the runtime disconnects before this copy finishes, Colab deletes the runtime files. Reconnect, rerun the setup/training/monitoring cells, and execute the Drive cell again.
