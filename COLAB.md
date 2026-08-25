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

Process an uploaded traffic video:

```python
from google.colab import files
uploaded = files.upload()
video_name = next(iter(uploaded))

!python src/traffic_monitor.py \
  --model runs/detect/road_safety_yolo11s/weights/best.pt \
  --source "$video_name" \
  --output runs/traffic_monitor/annotated.mp4 \
  --device 0
```

Colab storage is temporary. Copy trained weights and reports to Google Drive before disconnecting:

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p "/content/drive/MyDrive/yolo-road-safety"
!cp -r runs "/content/drive/MyDrive/yolo-road-safety/"
```
