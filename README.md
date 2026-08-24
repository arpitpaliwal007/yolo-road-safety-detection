# YOLO Custom Object Detection

A complete road-safety object-detection project using **Ultralytics YOLO**. It builds a compact, repeatable six-class dataset from COCO128, then trains, evaluates, and runs inference with a fine-tuned detector.

## Highlights

- Prepare a six-class road-safety dataset (person, bicycle, car, motorcycle, bus, truck)
- Fine-tune a pretrained YOLO model and evaluate mAP metrics
- Run inference on images, folders, videos, or a webcam
- Save annotated predictions and export standard evaluation metrics
- Keep large datasets and model checkpoints out of Git

## Tech stack

Python · Ultralytics YOLO · PyTorch · OpenCV · YAML · Apple Silicon (MPS) support

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Or, with Conda:

```bash
conda env create -f environment.yml
conda activate yolo-resume
```

Prepare the dataset (downloads the compact public COCO128 sample, filters it, and remaps labels):

```bash
python src/prepare_dataset.py
```

The generated data is YOLO formatted:

```text
data/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Each label file has one object per line:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to values from 0 to 1.

## Train

```bash
python src/train.py --data data/dataset.yaml --model yolo11n.pt --epochs 50 --imgsz 640
```

Results, charts, and the best checkpoint are saved under `runs/detect/`.

## Demo result

The included 3-epoch Apple Silicon baseline uses 102 training images and 26 validation images at 320px. It reached **mAP50-95: 0.0313**. This is a smoke-test baseline, not a production-quality accuracy claim; train for more epochs on a substantially larger, independently labelled dataset before using it for safety decisions.

## Validate

```bash
python src/validate.py --model runs/detect/custom_detector/weights/best.pt --data data/dataset.yaml
```

## Predict

```bash
python src/predict.py --model runs/detect/custom_detector/weights/best.pt --source path/to/image.jpg
```

For a webcam, use `--source 0`. Add `--show` to display predictions while running.

## Reproducibility notes

- The training seed defaults to `42` and can be overridden with `--seed`.
- Dataset images, labels, checkpoints, and experiment outputs are ignored by Git.
- Record the final model version, dataset version, mAP, and precision/recall in your résumé or project portfolio after training.

## Suggested résumé bullet

> Built an end-to-end road-safety object-detection pipeline using Ultralytics YOLO, including automated COCO data filtering/remapping, transfer learning, mAP evaluation, and real-time image/video inference.

## License

This repository is available under the MIT License. Check the [Ultralytics license](https://www.ultralytics.com/license) before commercial use of its models or software.
