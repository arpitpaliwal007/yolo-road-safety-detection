# Baseline experiment results

## Experiment setup

| Setting | Value |
| --- | --- |
| Model | YOLO11n (pretrained) |
| Dataset | Filtered COCO128 road-safety subset |
| Classes | person, bicycle, car, motorcycle, bus, truck |
| Train / validation images | 102 / 26 |
| Image size | 320 × 320 |
| Epochs | 3 |
| Hardware | Apple M2 (MPS) |

## Metrics by epoch

| Epoch | Precision | Recall | mAP@50 | mAP@50–95 | Train box loss | Train class loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0042 | 0.1246 | 0.0203 | 0.0117 | 1.2630 | 3.8487 |
| 2 | 0.0043 | 0.1219 | 0.0333 | 0.0196 | 1.3115 | 3.7141 |
| 3 | 0.0048 | 0.1219 | 0.0474 | 0.0313 | 1.2657 | 3.5392 |

This is a smoke-test baseline using a small, automatically filtered sample and only three epochs. It demonstrates that the full data-to-inference pipeline runs; it is not suitable for safety-critical deployment. A stronger result needs a larger purpose-labelled dataset and substantially longer training.

## Visual artifacts

| Training curves | Precision / recall | Confusion matrix |
| --- | --- | --- |
| ![Training curves](results.png) | ![PR curve](BoxPR_curve.png) | ![Confusion matrix](confusion_matrix.png) |

| Validation labels | Validation predictions |
| --- | --- |
| ![Validation labels](val_batch0_labels.jpg) | ![Validation predictions](val_batch0_pred.jpg) |
