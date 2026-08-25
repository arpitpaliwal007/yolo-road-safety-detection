# Dataset guide

`src/prepare_dataset.py` creates a tiny COCO128-derived dataset for verifying that the pipeline works. It is not intended to produce a deployment-quality detector.

For a serious experiment, replace it with a purpose-labelled road dataset:

```text
data/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Keep the six class IDs consistent with `dataset.yaml`:

| ID | Class |
| ---: | --- |
| 0 | person |
| 1 | bicycle |
| 2 | car |
| 3 | motorcycle |
| 4 | bus |
| 5 | truck |

Recommended data-quality targets:

- At least 1,000 images, with hundreds of instances per class
- Separate locations or videos between train and validation splits to prevent leakage
- Day, night, rain, congestion, occlusion, and small/distant objects
- Review every label visually and remove duplicate frames
- Keep a final test split that is never used while tuning

Generate a class-balance report after adding data:

```bash
python src/dataset_report.py --data data/dataset.yaml
```
