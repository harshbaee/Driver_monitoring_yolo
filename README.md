# Driver Monitoring System using YOLO

A computer vision project for real-time driver monitoring using YOLO object detection models. The project benchmarks YOLOv8 and YOLOv9 variants on a custom dataset to detect driver distraction and fatigue-related behaviors.

---

## Overview

Driver inattention is one of the leading causes of road accidents. This project builds a detection pipeline that identifies unsafe driver states (drowsiness, distraction, phone use, etc.) from camera footage using fine-tuned YOLO models. Three model variants were trained and evaluated to compare performance, speed, and accuracy tradeoffs.

---

## Repository Structure

```
├── train/                          # Training dataset (images + labels)
├── valid/                          # Validation dataset
├── test/                           # Test dataset
├── runs/                           # YOLO training output (weights, metrics, plots)
├── comparison_results/             # Side-by-side prediction outputs across models
├── data.yaml                       # Dataset config (class names, paths)
├── dms-v2-dataset-notebook.ipynb   # Dataset preparation and EDA
├── dms-v9.ipynb                    # Model training and evaluation notebook
├── test.py                         # Inference script for running predictions
├── yolo_comparison_results.csv     # Quantitative comparison of all three models
├── Figure_1.png                    # Results figure
└── Driver_Monitoring_YOLO_Presentation.pptx  # Project presentation slides
```

---

## Models

Three YOLO model variants were trained and compared:

| Model | Weights File | Size |
|-------|-------------|------|
| YOLOv8n (nano) | `yolov8n.pt` | ~6.4 MB |
| YOLOv9s (small) | `yolov9s.pt` | ~15 MB |
| YOLOv9t (tiny) | `yolov9t.pt` | ~4.9 MB |

Detailed per-model metrics (mAP, precision, recall, inference time) are in `yolo_comparison_results.csv`.

---

## Getting Started

### Prerequisites

```bash
pip install ultralytics opencv-python pandas matplotlib
```

### Running Inference

```bash
python test.py
```

You can modify the model path and source inside `test.py` to switch between the three trained weights.

### Training from Scratch

Open `dms-v9.ipynb` and follow the notebook cells. The dataset config is in `data.yaml` — update the paths if needed for your local environment.

---

## Dataset

The dataset is organized into `train/`, `valid/`, and `test/` splits with YOLO-format annotations. Class definitions and split paths are defined in `data.yaml`.

If the dataset is not included in this repo due to size, you can either:
- Download it separately (link to be added)
- Use your own annotated dataset following the same structure

---

## Results

See `yolo_comparison_results.csv` for a full quantitative breakdown and `comparison_results/` for visual prediction samples across models.

---

## Tech Stack

- Python
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV
- PyTorch
- Jupyter Notebook

---

## Author

Harsh — [github.com/harshbaee](https://github.com/harshbaee)
