# BlueprintCV

### Automated detection and GD&T symbol recognition in industrial technical drawings

---

## Overview

**BlueprintCV** is an end-to-end computer vision pipeline developed as a master's thesis project in collaboration with an industrial partner. The system processes scanned engineering blueprints and automatically:

1. **Detects dimension annotations** (quotes) across the entire drawing
2. **Classifies the associated GD&T symbol** for each detected dimension

The project progresses from baseline CNN detectors to Transformer-based architectures, benchmarking multiple models on a custom dataset built entirely from real industrial drawings provided by the partner company.

> The dataset used for training and evaluation is confidential and not included in this repository.

---

## Pipeline

```
Scanned blueprint
        │
        ▼
┌──────────────────────────┐
│  Stage 1 — Quote Detector │   YOLOv8s / YOLO11s / RF-DETR
│  Finds dimension boxes    │   mAP@50: 0.972
└────────────┬─────────────┘
             │  crop per detected quote
             ▼
┌──────────────────────────┐
│  Stage 2 — Symbol Detector│   YOLO11n
│  Finds GD&T symbol inside │   16 classes · Top-1: 0.937
└──────────────────────────┘
```

---

## Results

### Stage 1 — Quote Detection

| Model | Dataset | mAP@50 | F1 | Threshold |
|---|---|---|---|---|
| YOLOv8n | baseline | 0.911 | 0.85 | 0.575 |
| YOLOv8s | baseline | 0.950 | 0.91 | 0.618 |
| YOLO11s | augmented v1 | 0.881 | 0.87 | 0.432 |
| **YOLO11s** | **augmented v2** | **0.972** | **0.96** | **0.422** |
| RF-DETR Base | in progress | — | — | — |

**Key finding:** dataset quality and augmentation strategy had a larger impact on performance than architecture choice alone.

### Stage 2 — GD&T Symbol Recognition

| Model | Accuracy Top-1 | Accuracy Top-5 |
|---|---|---|
| YOLO11s-cls (classify) | 0.937 | 0.995 |
| YOLO11n (detect) | in progress | — |

---

## GD&T Symbol Classes

```
diameter · radius · angle · surface_finish · concentricity
cylindricity · position · flatness · perpendicularity
total_runout · circular_runout · slope · conical_taper
symmetry · surface_profile · linear
```

---

## Project Structure

```
BlueprintCV/
│
├── data/
│   ├── convert_to_yolo_global.py       LabelMe JSON → YOLO format
│   ├── augment_dataset.py              Scan-quality augmentation pipeline
│   ├── generate_blueprint_strutturato.py  Structured synthetic generation
│   ├── generate_mosaics.py             Real crop mosaics
│   ├── merge_datasets.py               Dataset merge utility
│   ├── resize_dataset.py               Batch image resizing
│   ├── convert_to_coco.py              YOLO → COCO JSON for RF-DETR
│   ├── convert_symbols_to_yolo.py      Symbol annotations → YOLO detection
│   ├── balance_symbol_dataset.py       Synthetic balancing for rare classes
│   └── generate_symbol_dataset.py      Symbol classification dataset
│
├── training/
│   ├── train_yolov8.py                 YOLO training (v8n / v8s / v11s)
│   ├── train_rfdetr.py                 RF-DETR training
│   ├── train_symbol_detector.py        GD&T symbol detector
│   └── train_symbols.py               GD&T symbol classifier
│
├── evaluation/
│   ├── visualize_predictions.py        Predictions with GT overlay
│   ├── test_symbol_detector.py         Symbol detector evaluation
│   ├── analyze_symbol_dataset.py       Dataset distribution analysis
│   └── generate_latex_tables.py        Auto LaTeX tables from results
│
└── inference/
    ├── inspector_gui.py                Live inference GUI (zoom/pan)
    └── pipeline_gui.py                 Full two-stage pipeline GUI
```

---

## Setup

### Requirements

- Python 3.10
- CUDA 12.1+
- GPU with 8GB+ VRAM recommended (16GB+ for RF-DETR)

Tested on:
- Windows 11 — NVIDIA GeForce GTX 1660 6GB
- Ubuntu 22.04 — NVIDIA GeForce RTX 2080 8GB

### Installation

```bash
conda create -n blueprintcv python=3.10 -y
conda activate blueprintcv

# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install ultralytics rfdetr supervision
pip install opencv-python pillow numpy matplotlib seaborn pandas scipy

# Annotation tool
pip install labelme
```

---

## Dataset Pipeline

> The dataset is confidential. The pipeline below assumes you have your own
> annotated dataset in LabelMe JSON format with the same folder structure.

```bash
# 1. Convert LabelMe annotations to YOLO format + resize
python data/convert_to_yolo_global.py

# 2. Apply augmentation
python data/augment_dataset.py

# 3. Generate synthetic structured blueprints
python data/generate_blueprint_strutturato.py

# 4. Merge all sources
python data/merge_datasets.py

# 5. Train
python training/train_yolov8.py
```

---

## Augmentation Strategy

A custom augmentation pipeline was designed to replicate the visual characteristics of real scanner output on industrial drawings. All augmentations operate on training images only and preserve YOLO bounding box coordinates.

**Photometric degradation** simulates scanner-specific artifacts: paper yellowing and tonal shift, gamma correction for under/over-exposed scans, contrast reduction to simulate faded ink, non-uniform illumination gradients across the page, and fine-grained Gaussian noise matching real scanner granularity.

**Geometric transforms** include aggressive zoom crops centered on dense dimension regions (scale 0.35–1.5×), small rotations (±15°), and perspective distortion to simulate documents not lying flat on the scanner bed.

**Blur modes** cover three distinct types: Gaussian blur for general focus loss, motion blur with random orientation for document movement during scanning, and circular defocus blur for a lifted document corner effect.

**Construction line overlay** draws 50–120 parallel black lines (1px, variable spacing and angle) over the image, replicating the dense hatching patterns found in section views of real mechanical drawings — one of the main sources of false positives in baseline models.

**Compression and ink artifacts** apply JPEG compression at quality 40–75 to simulate degraded scan exports, morphological erosion on dark pixels to simulate broken ink lines and partially illegible numbers, and partial occlusion using paper-colored rectangles on random regions of each bounding box.

**Synthetic data generation** produces structured blueprint images containing a border frame, title blocks positioned on the inner edges, and quote crops sampled from the real dataset — both scattered and in dense overlapping layouts — to diversify training examples without manual annotation.

---

## Training

### Stage 1 — Quote Detector

```bash
# Configure MODEL_SIZE in training/train_yolov8.py
# Options: "yolov8n", "yolov8s", "yolo11s"
python training/train_yolov8.py

# RF-DETR (requires COCO format — convert first)
python data/convert_to_coco.py
python training/train_rfdetr.py
```

### Stage 2 — Symbol Detector

```bash
# Convert symbol annotations
python data/convert_symbols_to_yolo.py

# Balance rare classes with synthetic symbols
python data/balance_symbol_dataset.py

# Train
python training/train_symbol_detector.py
```

---

## Inference GUI

A Tkinter-based GUI runs the full two-stage pipeline interactively:

```bash
python inference/pipeline_gui.py
```

- Load detector and symbol detector weights independently
- Load any blueprint image
- Visualize annotated results with color-coded GD&T classes
- Browse individual detected quote crops in a scrollable grid
- Zoom and pan on the annotated image

---

## Architecture Progression

| Phase | Model | Type | Notes |
|---|---|---|---|
| 1 | YOLOv8n | CNN anchor-free | Baseline |
| 2 | YOLOv8s | CNN anchor-free | Larger backbone |
| 3 | YOLO11s | CNN + attention | Hybrid |
| 4 | RF-DETR Base | Transformer (DINOv2) | End-to-end, no NMS |

---

## License

This repository contains only the code. The dataset used for training is proprietary and owned by the industrial partner. No data is included or distributed with this project.

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{blueprintcv2025,
  title  = {BlueprintCV: Automated Detection and GD&T Symbol Recognition
             in Industrial Technical Drawings},
  author = {Eduardo Pane},
  year   = {2026},
  school = {University}
}
```
