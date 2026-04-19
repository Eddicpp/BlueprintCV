# BlueprintCV

### Automated detection and GD&T symbol recognition in industrial technical drawings

---

## Overview

**BlueprintCV** is an end-to-end computer vision pipeline developed as a master's thesis project in collaboration with an industrial partner. The system processes scanned engineering blueprints and automatically:

1. **Detects dimension annotations** (quotes) across the entire drawing
2. **Identifies the associated GD&T symbol** for each detected dimension

The project benchmarks multiple YOLO architectures on a custom dataset built entirely from real industrial drawings provided by the partner company.

> The dataset used for training and evaluation is confidential and not included in this repository.

---

## Pipeline

```
Scanned blueprint
        │
        ▼
┌──────────────────────────┐
│  Stage 1 — Quote Detector │   YOLOv8s / YOLO11s
│  Finds dimension boxes    │   mAP@50: 0.972
└────────────┬─────────────┘
             │  crop per detected quote
             ▼
┌──────────────────────────┐
│  Stage 2 — Symbol Detector│   YOLO11n
│  Finds GD&T symbol inside │   16 classes
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

**Key finding:** dataset quality and augmentation strategy had a larger impact on performance than architecture choice alone.

### Stage 2 — GD&T Symbol Detection

| Model | mAP@50 |
|---|---|
| YOLO11n | in progress |

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
├── convert_to_yolo_global.py          LabelMe JSON → YOLO format + resize
├── augment_dataset.py                 Scan-quality augmentation pipeline
├── generate_blueprint_strutturato.py  Structured synthetic blueprint generation
├── merge_datasets.py                  Dataset merge utility
├── resize_dataset.py                  Batch image resizing
├── train_yolov8.py                    YOLO training (v8n / v8s / v11s)
├── extract_quotes_for_labeling.py     Extract quote crops for symbol labeling
├── analyze_symbol_dataset.py          Dataset class distribution analysis
├── convert_symbols_to_yolo.py         LabelMe symbol annotations → YOLO detection
├── balance_symbol_dataset.py          Synthetic generation for rare symbol classes
├── train_symbol_detector.py           GD&T symbol detector training
├── visualize_predictions.py           Predictions with GT overlay visualization
├── generate_latex_tables.py           Auto LaTeX tables from training results
└── pipeline_gui.py                    Full two-stage interactive inference GUI
```

---

## Setup

### Requirements

- Python 3.10
- CUDA 12.1+
- GPU with 6GB+ VRAM

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
pip install ultralytics
pip install opencv-python pillow numpy matplotlib seaborn pandas scipy

# Annotation tool
pip install labelme
```

---

## Full Pipeline — Step by Step

> The pipeline below assumes you have your own annotated dataset
> in LabelMe JSON format with the following structure:
>
> ```
> project_folder/
> ├── project_1/
> │   ├── immagini/    ← images
> │   └── labels/      ← LabelMe JSON files
> ├── project_2/
> │   ├── immagini/
> │   └── labels/
> ...
> ```

### Phase 1 — Quote Detection Dataset

```bash
# Step 1 — Convert LabelMe annotations to YOLO format
# Images are resized to MAX_SIDE=1600px during copy.
# Val and test sets are taken from separate project folders
# to measure generalisation across different companies.
python convert_to_yolo_global.py

# Step 2 — Apply scan-quality augmentation (x10 on train set)
# Adds photometric degradation, blur, zoom, perspective,
# construction lines, JPEG artifacts, and partial occlusion.
python augment_dataset.py

# Step 3 — Generate structured synthetic blueprints
# Creates blueprint-like images with border, title blocks,
# and real quote crops arranged in various layouts.
python generate_blueprint_strutturato.py

# Step 4 — Merge all dataset sources into a single dataset
# Edit DATASETS list in merge_datasets.py before running.
python merge_datasets.py

# Step 5 — (Optional) Resize images if any exceed MAX_SIDE
python resize_dataset.py
```

### Phase 2 — Train Quote Detector

```bash
# Edit MODEL_SIZE in train_yolov8.py before running.
# Available options: "yolov8n", "yolov8s", "yolo11s"
# Results saved to: runs/detect/sintesi_genesi/<RUN_NAME>/
python train_yolov8.py

# Visualize predictions on test set with GT overlay
python visualize_predictions.py

# Generate LaTeX tables from training results (for thesis)
python generate_latex_tables.py
```

### Phase 3 — Symbol Detection Dataset

```bash
# Step 1 — Extract all quote crops for manual labeling
python extract_quotes_for_labeling.py

# Step 2 — Label with LabelMe
# Draw bounding boxes around GD&T symbols and assign class.
# Skip quotes with no symbol (they will be class "linear").
labelme quote_per_labeling --labels quote_per_labeling/classes.txt --nodata

# Step 3 — Check class distribution
python analyze_symbol_dataset.py

# Step 4 — Convert LabelMe symbol annotations to YOLO detection format
python convert_symbols_to_yolo.py

# Step 5 — Balance rare classes with synthetic symbol generation
# Generates synthetic crops to reach TARGET_N examples per class.
python balance_symbol_dataset.py
```

### Phase 4 — Train Symbol Detector

```bash
# Train YOLO11n symbol detector
# Results saved to: runs/detect/sintesi_genesi/simboli_detector_run1/
python train_symbol_detector.py
```

### Phase 5 — Run Full Pipeline

```bash
# Launch interactive GUI
# Load Stage 1 weights (quote detector)
# Load Stage 2 weights (symbol detector)
# Load any blueprint image and run the full pipeline
python pipeline_gui.py
```

---

## Augmentation Strategy

A custom augmentation pipeline was designed to replicate the visual degradation of real industrial scanner output. All transforms apply to training images only and preserve YOLO bounding box coordinates.

**Photometric degradation** simulates scanner-specific artifacts: paper yellowing, gamma correction for under/over-exposed scans, contrast reduction for faded ink, non-uniform illumination gradients, and fine-grained Gaussian noise.

**Geometric transforms** include aggressive zoom crops centered on dense dimension regions (scale 0.35–1.5×), small rotations (±15°), and perspective distortion to simulate documents not lying flat on the scanner.

**Blur modes** cover three types: Gaussian blur, motion blur with random orientation, and circular defocus blur for a lifted document corner effect.

**Construction line overlay** draws 50–120 parallel black lines (1px) over the image, replicating the dense hatching found in section views — one of the main sources of false positives in baseline models.

**Compression and ink artifacts** apply JPEG compression (quality 40–75), morphological erosion on dark pixels to simulate broken ink lines, and partial occlusion with paper-colored rectangles.

**Synthetic data generation** produces structured blueprint images with a border frame, title blocks on inner edges, and real quote crops in scattered or overlapping layouts, to increase training diversity without additional manual annotation.

---

## Inference GUI

```bash
python pipeline_gui.py
```

- Load Stage 1 and Stage 2 model weights via file browser
- Load any blueprint image
- Run the full two-stage pipeline with a single click
- View annotated results with color-coded GD&T classes
- Browse individual detected quote crops in a scrollable grid
- Zoom and pan on the result image (scroll wheel + drag)

---

## Architecture Progression

| Phase | Model | Type | Notes |
|---|---|---|---|
| 1 | YOLOv8n | CNN anchor-free | Baseline |
| 2 | YOLOv8s | CNN anchor-free | Larger backbone |
| 3 | YOLO11s | CNN + attention | Best performing |

---

## License

This repository contains only the code. The dataset used for training is proprietary and owned by the industrial partner. No data is included or distributed with this project.

---

## Citation

```bibtex
@mastersthesis{blueprintcv2025,
  title  = {BlueprintCV: Automated Detection and GD&T Symbol Recognition
             in Industrial Technical Drawings},
  author = {Eduardo Pane},
  year   = {2026},
  school = {UNIPD}
}
```
