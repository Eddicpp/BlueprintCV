# BlueprintCV

### Automated detection and GD&T symbol recognition in industrial technical drawings

---

## Overview

**BlueprintCV** is an end-to-end computer vision pipeline developed as a master's thesis project in collaboration with an industrial partner. The system processes scanned engineering blueprints and automatically:

1. **Detects dimension annotations** (quotes) across the entire drawing
2. **Locates the GD&T symbol** inside each detected dimension
3. **Classifies the symbol** into one of 15 GD&T categories

The project benchmarks multiple YOLO architectures on a custom dataset built entirely from real industrial drawings provided by the partner company.

> The dataset used for training and evaluation is confidential and not included in this repository.

---

## Pipeline

```
Scanned blueprint
        │
        ▼
┌──────────────────────────┐
│  M1 — Quote Detector     │   YOLO11s   mAP@50: 0.972
│  Finds dimension boxes   │
└────────────┬─────────────┘
             │  crop per quote  (+40% context margin)
             ▼
┌──────────────────────────┐
│  M2 — Symbol Detector    │   YOLO11n   mAP@50: 0.966
│  Locates symbol in crop  │
└────────────┬─────────────┘
             │  symbol crop
             ▼
┌──────────────────────────┐
│  M3 — Symbol Classifier  │   YOLO11s-cls   Top-1: 0.937
│  Classifies symbol type  │
└──────────────────────────┘
```

**Key design choices:**
- M1 uses **tiling** for large blueprints (>2000px) — the image is split into overlapping tiles, predictions are merged with NMS
- M2 receives a **context-expanded crop** (40% margin around the quote box) to avoid missing symbols cut at the border
- Symbols whose center falls **outside** the original quote box are discarded — they belong to adjacent quotes
- M3 includes a **background class** to reject M2 false positives; detections below 0.50 confidence are suppressed

---

## Results

### M1 — Quote Detection

| Model | Dataset | mAP@50 | F1 | Threshold |
|---|---|---|---|---|
| YOLOv8n | baseline | 0.911 | 0.85 | 0.575 |
| YOLOv8s | baseline | 0.950 | 0.91 | 0.618 |
| YOLO11s | augmented v1 | 0.881 | 0.87 | 0.432 |
| **YOLO11s** | **augmented v2** | **0.972** | **0.96** | **0.422** |

### M2 — Symbol Detection

| Model | mAP@50 | mAP@50-95 |
|---|---|---|
| **YOLO11n** | **0.966** | **0.621** |

### M3 — Symbol Classification

| Model | Top-1 | Top-5 |
|---|---|---|
| **YOLO11s-cls** | **0.937** | **0.995** |

---

## GD&T Symbol Classes (15 + background)

```
diameter · radius · surface_finish · concentricity · cylindricity
position · flatness · perpendicularity · total_runout · circular_runout
slope · conical_taper · symmetry · surface_profile · linear
```

> `angle` was removed — the degree symbol (°) caused systematic false positives
> due to confusion with decimal points in numeric values.

---

## Project Structure

```
BlueprintCV/
│
├── ── Quote Detection (M1) ──
├── convert_to_yolo_global.py          LabelMe JSON → YOLO format + resize
├── augment_dataset.py                 Scan-quality augmentation pipeline
├── generate_blueprint_strutturato.py  Structured synthetic blueprint generation
├── augment_angular_quotes.py          Augmentation for angular/arc dimensions
├── merge_datasets.py                  Dataset merge utility
├── resize_dataset.py                  Batch image resizing
├── train_yolov8.py                    YOLO11s training script
├── visualize_predictions.py           Predictions with GT overlay
├── generate_latex_tables.py           Auto LaTeX tables from results
│
├── ── Symbol Detection (M2) ──
├── extract_quotes_for_labeling.py     Extract quote crops for labeling
├── remove_angle_labels.py             Remove angle class from LabelMe JSONs
├── convert_symbols_to_yolo.py         LabelMe annotations → YOLO detection
├── balance_symbol_dataset.py          Synthetic generation for rare classes
├── train_symbol_detector.py           YOLO11n symbol detector training
├── test_symbol_detector.py            Evaluation + preview on blueprints
│
├── ── Symbol Classification (M3) ──
├── rebuild_symbol_dataset.py          Build classification dataset from LabelMe
├── generate_background.py             Generate background class for M3
├── augment_arrow_tip.py               Arrow tip augmentation → linear class
├── augment_surface_finish.py          Surface finish real crop augmentation
├── augment_angle.py                   Angle real crop augmentation
├── train_symbol_classifier.py         YOLO11s-cls classifier training
├── analyze_symbol_dataset.py          Dataset class distribution analysis
│
└── ── Inference ──
    └── pipeline_gui.py                Full M1+M2+M3 interactive GUI
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

> The pipeline assumes an annotated dataset in LabelMe JSON format:
>
> ```
> project_folder/
> ├── project_1/
> │   ├── immagini/    ← images
> │   └── labels/      ← LabelMe JSON files
> └── project_2/ ...
> ```

### Phase 1 — Quote Detection Dataset

```bash
# Convert LabelMe annotations to YOLO format (MAX_SIDE=1600px)
# Val/test taken from separate company folders for generalisation
python convert_to_yolo_global.py

# Scan-quality augmentation x10 on train set
python augment_dataset.py

# Generate structured synthetic blueprints
python generate_blueprint_strutturato.py

# (Optional) Add angular/arc dimension examples
python augment_angular_quotes.py

# Merge all sources
python merge_datasets.py
python resize_dataset.py
```

### Phase 2 — Train M1

```bash
# Edit MODEL_SIZE and RUN_NAME in train_yolov8.py
python train_yolov8.py

python visualize_predictions.py
python generate_latex_tables.py
```

### Phase 3 — Symbol Detection Dataset

```bash
# Extract quote crops for labeling
python extract_quotes_for_labeling.py

# Label with LabelMe — draw bbox around each symbol, assign class
labelme quote_per_labeling --labels quote_per_labeling/classes.txt --nodata

# Remove angle class (causes false positives with decimal points)
python remove_angle_labels.py

# Convert to YOLO detection format
python convert_symbols_to_yolo.py

# Balance rare classes with synthetic generation
python balance_symbol_dataset.py
```

### Phase 4 — Train M2

```bash
python train_symbol_detector.py

# Test on blueprints
python test_symbol_detector.py --dir path/to/blueprints/
```

### Phase 5 — Symbol Classification Dataset

```bash
# Rebuild dataset from LabelMe annotations (pure symbol crops)
python rebuild_symbol_dataset.py

# Generate background class (M2 false positives rejection)
python generate_background.py

# Augment specific classes from real crops
python augment_arrow_tip.py        # arrow tips → linear
python augment_surface_finish.py   # real surface finish crops
```

### Phase 6 — Train M3

```bash
python train_symbol_classifier.py
```

### Phase 7 — Run Full Pipeline

```bash
python pipeline_gui.py
```

---

## Augmentation Strategy

### Quote Detection (M1)

A custom augmentation pipeline replicates real industrial scanner degradation. All transforms preserve YOLO bounding box coordinates.

**Photometric degradation** — paper yellowing, gamma correction, contrast reduction, non-uniform illumination, Gaussian noise.

**Geometric transforms** — aggressive zoom (0.35–1.5×), small rotations (±15°), perspective distortion for documents not flat on scanner.

**Blur modes** — Gaussian, motion blur (random orientation), circular defocus.

**Construction line overlay** — 50–120 parallel black lines (1px) simulating section view hatching, the main source of false positives in baseline models.

**Compression and ink artifacts** — JPEG compression (quality 40–75), morphological erosion for broken ink lines, partial occlusion.

**Synthetic blueprints** — structured images with border, title blocks, and real quote crops in scattered/overlapping layouts.

### Symbol Classification (M3)

Additional augmentation designed for small GD&T symbols:

**Faded ink** — dark pixels are lightened to simulate old or low-quality ink, producing grey strokes instead of black.

**Light construction strokes** — 1–5 grey lines (160–220 brightness) overlaid to simulate blueprint context lines passing over the symbol.

**Morphological variation** — both erosion (consumed ink) and dilation (ink bleed), shear distortion.

**Salt and pepper noise** — scattered white/black pixels simulating scanner sensor noise.

---

## Inference GUI

```bash
python pipeline_gui.py
```

- Load M1, M2, M3 weights independently via file browser
- Load any blueprint image (large images are automatically tiled)
- **Tab 1 — QUOTE (M1)**: all detected dimension boxes highlighted in blue with index and confidence
- **Tab 2 — IMMAGINE ANNOTATA**: full M1+M2+M3 result with color-coded symbol labels
- **Tab 3 — SIMBOLI TROVATI**: scrollable grid of all detected symbol crops with class and confidence
- Zoom (scroll wheel) and pan (drag) on both image tabs independently
- Context margin around each quote ensures border symbols are not cut off
- M3 background class suppresses M2 false positives automatically

---

## Architecture Progression

| Phase | Model | Type | Task | Notes |
|---|---|---|---|---|
| 1 | YOLOv8n | CNN anchor-free | Quote detection | Baseline |
| 2 | YOLOv8s | CNN anchor-free | Quote detection | Larger backbone |
| 3 | YOLO11s | CNN + attention | Quote detection | Best — 0.972 mAP@50 |
| 4 | YOLO11n | CNN + attention | Symbol detection | 0.966 mAP@50 |
| 5 | YOLO11s-cls | CNN + attention | Symbol classification | 0.937 Top-1 |

---

## License

This repository contains only the code. The dataset is proprietary and owned by the industrial partner. No data is included or distributed.

---

## Citation

```bibtex
@mastersthesis{blueprintcv2026,
  title  = {BlueprintCV: Automated Detection and GD&T Symbol Recognition
             in Industrial Technical Drawings},
  author = {Eduardo Pane},
  year   = {2026},
  school = {UNIPD}
}
```
