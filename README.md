# BlueprintCV

<img width="466" height="331" alt="image" src="https://github.com/user-attachments/assets/5627ace8-cb7a-44c7-b73f-cad97eb29171" />

<img width="258" height="334" alt="image" src="https://github.com/user-attachments/assets/09c9b6f5-6817-4fd9-a507-6b44ba986b3a" />

**BlueprintCV** automatically detects and classifies GD&T (Geometric Dimensioning and Tolerancing) symbols on scanned industrial blueprints, turning manual transcription into a fast, repeatable computer-vision pipeline.

---

## Objective

Industrial blueprints carry dozens of dimensional annotations, each potentially tagged with a GD&T symbol (flatness, perpendicularity, position, diameter, surface finish, etc.). Reading and transcribing these by hand is slow and error-prone.

BlueprintCV decomposes the problem into three specialized stages, because "find an annotation region", "locate the symbol inside it", and "identify the symbol class" are distinct visual tasks:

- **M1 — Quote Detector**: finds every dimensional annotation region on the blueprint
- **M2 — Symbol Detector**: locates the GD&T symbol within each M1 crop
- **M3 — Symbol Classifier**: recognizes the symbol class from the M2 crop

**Output**: a structured list of `{quote_box, symbol, confidence}` entries per blueprint — suitable for building automated inspection checklists or feeding downstream PLM/ERP systems.

---

## Pipeline

```
1. Input        scanned blueprint (.jpg / .png)
2. M1           detect quote bounding boxes
3. NMS merge    overlapping boxes merged (IoU > 0.6)
4. Crop         each quote box cropped with +40 % context margin
5. M2           locate symbol bounding box within each crop
6. Crop         symbol region extracted from the M2 bounding box
7. M3           classify the GD&T symbol class
8. Filter       low-confidence or "background" predictions rejected
9. Output       {quote_box, symbol, confidence} per blueprint
```

---

## Dataset

Real labeled blueprints were combined with synthetic data generated to cover symbol classes that are too rare in the real dataset.

**M1 (quotes)** — real scans annotated with LabelMe, augmented with gaussian-sampled faded lines to simulate cluttered backgrounds, and supplemented with procedurally generated blueprint layouts (`generate_blueprint_strutturato.py`) for broader generalization.

**M2 (symbol detection)** — YOLO-format crops derived from the M1 annotations, with a single `symbol` class. Rare classes are balanced via `balance_symbol_dataset.py`.

**M3 (symbol classification)** — 15 symbol classes plus a `background` class for rejecting false M2 detections:

| Classes | | | |
|---|---|---|---|
| `diameter` | `radius` | `surface_finish` | `concentricity` |
| `cylindricity` | `position` | `flatness` | `perpendicularity` |
| `total_runout` | `circular_runout` | `slope` | `conical_taper` |
| `symmetry` | `surface_profile` | `linear` | `background` |

Synthetic samples are generated procedurally across multiple layouts (single-symbol, stacked, inline, grouped) with gaussian noise on shape and position to mimic real-world variation. **Adaptive augmentation** scales the synthetic-to-real ratio per class — abundant classes receive minimal augmentation while scarce ones receive up to 10×.

| Split | Source |
|---|---|
| Train | Real (LabelMe-labeled) + adaptive synthetic augmentation |
| Test  | Held-out real blueprints, manually labeled |

---

## Model

| Stage | Architecture | Task | Classes |
|---|---|---|---|
| M1 — Quote Detector | YOLO11s | Object detection | 1 (`quote`) |
| M2 — Symbol Detector | YOLO11s | Object detection | 1 (`symbol`) |
| M3 — Symbol Classifier | YOLO11s-cls | Classification | 16 (15 symbols + `background`) |

YOLO11 was chosen for its accuracy/speed trade-off and ease of fine-tuning on a small-to-medium custom dataset, supporting both local-GPU and cloud training.

---

## Results

| Model | mAP@50 | Target |
|---|---|---|
| M1 — Quote Detector | **0.934** | ≥ 0.90 |
| M2 — Symbol Detector | **0.852** | ≥ 0.80 |

End-to-end metrics on a held-out test set of 180 real blueprints:

| Metric | Value |
|---|---|
| Precision | 0.933 |
| Recall | 0.909 |
| F1 | 0.921 |

---

## Installation

```bash
git clone https://github.com/Eddicpp/BlueprintCV.git
cd BlueprintCV
pip install ultralytics opencv-python pillow numpy easyocr
```

Place pretrained weights (`best_m1.pt`, `best_m2.pt`, `best_m3.pt`) under `weights/`.

---

## Usage

### Interactive GUI

```bash
# Full M1 → M2 → M3 pipeline with tabbed visualisation
python pipeline_gui.py

# Inspection GUI — browse predictions on single images or folders
python inspector_gui.py
```

### Training

```bash
# Train the M1 Quote Detector
python train_yolov8.py

# Train the M2 Symbol Detector
python train_symbol_detector.py

# Train the M3 Symbol Classifier
python train_symbol_classifier.py
```

### Dataset preparation

```bash
# Generate the synthetic M3 classification dataset
python generate_symbol_dataset.py

# Generate synthetic blueprint layouts for M1 training
python generate_blueprint_strutturato.py

# Augment the real blueprint dataset
python augment_dataset.py

# Convert LabelMe annotations to YOLO format (global)
python convert_to_yolo_global.py

# Convert symbol annotations to YOLO format
python convert_symbols_to_yolo.py
```

### Evaluation and visualisation

```bash
# Test the M2 symbol detector on single crops or a full dataset
python test_symbol_detector.py

# Visualise predictions overlaid on the test set
python visualize_predictions.py

# Export per-class metrics as LaTeX tables (for reports/papers)
python generate_latex_tables.py
```

---

## Repository Structure

```
BlueprintCV/
│
├── pipeline_gui.py               # Interactive M1→M2→M3 pipeline GUI
├── inspector_gui.py              # Live inference inspection GUI
│
├── train_yolov8.py               # M1 Quote Detector training
├── train_symbol_detector.py      # M2 Symbol Detector training
├── train_symbol_classifier.py    # M3 Symbol Classifier training
├── tune_quote_detector.py        # Hyperparameter tuning for M1
│
├── generate_blueprint_strutturato.py  # Synthetic blueprint generator (M1 data)
├── generate_symbol_dataset.py    # Synthetic symbol dataset generator (M3 data)
├── generate_background.py        # Background class generator for M3
│
├── augment_dataset.py            # Real-data augmentation pipeline
├── augment_angle.py              # Per-class augmentation helpers
├── augment_angular_quotes.py     #   ↳
├── augment_arrow_tip.py          #   ↳
├── augment_surface_finish.py     #   ↳
├── balance_symbol_dataset.py     # Adaptive class balancing
├── rebuild_symbol_dataset.py     # Rebuild M3 dataset from LabelMe annotations
│
├── convert_to_yolo_global.py     # LabelMe → YOLO conversion (M1)
├── convert_symbols_to_yolo.py    # LabelMe → YOLO conversion (M2/M3)
├── merge_datasets.py             # Dataset merging utility
├── resize_dataset.py             # Image resizing utility
├── check_dataset.py              # Dataset integrity check
├── clean.py                      # Remove orphaned label files
├── analyze_symbol_dataset.py     # Class distribution analysis
│
├── extract_quotes_for_labeling.py # Export quote crops for manual labeling
├── remove_angle_class.py         # Dataset maintenance utility
├── remove_angle_labels.py        #   ↳
│
├── test_symbol_detector.py       # M2 inference tests
├── visualize_predictions.py      # Prediction visualisation
├── run_preview.py                # Quick preview helper
├── generate_latex_tables.py      # Export metrics to LaTeX
│
└── README.md
```

---

## Limitations & Next Steps

- **Radius / perpendicularity ambiguity**: M2 occasionally confuses the "R" and "T" glyphs in noisy scans. The M3 `background` class reduces false positives but does not fully resolve the ambiguity.
- **Limited real data for rare classes**: performance on infrequent symbols (e.g. `conical_taper`, `surface_profile`) still relies heavily on synthetic data quality.
- **Windows-centric paths**: some scripts contain hardcoded Windows paths that need adjustment when running on Linux/macOS.

**Planned improvements:**
- Collect more real-world labeled data, especially for rare symbol classes
- Lightweight model variant for edge/embedded deployment
- Simple web demo (Streamlit or Gradio) for interactive testing without a local setup

---

## Tech Stack

Python · PyTorch · Ultralytics YOLO11 · OpenCV · NumPy · Pillow · EasyOCR · LabelMe · Matplotlib

---

## Author

**Eduardo Pane**  
[GitHub](https://github.com/Eddicpp) · [Kaggle](https://kaggle.com/eduardopane)
