# BlueprintCV

**BlueprintCV** automatically detects and classifies GD&T (Geometric Dimensioning and Tolerancing) symbols on scanned industrial blueprints, turning manual transcription into a fast, repeatable detection pipeline.

---

## Demo

<img width="466" height="331" alt="image" src="https://github.com/user-attachments/assets/5627ace8-cb7a-44c7-b73f-cad97eb29171" />
<img width="258" height="334" alt="image" src="https://github.com/user-attachments/assets/09c9b6f5-6817-4fd9-a507-6b44ba986b3a" />

```
[ blueprint scan ]  →  [ quotes detected ]  →  [ GD&T symbols classified ]
```

**Output**: a structured list of `{quote_box, symbol, confidence}` per blueprint — JSON-ready, suitable for ingestion into PLM/ERP systems or for generating an inspection checklist automatically instead of by hand.

**Typical use case**: a quality or design engineer uploads a scanned drawing and gets back every GD&T callout pre-located and labeled, instead of reading the drawing line by line.

---

## Objective

Industrial blueprints carry dozens of dimensional annotations, each one possibly tagged with a GD&T symbol (flatness, perpendicularity, position, diameter, surface finish, etc.). Reading and transcribing these by hand doesn't scale.

BlueprintCV splits the problem into two specialized detectors, since "find an annotation" and "recognize a tiny symbol inside it" are different visual problems:

- **M1 — Quote Detector**: finds every dimensional annotation region
- **M2 — Symbol Detector**: detects and classifies the GD&T symbol inside each region M1 finds

---

## Dataset

Real labeled blueprints were combined with synthetic data, generated to cover symbol classes too rare in the real dataset.

**M1 (quotes)** — real scans + custom augmentation (gaussian-sampled faded lines simulating cluttered backgrounds) + synthetic blueprint layouts for generalization, later expanded with lower-quality scans and a few hand-drawn sketches.

**M2 (symbols)** — 13 classes (`diameter`, `radius`, `surface_finish`, `concentricity`, `cylindricity`, `position`, `flatness`, `perpendicularity`, `total_runout`, `circular_runout`, `slope`, `conical_taper`, `symmetry`), generated procedurally across 12 layouts (single-symbol, stacked, inline, grouped), with gaussian noise on shape/position to mimic real-world imperfections. **Adaptive augmentation** scales the synthetic-to-real ratio per class — abundant classes get little augmentation, scarce ones get up to 10x.

| Split | Source |
|---|---|
| Train | Real (labeled) + adaptive synthetic augmentation |
| Test  | Held-out real blueprints, manually labeled |

---

## Model

| Component | Architecture | Notes |
|---|---|---|
| M1 — Quote Detector | YOLO11s | 1 class, real + augmented data |
| M2 — Symbol Detector | YOLOv11s | 13 classes, real + synthetic data |

YOLO was chosen for its accuracy/speed trade-off and ease of fine-tuning on a small/medium custom dataset, with both local-GPU and cloud training as options.

---

## Pipeline

```
1. Input        scanned blueprint (.jpg / .png)
2. M1            detect quote boxes
3. Merge         duplicate overlapping boxes merged (overlap_ratio > 60%)
4. Crop          each quote box cropped with +40% context margin
5. M2            detect & classify symbol inside each crop
6. OCR filter    ambiguous symbols (radius "R" / perpendicularity "T")
                  checked against OCR'd text to reject false positives
7. Output        {quote_box, symbol, confidence} per blueprint
```

---

## Results

| Model | mAP@50 | Target |
|---|---|---|
| M1 — Quote Detector | **0.934** | ≥ 0.90 |
| M2 — Symbol Detector | **0.852** | ≥ 0.80 |

On a held-out test set of 180 real blueprints:

| Metric | Value |
|---|---|
| Precision | 0.933 |
| Recall | 0.909 |
| F1 | 0.921 |

---

## Installation

```bash
git clone https://github.com/<your-username>/BlueprintCV.git
cd BlueprintCV
pip install -r requirements.txt
```

Place pretrained weights (`best_m1.pt`, `best_m2.pt`) under `weights/`.

---

## Usage

```bash
# Full M1 → M2 pipeline on a single blueprint
python test_symbol_detector.py --image path/to/image.jpg

# Train M1 (quote detector)
python tune_quote_detector.py

# Train M2 (symbol detector)
python train_symbol_detector.py

# Build the synthetic training dataset for M2
python generate_symbol_dataset.py

# Interactive GUIs
python pipeline_gui.py       # run the full pipeline visually
python inspector_gui.py      # inspect predictions / dataset samples
```

---

## Repository Structure

Scripts are grouped by what they do in the project lifecycle: data preparation, dataset generation, augmentation, training, inference, and tooling.

```
BlueprintCV/
│
├── Data preparation & cleaning
│   ├── extract_quotes_for_labeling.py   # Crops quotes out of full blueprints for annotation
│   ├── check_dataset.py                 # Sanity-checks labels/images consistency
│   ├── clean.py                         # General dataset cleanup utility
│   ├── remove_angle_class.py            # Removes a deprecated class from the dataset
│   ├── remove_angle_labels.py           # Removes a deprecated label from annotation files
│   ├── resize_dataset.py                # Batch image resizing
│   └── analyze_symbol_dataset.py        # Class distribution / dataset statistics
│
├── Dataset generation (synthetic)
│   ├── generate_symbol_dataset.py       # Main synthetic symbol dataset generator
│   ├── generate_blueprint_strutturato.py# Generates structured synthetic blueprints
│   ├── generate_background.py           # Generates background textures for synthetic data
│   ├── rebuild_symbol_dataset.py        # Rebuilds dataset after generator changes
│   └── balance_symbol_dataset.py        # Balances class counts across the dataset
│
├── Format conversion & merging
│   ├── convert_symbols_to_yolo.py       # Converts symbol annotations to YOLO format
│   ├── convert_to_yolo_global.py        # Converts full-blueprint annotations to YOLO format
│   └── merge_datasets.py                # Merges multiple dataset sources into one
│
├── Augmentation
│   ├── augment_dataset.py               # Main augmentation + train/val/test split
│   ├── augment_angle.py                 # Rotation-based augmentation
│   ├── augment_angular_quotes.py        # Augmentation specific to angular quotes
│   ├── augment_arrow_tip.py             # Augmentation specific to arrow-tip symbols
│   └── augment_surface_finish.py        # Augmentation specific to surface finish symbols
│
├── Training
│   ├── tune_quote_detector.py           # M1 training / hyperparameter tuning
│   ├── train_symbol_detector.py         # M2 training
│   └── train_yolov8.py                 # Generic YOLOv8 training entry point
│
├── Inference & evaluation
│   ├── test_symbol_detector.py          # Full M1→M2 pipeline test script
│   ├── visualize_predictions.py         # Draws predicted boxes/labels on images
│   ├── run_preview.py                   # Quick preview of model outputs
│   └── filter_ambiguous_symbols.py      # OCR-based false positive filter (R / T)
│
├── Tools & GUIs
│   ├── pipeline_gui.py                  # GUI to run the full pipeline interactively
│   └── inspector_gui.py                 # GUI to inspect dataset/predictions
│
├── Reporting
│   └── generate_latex_tables.py         # Generates LaTeX tables for the thesis report
│
├── weights/                             # Pretrained model weights
├── .gitignore
└── README.md
```

---

## Limitations & Next Steps

M2 occasionally confuses `radius`/`perpendicularity` with the letters "R"/"T" in noisy scans — an OCR filter helps but doesn't fully solve it. Real labeled data is still limited for some symbol classes, so performance there leans on synthetic data quality.

Next: more real-world data, a lighter model for edge deployment, and a simple web demo (Streamlit/Gradio) for interactive testing.

---

## Tech Stack

Python · PyTorch · Ultralytics YOLO · OpenCV · NumPy · EasyOCR · LabelMe

---

## Author

**Eduardo Pane**
[GitHub](https://github.com/Eddicpp) · [Kaggle](https://kaggle.com/eduardopane)
