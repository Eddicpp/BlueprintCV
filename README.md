# BlueprintCV

**BlueprintCV** automatically detects and classifies GD&T (Geometric Dimensioning and Tolerancing) symbols on scanned industrial blueprints, turning manual transcription into a fast, repeatable detection pipeline.

---

## Demo

> *Add a screenshot here: a real blueprint with M1 quote boxes (gray) and M2 symbol boxes (colored) overlaid — output of `run_pipeline.py`.*

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
- **M2 — Symbol Detector**: classifies the GD&T symbol inside each region M1 finds

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
| M2 — Symbol Detector | YOLO11s | 13 classes, real + synthetic data |

YOLO11 was chosen for its accuracy/speed trade-off and ease of fine-tuning on a small/medium custom dataset, with both local-GPU and cloud training as options.

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

> *Add confusion matrix image here.*

---

## Example Outputs

> *Add 2-3 side-by-side examples: a correct detection, and a failure case (e.g. a symbol missed in a low-contrast scan, or an R/T ambiguity caught by the OCR filter).*

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
python run_pipeline.py --image path/to/image.jpg

# Run on a folder of blueprints
python run_pipeline.py --dir path/to/folder/

# Run M1 only
python evaluate_m1.py --dir path/to/folder/

# Build the synthetic training dataset for M2
python generate_symbol_dataset.py

# Train M1 / M2 locally
python train_m1.py
python train_m2.py
```

---

## Repository Structure

```
BlueprintCV/
├── generate_symbol_dataset.py   # Synthetic dataset generator for M2
├── augment_dataset.py           # Real-data augmentation + train/val/test split
├── filter_labels.py             # YOLO label filtering utility
├── merge_datasets.py            # Dataset merging utility
├── labeling_gui.py              # Annotation GUI (pre-labeling + correction modes)
├── filter_ambiguous_symbols.py  # OCR-based false positive filter (R / T)
├── train_m1.py                  # M1 training
├── train_m2.py                  # M2 training
├── evaluate_m1.py               # M1 evaluation script
├── run_pipeline.py              # Full M1→M2 pipeline
├── weights/                     # Pretrained model weights
└── README.md
```

---

## Limitations & Next Steps

M2 occasionally confuses `radius`/`perpendicularity` with the letters "R"/"T" in noisy scans — an OCR filter helps but doesn't fully solve it. Real labeled data is still limited for some symbol classes, so performance there leans on synthetic data quality.

Next: more real-world data, a lighter model for edge deployment, and a simple web demo (Streamlit/Gradio) for interactive testing.

---

## Tech Stack

Python · PyTorch · Ultralytics YOLO11 · OpenCV · NumPy · EasyOCR · LabelMe

---

## Author

**Eduardo Pane**
[GitHub](https://github.com/Eddicpp) · [Kaggle](https://kaggle.com/eduardopane)
