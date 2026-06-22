# BlueprintCV

**BlueprintCV** is a two-stage computer vision pipeline for automatic detection and recognition of **GD&T (Geometric Dimensioning and Tolerancing)** annotations on scanned industrial technical drawings (blueprints).

The goal is to automatically localize every dimensional callout in a blueprint and classify the GD&T symbol it contains — replacing slow, error-prone manual transcription with an automated detection pipeline.

---

## Demo

> *Add a screenshot or short GIF here showing a blueprint before/after running through the full M1 → M2 pipeline.*

```
[ original blueprint ]   →   [ M1: quote regions detected ]   →   [ M2: GD&T symbols classified ]
```

---

## Objective

Industrial blueprints contain dozens to hundreds of dimensional annotations, each one possibly carrying a GD&T symbol (flatness, perpendicularity, position, diameter, surface finish, etc.). Reading and transcribing these by hand is slow and inconsistent.

BlueprintCV addresses this with a **cascaded two-model detection pipeline**:

- **M1 — Quote Detector**: localizes every dimensional annotation region in the blueprint
- **M2 — Symbol Detector**: detects and classifies the GD&T symbol inside each region found by M1

This is an **object detection** task end-to-end, built around two specialized YOLO11 models rather than a single general-purpose detector, since the two sub-problems (finding annotations vs. recognizing symbols) have very different visual scales and failure modes.

---

## Dataset

Real annotated blueprints were combined with a large volume of **synthetically generated training data**, since real labeled examples were scarce for several GD&T symbol classes.

**M1 (quotes):**
- Real scanned blueprints, manually labeled (LabelMe)
- Custom augmentation: gaussian-sampled parallel lines simulating faded strokes and cluttered backgrounds
- Synthetic blueprint-like layouts added to improve generalization
- Dataset expanded with lower-quality scans, heterogeneous drawings, and a limited set of hand-drawn sketches

**M2 (symbols):**
- 13 GD&T symbol classes: `diameter`, `radius`, `surface_finish`, `concentricity`, `cylindricity`, `position`, `flatness`, `perpendicularity`, `total_runout`, `circular_runout`, `slope`, `conical_taper`, `symmetry`
- Procedural synthetic generator producing realistic feature control frames across 12 distinct layouts (single-symbol, multi-symbol stacked/inline, large+small grouped symbols)
- Symbols rendered with gaussian noise on shape and position, to simulate real-world imperfections (manufacturers rarely draw perfectly symmetric/centered symbols, especially by hand)
- "Chaos" synthetic samples (grids and aligned rows of symbols/characters) to harden the model against densely packed, cluttered drawings
- **Adaptive augmentation**: the augmentation factor scales inversely with the number of real examples available per class — classes with abundant real data get little to no synthetic augmentation, scarce classes get up to 10x

| Split | Source |
|---|---|
| Train | Real (labeled) + adaptive synthetic augmentation + chaos samples |
| Test  | Held-out real blueprints, manually labeled |

---

## Model / Architecture

| Component | Architecture | Notes |
|---|---|---|
| M1 — Quote Detector | YOLO11s | Single class (`quote`), trained on real + augmented data |
| M2 — Symbol Detector | YOLO11s | 13 classes, trained on real + synthetic data |

YOLO11 was chosen over alternatives (Faster R-CNN, DETR-style detectors) for its strong accuracy/speed trade-off and straightforward fine-tuning on small/medium custom datasets — both relevant given the project's compute constraints (local GPU + cloud training).

---

## Pipeline

```
1. Input            → scanned blueprint image (.jpg / .png)
2. M1 inference      → detect all quote bounding boxes
3. Box merging       → overlapping duplicate detections merged (overlap_ratio > 60%)
4. Crop + context    → each quote box cropped with +40% context margin
5. M2 inference       → detect & classify GD&T symbol inside each crop
6. OCR post-filter    → ambiguous symbols (radius "R" / perpendicularity "T") verified
                        against OCR'd text in the quote cell to reject false positives
7. Output            → structured list of {quote_box, symbol, confidence} per blueprint
```

---

## Results

| Model | mAP@50 | Target | Notes |
|---|---|---|---|
| M1 — Quote Detector | **0.934** | ≥ 0.90 | Trained locally, real + augmented dataset |
| M2 — Symbol Detector | **0.852** | ≥ 0.80 | Trained on synthetic (12 layouts) + real data |

On a held-out test set of 180 real blueprints:

| Metric | Value |
|---|---|
| Precision | 0.933 |
| Recall | 0.909 |
| F1 | 0.921 |
| True Positives | 3198 |
| False Positives | 230 |
| False Negatives | 320 |

> *Add confusion matrix / PR curve images here.*

---

## Example Outputs

> *Add side-by-side examples here: correct detections, and a couple of failure cases (e.g. symbols missed in very low-contrast scans, or ambiguous R/T letters before OCR filtering).*

---

## Installation

```bash
git clone https://github.com/<your-username>/BlueprintCV.git
cd BlueprintCV
pip install -r requirements.txt
```

Pretrained weights (`best_m1.pt`, `best_m2.pt`) should be placed under `weights/` — see `requirements.txt` for the exact `ultralytics` / `opencv-python` versions used.

---

## Usage

```bash
# Run the full M1 → M2 pipeline on a single blueprint
python test_symbol_detector.py --blueprint path/to/image.jpg

# Run on a folder of blueprints
python test_symbol_detector.py --dir path/to/folder/

# Run only M1 (quote detector)
python test_m1.py --dir path/to/folder/

# Build the synthetic training dataset for M2
python build_dataset_simboli.py

# Train M1 / M2 locally
python train_m1_local.py
python train_m2_local.py
```

---

## Repository Structure

```
BlueprintCV/
├── build_dataset_simboli.py     # Synthetic dataset generator for M2
├── augment_and_split.py         # Real-data augmentation + train/val/test split
├── filter_labels.py             # YOLO label filtering utility
├── merge_datasets.py            # Dataset merging utility
├── prelabeling_gui.py           # Annotation GUI (pre-labeling + correction modes)
├── filter_ambiguous_symbols.py  # OCR-based false positive filter (R / T)
├── train_m1_local.py            # M1 training (local GPU)
├── train_m1_kaggle.py           # M1 training (Kaggle/cloud)
├── train_m2_local.py            # M2 training (local GPU)
├── train_m2_kaggle.py           # M2 training (Kaggle/cloud)
├── test_m1.py                   # M1 evaluation script
├── test_symbol_detector.py      # Full M1→M2 pipeline test script
├── weights/                     # Pretrained model weights
└── README.md
```

---

## Limitations & Future Work

- **M2 still confuses some ambiguous symbols** (`radius` vs. the letter "R", `perpendicularity` vs. the letter "T") in low-quality scans; an OCR-based post-filter mitigates but does not fully eliminate this
- **Real labeled data remains limited** for several GD&T classes — performance on those classes still depends heavily on synthetic augmentation quality
- Hand-drawn sketches are only partially represented in training data; robustness on fully hand-drawn blueprints is limited
- Possible improvements:
  - Larger real-world dataset, especially for underrepresented symbol classes
  - Lighter model variants (YOLO11n) for faster/edge deployment
  - End-to-end hyperparameter tuning across both M1 and M2 jointly
  - Web-based demo (Streamlit/Gradio) for interactive testing
  - Direct integration with PLM/ERP systems for structured output ingestion

---

## Tech Stack

- Python
- PyTorch
- Ultralytics YOLO11
- OpenCV
- NumPy
- EasyOCR
- LabelMe (annotation)

---

## Author

**Eduardo Pane**
[GitHub](https://github.com/Eddicpp) · [Kaggle](https://kaggle.com/eduardopane)
