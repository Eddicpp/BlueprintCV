"""
SINTESI: Genesi — Training Symbol Classifier (Modello 3)
Classifica il simbolo GD&T ritagliato dal Symbol Detector (M2).

Pipeline completa:
  Blueprint → [M1 Quote Detector]  → crop quota
           → [M2 Symbol Detector]  → crop simbolo
           → [M3 Symbol Classifier] → classe simbolo

Dataset: dataset_simboli/
  Struttura per cartelle per classe (formato YOLO classification):
    train/diameter/, train/radius/, train/angle/, ...
"""

import cv2
import torch
import numpy as np
import random
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

DATASET_DIR  = "./dataset_simboli"
MODEL_SIZE   = "yolo11s-cls"
PROJECT      = "runs/classify/sintesi_genesi"
RUN_NAME     = "symbol_classifier_run1"
EPOCHS       = 100
IMG_SIZE     = 128
BATCH_SIZE   = 32
PATIENCE     = 15
REJECT_CONF  = 0.50   # sotto questa soglia M3 rifiuta il crop → nessuna classe

# Percorso dove YOLO salva effettivamente
RUN_DIR = Path("runs/classify/runs/classify/sintesi_genesi") / RUN_NAME

SYMBOL_CLASSES = [
    "diameter", "radius", "surface_finish",
    "concentricity", "cylindricity", "position", "flatness",
    "perpendicularity", "total_runout", "circular_runout",
    "slope", "conical_taper", "symmetry", "surface_profile",
    "linear",
    "background",   # rigetta i falsi positivi di M2
]

# ─────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────

def train():
    print("=" * 55)
    print("SINTESI: Genesi — Training Symbol Classifier (M3)")
    print("=" * 55)

    if torch.cuda.is_available():
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU:     {gpu} ({vram:.1f} GB)")
    else:
        print("ATTENZIONE: GPU non disponibile")

    print(f"Dataset: {DATASET_DIR}")
    print(f"Modello: {MODEL_SIZE}")
    print(f"Epoche:  {EPOCHS}  |  ImgSize: {IMG_SIZE}  |  Batch: {BATCH_SIZE}\n")

    # Conta immagini per classe
    train_dir = Path(DATASET_DIR) / "train"
    if train_dir.exists():
        print("Distribuzione classi nel train set:")
        for cls_dir in sorted(train_dir.iterdir()):
            if cls_dir.is_dir():
                n = len(list(cls_dir.glob("*.jpg")) +
                        list(cls_dir.glob("*.png")))
                print(f"  {cls_dir.name:25s}: {n}")
        print()

    model = YOLO(f"{MODEL_SIZE}.pt")
    model.train(
        data        = DATASET_DIR,
        epochs      = EPOCHS,
        imgsz       = IMG_SIZE,
        batch       = BATCH_SIZE,
        patience    = PATIENCE,
        project     = PROJECT,
        name        = RUN_NAME,
        exist_ok    = True,
        device      = 0,
        workers     = 4,
        verbose     = True,
        plots       = True,
    )

    print(f"\n✓ Training completato → {RUN_DIR}")
    return RUN_DIR


# ─────────────────────────────────────────
# VALUTAZIONE
# ─────────────────────────────────────────

def evaluate():
    weights = RUN_DIR / "weights" / "best.pt"
    if not weights.exists():
        print(f"Weights non trovati: {weights}")
        return

    print(f"\nValutazione test set: {weights}")
    model   = YOLO(str(weights))
    metrics = model.val(
        data   = DATASET_DIR,
        split  = "test",
        imgsz  = IMG_SIZE,
        batch  = BATCH_SIZE,
    )

    print(f"\n── Metriche test set ──")
    print(f"  Top-1 accuracy: {metrics.top1:.4f}")
    print(f"  Top-5 accuracy: {metrics.top5:.4f}")

    # Confusion matrix
    test_dir  = Path(DATASET_DIR) / "test"
    cls_dirs  = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    cls_names = [d.name for d in cls_dirs]
    n         = len(cls_names)
    cm        = np.zeros((n, n), dtype=np.int32)

    print("\nCalcolo confusion matrix...")
    for i, cls_dir in enumerate(cls_dirs):
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        for img_path in imgs:
            res = model.predict(str(img_path), imgsz=IMG_SIZE, verbose=False)
            if res and res[0].probs is not None:
                pred_cls = res[0].names[int(res[0].probs.top1)]
                j = cls_names.index(pred_cls) if pred_cls in cls_names else -1
                if j >= 0:
                    cm[i, j] += 1

    # Disegna confusion matrix
    cell   = 55
    margin = 150
    H = margin + n*cell + 20
    W = margin + n*cell + 20
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    font   = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(n):
        for j in range(n):
            val     = cm[i,j]
            row_sum = cm[i].sum()
            ratio   = val/row_sum if row_sum > 0 else 0
            intensity = int(255*(1-ratio))
            color   = (intensity, int(intensity*0.6), 0)
            x1 = margin+j*cell; y1 = margin+i*cell
            cv2.rectangle(canvas,(x1,y1),(x1+cell,y1+cell),color,-1)
            cv2.rectangle(canvas,(x1,y1),(x1+cell,y1+cell),(180,180,180),1)
            txt_c = (255,255,255) if ratio>0.4 else (0,0,0)
            cv2.putText(canvas,str(val),(x1+cell//2-8,y1+cell//2+5),
                        font,0.35,txt_c,1)

    for i,name in enumerate(cls_names):
        short = name[:10]
        cv2.putText(canvas,short,(5,margin+i*cell+cell//2+5),
                    font,0.32,(0,0,0),1)
        cv2.putText(canvas,short,(margin+i*cell+3,margin-5),
                    font,0.28,(0,0,0),1)

    cv2.putText(canvas,"True",(5,margin//2),font,0.5,(0,0,0),1)
    cv2.putText(canvas,"Predicted",(margin,20),font,0.5,(0,0,0),1)

    out_path = RUN_DIR / "confusion_matrix.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"  Confusion matrix → {out_path}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    train()
    evaluate()