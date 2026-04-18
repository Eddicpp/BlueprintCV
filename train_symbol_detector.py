"""
SINTESI: Genesi — Training detector simboli quote
Allena YOLO11n per rilevare e classificare i simboli GD&T
nei crop delle quote.
"""

import torch
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

DATASET_YAML = "./dataset_simboli_detection/data.yaml"
MODEL_SIZE   = "yolo11n"
PROJECT      = "sintesi_genesi"
RUN_NAME     = "simboli_detector_run1"
EPOCHS       = 100
IMG_SIZE     = 640
BATCH_SIZE   = 16
PATIENCE     = 20

# ─────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("SINTESI: Genesi — Training detector simboli")
    print("=" * 55)

    if torch.cuda.is_available():
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU:     {gpu} ({vram:.1f} GB)")
    else:
        print("ATTENZIONE: GPU non disponibile")

    print(f"Dataset: {DATASET_YAML}")
    print(f"Modello: {MODEL_SIZE}")
    print(f"Epoche:  {EPOCHS}  |  ImgSize: {IMG_SIZE}  |  Batch: {BATCH_SIZE}\n")

    model = YOLO(f"{MODEL_SIZE}.pt")
    model.train(
        data        = DATASET_YAML,
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
        amp         = False,   # disabilita AMP per GTX 1660
    )

    run_dir = Path(PROJECT) / RUN_NAME
    print(f"\n✓ Training completato → {run_dir}")
    print(f"  Weights: {run_dir}/weights/best.pt")
    print(f"\nOra testa con:")
    print(f"  python test_symbol_detector.py")
