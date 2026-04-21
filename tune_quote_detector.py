"""
SINTESI: Genesi — Hyperparameter Tuning Quote Detector (M1)
Usa il tuner integrato di YOLO per trovare i migliori iperparametri
per il detector di quote.

Dopo il tuning rilancia train_yolov8.py con i parametri trovati.
"""

import torch
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

DATASET_YAML = "./dataset_finale/data.yaml"
MODEL_SIZE   = "yolo11s"
PROJECT      = "sintesi_genesi"
RUN_NAME     = "yolo11s_tune2"
EPOCHS       = 50
ITERATIONS   = 30
IMG_SIZE     = 640
BATCH_SIZE   = 8
AMP          = False

# ─────────────────────────────────────────
# TUNING
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("SINTESI: Genesi — Hyperparameter Tuning M1")
    print("=" * 55)

    if torch.cuda.is_available():
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU:        {gpu} ({vram:.1f} GB)")
    else:
        print("ATTENZIONE: GPU non disponibile")

    print(f"Dataset:    {DATASET_YAML}")
    print(f"Modello:    {MODEL_SIZE}")
    print(f"Trial:      {ITERATIONS} × {EPOCHS} epoche\n")

    model = YOLO(f"{MODEL_SIZE}.pt")

    model.tune(
        data       = DATASET_YAML,
        epochs     = EPOCHS,
        iterations = ITERATIONS,
        imgsz      = IMG_SIZE,
        batch      = BATCH_SIZE,
        amp        = AMP,
        project    = PROJECT,
        name       = RUN_NAME,
        exist_ok   = True,
        device     = 0,
        plots      = True,
        save       = True,
        optimizer  = "AdamW",
    )

    # I migliori parametri vengono salvati in:
    tune_dir = Path(PROJECT) / RUN_NAME
    best_cfg = tune_dir / "best_hyperparameters.yaml"

    print(f"\n✓ Tuning completato")
    print(f"  Risultati: {tune_dir}")
    if best_cfg.exists():
        print(f"\nMigliori iperparametri ({best_cfg}):")
        print(best_cfg.read_text())
    print(f"\nOra riallena con i parametri ottimali:")
    print(f"  python train_yolov8.py")
