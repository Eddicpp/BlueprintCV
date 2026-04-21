"""
<<<<<<< HEAD
SINTESI: Genesi — Training detector simboli quote
Allena YOLO11n per rilevare e classificare i simboli GD&T
nei crop delle quote.
"""

import torch
=======
SINTESI: Genesi — Training Symbol Detector (Modello 2)
Trova la posizione dei simboli GD&T nei crop delle quote.
NON classifica — dice solo dove sono i simboli.
Il Modello 3 (classificatore) si occupa di riconoscere la classe.

Una quota può avere più simboli → il modello restituisce
tutti i bounding box trovati nel crop.

Dataset: dataset_simboli_detection/
  Classe unica: "symbol" (classe 0)
  Label: bounding box di ogni simbolo nel crop

Pipeline completa:
  Blueprint → [M1 Quote Detector] → crop quota
           → [M2 Symbol Detector] → crop simbolo
           → [M3 Symbol Classifier] → classe simbolo
"""

import cv2
import torch
import random
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

DATASET_YAML = "./dataset_simboli_detection/data.yaml"
MODEL_SIZE   = "yolo11n"
PROJECT      = "sintesi_genesi"
<<<<<<< HEAD
RUN_NAME     = "simboli_detector_run1"
=======
RUN_NAME     = "symbol_detector_run1"
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
EPOCHS       = 100
IMG_SIZE     = 640
BATCH_SIZE   = 16
PATIENCE     = 20
<<<<<<< HEAD
=======
CONF         = 0.20   # soglia bassa — meglio trovare troppo che troppo poco

# Per la preview
QUOTE_DETECTOR = r".\runs\detect\sintesi_genesi\yolo11s_run3_augmented\weights\best.pt"
TEST_DIR       = "./dataset_yolo/test/images"
N_PREVIEW      = 10
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)

# ─────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────

<<<<<<< HEAD
if __name__ == "__main__":
    print("=" * 55)
    print("SINTESI: Genesi — Training detector simboli")
=======
def train():
    print("=" * 55)
    print("SINTESI: Genesi — Training Symbol Detector (M2)")
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
    print("=" * 55)

    if torch.cuda.is_available():
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU:     {gpu} ({vram:.1f} GB)")
    else:
        print("ATTENZIONE: GPU non disponibile")

    print(f"Dataset: {DATASET_YAML}")
    print(f"Modello: {MODEL_SIZE}")
<<<<<<< HEAD
    print(f"Epoche:  {EPOCHS}  |  ImgSize: {IMG_SIZE}  |  Batch: {BATCH_SIZE}\n")
=======
    print(f"Epoche:  {EPOCHS}  |  ImgSize: {IMG_SIZE}  |  Batch: {BATCH_SIZE}")
    print(f"Classe unica: symbol (trova posizione, non classifica)\n")
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)

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
<<<<<<< HEAD
        amp         = False,   # disabilita AMP per GTX 1660
    )

    run_dir = Path(PROJECT) / RUN_NAME
    print(f"\n✓ Training completato → {run_dir}")
    print(f"  Weights: {run_dir}/weights/best.pt")
    print(f"\nOra testa con:")
    print(f"  python test_symbol_detector.py")
=======
        amp         = False,
    )

    run_dir = Path("runs/detect/runs/detect/sintesi_genesi") / RUN_NAME
    print(f"\n✓ Training completato → {run_dir}")
    print(f"  Weights: {run_dir}/weights/best.pt")
    return run_dir


# ─────────────────────────────────────────
# PREVIEW — mostra tutti i simboli trovati per quota
# ─────────────────────────────────────────

def preview(run_dir: Path):
    sym_weights = run_dir / "weights" / "best.pt"
    det_weights = Path(QUOTE_DETECTOR)

    if not sym_weights.exists():
        print(f"Weights non trovati: {sym_weights}")
        return
    if not det_weights.exists():
        print(f"Weights quote non trovati: {det_weights}")
        return

    print(f"\n── Preview Symbol Detector ──")
    det_quotes  = YOLO(str(det_weights))
    det_symbols = YOLO(str(sym_weights))

    test_dir  = Path(TEST_DIR)
    img_paths = sorted(list(test_dir.glob("*.jpg")) +
                       list(test_dir.glob("*.png")))
    random.shuffle(img_paths)
    img_paths = img_paths[:N_PREVIEW]

    out_dir = run_dir / "preview"
    out_dir.mkdir(exist_ok=True)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        vis = img.copy()

        # M1 — trova le quote
        det_res = det_quotes.predict(img, conf=0.40, verbose=False)[0]
        if det_res.boxes is None:
            continue

        print(f"\n  {img_path.name}")

        for box in det_res.boxes:
            if int(box.cls[0]) != 2:  # 2 = quote
                continue
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            x1=max(0,x1); y1=max(0,y1)
            x2=min(iw,x2); y2=min(ih,y2)
            if x2-x1 < 5 or y2-y1 < 5:
                continue

            # Box quota in grigio
            cv2.rectangle(vis,(x1,y1),(x2,y2),(150,150,150),1)

            crop = img[y1:y2, x1:x2]

            # M2 — trova tutti i simboli nel crop
            sym_res = det_symbols.predict(
                crop, imgsz=IMG_SIZE,
                conf=CONF, verbose=False)[0]

            n_symbols = 0
            if sym_res.boxes is not None:
                for sb in sym_res.boxes:
                    sc = float(sb.conf[0])
                    # Coordinate simbolo nel sistema dell'immagine originale
                    sx1,sy1,sx2,sy2 = map(int, sb.xyxy[0].tolist())
                    sx1 = max(0, x1+sx1); sy1 = max(0, y1+sy1)
                    sx2 = min(iw, x1+sx2); sy2 = min(ih, y1+sy2)

                    # Box simbolo in verde
                    cv2.rectangle(vis,(sx1,sy1),(sx2,sy2),(0,200,80),2)
                    cv2.putText(vis,f"sym {sc:.2f}",
                                (sx1,max(sy1-3,8)),
                                font,0.35,(0,200,80),1,cv2.LINE_AA)
                    n_symbols += 1

            print(f"    quota [{x1},{y1},{x2},{y2}] → {n_symbols} simboli trovati")

        cv2.imwrite(str(out_dir/img_path.name), vis)

    print(f"\n  Preview → {out_dir}")
    print("  Box grigio = quota  |  Box verde = simbolo rilevato")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    run_dir = train()
    preview(run_dir)
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
