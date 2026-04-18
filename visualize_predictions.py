"""
SINTESI: Genesi — Visualizzazione predizioni sul test set
Salva ogni immagine con i bounding box predetti colorati per classe,
confrontati con le ground truth.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

WEIGHTS    = "./runs/detect/sintesi_genesi/yolo11s_run3_augmented/weights/best.pt"
OUTPUT_DIR = "./runs/detect/sintesi_genesi/yolo11s_run3_augmented/test_predictions"
TEST_IMAGES = "./dataset_yolo/test/images"
TEST_LABELS = "./dataset_yolo/test/labels"
CLASSES      = ["border", "table", "quote"]
CONF_THRESH  = 0.422        # soglia ottimale dal F1-curve
IOU_THRESH   = 0.45
IMG_SIZE     = 1024
MAX_IMAGES   = None         # None = tutte, oppure un numero es. 50

# Colori BGR per OpenCV — predizioni
PRED_COLORS = {
    "border": (209, 144, 74),   # blu
    "table":  (56, 168, 232),   # arancione
    "quote":  (122, 184, 93),   # verde
}
# Colori ground truth (più chiari)
GT_COLORS = {
    "border": (255, 200, 150),
    "table":  (150, 220, 255),
    "quote":  (180, 230, 160),
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS  = 2

# ─────────────────────────────────────────
# UTILITÀ
# ─────────────────────────────────────────

def yolo_to_abs(cx, cy, w, h, img_w, img_h):
    """Converte coordinate YOLO normalizzate in pixel assoluti"""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def draw_box(img, x1, y1, x2, y2, label, color, conf=None):
    """Disegna un bounding box con etichetta"""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, THICKNESS)
    text = label if conf is None else f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
    # Sfondo etichetta
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4),
                FONT, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)


def load_ground_truth(label_path: Path, img_w: int, img_h: int):
    """Legge le label YOLO e restituisce lista di (class_name, x1, y1, x2, y2)"""
    gt = []
    if not label_path.exists():
        return gt
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1, y1, x2, y2 = yolo_to_abs(cx, cy, w, h, img_w, img_h)
            if cls_id < len(CLASSES):
                gt.append((CLASSES[cls_id], x1, y1, x2, y2))
    return gt


# ─────────────────────────────────────────
# INFERENZA E SALVATAGGIO
# ─────────────────────────────────────────

def run_inference():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sottocartelle per classe
    for cls in CLASSES + ["all"]:
        (output_dir / cls).mkdir(exist_ok=True)

    model      = YOLO(WEIGHTS)
    img_paths  = sorted(Path(TEST_IMAGES).glob("*.png"))
    img_paths += sorted(Path(TEST_IMAGES).glob("*.jpg"))

    if MAX_IMAGES:
        img_paths = img_paths[:MAX_IMAGES]

    print(f"Immagini da processare: {len(img_paths)}")
    print(f"Output in: {output_dir}\n")

    stats = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in CLASSES}

    for idx, img_path in enumerate(img_paths):
        img    = cv2.imread(str(img_path))
        if img is None:
            print(f"  ✗ Impossibile leggere {img_path.name}")
            continue

        img_h, img_w = img.shape[:2]
        canvas       = img.copy()

        # ── Ground truth (tratteggiato) ──
        label_path = Path(TEST_LABELS) / f"{img_path.stem}.txt"
        gt_boxes   = load_ground_truth(label_path, img_w, img_h)

        for cls_name, x1, y1, x2, y2 in gt_boxes:
            color = GT_COLORS.get(cls_name, (200, 200, 200))
            # Rettangolo tratteggiato — disegno con linee corte
            for i in range(x1, x2, 10):
                cv2.line(canvas, (i, y1), (min(i+5, x2), y1), color, 1)
                cv2.line(canvas, (i, y2), (min(i+5, x2), y2), color, 1)
            for i in range(y1, y2, 10):
                cv2.line(canvas, (x1, i), (x1, min(i+5, y2)), color, 1)
                cv2.line(canvas, (x2, i), (x2, min(i+5, y2)), color, 1)

        # ── Predizioni ──
        results     = model.predict(
            source  = str(img_path),
            imgsz   = IMG_SIZE,
            conf    = CONF_THRESH,
            iou     = IOU_THRESH,
            device  = 0,
            verbose = False,
        )

        classes_found = set()
        for r in results:
            for box in r.boxes:
                cls_id   = int(box.cls)
                conf     = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else "unknown"
                color    = PRED_COLORS.get(cls_name, (128, 128, 128))

                draw_box(canvas, x1, y1, x2, y2, cls_name, color, conf)
                classes_found.add(cls_name)

        # ── Legenda ──
        legend_y = 30
        for cls_name, color in PRED_COLORS.items():
            cv2.rectangle(canvas, (10, legend_y - 12), (28, legend_y + 4), color, -1)
            cv2.putText(canvas, f"PRED {cls_name}", (34, legend_y),
                        FONT, 0.55, color, 1, cv2.LINE_AA)
            legend_y += 24
        for cls_name, color in GT_COLORS.items():
            cv2.putText(canvas, f"GT   {cls_name}", (34, legend_y),
                        FONT, 0.55, color, 1, cv2.LINE_AA)
            legend_y += 24

        # ── Salvataggio ──
        # 1. Immagine completa nella cartella "all"
        out_all = output_dir / "all" / img_path.name
        cv2.imwrite(str(out_all), canvas)

        # 2. Copia nelle cartelle delle classi trovate
        for cls_name in classes_found:
            out_cls = output_dir / cls_name / img_path.name
            cv2.imwrite(str(out_cls), canvas)

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1:4d}/{len(img_paths)}] {img_path.name} "
                  f"— classi rilevate: {', '.join(sorted(classes_found)) or 'nessuna'}")

    print(f"\n✓ Completato. Immagini salvate in: {output_dir}")
    print(f"  all/       → tutte le immagini")
    for cls in CLASSES:
        n = len(list((output_dir / cls).glob("*")))
        print(f"  {cls}/{'':8s} → {n} immagini con almeno una {cls} rilevata")

    print("\nLegenda colori:")
    print("  Box SOLIDO    = predizione del modello (con confidence)")
    print("  Box TRATTEGGIATO = ground truth (label vera)")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    run_inference()
