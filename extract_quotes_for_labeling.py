"""
SINTESI: Genesi — Estrazione quote per labeling simboli

Estrae tutti i crop delle quote dal dataset annotato
in una cartella flat pronta per il labeling con LabelMe.

Ogni crop viene salvato come immagine singola con nome univoco.
Dopo il labeling, ogni crop avrà un JSON con la classe del simbolo.

Uso:
    python extract_quotes_for_labeling.py
    labelme quote_per_labeling --labels quote_per_labeling/classes.txt --nodata
"""

import cv2
import random
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

TRAIN_IMAGES = "./dataset_yolo/train/images"
TRAIN_LABELS = "./dataset_yolo/train/labels"
OUTPUT_DIR   = "./quote_per_labeling"

# Classi simboli da riconoscere
SYMBOL_CLASSES = [
    "diameter",       # Ø
    "radius",         # R
<<<<<<< HEAD
    "angle",          # arco con trattino °
=======
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
    "tolerance",      # ±
    "depth",          # ↓
    "surface_finish", # segno finitura √
    "counterbore",    # ⌴
    "countersink",    # V rovesciato
    "linear",         # nessun simbolo
]

MAX_CROPS    = 1500   # estrai al massimo N crop (puoi aumentare)
MARGIN       = 8      # margine attorno al bbox in pixel
RANDOM_SEED  = 42

# ─────────────────────────────────────────
# ESTRAZIONE
# ─────────────────────────────────────────

def yolo_to_abs(cx, cy, w, h, iw, ih):
    x1 = int((cx-w/2)*iw); y1 = int((cy-h/2)*ih)
    x2 = int((cx+w/2)*iw); y2 = int((cy+h/2)*ih)
    return max(0,x1), max(0,y1), min(iw,x2), min(ih,y2)


def extract_crops(images_dir: Path, labels_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(list(images_dir.glob("*.png")) +
                       list(images_dir.glob("*.jpg")))
    random.shuffle(img_paths)

    print(f"Scansione {len(img_paths)} immagini...")

    crops_saved = 0
    crop_idx    = 0

    for img_path in img_paths:
        if crops_saved >= MAX_CROPS:
            break

        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        for line in lbl_path.read_text().strip().splitlines():
            if crops_saved >= MAX_CROPS:
                break

            parts = line.strip().split()
            if len(parts) != 5 or int(parts[0]) != 2:
                continue

            cx,cy,bw,bh = map(float, parts[1:])
            x1,y1,x2,y2 = yolo_to_abs(cx,cy,bw,bh,iw,ih)

            if x2-x1 < 20 or y2-y1 < 8:
                continue

            # Aggiungi margine
            rx1 = max(0, x1-MARGIN)
            ry1 = max(0, y1-MARGIN)
            rx2 = min(iw, x2+MARGIN)
            ry2 = min(ih, y2+MARGIN)

            crop = img[ry1:ry2, rx1:rx2].copy()
            if crop.size == 0:
                continue

            # Nome univoco: img_stem + indice crop
            fname = f"{img_path.stem}_q{crop_idx:05d}.jpg"
            cv2.imwrite(str(output_dir / fname), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            crop_idx    += 1
            crops_saved += 1

    return crops_saved


# ─────────────────────────────────────────
# GENERA FILE CLASSI PER LABELME
# ─────────────────────────────────────────

def write_classes_file(output_dir: Path):
    classes_path = output_dir / "classes.txt"
    with open(classes_path, "w") as f:
        for cls in SYMBOL_CLASSES:
            f.write(cls + "\n")
    return classes_path


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    print("=" * 55)
    print("SINTESI: Genesi — Estrazione quote per labeling")
    print("=" * 55)
    print(f"Input:   {TRAIN_IMAGES}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Max:     {MAX_CROPS} crop\n")

    n = extract_crops(
        Path(TRAIN_IMAGES),
        Path(TRAIN_LABELS),
        Path(OUTPUT_DIR)
    )

    classes_path = write_classes_file(Path(OUTPUT_DIR))

    print(f"\n✓ Estratti: {n} crop in {OUTPUT_DIR}")
    print(f"  Classi:   {classes_path}")
    print(f"\nOra lancia LabelMe per il labeling:")
    print(f"\n  labelme {OUTPUT_DIR} --labels {classes_path} --nodata")
    print(f"\nIn LabelMe:")
    print(f"  - Apri ogni immagine")
    print(f"  - Edit → Create Rectangle (o premi 'R')")
    print(f"  - Disegna un box attorno al simbolo e assegna la classe")
    print(f"  - Per le quote senza simbolo assegna classe 'linear'")
    print(f"  - Salva con Ctrl+S")
    print(f"\nDopo il labeling torna qui per generare il dataset.")
