"""
SINTESI: Genesi — Resize dataset
Ridimensiona tutte le immagini del dataset a dimensione massima gestibile.
Le immagini già nella dimensione target vengono saltate.
Le label YOLO non cambiano (sono già normalizzate).
"""

import cv2
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

# Cartelle da processare — aggiungi tutti i tuoi dataset
DATASET_DIRS = [
    "./dataset_yolo",
    "./dataset_yolo_aug",
    "./dataset_mosaici",
]

# Dimensione massima del lato più lungo
# 2480 = A4 300dpi — gestibile ma pesante
# 1600 = buon compromesso qualità/memoria
# 1280 = leggero, training veloce
MAX_SIDE = 1600

# ─────────────────────────────────────────
# RESIZE
# ─────────────────────────────────────────

def resize_dataset(dataset_dir: str):
    root = Path(dataset_dir)
    if not root.exists():
        print(f"  ✗ {dataset_dir} non trovato, skip")
        return 0, 0

    img_paths = []
    for split in ["train", "val", "test"]:
        img_dir = root / split / "images"
        if img_dir.exists():
            img_paths += list(img_dir.glob("*.png"))
            img_paths += list(img_dir.glob("*.jpg"))

    resized = 0
    skipped = 0

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        max_dim = max(h, w)

        if max_dim <= MAX_SIDE:
            skipped += 1
            continue

        scale = MAX_SIDE / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_r = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(img_path), img_r)
        resized += 1

        if resized % 100 == 0:
            print(f"    {resized} ridimensionate...")

    return resized, skipped


if __name__ == "__main__":
    print("=" * 55)
    print(f"SINTESI: Genesi — Resize dataset (max {MAX_SIDE}px)")
    print("=" * 55)

    total_resized = 0
    total_skipped = 0

    for dataset_dir in DATASET_DIRS:
        print(f"\n{dataset_dir}...")
        r, s = resize_dataset(dataset_dir)
        print(f"  Ridimensionate: {r}  |  Già ok: {s}")
        total_resized += r
        total_skipped += s

    print(f"\n✓ Completato.")
    print(f"  Ridimensionate: {total_resized}")
    print(f"  Già ok:         {total_skipped}")
