"""
SINTESI: Genesi — Rimuovi classe angle da dataset_simboli_detection
Elimina tutti i bounding box di classe angle dalle label YOLO.
Le immagini che avevano solo angle vengono eliminate completamente.

Uso:
    python remove_angle_class.py
"""

import cv2
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

DATASET_DIR  = "./dataset_simboli_detection"

SYMBOL_CLASSES = [
    "diameter",        # 0
    "radius",          # 1
    "angle",           # 2  ← da rimuovere
    "surface_finish",  # 3
    "concentricity",   # 4
    "cylindricity",    # 5
    "position",        # 6
    "flatness",        # 7
    "perpendicularity",# 8
    "total_runout",    # 9
    "circular_runout", # 10
    "slope",           # 11
    "conical_taper",   # 12
    "symmetry",        # 13
    "surface_profile", # 14
]

ANGLE_CLS_ID = SYMBOL_CLASSES.index("angle")  # 2

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("SINTESI: Genesi — Rimozione classe angle")
    print("=" * 55)
    print(f"Dataset: {DATASET_DIR}")
    print(f"Classe da rimuovere: angle (id={ANGLE_CLS_ID})\n")

    total_removed_boxes = 0
    total_removed_imgs  = 0
    total_kept_imgs     = 0

    for split in ["train", "val", "test"]:
        lbl_dir = Path(DATASET_DIR) / split / "labels"
        img_dir = Path(DATASET_DIR) / split / "images"

        if not lbl_dir.exists():
            continue

        txts = sorted(lbl_dir.glob("*.txt"))
        print(f"{split}: {len(txts)} label")

        removed_boxes = 0
        removed_imgs  = 0
        kept_imgs     = 0

        for txt in txts:
            lines = txt.read_text().strip().splitlines()
            new_lines = []

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])

                # Salta i box di classe angle
                if cls_id == ANGLE_CLS_ID:
                    removed_boxes += 1
                    continue

                new_lines.append(line)

            if new_lines:
                # Riscrivi label senza angle
                txt.write_text("\n".join(new_lines))
                kept_imgs += 1
            else:
                # Nessun box rimasto — elimina label e immagine
                txt.unlink()
                for ext in [".jpg", ".png", ".jpeg"]:
                    img = img_dir / (txt.stem + ext)
                    if img.exists():
                        img.unlink()
                        break
                removed_imgs += 1

        print(f"  Box angle rimossi:  {removed_boxes}")
        print(f"  Immagini eliminate: {removed_imgs} (erano solo angle)")
        print(f"  Immagini rimaste:   {kept_imgs}\n")

        total_removed_boxes += removed_boxes
        total_removed_imgs  += removed_imgs
        total_kept_imgs     += kept_imgs

    print("─" * 55)
    print(f"Totale box angle rimossi:  {total_removed_boxes}")
    print(f"Totale immagini eliminate: {total_removed_imgs}")
    print(f"Totale immagini rimaste:   {total_kept_imgs}")
    print(f"\n✓ Fatto — riallena con: python train_symbol_detector.py")
