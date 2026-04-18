"""
SINTESI: Genesi — Conversione dataset simboli LabelMe → YOLO detection
Converte i JSON LabelMe dei crop delle quote in formato YOLO detection.
Le label contengono la classe specifica del simbolo + bounding box.

Output: dataset_simboli_detection/
  train/images/ + train/labels/
  val/images/   + val/labels/
  test/images/  + test/labels/
  data.yaml
"""

import json
import cv2
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

LABELED_DIR  = "./quote_per_labeling"
OUTPUT_DIR   = "./dataset_simboli_detection"

SYMBOL_CLASSES = [
    "diameter",
    "radius",
    "angle",
    "surface_finish",
    "concentricity",
    "cylindricity",
    "position",
    "flatness",
    "perpendicularity",
    "total_runout",
    "circular_runout",
    "slope",
    "conical_taper",
    "symmetry",
    "surface_profile",
]

TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
RANDOM_SEED  = 42
MAX_SIDE     = 640   # ridimensiona crop troppo grandi

# ─────────────────────────────────────────
# UTILITÀ
# ─────────────────────────────────────────

def points_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_to_yolo(x1, y1, x2, y2, iw, ih):
    cx = (x1+x2)/2/iw
    cy = (y1+y2)/2/ih
    w  = (x2-x1)/iw
    h  = (y2-y1)/ih
    return (max(0,min(1,cx)), max(0,min(1,cy)),
            max(0,min(1,w)),  max(0,min(1,h)))


def copy_and_resize(src: Path, dst: Path):
    img = cv2.imread(str(src))
    if img is None:
        return False, 1, 1
    h, w = img.shape[:2]
    if max(h,w) > MAX_SIDE:
        scale = MAX_SIDE / max(h,w)
        img   = cv2.resize(img, (int(w*scale), int(h*scale)),
                           interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img)
    return True, img.shape[1], img.shape[0]


# ─────────────────────────────────────────
# RACCOLTA COPPIE (json, immagine)
# ─────────────────────────────────────────

def collect_pairs(labeled_dir: Path):
    pairs  = []
    skipped = 0
    stats  = defaultdict(int)

    json_files = sorted(labeled_dir.glob("*.json"))
    print(f"JSON trovati: {len(json_files)}")

    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            skipped += 1
            continue

        shapes = data.get("shapes", [])
        if not shapes:
            skipped += 1
            continue

        # Trova immagine corrispondente
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = labeled_dir / (json_path.stem + ext)
            if p.exists():
                img_path = p
                break

        if img_path is None:
            skipped += 1
            continue

        # Converti shapes in label YOLO
        valid_shapes = []
        for shape in shapes:
            label = shape.get("label","").strip().lower().replace(" ","_")
            if label not in SYMBOL_CLASSES:
                continue
            points = shape.get("points", [])
            if len(points) < 2:
                continue
            valid_shapes.append((label, points))

        if not valid_shapes:
            skipped += 1
            continue

        for label, _ in valid_shapes:
            stats[label] += 1

        pairs.append((json_path, img_path, valid_shapes))

    print(f"Coppie valide: {len(pairs)}  |  Saltate: {skipped}")
    print("\nDistribuzione classi:")
    for cls, n in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {cls:25s}: {n}")

    return pairs


# ─────────────────────────────────────────
# BUILD DATASET
# ─────────────────────────────────────────

def build_dataset(pairs):
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    n       = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train+n_val],
        "test":  pairs[n_train+n_val:],
    }

    total_ann  = 0
    total_imgs = 0

    for split, split_pairs in splits.items():
        img_out = Path(OUTPUT_DIR) / split / "images"
        lbl_out = Path(OUTPUT_DIR) / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        print(f"\n{split}: {len(split_pairs)} immagini")

        for json_path, img_path, shapes in split_pairs:
            # Copia e ridimensiona immagine
            dst_img = img_out / img_path.name
            ok, iw, ih = copy_and_resize(img_path, dst_img)
            if not ok:
                continue

            # Genera label YOLO
            # Le coordinate dei JSON sono riferite all'immagine originale
            # Leggi dimensioni originali dal JSON
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            orig_w = data.get("imageWidth",  iw)
            orig_h = data.get("imageHeight", ih)

            lines = []
            for label, points in shapes:
                cls_id = SYMBOL_CLASSES.index(label)
                x1,y1,x2,y2 = points_to_bbox(points)

                # Adatta coordinate se immagine è stata ridimensionata
                sx = iw / orig_w
                sy = ih / orig_h
                x1 *= sx; x2 *= sx
                y1 *= sy; y2 *= sy

                cx,cy,bw,bh = bbox_to_yolo(x1,y1,x2,y2,iw,ih)
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                total_ann += 1

            # Salva label
            lbl_path = lbl_out / f"{img_path.stem}.txt"
            with open(lbl_path, "w") as f:
                f.write("\n".join(lines))

            total_imgs += 1

    return total_imgs, total_ann


# ─────────────────────────────────────────
# data.yaml
# ─────────────────────────────────────────

def write_yaml():
    content = f"""path: {Path(OUTPUT_DIR).resolve()}
train: train/images
val:   val/images
test:  test/images

nc: {len(SYMBOL_CLASSES)}
names: {SYMBOL_CLASSES}
"""
    yaml_path = Path(OUTPUT_DIR) / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"\ndata.yaml → {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("SINTESI: Genesi — Conversione simboli → YOLO detection")
    print("=" * 55)
    print(f"Input:  {LABELED_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Classi: {len(SYMBOL_CLASSES)}\n")

    pairs = collect_pairs(Path(LABELED_DIR))

    if not pairs:
        print("Nessuna coppia trovata.")
        exit(1)

    total_imgs, total_ann = build_dataset(pairs)
    yaml_path = write_yaml()

    print(f"\n✓ Completato!")
    print(f"  Immagini: {total_imgs}")
    print(f"  Annotazioni: {total_ann}")
    print(f"\nPer allenare il detector simboli:")
    print(f"  yolo detect train data={OUTPUT_DIR}/data.yaml "
          f"model=yolo11n.pt epochs=100 imgsz=640 batch=16 "
          f"project=sintesi_genesi name=simboli_detector_run1")
