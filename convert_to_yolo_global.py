"""
Convertitore GLOBALE LabelMe JSON → YOLO format
Raccoglie dati da tutti i progetti in una cartella radice,
include SOLO le immagini che hanno una label corrispondente,
e produce un unico split train/val/test.
Ridimensiona le immagini durante la copia per risparmiare spazio.

Struttura attesa:
    radice/
    ├── progetto_1/
    │   ├── immagini/
    │   └── labels/
    ├── progetto_2/
    │   ├── immagini/
    │   └── labels/
    ...

Uso:
    Metti questo script nella cartella radice e lancia:
    python convert_to_yolo_global.py
"""

import json
import cv2
import random
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

ROOT_DIR    = "."              # cartella radice per il TRAINING
OUTPUT_DIR  = "./dataset_yolo" # dataset finale con train/val/test
CLASSES     = ["border", "table", "quote"]
RANDOM_SEED = 42
MAX_SIDE    = 1600             # ridimensiona il lato più lungo a questo valore
                               # metti None per non ridimensionare

# Cartelle dedicate per val e test (aziende diverse dal training)
VAL_DIR  = r"C:\Users\Eduardo Pane\OneDrive - Giustizia\Desktop\drive-download-20260407T124738Z-3-001\EA6720A517-043-21-"
TEST_DIR = r"C:\Users\Eduardo Pane\OneDrive - Giustizia\Desktop\drive-download-20260407T124738Z-3-001\EA6720A543-103-22-"

# ─────────────────────────────────────────
# CONVERSIONE
# ─────────────────────────────────────────

def points_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h):
    x_center = (x_min + x_max) / 2 / img_w
    y_center  = (y_min + y_max) / 2 / img_h
    width     = (x_max - x_min) / img_w
    height    = (y_max - y_min) / img_h
    x_center  = max(0.0, min(1.0, x_center))
    y_center  = max(0.0, min(1.0, y_center))
    width     = max(0.0, min(1.0, width))
    height    = max(0.0, min(1.0, height))
    return x_center, y_center, width, height


def convert_json(json_path: Path):
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ✗ Errore lettura {json_path.name}: {e}")
        return None

    img_w  = data.get("imageWidth")
    img_h  = data.get("imageHeight")
    shapes = data.get("shapes", [])

    if not img_w or not img_h:
        print(f"  ✗ Dimensioni immagine mancanti in {json_path.name}")
        return None

    lines  = []
    stats  = {c: 0 for c in CLASSES}
    stats["unknown"] = 0

    for shape in shapes:
        label      = shape.get("label", "").lower().strip()
        shape_type = shape.get("shape_type", "")
        points     = shape.get("points", [])

        if label not in CLASSES:
            stats["unknown"] += 1
            continue
        if shape_type not in ("rectangle", "polygon"):
            continue
        if len(points) < 2:
            continue

        x_min, y_min, x_max, y_max = points_to_bbox(points)

        if x_max <= x_min or y_max <= y_min:
            continue

        class_id        = CLASSES.index(label)
        x_c, y_c, w, h = bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)
        lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        stats[label] += 1

    return lines, img_w, img_h, stats


# ─────────────────────────────────────────
# COPIA CON RIDIMENSIONAMENTO
# ─────────────────────────────────────────

def copy_and_resize(img_path: Path, dest_path: Path):
    """
    Legge l'immagine, la ridimensiona se necessario e la salva.
    Le label non cambiano — usano coordinate normalizzate.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False

    if MAX_SIDE is not None:
        h, w = img.shape[:2]
        if max(h, w) > MAX_SIDE:
            scale = MAX_SIDE / max(h, w)
            img   = cv2.resize(img,
                               (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(dest_path), img)
    return True


# ─────────────────────────────────────────
# RACCOLTA GLOBALE
# ─────────────────────────────────────────

def collect_all_pairs(root: Path, exclude_dirs: list = None):
    """
    Scansiona tutte le sottocartelle progetto escludendo quelle
    riservate a val e test.
    """
    pairs    = []
    missing  = 0
    exclude  = [Path(d).resolve() for d in (exclude_dirs or [])]

    projects = sorted([d for d in root.iterdir() if d.is_dir()
                       and (d / "labels").exists()
                       and (d / "immagini").exists()
                       and d.resolve() not in exclude])

    print(f"Progetti trovati per training: {len(projects)}\n")

    for project in projects:
        labels_dir   = project / "labels"
        immagini_dir = project / "immagini"
        json_files   = sorted(labels_dir.glob("*.json"))

        project_pairs = 0
        for json_path in json_files:
            stem     = json_path.stem
            img_path = immagini_dir / f"{stem}.png"
            if not img_path.exists():
                img_path = immagini_dir / f"{stem}.jpg"
            if not img_path.exists():
                missing += 1
                continue
            pairs.append((json_path, img_path))
            project_pairs += 1

        print(f"  {project.name:40s}  {project_pairs} coppie label+immagine")

    print(f"\nTotale coppie training: {len(pairs)}")
    print(f"Label senza immagine:   {missing}")
    return pairs


def collect_pairs_from_dir(folder: Path):
    """
    Raccoglie coppie (json, immagine) da una singola cartella progetto.
    Usata per val e test.
    """
    pairs  = []
    labels_dir   = folder / "labels"
    immagini_dir = folder / "immagini"

    if not labels_dir.exists() or not immagini_dir.exists():
        print(f"  ✗ Struttura non trovata in {folder}")
        return pairs

    for json_path in sorted(labels_dir.glob("*.json")):
        stem     = json_path.stem
        img_path = immagini_dir / f"{stem}.png"
        if not img_path.exists():
            img_path = immagini_dir / f"{stem}.jpg"
        if img_path.exists():
            pairs.append((json_path, img_path))

    print(f"  {folder.name:40s}  {len(pairs)} coppie")
    return pairs


# ─────────────────────────────────────────
# SPLIT E COPIA
# ─────────────────────────────────────────

def build_dataset(train_pairs, val_pairs, test_pairs, output_dir: Path):
    random.seed(RANDOM_SEED)
    random.shuffle(train_pairs)

    splits = {
        "train": train_pairs,
        "val":   val_pairs,
        "test":  test_pairs,
    }

    global_stats = {c: 0 for c in CLASSES}
    global_stats["unknown"] = 0
    total_ann  = 0
    total_skip = 0
    resized    = 0

    for split, split_pairs in splits.items():
        img_out   = output_dir / split / "images"
        label_out = output_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {split} ({len(split_pairs)} immagini)...")

        for json_path, img_path in split_pairs:
            result = convert_json(json_path)
            if result is None:
                continue

            lines, orig_w, orig_h, stats = result

            if not lines:
                total_skip += 1
                continue

            # Copia e ridimensiona
            dest = img_out / img_path.name
            if not copy_and_resize(img_path, dest):
                total_skip += 1
                continue

            if MAX_SIDE and max(orig_w, orig_h) > MAX_SIDE:
                resized += 1

            # Scrivi label YOLO
            txt_path = label_out / f"{img_path.stem}.txt"
            with open(txt_path, "w") as f:
                f.write("\n".join(lines))

            total_ann += len(lines)
            for c in CLASSES:
                global_stats[c] += stats.get(c, 0)
            global_stats["unknown"] += stats.get("unknown", 0)

    return splits, global_stats, total_ann, total_skip, resized


# ─────────────────────────────────────────
# data.yaml
# ─────────────────────────────────────────

def write_yaml(output_dir: Path):
    content = f"""path: {output_dir.resolve()}
train: train/images
val:   val/images
test:  test/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(content)
    return yaml_path


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    root       = Path(ROOT_DIR).resolve()
    output_dir = Path(OUTPUT_DIR)
    val_dir    = Path(VAL_DIR)
    test_dir   = Path(TEST_DIR)

    print("=" * 60)
    print("SINTESI: Genesi — Conversione dataset globale")
    print("=" * 60)
    print(f"Training:   {root}")
    print(f"Val:        {val_dir}")
    print(f"Test:       {test_dir}")
    print(f"Output:     {output_dir.resolve()}")
    print(f"Classi:     {CLASSES}")
    print(f"Max side:   {MAX_SIDE}px\n")

    # Training — tutte le cartelle ESCLUSE val e test
    print("── TRAINING ──")
    train_pairs = collect_all_pairs(root,
                                    exclude_dirs=[str(val_dir), str(test_dir)])

    # Validation — cartella dedicata
    print("\n── VALIDATION ──")
    val_pairs = collect_pairs_from_dir(val_dir)

    # Test — cartella dedicata
    print("\n── TEST ──")
    test_pairs = collect_pairs_from_dir(test_dir)

    if not train_pairs:
        print("\nNessuna coppia di training trovata.")
        exit(1)

    splits, stats, total_ann, total_skip, resized = build_dataset(
        train_pairs, val_pairs, test_pairs, output_dir)

    yaml_path = write_yaml(output_dir)

    print("\n" + "=" * 60)
    print("RIEPILOGO FINALE")
    print("=" * 60)
    print(f"  train: {len(train_pairs)} immagini  ← mix aziende")
    print(f"  val:   {len(val_pairs)} immagini  ← {val_dir.name}")
    print(f"  test:  {len(test_pairs)} immagini  ← {test_dir.name}")
    print(f"\nRidimensionate (>{MAX_SIDE}px): {resized}")
    print(f"Annotazioni totali:         {total_ann}")
    print(f"Immagini senza ann.:        {total_skip}  (escluse)")
    print(f"\nPer classe:")
    for c in CLASSES:
        print(f"  {c:10s}: {stats[c]}")
    if stats["unknown"] > 0:
        print(f"  {'unknown':10s}: {stats['unknown']}  ← label non riconosciute")
    print(f"\ndata.yaml:  {yaml_path}")
    print(f"\nPronti per il training → python train_yolov8.py")