"""
SINTESI: Genesi — Generazione e augmentation dataset simboli
1. Legge i crop reali labellati da LabelMe
2. Per le classi rare genera simboli sintetici per compensare
3. Applica x4 augmentation su tutto (zoom in/out, blur, rotate, erase)

Output: dataset_simboli/ (formato YOLO classification)
"""

import cv2
import numpy as np
import random
import json
import math
import shutil
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

LABELED_DIR  = "./quote_per_labeling"
OUTPUT_DIR   = "./dataset_simboli"

TARGET_N     = 256     # immagini reali target per classe
AUGMENT_X    = 4       # moltiplicatore augmentation finale
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
RANDOM_SEED  = 42

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
    "linear",   # quote senza simbolo
]

# Crop senza JSON = quote non annotate = probabilmente linear
N_LINEAR_UNLABELED = 600

# ─────────────────────────────────────────
# LETTURA CROP LABELLATI
# ─────────────────────────────────────────

def load_linear_crops(labeled_dir: Path):
    """
    Raccoglie i crop senza JSON corrispondente —
    sono le quote non annotate, probabilmente senza simbolo (linear).
    """
    all_imgs  = set(p.stem for p in labeled_dir.glob("*.jpg"))
    all_imgs |= set(p.stem for p in labeled_dir.glob("*.png"))
    annotated = set(p.stem for p in labeled_dir.glob("*.json"))
    unannotated = list(all_imgs - annotated)
    random.shuffle(unannotated)
    unannotated = unannotated[:N_LINEAR_UNLABELED]

    crops = []
    for stem in unannotated:
        for ext in [".jpg", ".png"]:
            p = labeled_dir / (stem + ext)
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    crops.append(img)
                break

    print(f"  linear (non annotate): {len(crops)} crop")
    return crops


def load_labeled_crops(labeled_dir: Path):
    """
    Legge i JSON LabelMe e ritorna dict {classe: [img, img, ...]}
    """
    crops   = defaultdict(list)
    jsons   = sorted(labeled_dir.glob("*.json"))
    print(f"Lettura {len(jsons)} JSON...")

    for json_path in jsons:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        shapes = data.get("shapes", [])
        if not shapes:
            continue

        img_path = labeled_dir / json_path.stem
        for ext in [".jpg", ".jpeg", ".png"]:
            if (labeled_dir / (json_path.stem + ext)).exists():
                img_path = labeled_dir / (json_path.stem + ext)
                break

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for shape in shapes:
            label = shape.get("label","").strip().lower().replace(" ","_")
            if label not in SYMBOL_CLASSES:
                continue
            crops[label].append(img.copy())

    for cls, lst in crops.items():
        print(f"  {cls:25s}: {len(lst)} crop reali")
    return crops


# ─────────────────────────────────────────
# DISEGNO SIMBOLI SINTETICI
# ─────────────────────────────────────────

def make_symbol_patch(cls: str, size: int = 64):
    """
    Genera patch bianca con il simbolo GD&T disegnato dritto.
    Fedele alla legenda della norma.
    """
    p  = np.ones((size, size, 3), dtype=np.uint8) * 255
    cx = size // 2
    cy = size // 2
    r  = int(size * 0.30)
    c  = (0, 0, 0)
    th = max(1, size // 28)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs   = size / 52.0

    if cls == "diameter":
        # Ø — cerchio con barra diagonale
        cv2.circle(p, (cx,cy), r, c, th, cv2.LINE_AA)
        cv2.line(p, (cx-int(r*.8),cy+int(r*.8)),
                    (cx+int(r*.8),cy-int(r*.8)), c, th, cv2.LINE_AA)

    elif cls == "radius":
        # R dentro cerchio opzionale
        if random.random() > 0.4:
            cv2.circle(p, (cx,cy), r, c, th, cv2.LINE_AA)
        (tw, th2), _ = cv2.getTextSize("R", font, fs*1.3, th)
        cv2.putText(p, "R", (cx-tw//2, cy+th2//2),
                    font, fs*1.3, c, th, cv2.LINE_AA)

    elif cls == "angle":
        # Arco con trattino sopra
        cv2.ellipse(p, (cx, cy+r//3), (r,r), 0, 200, 340, c, th, cv2.LINE_AA)
        cv2.line(p, (cx-r//2, cy-r//2), (cx+r//2, cy-r//2), c, th)

    elif cls == "surface_finish":
        # √ con gambo allungato
        cv2.line(p, (cx-r, cy+r//3), (cx-r//4, cy-r//2), c, th, cv2.LINE_AA)
        cv2.line(p, (cx-r//4, cy-r//2), (cx+r, cy-r//2), c, th, cv2.LINE_AA)
        val = str(random.choice([0.8,1.6,3.2,6.3,12.5,25]))
        cv2.putText(p, val, (cx-r//4, cy-r//2-4),
                    font, fs*0.55, c, 1, cv2.LINE_AA)

    elif cls == "concentricity":
        # Cerchio con punto al centro (due cerchi concentrici)
        cv2.circle(p, (cx,cy), r, c, th, cv2.LINE_AA)
        cv2.circle(p, (cx,cy), r//2, c, th, cv2.LINE_AA)
        cv2.circle(p, (cx,cy), 2, c, -1)

    elif cls == "cylindricity":
        # Cerchio con due linee verticali tangenti
        cv2.circle(p, (cx,cy), r, c, th, cv2.LINE_AA)
        cv2.line(p, (cx-r, cy-r), (cx-r, cy+r), c, th)
        cv2.line(p, (cx+r, cy-r), (cx+r, cy+r), c, th)

    elif cls == "position":
        # Cerchio con croce interna ⊕
        cv2.circle(p, (cx,cy), r, c, th, cv2.LINE_AA)
        cv2.line(p, (cx-r, cy), (cx+r, cy), c, th)
        cv2.line(p, (cx, cy-r), (cx, cy+r), c, th)

    elif cls == "flatness":
        # Due linee parallele orizzontali
        cv2.line(p, (cx-r, cy-r//3), (cx+r, cy-r//3), c, th)
        cv2.line(p, (cx-r, cy+r//3), (cx+r, cy+r//3), c, th)

    elif cls == "perpendicularity":
        # ⊥ — T rovesciata
        cv2.line(p, (cx-r, cy+r//2), (cx+r, cy+r//2), c, th)
        cv2.line(p, (cx, cy+r//2), (cx, cy-r), c, th)

    elif cls == "total_runout":
        # Due frecce diagonali parallele ↗↗
        for dx in [-r//3, r//3]:
            cv2.arrowedLine(p,
                (cx+dx-r//2, cy+r//2),
                (cx+dx+r//2, cy-r//2),
                c, th, tipLength=0.3, line_type=cv2.LINE_AA)

    elif cls == "circular_runout":
        # Una freccia diagonale ↗
        cv2.arrowedLine(p,
            (cx-r//2, cy+r//2), (cx+r//2, cy-r//2),
            c, th, tipLength=0.3, line_type=cv2.LINE_AA)

    elif cls == "slope":
        # Triangolo rettangolo con tratto
        pts = np.array([[cx-r, cy+r//2],
                        [cx+r, cy+r//2],
                        [cx+r, cy-r//2]], np.int32)
        cv2.polylines(p, [pts], True, c, th, cv2.LINE_AA)

    elif cls == "conical_taper":
        # Triangolo isoscele (cono)
        pts = np.array([[cx,    cy-r],
                        [cx+r,  cy+r//2],
                        [cx-r,  cy+r//2]], np.int32)
        cv2.polylines(p, [pts], True, c, th, cv2.LINE_AA)
        cv2.line(p, (cx, cy-r), (cx, cy+r//2), c, th)

    elif cls == "symmetry":
        # Due linee parallele orizzontali con tratto verticale centrale
        cv2.line(p, (cx-r, cy-r//3), (cx+r, cy-r//3), c, th)
        cv2.line(p, (cx-r, cy+r//3), (cx+r, cy+r//3), c, th)
        cv2.line(p, (cx, cy-r//3), (cx, cy+r//3), c, th)

    elif cls == "surface_profile":
        # Mezza luna (profilo di superficie) ⌓
        cv2.ellipse(p, (cx, cy+r//4), (r, r), 0, 180, 360,
                    c, th, cv2.LINE_AA)

    return p


# ─────────────────────────────────────────
# AUGMENTATION (zoom in/out, blur, rotate, erase)
# ─────────────────────────────────────────

def augment(img):
    out = img.copy()
    h, w = out.shape[:2]
    paper = int(np.percentile(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), 90))
    paper = max(200, min(255, paper))

    # ── Zoom in/out (soprattutto out) ──
    scale = random.choices(
        [random.uniform(0.30, 0.55),   # zoom out forte
         random.uniform(0.55, 0.80),   # zoom out moderato
         random.uniform(0.80, 1.10),   # neutro
         random.uniform(1.10, 1.60)],  # zoom in
        weights=[0.30, 0.30, 0.25, 0.15]
    )[0]
    nw = max(8, int(w*scale))
    nh = max(8, int(h*scale))
    resized = cv2.resize(out, (nw,nh), interpolation=cv2.INTER_AREA)
    canvas  = np.ones((h,w,3), dtype=np.uint8)*paper
    y_off   = max(0, (h-nh)//2)
    x_off   = max(0, (w-nw)//2)
    nh2 = min(nh, h-y_off); nw2 = min(nw, w-x_off)
    canvas[y_off:y_off+nh2, x_off:x_off+nw2] = resized[:nh2,:nw2]
    out = canvas

    # ── Rotazione ──
    if random.random() < 0.50:
        angle = random.uniform(-25, 25)
        M     = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
        out   = cv2.warpAffine(out, M, (w,h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(paper,paper,paper))

    # ── Scan quality ──
    if random.random() < 0.85:
        f  = out.astype(np.float32)
        f  = f*(random.randint(210,252)/255.0)
        f[:,:,0] *= random.uniform(0.87,0.98)
        f  = np.clip(f/255.0,0,1)**(1/random.uniform(0.65,1.40))*255.0
        f  = 128+(f-128)*random.uniform(0.72,0.96)
        f += np.random.normal(0,random.uniform(1.0,5.5),f.shape)
        out = np.clip(f,0,255).astype(np.uint8)

    # ── Blur ──
    if random.random() < 0.45:
        mode = random.choice(["gaussian","motion","defocus"])
        if mode == "gaussian":
            k   = random.choice([3,5,7])
            out = cv2.GaussianBlur(out,(k,k),random.uniform(0.5,2.0))
        elif mode == "motion":
            k      = random.choice([5,7,9])
            kernel = np.zeros((k,k), dtype=np.float32)
            kernel[k//2,:] = 1.0/k
            M2     = cv2.getRotationMatrix2D((k//2,k//2),
                         random.choice([0,45,90,135]),1.0)
            kernel = cv2.warpAffine(kernel,M2,(k,k))
            kernel /= kernel.sum()+1e-6
            out    = cv2.filter2D(out,-1,kernel)
        else:
            k      = random.choice([3,5,7])
            kernel = np.zeros((k,k), dtype=np.float32)
            cv2.circle(kernel,(k//2,k//2),k//2,1,-1)
            kernel /= kernel.sum()
            out    = cv2.filter2D(out,-1,kernel)

    # ── JPEG compression ──
    if random.random() < 0.40:
        _,buf = cv2.imencode('.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,
                                        random.randint(38,78)])
        out   = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # ── Erode ──
    if random.random() < 0.30:
        kernel   = np.ones((2,2),np.uint8)
        dark     = out < 80
        eroded   = cv2.erode(out,kernel,iterations=1)
        out[dark] = eroded[dark]

    # ── Cancellazioni con colore sfondo ──
    if random.random() < 0.35:
        n_erase = random.randint(1,3)
        for _ in range(n_erase):
            ew = random.randint(3, max(4,w//4))
            eh = random.randint(3, max(4,h//4))
            ex = random.randint(0, max(0,w-ew))
            ey = random.randint(0, max(0,h-eh))
            cv2.rectangle(out,(ex,ey),(ex+ew,ey+eh),
                          (paper,paper,paper),-1)

    return out


# ─────────────────────────────────────────
# GENERAZIONE DATASET
# ─────────────────────────────────────────

def generate_dataset(crops_by_class: dict):
    n_train = int(TARGET_N * AUGMENT_X * TRAIN_RATIO)
    n_val   = int(TARGET_N * AUGMENT_X * VAL_RATIO)
    n_test  = TARGET_N * AUGMENT_X - n_train - n_val

    print(f"\nTarget per classe: {TARGET_N} base × {AUGMENT_X} aug = "
          f"{TARGET_N*AUGMENT_X} totali")
    print(f"  train={n_train}  val={n_val}  test={n_test}\n")

    for cls in SYMBOL_CLASSES:
        real_crops = crops_by_class.get(cls, [])
        n_real     = len(real_crops)
        n_needed   = max(0, TARGET_N - n_real)

        print(f"  {cls:25s}: {n_real} reali + {n_needed} sintetici", end="")

        # Genera sintetici se necessario
        synthetic = []
        if n_needed > 0:
            for _ in range(n_needed):
                sym_size = random.randint(32, 96)
                sym      = make_symbol_patch(cls, sym_size)
                # Ridimensiona a dimensione simile ai crop reali
                tw = random.randint(30, 100)
                th = random.randint(30, 100)
                synthetic.append(cv2.resize(sym, (tw,th)))

        all_base = real_crops + synthetic
        random.shuffle(all_base)

        # Augmentation x4
        augmented = []
        for base in all_base:
            augmented.append(base.copy())  # originale
            for _ in range(AUGMENT_X - 1):
                augmented.append(augment(base.copy()))

        random.shuffle(augmented)

        # Split e salva
        splits = {
            "train": augmented[:n_train],
            "val":   augmented[n_train:n_train+n_val],
            "test":  augmented[n_train+n_val:n_train+n_val+n_test],
        }

        saved = 0
        for split, imgs in splits.items():
            out_dir = Path(OUTPUT_DIR) / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(imgs):
                cv2.imwrite(
                    str(out_dir / f"{cls}_{split}_{i:05d}.jpg"),
                    img,
                    [cv2.IMWRITE_JPEG_QUALITY, random.randint(72,95)])
                saved += 1

        print(f" → {saved} immagini salvate")

    return


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("SINTESI: Genesi — Generazione dataset simboli")
    print("=" * 55)

    crops = load_labeled_crops(Path(LABELED_DIR))

    # Aggiungi crop linear (non annotate)
    linear_crops = load_linear_crops(Path(LABELED_DIR))
    crops["linear"] = linear_crops

    generate_dataset(crops)

    # Riepilogo finale
    all_classes = SYMBOL_CLASSES
    total = 0
    print(f"\nRiepilogo output:")
    for split in ["train","val","test"]:
        n = sum(len(list((Path(OUTPUT_DIR)/split/cls).glob("*.jpg")))
                for cls in all_classes
                if (Path(OUTPUT_DIR)/split/cls).exists())
        print(f"  {split:6s}: {n}")
        total += n

    print(f"  TOTALE: {total}")
    print(f"\n✓ Dataset in: {OUTPUT_DIR}")
    print(f"\nPer allenare:")
    print(f"  python train_symbols.py")