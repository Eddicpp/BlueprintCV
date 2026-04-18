"""
SINTESI: Genesi — Bilanciamento dataset simboli detection
Per le classi rare genera crop sintetici aggiungendo il simbolo
nella posizione corretta rispetto alla quota:
  - sinistra o destra: diameter, radius, surface_finish,
                       concentricity, cylindricity, position,
                       flatness, perpendicularity, total_runout,
                       circular_runout, slope, conical_taper,
                       symmetry, surface_profile
  - centro o sopra:    angle (gradi), surface_finish (finitura)

Aggiunge i nuovi crop al dataset_simboli_detection/train/
"""

import cv2
import numpy as np
import random
import math
import shutil
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

DETECTION_DATASET = "./dataset_simboli_detection"
LABELED_DIR       = "./quote_per_labeling"
OUTPUT_DIR        = "./dataset_simboli_detection"  # stesso dataset, aggiungi a train

TARGET_N     = 80    # esempi target per classe
RANDOM_SEED  = 42

SYMBOL_CLASSES = [
    "diameter", "radius", "angle", "surface_finish",
    "concentricity", "cylindricity", "position", "flatness",
    "perpendicularity", "total_runout", "circular_runout",
    "slope", "conical_taper", "symmetry", "surface_profile",
]

# Posizione simbolo per classe
POSITION_MAP = {
    "diameter":        ["left", "right"],
    "radius":          ["left", "right"],
    "angle":           ["center", "above"],
    "surface_finish":  ["center", "left", "right"],
    "concentricity":   ["left", "right"],
    "cylindricity":    ["left", "right"],
    "position":        ["left", "right"],
    "flatness":        ["left", "right"],
    "perpendicularity":["left", "right"],
    "total_runout":    ["left", "right"],
    "circular_runout": ["left", "right"],
    "slope":           ["left", "right"],
    "conical_taper":   ["left", "right"],
    "symmetry":        ["left", "right"],
    "surface_profile": ["left", "right"],
}

# ─────────────────────────────────────────
# RACCOLTA CROP REALI
# ─────────────────────────────────────────

def yolo_to_abs(cx, cy, w, h, iw, ih):
    x1=int((cx-w/2)*iw); y1=int((cy-h/2)*ih)
    x2=int((cx+w/2)*iw); y2=int((cy+h/2)*ih)
    return max(0,x1), max(0,y1), min(iw,x2), min(ih,y2)


def abs_to_yolo(x1, y1, x2, y2, iw, ih):
    cx=(x1+x2)/2/iw; cy=(y1+y2)/2/ih
    w=(x2-x1)/iw;    h=(y2-y1)/ih
    return (max(0,min(1,cx)), max(0,min(1,cy)),
            max(0,min(1,w)),  max(0,min(1,h)))


def collect_base_crops(labeled_dir: Path):
    """Raccoglie crop reali senza simbolo (non annotati)."""
    all_imgs  = set(p.stem for p in labeled_dir.glob("*.jpg"))
    all_imgs |= set(p.stem for p in labeled_dir.glob("*.png"))
    annotated = set(p.stem for p in labeled_dir.glob("*.json"))
    unannotated = list(all_imgs - annotated)
    random.shuffle(unannotated)

    crops = []
    for stem in unannotated:
        for ext in [".jpg", ".png"]:
            p = labeled_dir / (stem + ext)
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None and img.shape[0]>15 and img.shape[1]>30:
                    crops.append(img)
                break

    # Aggiungi anche crop dal train set esistente
    train_img_dir = Path(DETECTION_DATASET) / "train" / "images"
    if train_img_dir.exists():
        for p in list(train_img_dir.glob("*.jpg"))[:200]:
            img = cv2.imread(str(p))
            if img is not None:
                crops.append(img)

    print(f"Crop base disponibili: {len(crops)}")
    return crops


# ─────────────────────────────────────────
# CONTA ESISTENTI
# ─────────────────────────────────────────

def count_existing(train_lbl_dir: Path):
    counts = defaultdict(int)
    for lbl in train_lbl_dir.glob("*.txt"):
        for line in lbl.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                if cls_id < len(SYMBOL_CLASSES):
                    counts[SYMBOL_CLASSES[cls_id]] += 1
    return counts


# ─────────────────────────────────────────
# DISEGNO SIMBOLI
# ─────────────────────────────────────────

def draw_symbol_on_patch(patch, cls, color, font_scale):
    """Disegna il simbolo GD&T sulla patch e ritorna il bbox del simbolo."""
    h, w  = patch.shape[:2]
    cx    = w // 2
    cy    = h // 2
    r     = max(4, min(h//3, w//3, 20))
    th    = max(1, int(font_scale * 1.5))
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # Box del simbolo — sarà aggiornato per ogni tipo
    sx1, sy1, sx2, sy2 = 2, 2, w-2, h-2

    if cls == "diameter":
        cv2.circle(patch, (cx,cy), r, color, th, cv2.LINE_AA)
        cv2.line(patch, (cx-int(r*.8),cy+int(r*.8)),
                        (cx+int(r*.8),cy-int(r*.8)), color, th, cv2.LINE_AA)
        sx1,sy1,sx2,sy2 = cx-r-1,cy-r-1,cx+r+1,cy+r+1

    elif cls == "radius":
        if random.random() > 0.4:
            cv2.circle(patch, (cx,cy), r, color, th, cv2.LINE_AA)
        (tw,th2),_ = cv2.getTextSize("R", font, font_scale*1.2, th)
        cv2.putText(patch, "R", (cx-tw//2, cy+th2//2),
                    font, font_scale*1.2, color, th, cv2.LINE_AA)
        sx1,sy1,sx2,sy2 = cx-r-1,cy-r-1,cx+r+1,cy+r+1

    elif cls == "angle":
        cv2.ellipse(patch,(cx,cy+r//3),(r,r),0,200,340,color,th,cv2.LINE_AA)
        cv2.line(patch,(cx-r//2,cy-r//2),(cx+r//2,cy-r//2),color,th)
        sx1,sy1,sx2,sy2 = cx-r-1,cy-r-1,cx+r+1,cy+r//3+1

    elif cls == "surface_finish":
        cv2.line(patch,(cx-r,cy+r//3),(cx-r//4,cy-r//2),color,th,cv2.LINE_AA)
        cv2.line(patch,(cx-r//4,cy-r//2),(cx+r,cy-r//2),color,th,cv2.LINE_AA)
        val = str(random.choice([0.8,1.6,3.2,6.3,12.5]))
        cv2.putText(patch,val,(cx-r//4,cy-r//2-3),font,font_scale*0.6,color,1)
        sx1,sy1,sx2,sy2 = cx-r-1,cy-r//2-8,cx+r+1,cy+r//3+1

    elif cls == "concentricity":
        cv2.circle(patch,(cx,cy),r,color,th,cv2.LINE_AA)
        cv2.circle(patch,(cx,cy),r//2,color,th,cv2.LINE_AA)
        cv2.circle(patch,(cx,cy),2,color,-1)
        sx1,sy1,sx2,sy2 = cx-r-1,cy-r-1,cx+r+1,cy+r+1

    elif cls == "cylindricity":
        cv2.circle(patch,(cx,cy),r,color,th,cv2.LINE_AA)
        cv2.line(patch,(cx-r,cy-r),(cx-r,cy+r),color,th)
        cv2.line(patch,(cx+r,cy-r),(cx+r,cy+r),color,th)
        sx1,sy1,sx2,sy2 = cx-r-2,cy-r-1,cx+r+2,cy+r+1

    elif cls == "position":
        cv2.circle(patch,(cx,cy),r,color,th,cv2.LINE_AA)
        cv2.line(patch,(cx-r,cy),(cx+r,cy),color,th)
        cv2.line(patch,(cx,cy-r),(cx,cy+r),color,th)
        sx1,sy1,sx2,sy2 = cx-r-1,cy-r-1,cx+r+1,cy+r+1

    elif cls == "flatness":
        cv2.line(patch,(cx-r,cy-r//3),(cx+r,cy-r//3),color,th)
        cv2.line(patch,(cx-r,cy+r//3),(cx+r,cy+r//3),color,th)
        sx1,sy1,sx2,sy2 = cx-r-1,cy-r//3-2,cx+r+1,cy+r//3+2

    elif cls == "perpendicularity":
        cv2.line(patch,(cx-r,cy+r//2),(cx+r,cy+r//2),color,th)
        cv2.line(patch,(cx,cy+r//2),(cx,cy-r),color,th)
        sx1,sy1,sx2,sy2 = cx-r-1,cy-r-1,cx+r+1,cy+r//2+2

    elif cls == "total_runout":
        for dx in [-r//3, r//3]:
            cv2.arrowedLine(patch,(cx+dx-r//2,cy+r//2),
                            (cx+dx+r//2,cy-r//2),color,th,
                            tipLength=0.3,line_type=cv2.LINE_AA)
        sx1,sy1,sx2,sy2 = cx-r-1,cy-r//2-1,cx+r+1,cy+r//2+1

    elif cls == "circular_runout":
        cv2.arrowedLine(patch,(cx-r//2,cy+r//2),(cx+r//2,cy-r//2),
                        color,th,tipLength=0.3,line_type=cv2.LINE_AA)
        sx1,sy1,sx2,sy2 = cx-r//2-1,cy-r//2-1,cx+r//2+1,cy+r//2+1

    elif cls == "slope":
        pts = np.array([[cx-r,cy+r//2],[cx+r,cy+r//2],[cx+r,cy-r//2]],np.int32)
        cv2.polylines(patch,[pts],True,color,th,cv2.LINE_AA)
        sx1,sy1,sx2,sy2 = cx-r-1,cy-r//2-1,cx+r+1,cy+r//2+1

    elif cls == "conical_taper":
        pts = np.array([[cx,cy-r],[cx+r,cy+r//2],[cx-r,cy+r//2]],np.int32)
        cv2.polylines(patch,[pts],True,color,th,cv2.LINE_AA)
        cv2.line(patch,(cx,cy-r),(cx,cy+r//2),color,th)
        sx1,sy1,sx2,sy2 = cx-r-1,cy-r-1,cx+r+1,cy+r//2+1

    elif cls == "symmetry":
        cv2.line(patch,(cx-r,cy-r//3),(cx+r,cy-r//3),color,th)
        cv2.line(patch,(cx-r,cy+r//3),(cx+r,cy+r//3),color,th)
        cv2.line(patch,(cx,cy-r//3),(cx,cy+r//3),color,th)
        sx1,sy1,sx2,sy2 = cx-r-1,cy-r//3-2,cx+r+1,cy+r//3+2

    elif cls == "surface_profile":
        cv2.ellipse(patch,(cx,cy+r//4),(r,r),0,180,360,color,th,cv2.LINE_AA)
        sx1,sy1,sx2,sy2 = cx-r-1,cy-r//4-1,cx+r+1,cy+r//4+r+1

    # Clamp bbox
    sx1=max(0,sx1); sy1=max(0,sy1)
    sx2=min(w,sx2); sy2=min(h,sy2)
    return patch, (sx1,sy1,sx2,sy2)


# ─────────────────────────────────────────
# AUGMENTATION SIMBOLO
# ─────────────────────────────────────────

def augment_img(img):
    out = img.copy()
    h,w = out.shape[:2]
    paper = max(200, int(np.percentile(
        cv2.cvtColor(out,cv2.COLOR_BGR2GRAY), 90)))

    # Scan quality
    f  = out.astype(np.float32)*(random.randint(210,252)/255.0)
    f[:,:,0] *= random.uniform(0.87,0.98)
    f  = np.clip(f/255.0,0,1)**(1/random.uniform(0.7,1.3))*255.0
    f  = 128+(f-128)*random.uniform(0.75,0.96)
    f += np.random.normal(0,random.uniform(1,5),f.shape)
    out = np.clip(f,0,255).astype(np.uint8)

    # Blur leggero
    if random.random() < 0.4:
        k   = random.choice([3,5])
        out = cv2.GaussianBlur(out,(k,k),random.uniform(0.3,1.5))

    # JPEG
    if random.random() < 0.35:
        _,buf = cv2.imencode('.jpg',out,
                             [cv2.IMWRITE_JPEG_QUALITY,random.randint(50,80)])
        out   = cv2.imdecode(buf,cv2.IMREAD_COLOR)

    return out


# ─────────────────────────────────────────
# GENERAZIONE SINTETICA
# ─────────────────────────────────────────

def generate_synthetic(cls, base_crops, n_needed):
    """
    Genera n_needed crop con il simbolo cls aggiunto nella
    posizione corretta (sinistra/destra o centro/sopra).
    Ritorna lista di (img, yolo_label_string).
    """
    results  = []
    positions = POSITION_MAP.get(cls, ["left","right"])

    for _ in range(n_needed):
        base = random.choice(base_crops).copy()
        h, w = base.shape[:2]

        # Ridimensiona a dimensione ragionevole
        tw = random.randint(80, 220)
        th = random.randint(30, 80)
        base = cv2.resize(base, (tw,th), interpolation=cv2.INTER_AREA)
        h, w = base.shape[:2]

        # Stima colore testo dal crop
        gray     = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        darkness = random.randint(15, 60)
        color    = (darkness, darkness, darkness)

        # Dimensione simbolo proporzionale all'altezza
        sym_h = max(10, int(h * random.uniform(0.5, 0.9)))
        sym_w = sym_h

        pos = random.choice(positions)

        if pos == "left":
            # Simbolo a sinistra — allarga canvas a sinistra
            gap    = random.randint(3, 12)
            new_w  = w + sym_w + gap
            canvas = np.ones((h, new_w, 3), dtype=np.uint8) * 245
            # Incolla crop a destra
            canvas[:, sym_w+gap:] = base
            # Patch simbolo centrata verticalmente
            sym_patch = np.ones((h, sym_w, 3), dtype=np.uint8) * 245
            cy_sym    = h//2
            cx_sym    = sym_w//2
            fs = sym_h / 50.0
            sym_patch, (sx1,sy1,sx2,sy2) = draw_symbol_on_patch(
                sym_patch, cls, color, fs)
            canvas[:, :sym_w] = sym_patch
            # Label YOLO riferita al canvas intero
            lx1 = sx1; ly1 = sy1
            lx2 = sx2; ly2 = sy2

        elif pos == "right":
            gap    = random.randint(3, 12)
            new_w  = w + sym_w + gap
            canvas = np.ones((h, new_w, 3), dtype=np.uint8) * 245
            canvas[:, :w] = base
            sym_patch = np.ones((h, sym_w, 3), dtype=np.uint8) * 245
            fs = sym_h / 50.0
            sym_patch, (sx1,sy1,sx2,sy2) = draw_symbol_on_patch(
                sym_patch, cls, color, fs)
            canvas[:, w+gap:w+gap+sym_w] = sym_patch
            lx1 = w+gap+sx1; ly1 = sy1
            lx2 = w+gap+sx2; ly2 = sy2

        elif pos == "center":
            canvas = base.copy()
            new_w  = w
            fs = min(h,w) / 50.0
            canvas, (sx1,sy1,sx2,sy2) = draw_symbol_on_patch(
                canvas, cls, color, fs)
            lx1=sx1; ly1=sy1; lx2=sx2; ly2=sy2

        else:  # above
            gap    = random.randint(3,8)
            sym_row_h = int(h * 0.4)
            new_h  = h + sym_row_h + gap
            canvas = np.ones((new_h, w, 3), dtype=np.uint8) * 245
            canvas[sym_row_h+gap:,:] = base
            sym_patch = np.ones((sym_row_h, w, 3), dtype=np.uint8)*245
            fs = sym_row_h / 50.0
            sym_patch, (sx1,sy1,sx2,sy2) = draw_symbol_on_patch(
                sym_patch, cls, color, fs)
            canvas[:sym_row_h,:] = sym_patch
            lx1=sx1; ly1=sy1; lx2=sx2; ly2=sy2
            h = new_h

        ih, iw = canvas.shape[:2]

        # Clamp
        lx1=max(0,lx1); ly1=max(0,ly1)
        lx2=min(iw,lx2); ly2=min(ih,ly2)

        if lx2-lx1 < 3 or ly2-ly1 < 3:
            continue

        # Augmentation
        canvas = augment_img(canvas)

        # YOLO label
        cls_id = SYMBOL_CLASSES.index(cls)
        cx,cy,bw,bh = abs_to_yolo(lx1,ly1,lx2,ly2,iw,ih)
        label = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

        results.append((canvas, label))

    return results


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("SINTESI: Genesi — Bilanciamento dataset simboli detection")
    print("=" * 55)

    train_img = Path(OUTPUT_DIR) / "train" / "images"
    train_lbl = Path(OUTPUT_DIR) / "train" / "labels"
    train_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)

    # Conta esistenti
    existing = count_existing(train_lbl)
    print(f"\nEsistenti nel train set:")
    for cls in SYMBOL_CLASSES:
        n = existing.get(cls, 0)
        flag = " ← RARA" if n < TARGET_N else ""
        print(f"  {cls:25s}: {n:>4}{flag}")

    # Raccogli crop base
    base_crops = collect_base_crops(Path(LABELED_DIR))
    if not base_crops:
        print("Nessun crop base trovato.")
        exit(1)

    # Genera sintetici per classi rare
    total_added = 0
    print(f"\nGenerazione sintetici (target={TARGET_N}/classe)...")

    for cls in SYMBOL_CLASSES:
        n_existing = existing.get(cls, 0)
        n_needed   = max(0, TARGET_N - n_existing)

        if n_needed == 0:
            print(f"  {cls:25s}: già ok ({n_existing})")
            continue

        print(f"  {cls:25s}: +{n_needed} sintetici...", end="", flush=True)
        generated = generate_synthetic(cls, base_crops, n_needed)

        for i, (img, label) in enumerate(generated):
            name = f"syn_{cls}_{i:04d}"
            cv2.imwrite(str(train_img / f"{name}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, random.randint(75,95)])
            (train_lbl / f"{name}.txt").write_text(label)
            total_added += 1

        print(f" ✓ {len(generated)}")

    print(f"\n✓ Aggiunte {total_added} immagini sintetiche al train set")
    print(f"  Output: {OUTPUT_DIR}/train/")
    print(f"\nOra riallena il detector:")
    print(f"  yolo detect train data={OUTPUT_DIR}/data.yaml "
          f"model=yolo11n.pt epochs=100 imgsz=640 batch=16 "
          f"project=sintesi_genesi name=simboli_detector_run1")
