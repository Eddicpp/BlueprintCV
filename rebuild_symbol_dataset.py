"""
SINTESI: Genesi — Rebuild dataset_simboli da annotazioni reali
Legge i JSON LabelMe da quote_per_labeling/,
ritaglia SOLO il simbolo usando il bounding box annotato
e popola dataset_simboli/ con i crop puri del simbolo.

Poi applica augmentation per raggiungere TARGET_N per classe.
"""

import cv2
import json
import numpy as np
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

LABELED_DIR  = "./quote_per_labeling"
OUTPUT_DIR   = "./dataset_simboli"
TARGET_N     = 1500     # immagini target per classe
AUGMENT_X    = 10      # aumentato se servono più esempi
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
RANDOM_SEED  = 42
MARGIN       = 4       # pixel di margine attorno al simbolo ritagliato

SYMBOL_CLASSES = [
    "diameter", "radius", "surface_finish",
    "concentricity", "cylindricity", "position", "flatness",
    "perpendicularity", "total_runout", "circular_runout",
    "slope", "conical_taper", "symmetry", "surface_profile",
    "linear",
]

# ─────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────

def augment(img, seed_offset: int):
    random.seed(seed_offset)
    np.random.seed(seed_offset % (2**32))

    out  = img.copy()
    h, w = out.shape[:2]
    paper = max(200, int(np.percentile(
        cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), 90)))

    # ── Zoom ──
    scale = random.choices(
        [random.uniform(0.35, 0.60),
         random.uniform(0.60, 0.85),
         random.uniform(0.85, 1.15),
         random.uniform(1.15, 1.60)],
        weights=[0.20, 0.35, 0.30, 0.15]
    )[0]
    nw = max(8, int(w*scale))
    nh = max(8, int(h*scale))
    resized = cv2.resize(out,(nw,nh),interpolation=cv2.INTER_AREA)
    canvas  = np.ones((h,w,3),dtype=np.uint8)*paper
    y_off   = max(0,(h-nh)//2); x_off = max(0,(w-nw)//2)
    canvas[y_off:y_off+min(nh,h-y_off),
           x_off:x_off+min(nw,w-x_off)] = resized[:min(nh,h-y_off),
                                                    :min(nw,w-x_off)]
    out = canvas

    # ── Rotazione ──
    if random.random() < 0.45:
        angle = random.uniform(-20, 20)
        M     = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
        out   = cv2.warpAffine(out, M, (w,h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(paper,paper,paper))

    # ── Flip ──
    if random.random() < 0.50:
        out = cv2.flip(out, 1)

    # ── Scan quality ──
    f  = out.astype(np.float32)
    f  = f*(random.randint(198,252)/255.0)
    f[:,:,0] *= random.uniform(0.85,0.99)
    f  = np.clip(f/255.0,0,1)**(1/random.uniform(0.60,1.50))*255.0
    f  = 128+(f-128)*random.uniform(0.68,0.97)
    f += np.random.normal(0, random.uniform(0.5,6.0), f.shape)
    out = np.clip(f,0,255).astype(np.uint8)

    # ── Inchiostro sbiadito — rende le linee grigie invece di nere ──
    if random.random() < 0.40:
        # Schiarisce i pixel scuri simulando inchiostro vecchio/sbiadito
        fade = random.uniform(0.25, 0.65)
        dark_mask = out < 120
        out = out.astype(np.float32)
        out[dark_mask] = out[dark_mask] + (paper - out[dark_mask]) * fade
        out = np.clip(out, 0, 255).astype(np.uint8)

    # ── Tratti leggeri sovrapposti (linee grigie di contesto) ──
    if random.random() < 0.50:
        n_lines = random.randint(1, 5)
        for _ in range(n_lines):
            gray_val = random.randint(160, 220)
            lcolor   = (gray_val, gray_val, gray_val)
            x1l = random.randint(0, w); y1l = random.randint(0, h)
            x2l = random.randint(0, w); y2l = random.randint(0, h)
            cv2.line(out, (x1l,y1l), (x2l,y2l), lcolor, 1, cv2.LINE_AA)

    # ── Blur ──
    if random.random() < 0.45:
        mode = random.choice(["gaussian","motion","defocus"])
        if mode == "gaussian":
            k   = random.choice([3,5,7])
            out = cv2.GaussianBlur(out,(k,k),random.uniform(0.4,2.5))
        elif mode == "motion":
            k      = random.choice([5,7,9])
            kernel = np.zeros((k,k),dtype=np.float32)
            kernel[k//2,:] = 1.0/k
            M2     = cv2.getRotationMatrix2D((k//2,k//2),
                         random.uniform(0,180),1.0)
            kernel = cv2.warpAffine(kernel,M2,(k,k))
            kernel /= kernel.sum()+1e-6
            out    = cv2.filter2D(out,-1,kernel)
        else:
            k      = random.choice([3,5,7])
            kernel = np.zeros((k,k),dtype=np.float32)
            cv2.circle(kernel,(k//2,k//2),k//2,1,-1)
            kernel /= kernel.sum()
            out    = cv2.filter2D(out,-1,kernel)

    # ── JPEG compression ──
    if random.random() < 0.40:
        q     = random.randint(35,80)
        _,buf = cv2.imencode('.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,q])
        out   = cv2.imdecode(buf,cv2.IMREAD_COLOR)

    # ── Erode (inchiostro consumato) ──
    if random.random() < 0.35:
        k_e  = np.ones((random.choice([2,2,3]),
                        random.choice([2,2,3])),np.uint8)
        dark = out < random.randint(60,100)
        out[dark] = cv2.erode(out,k_e,iterations=1)[dark]

    # ── Dilate (inchiostro allargato) ──
    if random.random() < 0.20:
        k_d  = np.ones((2,2),np.uint8)
        dark = out < 80
        out[dark] = cv2.dilate(out,k_d,iterations=1)[dark]

    # ── Rumore sale e pepe ──
    if random.random() < 0.25:
        n_px = random.randint(1, max(2,int(w*h*0.015)))
        for _ in range(n_px):
            px = random.randint(0,w-1)
            py = random.randint(0,h-1)
            out[py,px] = random.choice([0,255])

    # ── Shear leggero ──
    if random.random() < 0.25:
        shear = random.uniform(-0.12, 0.12)
        M_sh  = np.float32([[1,shear,0],[0,1,0]])
        out   = cv2.warpAffine(out,M_sh,(w,h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(paper,paper,paper))

    # ── Cancellazione parziale ──
    if random.random() < 0.25:
        ew = random.randint(2, max(3,w//4))
        eh = random.randint(2, max(3,h//4))
        ex = random.randint(0, max(0,w-ew))
        ey = random.randint(0, max(0,h-eh))
        cv2.rectangle(out,(ex,ey),(ex+ew,ey+eh),(paper,paper,paper),-1)

    return out


# ─────────────────────────────────────────
# RACCOLTA CROP SIMBOLI PURI
# ─────────────────────────────────────────

def collect_symbol_crops(labeled_dir: Path):
    """
    Legge i JSON LabelMe, ritaglia SOLO il simbolo
    usando il bounding box annotato.
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

        # Trova immagine
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = labeled_dir / (json_path.stem + ext)
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        for shape in shapes:
            label  = shape.get("label","").strip().lower().replace(" ","_")
            if label not in SYMBOL_CLASSES:
                continue

            points = shape.get("points", [])
            if len(points) < 2:
                continue

            # Bounding box del simbolo annotato
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1 = max(0, int(min(xs)) - MARGIN)
            y1 = max(0, int(min(ys)) - MARGIN)
            x2 = min(iw, int(max(xs)) + MARGIN)
            y2 = min(ih, int(max(ys)) + MARGIN)

            if x2-x1 < 5 or y2-y1 < 5:
                continue

            # Ritaglia SOLO il simbolo
            crop = img[y1:y2, x1:x2].copy()
            if crop.size > 0:
                crops[label].append(crop)

    print("\nCrop simboli puri raccolti:")
    for cls in SYMBOL_CLASSES:
        n = len(crops.get(cls, []))
        flag = " ← POCHI" if n < 20 else ""
        print(f"  {cls:25s}: {n}{flag}")

    return crops


# ─────────────────────────────────────────
# GENERA DATASET
# ─────────────────────────────────────────

def build_dataset(crops: dict):
    n_train = int(TARGET_N * TRAIN_RATIO)
    n_val   = int(TARGET_N * VAL_RATIO)
    n_test  = TARGET_N - n_train - n_val

    print(f"\nTarget: {TARGET_N}/classe → train={n_train} val={n_val} test={n_test}\n")

    for cls in SYMBOL_CLASSES:
        real_crops = crops.get(cls, [])
        n_real     = len(real_crops)

        if n_real == 0:
            print(f"  {cls:25s}: nessun crop reale — skipping")
            continue

        # Calcola augmentation necessaria
        aug_factor = max(1, -(-TARGET_N // n_real))
        print(f"  {cls:25s}: {n_real} reali × {aug_factor} aug = {n_real*aug_factor}", end="")

        # Genera augmented
        all_crops = []
        for crop in real_crops:
            all_crops.append(crop.copy())
            for i in range(aug_factor - 1):
                seed = RANDOM_SEED + hash(cls) + id(crop) + i * 997
                all_crops.append(augment(crop.copy(), abs(seed) % (2**31)))

        random.shuffle(all_crops)

        # Split e salva
        splits = {
            "train": all_crops[:n_train],
            "val":   all_crops[n_train:n_train+n_val],
            "test":  all_crops[n_train+n_val:n_train+n_val+n_test],
        }

        saved = 0
        for split, imgs in splits.items():
            out_dir = Path(OUTPUT_DIR) / split / cls
            # Svuota e ricrea
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)

            for i, img in enumerate(imgs):
                cv2.imwrite(
                    str(out_dir / f"{cls}_{split}_{i:05d}.jpg"),
                    img,
                    [cv2.IMWRITE_JPEG_QUALITY, random.randint(72,95)])
                saved += 1

        print(f" → {saved} salvate")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("SINTESI: Genesi — Rebuild dataset_simboli")
    print("=" * 55)
    print(f"Input:  {LABELED_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Target: {TARGET_N} immagini per classe\n")

    crops = collect_symbol_crops(Path(LABELED_DIR))

    if not any(crops.values()):
        print("Nessun crop trovato.")
        exit(1)

    build_dataset(crops)

    # Riepilogo
    total = 0
    print(f"\nRiepilogo finale:")
    for split in ["train","val","test"]:
        n = sum(len(list((Path(OUTPUT_DIR)/split/cls).glob("*.jpg")))
                for cls in SYMBOL_CLASSES
                if (Path(OUTPUT_DIR)/split/cls).exists())
        print(f"  {split:6s}: {n}")
        total += n
    print(f"  TOTALE: {total}")
    print(f"\n✓ Dataset pronto in {OUTPUT_DIR}")
    print(f"\nOra allena il classificatore:")
    print(f"  python train_symbol_classifier.py")