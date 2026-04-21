"""
SINTESI: Genesi — Genera classe background per classificatore M3
Raccoglie crop di quote senza simbolo da:
  1. quote_per_labeling/ — immagini senza JSON (non annotate = probabilmente linear)
  2. Falsi positivi di M2 — lancia M2 su quote_per_labeling e prende
     i crop dove M2 trova un simbolo ma la quota non ha annotazione

Output: dataset_simboli/train/background/
        dataset_simboli/val/background/
        dataset_simboli/test/background/
"""

import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

LABELED_DIR   = "./quote_per_labeling"
OUTPUT_DIR    = "./dataset_simboli"
M2_WEIGHTS    = r".\runs\detect\sintesi_genesi\symbol_detector_run1\weights\best.pt"
TARGET_N      = 800
TRAIN_RATIO   = 0.80
VAL_RATIO     = 0.10
RANDOM_SEED   = 42
AUGMENT_X     = 4
CONF_M2       = 0.20
IMG_SIZE_M2   = 640

# ─────────────────────────────────────────
# AUGMENTATION LEGGERA
# ─────────────────────────────────────────

def augment(img, seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    out  = img.copy()
    h, w = out.shape[:2]
    paper = max(200, int(np.percentile(
        cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), 90)))

    # Zoom
    scale = random.uniform(0.6, 1.4)
    nw = max(8, int(w*scale)); nh = max(8, int(h*scale))
    resized = cv2.resize(out,(nw,nh),interpolation=cv2.INTER_AREA)
    canvas  = np.ones((h,w,3),dtype=np.uint8)*paper
    y_off = max(0,(h-nh)//2); x_off = max(0,(w-nw)//2)
    canvas[y_off:y_off+min(nh,h-y_off),
           x_off:x_off+min(nw,w-x_off)] = resized[:min(nh,h-y_off),
                                                    :min(nw,w-x_off)]
    out = canvas

    # Flip
    if random.random() < 0.5:
        out = cv2.flip(out, 1)

    # Scan quality
    f  = out.astype(np.float32)*(random.randint(205,252)/255.0)
    f  = np.clip(f/255.0,0,1)**(1/random.uniform(0.7,1.3))*255.0
    f += np.random.normal(0,random.uniform(1,5),f.shape)
    out = np.clip(f,0,255).astype(np.uint8)

    # Blur leggero
    if random.random() < 0.4:
        k   = random.choice([3,5])
        out = cv2.GaussianBlur(out,(k,k),random.uniform(0.5,1.5))

    # JPEG
    if random.random() < 0.4:
        _,buf = cv2.imencode('.jpg',out,
                             [cv2.IMWRITE_JPEG_QUALITY,random.randint(45,80)])
        out   = cv2.imdecode(buf,cv2.IMREAD_COLOR)

    return out


# ─────────────────────────────────────────
# RACCOLTA CROP BACKGROUND
# ─────────────────────────────────────────

def collect_background_crops():
    labeled_dir = Path(LABELED_DIR)

    # Immagini senza JSON = quote non annotate
    all_imgs  = set(p.stem for p in labeled_dir.glob("*.jpg"))
    all_imgs |= set(p.stem for p in labeled_dir.glob("*.png"))
    annotated = set(p.stem for p in labeled_dir.glob("*.json"))
    unannotated = list(all_imgs - annotated)
    random.shuffle(unannotated)

    crops = []
    print(f"Quote non annotate trovate: {len(unannotated)}")

    for stem in unannotated:
        for ext in [".jpg",".png"]:
            p = labeled_dir / (stem+ext)
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    crops.append(img)
                break

    print(f"Crop background da quote non annotate: {len(crops)}")

    # Falsi positivi M2 — lancia M2 su quote annotate e prende
    # crop dove M2 trova qualcosa ma a bassa confidence
    print("\nRicerca falsi positivi M2...")
    try:
        m2 = YOLO(M2_WEIGHTS)
        annotated_imgs = []
        for stem in list(annotated)[:200]:  # max 200
            for ext in [".jpg",".png"]:
                p = labeled_dir / (stem+ext)
                if p.exists():
                    annotated_imgs.append(p)
                    break

        fp_crops = []
        for img_path in annotated_imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            res = m2.predict(img, imgsz=IMG_SIZE_M2,
                             conf=CONF_M2, verbose=False)[0]
            if res.boxes is None:
                continue
            for sb in res.boxes:
                sc = float(sb.conf[0])
                # Prendi crop a confidence medio-bassa come potenziali FP
                if sc < 0.45:
                    sx1,sy1,sx2,sy2 = map(int,sb.xyxy[0].tolist())
                    sx1=max(0,sx1); sy1=max(0,sy1)
                    sx2=min(iw,sx2); sy2=min(ih,sy2)
                    if sx2-sx1 > 3 and sy2-sy1 > 3:
                        fp_crops.append(img[sy1:sy2,sx1:sx2].copy())

        print(f"Falsi positivi M2 raccolti: {len(fp_crops)}")
        crops.extend(fp_crops)
    except Exception as e:
        print(f"  Skip falsi positivi: {e}")

    random.shuffle(crops)
    print(f"\nTotale crop background: {len(crops)}")
    return crops


# ─────────────────────────────────────────
# SALVA DATASET
# ─────────────────────────────────────────

def save_background(crops):
    if not crops:
        print("Nessun crop background trovato.")
        return

    # Augmentation per raggiungere TARGET_N
    aug_x = max(1, -(-TARGET_N // len(crops)))
    print(f"Augmentation: x{aug_x} → {len(crops)*aug_x} totali")

    all_crops = []
    for crop in crops:
        all_crops.append(crop.copy())
        for i in range(aug_x-1):
            seed = RANDOM_SEED + hash(id(crop)) + i*997
            all_crops.append(augment(crop.copy(), abs(seed)%(2**31)))

    random.shuffle(all_crops)
    all_crops = all_crops[:TARGET_N*2]  # cap

    n_train = int(len(all_crops)*TRAIN_RATIO)
    n_val   = int(len(all_crops)*VAL_RATIO)

    splits = {
        "train": all_crops[:n_train],
        "val":   all_crops[n_train:n_train+n_val],
        "test":  all_crops[n_train+n_val:],
    }

    for split, imgs in splits.items():
        out_dir = Path(OUTPUT_DIR) / split / "background"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        for i, img in enumerate(imgs):
            cv2.imwrite(
                str(out_dir/f"bg_{split}_{i:05d}.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, random.randint(72,95)])
        print(f"  {split:6s}: {len(imgs)} immagini")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("SINTESI: Genesi — Genera classe background (M3)")
    print("=" * 55)

    crops = collect_background_crops()
    save_background(crops)

    print(f"\n✓ Background salvato in {OUTPUT_DIR}/*/background/")
    print(f"\nOra riallena M3:")
    print(f"  python train_symbol_classifier.py")
