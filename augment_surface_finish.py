"""
SINTESI: Genesi — Augmentation surface_finish reale
Prende i crop REALI di surface_finish, applica augmentation
e sostituisce la cartella nel dataset_simboli.

Uso:
    1. Metti i crop reali in: ./surface_finish_real/
    2. Lancia: python augment_surface_finish.py
    3. Riallena: python train_symbols.py
"""

import cv2
import numpy as np
import random
import shutil
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

SOURCE_DIR  = "./surface_finish_real"
OUTPUT_DIR  = "./dataset_simboli/train/surface_finish"
AUGMENT_X   = 10     # x10 — porta ~48 reali a ~480
RANDOM_SEED = 42

# ─────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────

def augment(img):
    out  = img.copy()
    h, w = out.shape[:2]
    paper = max(200, int(np.percentile(
        cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), 90)))

    # Zoom in/out
    scale = random.choices(
        [random.uniform(0.40, 0.65),
         random.uniform(0.65, 0.90),
         random.uniform(0.90, 1.20),
         random.uniform(1.20, 1.60)],
        weights=[0.25, 0.35, 0.25, 0.15]
    )[0]
    nw = max(8, int(w*scale))
    nh = max(8, int(h*scale))
    resized = cv2.resize(out, (nw,nh), interpolation=cv2.INTER_AREA)
    canvas  = np.ones((h,w,3), dtype=np.uint8)*paper
    y_off   = max(0,(h-nh)//2); x_off = max(0,(w-nw)//2)
    nh2=min(nh,h-y_off); nw2=min(nw,w-x_off)
    canvas[y_off:y_off+nh2, x_off:x_off+nw2] = resized[:nh2,:nw2]
    out = canvas

    # Rotazione leggera
    if random.random() < 0.40:
        angle = random.uniform(-12, 12)
        M     = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
        out   = cv2.warpAffine(out, M, (w,h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(paper,paper,paper))

    # Flip orizzontale
    if random.random() < 0.50:
        out = cv2.flip(out, 1)

    # Scan quality
    if random.random() < 0.90:
        f  = out.astype(np.float32)
        f  = f*(random.randint(208,252)/255.0)
        f[:,:,0] *= random.uniform(0.87,0.98)
        f  = np.clip(f/255.0,0,1)**(1/random.uniform(0.65,1.40))*255.0
        f  = 128+(f-128)*random.uniform(0.72,0.96)
        f += np.random.normal(0,random.uniform(1.5,5.5),f.shape)
        out = np.clip(f,0,255).astype(np.uint8)

    # Blur
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
                         random.choice([0,45,90]),1.0)
            kernel = cv2.warpAffine(kernel,M2,(k,k))
            kernel /= kernel.sum()+1e-6
            out    = cv2.filter2D(out,-1,kernel)
        else:
            k      = random.choice([3,5,7])
            kernel = np.zeros((k,k), dtype=np.float32)
            cv2.circle(kernel,(k//2,k//2),k//2,1,-1)
            kernel /= kernel.sum()
            out    = cv2.filter2D(out,-1,kernel)

    # JPEG
    if random.random() < 0.40:
        _,buf = cv2.imencode('.jpg',out,
                             [cv2.IMWRITE_JPEG_QUALITY,
                              random.randint(40,78)])
        out   = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Erode
    if random.random() < 0.35:
        kernel   = np.ones((2,2), np.uint8)
        dark     = out < 80
        eroded   = cv2.erode(out, kernel, iterations=1)
        out[dark] = eroded[dark]

    # Cancellazione parziale
    if random.random() < 0.25:
        ew = random.randint(2, max(3,w//4))
        eh = random.randint(2, max(3,h//4))
        ex = random.randint(0, max(0,w-ew))
        ey = random.randint(0, max(0,h-eh))
        cv2.rectangle(out,(ex,ey),(ex+ew,ey+eh),
                      (paper,paper,paper),-1)

    return out


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("SINTESI: Genesi — Augmentation surface_finish reale")
    print("=" * 55)

    src = Path(SOURCE_DIR)
    dst = Path(OUTPUT_DIR)

    imgs = sorted(list(src.glob("*.jpg")) +
                  list(src.glob("*.png")))

    if not imgs:
        print(f"Nessuna immagine trovata in {src}")
        print("Metti i crop reali di surface_finish in quella cartella.")
        exit(1)

    print(f"Crop reali trovati: {len(imgs)}")
    print(f"Augmentation: x{AUGMENT_X}")
    print(f"Totale atteso: {len(imgs) * AUGMENT_X}")

    # Svuota la cartella esistente (rimuove i sintetici precedenti)
    if dst.exists():
        shutil.rmtree(dst)
        print(f"\nRimossa cartella esistente: {dst}")
    dst.mkdir(parents=True)

    saved = 0
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Originale
        cv2.imwrite(str(dst / f"sf_{img_path.stem}_orig.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

        # Augmentation
        for i in range(AUGMENT_X - 1):
            aug = augment(img.copy())
            cv2.imwrite(
                str(dst / f"sf_{img_path.stem}_aug{i:03d}.jpg"),
                aug,
                [cv2.IMWRITE_JPEG_QUALITY, random.randint(72,95)])
            saved += 1

    print(f"\n✓ Salvate {saved} immagini in {dst}")
    print(f"\nOra riallena:")
    print(f"  python train_symbols.py")
