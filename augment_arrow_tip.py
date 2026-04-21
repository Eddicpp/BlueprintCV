"""
SINTESI: Genesi — Augmentation arrow_tip → merge in linear
Prende i crop della cartella arrow_tip/, applica augmentation
x40 con massima variabilità e li copia in dataset_simboli/train/linear/

Ogni delle 40 versioni è garantita diversa tramite seed_offset.

Uso:
    1. Metti i crop dei triangoli-freccia in: ./arrow_tip/
    2. Lancia: python augment_arrow_tip.py
    3. Riallena: python train_symbols.py
"""

import cv2
import numpy as np
import random
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

ARROW_TIP_DIR  = "./arrow_tip"
LINEAR_OUT_DIR = "./dataset_simboli/train/linear"
AUGMENT_X      = 40
RANDOM_SEED    = 42

# ─────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────

def augment(img, seed_offset: int):
    """seed_offset garantisce che ogni delle 40 versioni sia diversa."""
    random.seed(seed_offset)
    np.random.seed(seed_offset % (2**32))

    out  = img.copy()
    h, w = out.shape[:2]
    paper = max(200, int(np.percentile(
        cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), 90)))

    # ── Zoom — range molto ampio ──
    scale = random.choices(
        [random.uniform(0.25, 0.50),
         random.uniform(0.50, 0.75),
         random.uniform(0.75, 1.10),
         random.uniform(1.10, 1.60),
         random.uniform(1.60, 2.20)],
        weights=[0.20, 0.25, 0.25, 0.20, 0.10]
    )[0]
    nw = max(8, int(w*scale))
    nh = max(8, int(h*scale))
    resized = cv2.resize(out, (nw,nh), interpolation=cv2.INTER_AREA)
    canvas  = np.ones((h, w, 3), dtype=np.uint8) * paper
    y_off   = random.randint(0, max(0, h-nh))
    x_off   = random.randint(0, max(0, w-nw))
    nh2 = min(nh, h-y_off); nw2 = min(nw, w-x_off)
    canvas[y_off:y_off+nh2, x_off:x_off+nw2] = resized[:nh2,:nw2]
    out = canvas

    # ── Rotazione 360° — i triangoli appaiono in tutte le direzioni ──
    angle = random.uniform(0, 360)
    M     = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    out   = cv2.warpAffine(out, M, (w,h),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(paper,paper,paper))

    # ── Flip ──
    if random.random() < 0.50:
        out = cv2.flip(out, 1)
    if random.random() < 0.40:
        out = cv2.flip(out, 0)

    # ── Shear ──
    if random.random() < 0.30:
        shear = random.uniform(-0.15, 0.15)
        M_sh  = np.float32([[1, shear, 0],[0, 1, 0]])
        out   = cv2.warpAffine(out, M_sh, (w,h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(paper,paper,paper))

    # ── Scan quality ──
    f  = out.astype(np.float32)
    f  = f * (random.randint(195,255)/255.0)
    f[:,:,0] *= random.uniform(0.84,1.0)
    f[:,:,1] *= random.uniform(0.90,1.0)
    gamma = random.uniform(0.55, 1.55)
    f  = np.clip(f/255.0,0,1)**(1/gamma)*255.0
    f  = 128+(f-128)*random.uniform(0.60,0.98)
    f += np.random.normal(0, random.uniform(0.5,7.0), f.shape)
    out = np.clip(f,0,255).astype(np.uint8)

    # ── Blur ──
    if random.random() < 0.55:
        mode = random.choices(
            ["gaussian","motion","defocus","none"],
            weights=[0.35,0.30,0.20,0.15])[0]
        if mode == "gaussian":
            k   = random.choice([3,5,7,9])
            out = cv2.GaussianBlur(out,(k,k),random.uniform(0.3,3.0))
        elif mode == "motion":
            k      = random.choice([5,7,9,11])
            kernel = np.zeros((k,k), dtype=np.float32)
            kernel[k//2,:] = 1.0/k
            M2     = cv2.getRotationMatrix2D((k//2,k//2),
                         random.uniform(0,180),1.0)
            kernel = cv2.warpAffine(kernel,M2,(k,k))
            kernel /= kernel.sum()+1e-6
            out    = cv2.filter2D(out,-1,kernel)
        elif mode == "defocus":
            k      = random.choice([3,5,7,9])
            kernel = np.zeros((k,k), dtype=np.float32)
            cv2.circle(kernel,(k//2,k//2),k//2,1,-1)
            kernel /= kernel.sum()
            out    = cv2.filter2D(out,-1,kernel)

    # ── JPEG ──
    if random.random() < 0.50:
        q     = random.randint(25,85)
        _,buf = cv2.imencode('.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,q])
        out   = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # ── Erode ──
    if random.random() < 0.45:
        it       = random.choice([1,1,2])
        kernel_e = np.ones((random.choice([2,2,3]),
                            random.choice([2,2,3])), np.uint8)
        dark     = out < random.randint(60,100)
        eroded   = cv2.erode(out, kernel_e, iterations=it)
        out[dark] = eroded[dark]

    # ── Dilate ──
    if random.random() < 0.15:
        kernel_d = np.ones((2,2), np.uint8)
        dark2    = out < 80
        dilated  = cv2.dilate(out, kernel_d, iterations=1)
        out[dark2] = dilated[dark2]

    # ── Cancellazione parziale ──
    for _ in range(random.randint(0,3)):
        ew = random.randint(1, max(2,w//3))
        eh = random.randint(1, max(2,h//3))
        ex = random.randint(0, max(0,w-ew))
        ey = random.randint(0, max(0,h-eh))
        cv2.rectangle(out,(ex,ey),(ex+ew,ey+eh),
                      (random.randint(200,255),)*3,-1)

    # ── Sale e pepe ──
    if random.random() < 0.20:
        for _ in range(random.randint(1,max(2,int(w*h*0.02)))):
            out[random.randint(0,h-1),
                random.randint(0,w-1)] = random.choice([0,255])

    return out


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("SINTESI: Genesi — Augmentation arrow_tip → linear")
    print("=" * 55)

    src = Path(ARROW_TIP_DIR)
    dst = Path(LINEAR_OUT_DIR)
    dst.mkdir(parents=True, exist_ok=True)

    imgs = sorted(list(src.glob("*.jpg")) +
                  list(src.glob("*.png")))

    if not imgs:
        print(f"Nessuna immagine trovata in {src}")
        print("Metti i crop dei triangoli-freccia in quella cartella.")
        exit(1)

    print(f"Crop arrow_tip trovati: {len(imgs)}")
    print(f"Augmentation: x{AUGMENT_X}")
    print(f"Totale atteso: {len(imgs) * AUGMENT_X}")
    print(f"Output: {dst}\n")

    saved = 0
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Originale
        cv2.imwrite(str(dst / f"arrow_{img_path.stem}_orig.jpg"),
                    img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

        # 39 versioni diverse — seed unico per ognuna
        for i in range(AUGMENT_X - 1):
            seed = RANDOM_SEED + hash(img_path.stem) + i * 997
            aug  = augment(img.copy(), seed_offset=abs(seed) % (2**31))
            cv2.imwrite(
                str(dst / f"arrow_{img_path.stem}_aug{i:03d}.jpg"),
                aug,
                [cv2.IMWRITE_JPEG_QUALITY, random.randint(72,95)])
            saved += 1

    print(f"✓ Salvate {saved} immagini in {dst}")
    print(f"\nOra riallena:")
    print(f"  python train_symbols.py")