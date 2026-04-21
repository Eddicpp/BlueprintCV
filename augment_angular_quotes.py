"""
SINTESI: Genesi — Augmentation quote angolari
Prende crop reali di quote angolari da angular_quotes/,
applica x40 augmentation e genera sintetici simili.
Aggiunge tutto al dataset_yolo_aug/train/ con label corrette.

Uso:
    1. Metti i crop reali in: ./angular_quotes/
    2. Lancia: python augment_angular_quotes.py
    3. Rigenera dataset_finale e riallena
"""

import cv2
import numpy as np
import random
import shutil
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

SOURCE_DIR   = "./angular_quotes"
OUT_IMG_DIR  = "./dataset_yolo_aug/train/images"
OUT_LBL_DIR  = "./dataset_yolo_aug/train/labels"
AUGMENT_X    = 5
N_SYNTHETIC  = 200   # quanti sintetici generare
RANDOM_SEED  = 42
QUOTE_CLS_ID = 2     # classe quote nel detector M1

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

    # Zoom — soprattutto out perché le quote angolari sono piccole
    scale = random.choices(
        [random.uniform(0.40, 0.65),
         random.uniform(0.65, 0.90),
         random.uniform(0.90, 1.20),
         random.uniform(1.20, 1.80)],
        weights=[0.15, 0.30, 0.35, 0.20]
    )[0]
    nw = max(8, int(w*scale))
    nh = max(8, int(h*scale))
    resized = cv2.resize(out,(nw,nh),interpolation=cv2.INTER_AREA)
    canvas  = np.ones((h,w,3),dtype=np.uint8)*paper
    y_off   = random.randint(0, max(0,h-nh))
    x_off   = random.randint(0, max(0,w-nw))
    canvas[y_off:y_off+min(nh,h-y_off),
           x_off:x_off+min(nw,w-x_off)] = resized[:min(nh,h-y_off),
                                                    :min(nw,w-x_off)]
    out = canvas

    # Rotazione leggera — le quote angolari sono spesso inclinate
    if random.random() < 0.60:
        angle = random.uniform(-20, 20)
        M     = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
        out   = cv2.warpAffine(out, M, (w,h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(paper,paper,paper))

    # Flip orizzontale
    if random.random() < 0.50:
        out = cv2.flip(out, 1)

    # Scan quality
    f  = out.astype(np.float32)*(random.randint(195,252)/255.0)
    f[:,:,0] *= random.uniform(0.85,0.99)
    gamma = random.uniform(0.60, 1.45)
    f  = np.clip(f/255.0,0,1)**(1/gamma)*255.0
    f  = 128+(f-128)*random.uniform(0.65,0.97)
    f += np.random.normal(0, random.uniform(1.0,6.0), f.shape)
    out = np.clip(f,0,255).astype(np.uint8)

    # Blur
    if random.random() < 0.50:
        mode = random.choice(["gaussian","motion","defocus"])
        if mode == "gaussian":
            k   = random.choice([3,5,7])
            out = cv2.GaussianBlur(out,(k,k),random.uniform(0.5,2.0))
        elif mode == "motion":
            k      = random.choice([5,7,9])
            kernel = np.zeros((k,k), dtype=np.float32)
            kernel[k//2,:] = 1.0/k
            M2     = cv2.getRotationMatrix2D((k//2,k//2),
                         random.uniform(0,180),1.0)
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
    if random.random() < 0.45:
        q     = random.randint(30,80)
        _,buf = cv2.imencode('.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,q])
        out   = cv2.imdecode(buf,cv2.IMREAD_COLOR)

    # Erode
    if random.random() < 0.40:
        k_e  = np.ones((2,2),np.uint8)
        dark = out < random.randint(60,100)
        out[dark] = cv2.erode(out,k_e,iterations=1)[dark]

    return out


# ─────────────────────────────────────────
# GENERAZIONE SINTETICA
# ─────────────────────────────────────────

def make_angular_quote(seed: int):
    """
    Genera una quota angolare sintetica:
    numero piccolo con freccia/simbolo vicino,
    orientamento verticale o inclinato.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))

    # Dimensioni simili ai crop reali
    w = random.randint(20, 60)
    h = random.randint(30, 80)
    paper = random.randint(220, 255)
    img   = np.ones((h,w,3),dtype=np.uint8)*paper

    dark  = random.randint(10, 60)
    color = (dark,dark,dark)
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # Numero della quota
    val   = str(random.choice([5,6,8,9,10,12,15,16,20,25,30,45,60,90,120]))
    fs    = random.uniform(0.25, 0.45)
    (tw,th),_ = cv2.getTextSize(val, font, fs, 1)
    tx = max(0, (w-tw)//2)
    ty = random.randint(th+2, max(th+3, h-th-2))
    cv2.putText(img, val, (tx,ty), font, fs, color, 1, cv2.LINE_AA)

    # Simbolo sotto o sopra il numero
    sym_type = random.choice(["arrow_down","arrow_up","arc","dot","line"])
    cx = w//2

    if sym_type == "arrow_down":
        y_start = ty+3
        y_end   = min(h-2, ty+random.randint(5,12))
        cv2.arrowedLine(img,(cx,y_start),(cx,y_end),color,1,
                        tipLength=0.4,line_type=cv2.LINE_AA)

    elif sym_type == "arrow_up":
        y_start = ty-th-3
        y_end   = max(2, ty-th-random.randint(5,12))
        cv2.arrowedLine(img,(cx,y_start),(cx,y_end),color,1,
                        tipLength=0.4,line_type=cv2.LINE_AA)

    elif sym_type == "arc":
        r   = random.randint(4,10)
        cy2 = ty+r+2
        cv2.ellipse(img,(cx,cy2),(r,r),0,0,180,color,1,cv2.LINE_AA)

    elif sym_type == "dot":
        cv2.circle(img,(cx,ty+4),1,color,-1)

    else:  # line
        lw2 = random.randint(4, w-4)
        lx  = (w-lw2)//2
        cv2.line(img,(lx,ty+4),(lx+lw2,ty+4),color,1)

    # Rotazione leggera
    if random.random() < 0.5:
        angle = random.uniform(-15,15)
        M     = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
        img   = cv2.warpAffine(img, M, (w,h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(paper,paper,paper))

    return img


# ─────────────────────────────────────────
# SALVA IMMAGINE + LABEL YOLO
# ─────────────────────────────────────────

def save_as_blueprint_patch(crop_img, name: str,
                             img_dir: Path, lbl_dir: Path):
    """
    Inserisce il crop in un canvas bianco (simula un blueprint)
    e genera la label YOLO con la quota al centro.
    """
    ch, cw = crop_img.shape[:2]
    # Canvas più grande del crop
    canvas_w = random.randint(cw*3, cw*6)
    canvas_h = random.randint(ch*3, ch*6)
    paper    = random.randint(230, 255)
    canvas   = np.ones((canvas_h,canvas_w,3),dtype=np.uint8)*paper

    # Posizione random del crop nel canvas
    ox = random.randint(cw, canvas_w-2*cw)
    oy = random.randint(ch, canvas_h-2*ch)
    canvas[oy:oy+ch, ox:ox+cw] = crop_img

    # Aggiungi linee di sfondo — NON devono passare per il crop
    # Zona proibita: il rettangolo del crop con margine
    margin  = 4
    crop_x1 = ox - margin
    crop_y1 = oy - margin
    crop_x2 = ox + cw + margin
    crop_y2 = oy + ch + margin

    def line_intersects_crop(x1, y1, x2, y2):
        """Controlla se il segmento interseca il crop."""
        # Bounding box del segmento
        lx1=min(x1,x2); ly1=min(y1,y2)
        lx2=max(x1,x2); ly2=max(y1,y2)
        # Se i bbox non si sovrappongono, no intersezione
        if lx2 < crop_x1 or lx1 > crop_x2:
            return False
        if ly2 < crop_y1 or ly1 > crop_y2:
            return False
        return True

    n_lines = random.randint(15, 40)
    attempts = 0
    drawn    = 0
    while drawn < n_lines and attempts < n_lines * 10:
        attempts += 1
        x1 = random.randint(0, canvas_w)
        y1 = random.randint(0, canvas_h)
        x2 = random.randint(0, canvas_w)
        y2 = random.randint(0, canvas_h)
        if line_intersects_crop(x1,y1,x2,y2):
            continue
        cv2.line(canvas,(x1,y1),(x2,y2),
                 (random.randint(130,210),)*3, 1)
        drawn += 1

    # Salva immagine
    img_path = img_dir / f"{name}.jpg"
    cv2.imwrite(str(img_path), canvas,
                [cv2.IMWRITE_JPEG_QUALITY, random.randint(72,95)])

    # Label YOLO — bbox del crop nel canvas
    cx_n = (ox + cw/2) / canvas_w
    cy_n = (oy + ch/2) / canvas_h
    bw_n = cw / canvas_w
    bh_n = ch / canvas_h
    lbl_path = lbl_dir / f"{name}.txt"
    with open(lbl_path, "w") as f:
        f.write(f"{QUOTE_CLS_ID} {cx_n:.6f} {cy_n:.6f} "
                f"{bw_n:.6f} {bh_n:.6f}\n")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("SINTESI: Genesi — Augmentation quote angolari")
    print("=" * 55)

    src     = Path(SOURCE_DIR)
    img_dir = Path(OUT_IMG_DIR)
    lbl_dir = Path(OUT_LBL_DIR)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    reals = sorted(list(src.glob("*.jpg")) +
                   list(src.glob("*.png")))

    if not reals:
        print(f"Nessun crop trovato in {src}")
        print("Metti i crop reali in quella cartella.")
        exit(1)

    print(f"Crop reali trovati: {len(reals)}")
    saved = 0

    # ── Augmentation sui reali ──
    print(f"\nAugmentation x{AUGMENT_X} sui reali...")
    for img_path in reals:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for i in range(AUGMENT_X):
            seed = RANDOM_SEED + hash(img_path.stem) + i*997
            aug  = augment(img.copy(), abs(seed)%(2**31))
            name = f"angular_real_{img_path.stem}_aug{i:03d}"
            save_as_blueprint_patch(aug, name, img_dir, lbl_dir)
            saved += 1

    print(f"  Salvate {saved} immagini da reali")

    # ── Sintetici ──
    print(f"\nGenerazione {N_SYNTHETIC} sintetici...")
    syn_saved = 0
    for i in range(N_SYNTHETIC):
        seed = RANDOM_SEED + i*1337
        syn  = make_angular_quote(seed)
        # Applica augmentation anche al sintetico
        aug_seed = seed + 42
        syn = augment(syn, aug_seed % (2**31))
        name = f"angular_syn_{i:05d}"
        save_as_blueprint_patch(syn, name, img_dir, lbl_dir)
        syn_saved += 1

    print(f"  Salvati {syn_saved} sintetici")
    print(f"\n✓ Totale aggiunto al dataset: {saved+syn_saved}")
    print(f"\nOra rigenera dataset_finale e riallena M1:")
    print(f"  python merge_datasets.py")
    print(f"  python resize_dataset.py")
    print(f"  python train_yolov8.py")