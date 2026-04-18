"""
SINTESI: Genesi — Generatore blueprint strutturato (unificato)
Unisce generate_mosaics.py e generate_synthetic_structured.py.

Ogni immagine generata ha:
  - border: cornice del foglio tecnico
  - table:  1-3 cartigli posizionati sui bordi INTERNI (non fuori)
            negli angoli e sui lati del border
  - quote:  mix di ritagli reali (mosaici) + geometrie sintetiche
            con frecce e numeri, distribuite nell'area interna

Output: dataset_blueprint_strutturato/
"""

import cv2
import numpy as np
import random
import math
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

TRAIN_IMAGES = "./dataset_yolo/train/images"
TRAIN_LABELS = "./dataset_yolo/train/labels"
OUTPUT_DIR   = "./dataset_blueprint_strutturato"
CLASSES      = ["border", "table", "quote"]

N_IMAGES     = 600
IMG_W        = 1600
IMG_H        = 1200
RANDOM_SEED  = 42

TRAIN_RATIO  = 0.85
VAL_RATIO    = 0.10

MIN_QUOTES   = 10
MAX_QUOTES   = 35

# Probabilità tipo quota
P_REAL_CROP  = 0.60   # quota reale ritagliata
P_SYNTHETIC  = 0.40   # quota sintetica (geometria + freccia + numero)

# Modalità posizionamento quote
P_OVERLAP    = 0.35
P_TOUCHING   = 0.35
P_SCATTERED  = 0.30

# ─────────────────────────────────────────
# UTILITÀ
# ─────────────────────────────────────────

def save_labels(path: Path, boxes: list):
    with open(path, "w") as f:
        for b in boxes:
            f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")


def abs_to_yolo(x1, y1, x2, y2, W, H):
    cx = (x1+x2)/2/W; cy = (y1+y2)/2/H
    w  = (x2-x1)/W;   h  = (y2-y1)/H
    return (max(0,min(1,cx)), max(0,min(1,cy)),
            max(0,min(1,w)),  max(0,min(1,h)))


def yolo_to_abs(cx, cy, w, h, W, H):
    x1=int((cx-w/2)*W); y1=int((cy-h/2)*H)
    x2=int((cx+w/2)*W); y2=int((cy+h/2)*H)
    return max(0,x1), max(0,y1), min(W,x2), min(H,y2)


def iou(b1, b2):
    ix1=max(b1[0],b2[0]); iy1=max(b1[1],b2[1])
    ix2=min(b1[2],b2[2]); iy2=min(b1[3],b2[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1])
    a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    union=a1+a2-inter
    return inter/union if union>0 else 0


# ─────────────────────────────────────────
# RACCOLTA RITAGLI QUOTE REALI
# ─────────────────────────────────────────

def collect_quote_crops(images_dir: Path, labels_dir: Path):
    crops = []
    img_paths = sorted(list(images_dir.glob("*.png")) +
                       list(images_dir.glob("*.jpg")))
    print(f"Raccolta ritagli da {len(img_paths)} immagini...")

    for img_path in img_paths:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5 or int(parts[0]) != 2:
                continue
            cx,cy,bw,bh = map(float, parts[1:])
            x1,y1,x2,y2 = yolo_to_abs(cx,cy,bw,bh,iw,ih)
            if x2-x1<10 or y2-y1<10:
                continue
            m = random.randint(2,6)
            patch = img[max(0,y1-m):min(ih,y2+m),
                        max(0,x1-m):min(iw,x2+m)].copy()
            if patch.size > 0:
                crops.append(patch)

    print(f"  Ritagli: {len(crops)}")
    return crops


# ─────────────────────────────────────────
# SFONDO CARTA
# ─────────────────────────────────────────

def make_background(w, h):
    paper = random.randint(230, 250)
    bg    = np.ones((h, w, 3), dtype=np.uint8) * paper
    noise = np.random.normal(0, 2, (h, w, 3))
    bg    = np.clip(bg.astype(np.float32)+noise, 218, 255).astype(np.uint8)
    bg[:,:,0] = np.clip(bg[:,:,0].astype(np.int16)-random.randint(3,10),
                        0, 255).astype(np.uint8)

    # Linee di costruzione sparse (blueprint)
    if random.random() > 0.5:
        color = (random.randint(190,215),)*3
        for y in range(0, h, random.randint(40,150)):
            cv2.line(bg, (0,y), (w,y), color, 1)
        for x in range(0, w, random.randint(40,150)):
            cv2.line(bg, (x,0), (x,h), color, 1)

    return bg


# ─────────────────────────────────────────
# BORDER
# ─────────────────────────────────────────

def draw_border(canvas):
    h, w   = canvas.shape[:2]
    mx     = random.randint(30, 70)
    my     = random.randint(30, 70)
    color  = (random.randint(15,55),)*3
    thick  = random.choice([1,2])
    inner  = random.randint(8,16)

    x1,y1,x2,y2 = mx, my, w-mx, h-my
    cv2.rectangle(canvas, (x1,y1), (x2,y2), color, thick)
    cv2.rectangle(canvas, (x1+inner,y1+inner),
                  (x2-inner,y2-inner), color, 1)
    return x1,y1,x2,y2, inner


# ─────────────────────────────────────────
# TABELLE SUI BORDI INTERNI
# ─────────────────────────────────────────

def draw_tables_on_border(canvas, bx1, by1, bx2, by2, inner):
    """
    Disegna 1-3 tabelle posizionate sui bordi INTERNI del foglio.
    Posizioni disponibili: angolo BR, angolo TR, lato bottom, lato right, angolo BL.
    Mai fuori dal border.
    """
    h, w   = canvas.shape[:2]
    color  = (random.randint(15,55),)*3
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fields = ["Scala","Data","Foglio","Rev.","Titolo","Prog.","N°","Mat.","Peso"]

    # Posizioni disponibili sui bordi interni
    positions = random.sample(
        ["corner_br", "corner_tr", "corner_bl", "side_bottom", "side_right"],
        k=random.randint(1, 3)
    )

    table_boxes = []

    for pos in positions:
        cols   = random.randint(3, 5)
        rows   = random.randint(3, 5)
        col_w  = random.randint(60, 100)
        row_h  = random.randint(18, 30)
        tw     = cols * col_w
        th     = rows * row_h
        gap    = inner + random.randint(2, 8)

        if pos == "corner_br":
            tx1 = bx2 - tw - gap
            ty1 = by2 - th - gap
        elif pos == "corner_tr":
            tx1 = bx2 - tw - gap
            ty1 = by1 + gap
        elif pos == "corner_bl":
            tx1 = bx1 + gap
            ty1 = by2 - th - gap
        elif pos == "side_bottom":
            tx1 = bx1 + (bx2-bx1)//2 - tw//2
            ty1 = by2 - th - gap
        else:  # side_right
            tx1 = bx2 - tw - gap
            ty1 = by1 + (by2-by1)//2 - th//2

        # Clamp sempre dentro il border
        tx1 = max(bx1+gap, min(bx2-tw-2, tx1))
        ty1 = max(by1+gap, min(by2-th-2, ty1))
        tx2 = tx1 + tw
        ty2 = ty1 + th

        # Salta se troppo piccola
        if tx2-tx1 < 20 or ty2-ty1 < 20:
            continue

        # Sfondo bianco
        cv2.rectangle(canvas, (tx1,ty1), (tx2,ty2), (255,255,255), -1)
        # Griglia
        for c in range(cols+1):
            cv2.line(canvas, (tx1+c*col_w,ty1), (tx1+c*col_w,ty2), color, 1)
        for r in range(rows+1):
            cv2.line(canvas, (tx1,ty1+r*row_h), (tx2,ty1+r*row_h), color, 1)
        # Testo
        for r in range(rows):
            for c in range(cols):
                text = random.choice(fields) if r==0 else \
                       str(random.randint(1,999)) if r>1 else ""
                cv2.putText(canvas, text,
                            (tx1+c*col_w+3, ty1+r*row_h+row_h-5),
                            font, 0.28, color, 1, cv2.LINE_AA)

        table_boxes.append((tx1, ty1, tx2, ty2))

    return table_boxes


# ─────────────────────────────────────────
# QUOTA SINTETICA
# ─────────────────────────────────────────

def make_synthetic_quote(max_w=200, max_h=80):
    """
    Genera una quota sintetica su sfondo bianco/carta:
    linea di misura + frecce + numero.
    Tipi: lineare orizzontale, lineare verticale, angolare, radiale.
    """
    q_type = random.choice(["linear_h", "linear_v", "angular", "radial"])
    paper  = random.randint(230, 252)
    patch  = np.ones((max_h, max_w, 3), dtype=np.uint8) * paper
    color  = (random.randint(15,60),)*3
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fs     = random.uniform(0.3, 0.6)
    value  = random.choice([
        str(random.randint(1,9999)),
        f"{random.uniform(0.1,99.9):.1f}",
        f"Ø{random.randint(5,200)}",
        f"R{random.randint(2,100)}",
        f"{random.randint(1,180)}°",
    ])

    cx, cy = max_w//2, max_h//2

    if q_type == "linear_h":
        length = random.randint(max_w//3, max_w*3//4)
        x1l = cx - length//2; x2l = cx + length//2
        cv2.line(patch, (x1l,cy), (x2l,cy), color, 1, cv2.LINE_AA)
        cv2.arrowedLine(patch, (x2l,cy), (x1l,cy), color, 1,
                        tipLength=0.15, line_type=cv2.LINE_AA)
        cv2.arrowedLine(patch, (x1l,cy), (x2l,cy), color, 1,
                        tipLength=0.15, line_type=cv2.LINE_AA)
        cv2.line(patch, (x1l,cy-8),(x1l,cy+8), color, 1)
        cv2.line(patch, (x2l,cy-8),(x2l,cy+8), color, 1)
        (tw,th),_ = cv2.getTextSize(value, font, fs, 1)
        cv2.putText(patch, value, (cx-tw//2, cy-6), font, fs, color, 1, cv2.LINE_AA)

    elif q_type == "linear_v":
        length = random.randint(max_h//3, max_h*3//4)
        y1l = cy - length//2; y2l = cy + length//2
        cv2.line(patch, (cx,y1l), (cx,y2l), color, 1, cv2.LINE_AA)
        cv2.arrowedLine(patch, (cx,y2l), (cx,y1l), color, 1,
                        tipLength=0.15, line_type=cv2.LINE_AA)
        cv2.arrowedLine(patch, (cx,y1l), (cx,y2l), color, 1,
                        tipLength=0.15, line_type=cv2.LINE_AA)
        cv2.line(patch, (cx-8,y1l),(cx+8,y1l), color, 1)
        cv2.line(patch, (cx-8,y2l),(cx+8,y2l), color, 1)
        (tw,th),_ = cv2.getTextSize(value, font, fs, 1)
        cv2.putText(patch, value, (cx+6, cy+th//2), font, fs, color, 1, cv2.LINE_AA)

    elif q_type == "angular":
        r   = random.randint(20, min(cx,cy)-5)
        a1  = random.randint(0, 270)
        a2  = a1 + random.randint(15, 90)
        cv2.ellipse(patch, (cx,cy), (r,r), 0, a1, a2, color, 1, cv2.LINE_AA)
        for a in [a1, a2]:
            rad = math.radians(a)
            cv2.line(patch, (cx,cy),
                     (cx+int((r+15)*math.cos(rad)),
                      cy-int((r+15)*math.sin(rad))), color, 1)
        mid = math.radians((a1+a2)/2)
        tx  = cx + int((r+18)*math.cos(mid))
        ty  = cy - int((r+18)*math.sin(mid))
        (tw,th),_ = cv2.getTextSize(value, font, fs, 1)
        cv2.putText(patch, value, (tx-tw//2, ty+th//2),
                    font, fs, color, 1, cv2.LINE_AA)

    else:  # radial
        r = random.randint(15, min(cx,cy)-5)
        cv2.circle(patch, (cx,cy), r, color, 1, cv2.LINE_AA)
        angle = random.uniform(0, 2*math.pi)
        px2   = cx + int((r+30)*math.cos(angle))
        py2   = cy + int((r+30)*math.sin(angle))
        cv2.line(patch, (cx,cy), (px2,py2), color, 1)
        cv2.arrowedLine(patch, (cx,cy),
                        (cx+int(r*math.cos(angle)),
                         cy+int(r*math.sin(angle))),
                        color, 1, tipLength=0.2)
        (tw,th),_ = cv2.getTextSize(value, font, fs, 1)
        cv2.putText(patch, value, (px2+3, py2+th//2),
                    font, fs, color, 1, cv2.LINE_AA)

    # Ritaglia al contenuto effettivo
    gray    = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, paper-10, 255, cv2.THRESH_BINARY_INV)
    coords  = cv2.findNonZero(mask)
    if coords is not None:
        rx,ry,rw,rh = cv2.boundingRect(coords)
        pad = 4
        rx  = max(0, rx-pad); ry = max(0, ry-pad)
        rw  = min(max_w-rx, rw+2*pad)
        rh  = min(max_h-ry, rh+2*pad)
        patch = patch[ry:ry+rh, rx:rx+rw]

    return patch


# ─────────────────────────────────────────
# AUGMENTATION RITAGLIO
# ─────────────────────────────────────────

def augment_crop(patch):
    out = patch.copy()
    if random.random() < 0.6:
        s  = random.uniform(0.5, 1.8)
        nw = max(8, int(out.shape[1]*s))
        nh = max(8, int(out.shape[0]*s))
        out = cv2.resize(out, (nw,nh), interpolation=cv2.INTER_AREA)
    if random.random() < 0.3:
        angle = random.uniform(-15,15)
        h2,w2 = out.shape[:2]
        M  = cv2.getRotationMatrix2D((w2/2,h2/2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w2,h2),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(245,245,240))
    if random.random() < 0.25:
        out = cv2.GaussianBlur(out, (3,3), 0)
    if random.random() < 0.3:
        alpha = random.uniform(0.5,0.9)
        white = np.ones_like(out)*245
        out   = cv2.addWeighted(out, alpha, white, 1-alpha, 0)
    return out


# ─────────────────────────────────────────
# POSIZIONAMENTO QUOTE NELL'AREA INTERNA
# ─────────────────────────────────────────

def place_quotes(canvas, crops, area_x1, area_y1, area_x2, area_y2,
                 occupied, n_quotes):
    """
    Posiziona quote (reali o sintetiche) nell'area interna al border,
    evitando le zone occupate dalle tabelle.
    """
    placed    = list(occupied)
    yolo_boxes = []
    cls_quote = CLASSES.index("quote")

    mode = random.choices(
        ["overlap","touching","scattered"],
        weights=[P_OVERLAP, P_TOUCHING, P_SCATTERED]
    )[0]

    for _ in range(n_quotes):
        # Scegli tipo quota
        if crops and random.random() < P_REAL_CROP:
            patch = augment_crop(random.choice(crops))
        else:
            patch = augment_crop(make_synthetic_quote())

        ph, pw = patch.shape[:2]

        # Scala se troppo grande per l'area
        max_dim = min(area_x2-area_x1, area_y2-area_y1) // 3
        if max(ph,pw) > max_dim:
            s  = max_dim / max(ph,pw)
            pw = max(8, int(pw*s))
            ph = max(8, int(ph*s))
            patch = cv2.resize(patch, (pw,ph))

        placed_ok = False
        for _ in range(60):
            if mode == "scattered" or not placed:
                px = random.randint(area_x1+5, max(area_x1+6, area_x2-pw-5))
                py = random.randint(area_y1+5, max(area_y1+6, area_y2-ph-5))
                new_box = (px,py,px+pw,py+ph)
                if any(iou(new_box,b)>0.05 for b in placed):
                    continue

            elif mode == "touching":
                ref  = random.choice(placed[-5:] if len(placed)>5 else placed)
                side = random.choice(["right","below","left","above"])
                gap  = random.randint(1,8)
                if side=="right":   px=ref[2]+gap; py=ref[1]+random.randint(-ph//3,ph//3)
                elif side=="below": py=ref[3]+gap; px=ref[0]+random.randint(-pw//3,pw//3)
                elif side=="left":  px=ref[0]-pw-gap; py=ref[1]+random.randint(-ph//3,ph//3)
                else:               py=ref[1]-ph-gap; px=ref[0]+random.randint(-pw//3,pw//3)
                px = max(area_x1+5, min(area_x2-pw-5, px))
                py = max(area_y1+5, min(area_y2-ph-5, py))
                new_box = (px,py,px+pw,py+ph)
                if any(iou(new_box,b)>0.05 for b in placed):
                    continue

            else:  # overlap
                ref = random.choice(placed[-5:] if len(placed)>5 else placed)
                overlap_px = random.randint(int(pw*0.05), int(pw*0.20))
                overlap_py = random.randint(int(ph*0.05), int(ph*0.20))
                side = random.choice(["right","below","left","above"])
                if side=="right":  px=ref[2]-overlap_px; py=ref[1]+random.randint(-ph//4,ph//4)
                elif side=="below":py=ref[3]-overlap_py; px=ref[0]+random.randint(-pw//4,pw//4)
                elif side=="left": px=ref[0]-pw+overlap_px; py=ref[1]+random.randint(-ph//4,ph//4)
                else:              py=ref[1]-ph+overlap_py; px=ref[0]+random.randint(-pw//4,pw//4)
                px = max(area_x1+5, min(area_x2-pw-5, px))
                py = max(area_y1+5, min(area_y2-ph-5, py))
                new_box = (px,py,px+pw,py+ph)
                if any(iou(new_box,b)>0.20 for b in placed):
                    continue

            # Incolla
            x1,y1,x2,y2 = new_box
            x2=min(IMG_W,x2); y2=min(IMG_H,y2)
            apw=x2-x1; aph=y2-y1
            if apw<4 or aph<4:
                continue
            crop_r = cv2.resize(patch, (apw,aph)) \
                     if patch.shape[:2]!=(aph,apw) else patch
            alpha  = random.uniform(0.78,1.0)
            region = canvas[y1:y2,x1:x2].copy()
            canvas[y1:y2,x1:x2] = cv2.addWeighted(
                region,1-alpha, crop_r,alpha, 0)

            placed.append(new_box)
            yolo_boxes.append([cls_quote,
                                *abs_to_yolo(x1,y1,x2,y2,IMG_W,IMG_H)])
            placed_ok = True
            break

    return yolo_boxes


# ─────────────────────────────────────────
# GENERATORE IMMAGINE
# ─────────────────────────────────────────

def generate_image(idx: int, crops: list,
                   out_img: Path, out_lbl: Path):
    canvas     = make_background(IMG_W, IMG_H)
    yolo_boxes = []

    cls_border = CLASSES.index("border")
    cls_table  = CLASSES.index("table")

    # 1. BORDER
    bx1,by1,bx2,by2,inner = draw_border(canvas)
    yolo_boxes.append([cls_border,
                       *abs_to_yolo(bx1,by1,bx2,by2,IMG_W,IMG_H)])

    # 2. TABELLE sui bordi interni
    table_abs = draw_tables_on_border(canvas, bx1,by1,bx2,by2, inner)
    for tx1,ty1,tx2,ty2 in table_abs:
        yolo_boxes.append([cls_table,
                           *abs_to_yolo(tx1,ty1,tx2,ty2,IMG_W,IMG_H)])

    # 3. QUOTE nell'area interna (escluse zone tabelle)
    inner2  = inner + 5
    area_x1 = bx1 + inner2
    area_y1 = by1 + inner2
    area_x2 = bx2 - inner2
    area_y2 = by2 - inner2

    n_quotes = random.randint(MIN_QUOTES, MAX_QUOTES)
    q_boxes  = place_quotes(canvas, crops,
                            area_x1, area_y1, area_x2, area_y2,
                            table_abs, n_quotes)
    yolo_boxes.extend(q_boxes)

    # Salva
    name = f"blueprint_{idx:05d}"
    cv2.imwrite(str(out_img / f"{name}.jpg"), canvas,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    save_labels(out_lbl / f"{name}.txt", yolo_boxes)

    return len(q_boxes)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("SINTESI: Genesi — Generatore blueprint strutturato")
    print("=" * 55)
    print(f"Immagini: {N_IMAGES}  |  Quote: {MIN_QUOTES}-{MAX_QUOTES}/img")
    print(f"Quote reali: {P_REAL_CROP*100:.0f}%  "
          f"| Sintetiche: {P_SYNTHETIC*100:.0f}%\n")

    crops = collect_quote_crops(Path(TRAIN_IMAGES), Path(TRAIN_LABELS))
    if not crops:
        print("Nessun ritaglio trovato — verranno usate solo quote sintetiche")

    n_train = int(N_IMAGES * TRAIN_RATIO)
    n_val   = int(N_IMAGES * VAL_RATIO)
    n_test  = N_IMAGES - n_train - n_val
    splits  = {"train": n_train, "val": n_val, "test": n_test}

    total_q = 0
    idx     = 0

    for split, n in splits.items():
        out_img = Path(OUTPUT_DIR) / split / "images"
        out_lbl = Path(OUTPUT_DIR) / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        print(f"Generando {split} ({n} immagini)...")
        for i in range(n):
            q = generate_image(idx, crops, out_img, out_lbl)
            total_q += q
            idx     += 1
            if (i+1) % 100 == 0:
                print(f"  {i+1}/{n}")

    yaml = f"""path: {Path(OUTPUT_DIR).resolve()}
train: train/images
val:   val/images
test:  test/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    (Path(OUTPUT_DIR) / "data.yaml").write_text(yaml)

    print(f"\n✓ Completato!")
    print(f"  Immagini:  {N_IMAGES}")
    print(f"  Quote tot: {total_q}  (media {total_q/N_IMAGES:.1f}/img)")
    print(f"  Output:    {OUTPUT_DIR}")
    print(f"\nIn merge_datasets.py aggiungi:")
    print(f'  "./dataset_blueprint_strutturato"')
