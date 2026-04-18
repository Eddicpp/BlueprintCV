"""
SINTESI: Genesi — Augmentation dataset blueprint v4
Fedele alla qualità delle scansioni reali di disegni tecnici industriali.

Augmentation:
  - Qualità scansione (gamma, ingiallimento, contrasto, rumore)
  - Banding scanner
  - Blur (gaussiano e motion)
  - Linee dense nere (tratteggio sezioni)
  - Zoom aggressivo centrato sulle quote
  - Rotazione, flip
  - JPEG compression (artefatti da scansione salvata male)
  - Erode text (inchiostro sbiadito, linee spezzate)
  - Perspective distortion (foglio non piatto sullo scanner)
  - Erase parts (occlusioni parziali)
"""

import cv2
import numpy as np
import random
import shutil
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

INPUT_DIR      = "./dataset_yolo"
OUTPUT_DIR     = "./dataset_yolo_aug"
CLASSES        = ["border", "table", "quote"]
AUGMENT_FACTOR = 10
RANDOM_SEED    = 42

# Probabilità augmentation
P_FLIP          = 0.5
P_FLIP_V        = 0.1
P_SCAN_QUALITY  = 0.95
P_SCAN_BANDS    = 0.45
P_DENSE_LINES   = 0.80
P_ZOOM          = 0.70
P_ROTATE        = 0.30
P_BLUR          = 0.45
P_JPEG          = 0.50   # artefatti JPEG da scansione salvata male
P_ERODE         = 0.40   # inchiostro sbiadito / linee spezzate
P_PERSPECTIVE   = 0.35   # foglio non piatto sullo scanner
P_ERASE         = 0.45

# ─────────────────────────────────────────
# UTILITÀ LABEL
# ─────────────────────────────────────────

def load_labels(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append([int(parts[0])] + list(map(float, parts[1:])))
    return boxes


def save_labels(label_path: Path, boxes: list):
    with open(label_path, "w") as f:
        for box in boxes:
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")


def yolo_to_abs(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w/2) * img_w)
    y1 = int((cy - h/2) * img_h)
    x2 = int((cx + w/2) * img_w)
    y2 = int((cy + h/2) * img_h)
    return max(0,x1), max(0,y1), min(img_w,x2), min(img_h,y2)


def abs_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = (x1+x2)/2/img_w
    cy = (y1+y2)/2/img_h
    w  = (x2-x1)/img_w
    h  = (y2-y1)/img_h
    return (max(0,min(1,cx)), max(0,min(1,cy)),
            max(0,min(1,w)),  max(0,min(1,h)))


def clip_boxes(boxes, img_w, img_h):
    valid = []
    for box in boxes:
        cls, cx, cy, w, h = box
        x1,y1,x2,y2 = yolo_to_abs(cx,cy,w,h,img_w,img_h)
        if x2-x1 > 5 and y2-y1 > 5:
            valid.append([cls, *abs_to_yolo(x1,y1,x2,y2,img_w,img_h)])
    return valid


# ─────────────────────────────────────────
# QUALITÀ SCANSIONE — cuore dell'augmentation
# ─────────────────────────────────────────

def aug_scan_quality(img, boxes):
    """
    Riproduce fedelmente la qualità di una scansione reale:
    - Ingiallimento carta (tono caldo leggermente sporco)
    - Abbassamento contrasto (linee non nere puro ma grigio scuro)
    - Gamma per simulare scanner vecchio
    - Rumore fine uniforme (granulometria tipica da scanner)
    - Variazione di luminosità non uniforme sul foglio
    """
    out = img.copy().astype(np.float32)
    h, w = out.shape[:2]

    # 1. Tono carta — non bianco puro ma leggermente sporco/ingiallito
    #    I bianchi diventano grigi chiari caldi
    paper_tone = random.randint(220, 248)
    # Abbassa tutto verso il tono carta (i bianchi non sono più 255)
    out = out * (paper_tone / 255.0)

    # 2. Ingiallimento — canale blu leggermente più basso
    yellow_strength = random.uniform(0.88, 0.97)
    out[:,:,0] *= yellow_strength   # BGR: abbassa blue → carta più calda

    # 3. Gamma correction — scanner vecchio o sotto/sovraesposto
    gamma     = random.uniform(0.6, 1.4)
    inv_gamma = 1.0 / gamma
    out_norm  = np.clip(out / 255.0, 0, 1)
    out       = (out_norm ** inv_gamma) * 255.0

    # 4. Contrasto ridotto — le linee non sono nere puro
    #    Simula inchiostro sbiadito o scansione con contrasto basso
    contrast = random.uniform(0.75, 0.95)
    # Schiaccia verso il centro: contrasto < 1 avvicina bianchi e neri
    out = 128 + (out - 128) * contrast

    # 5. Gradiente di luminosità non uniforme (scanner non illumina uniformemente)
    if random.random() > 0.3:
        axis = random.choice(["h", "v", "diag"])
        if axis == "h":
            grad = np.linspace(random.uniform(0.92,1.0),
                               random.uniform(0.80,0.95), w, dtype=np.float32)
            out *= grad[np.newaxis,:,np.newaxis]
        elif axis == "v":
            grad = np.linspace(random.uniform(0.92,1.0),
                               random.uniform(0.80,0.95), h, dtype=np.float32)
            out *= grad[:,np.newaxis,np.newaxis]
        else:
            # Angolo del gradiente (simula scanner con luce laterale)
            for y in range(h):
                factor = 1.0 - (y/h) * random.uniform(0.05, 0.18)
                out[y] *= factor

    # 6. Rumore fine uniforme — granulometria scanner
    sigma = random.uniform(1.5, 6.0)
    noise = np.random.normal(0, sigma, out.shape)
    out   = out + noise

    return np.clip(out, 0, 255).astype(np.uint8), boxes


def aug_scan_bands(img, boxes):
    """
    Strisce orizzontali con variazione di luminosità sottile —
    banding tipico degli scanner flatbed a LED.
    Molto sottile, quasi impercettibile ma reale.
    """
    out = img.copy().astype(np.float32)
    h   = out.shape[0]
    n_bands = random.randint(3, 12)

    for _ in range(n_bands):
        band_h   = random.randint(h//30, h//8)
        y_start  = random.randint(0, h - band_h)
        factor   = random.uniform(0.92, 1.05)
        out[y_start:y_start+band_h] *= factor

    return np.clip(out, 0, 255).astype(np.uint8), boxes


def aug_blur(img, boxes):
    """
    Blur da scanner — simula messa a fuoco imperfetta,
    documento mosso o piano di vetro sporco.
    Tre modalità: gaussiano, motion, defocus circolare.
    """
    mode = random.choice(["gaussian", "motion", "defocus"])

    if mode == "gaussian":
        k   = random.choice([3, 5, 7])
        sig = random.uniform(0.5, 2.0)
        out = cv2.GaussianBlur(img, (k, k), sig)

    elif mode == "motion":
        k      = random.choice([5, 7, 9])
        angle  = random.choice([0, 45, 90, 135])
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k//2, :] = 1.0 / k
        M      = cv2.getRotationMatrix2D((k//2, k//2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        kernel /= kernel.sum() + 1e-6
        out    = cv2.filter2D(img, -1, kernel)

    else:
        k      = random.choice([3, 5, 7])
        kernel = np.zeros((k, k), dtype=np.float32)
        cv2.circle(kernel, (k//2, k//2), k//2, 1, -1)
        kernel = kernel / kernel.sum()
        out    = cv2.filter2D(img, -1, kernel)

    return out, boxes


# ─────────────────────────────────────────
# LINEE DENSE NERE — tratteggio sezioni
# ─────────────────────────────────────────

def aug_dense_lines(img, boxes):
    """
    Disegna 50-120 linee nere sottili principalmente nella stessa direzione —
    simula il tratteggio delle sezioni nei disegni tecnici industriali.
    Le linee sono nere (non grigie) e molto fitte, come nella scansione reale.
    """
    out = img.copy()
    h, w = out.shape[:2]

    n_lines = random.randint(50, 120)

    # Direzione predominante (come nella scansione reale)
    main_angle = random.choice([45, -45, 30, -30, 60, -60, 0, 90])
    # Piccola variazione attorno alla direzione principale
    angle_spread = random.uniform(0, 5)

    # Colore: nero o grigio molto scuro (non colorato)
    line_darkness = random.randint(20, 80)   # 0=nero, 255=bianco
    color = (line_darkness, line_darkness, line_darkness)

    # Spessore: sempre 1 pixel — sottile come nelle sezioni reali
    thickness = 1

    # Spaziatura — fitte ma non sovrapposte
    spacing = random.randint(4, 15)

    import math
    rad = math.radians(main_angle)

    # Zona del foglio dove appaiono le linee
    # Può essere l'intero foglio o una zona specifica (come una sezione)
    zone_type = random.choice(["full", "region", "region"])
    if zone_type == "full":
        zx1, zy1, zx2, zy2 = 0, 0, w, h
    else:
        # Zona rettangolare casuale (simula una singola sezione)
        zx1 = random.randint(0, w//2)
        zy1 = random.randint(0, h//2)
        zx2 = random.randint(w//2, w)
        zy2 = random.randint(h//2, h)

    # Disegna le linee parallele
    if abs(main_angle) == 90 or main_angle == 90:
        # Linee verticali
        for x in range(zx1, zx2, spacing):
            angle_jitter = random.uniform(-angle_spread, angle_spread)
            cv2.line(out, (x, zy1), (x, zy2), color, thickness,
                     cv2.LINE_AA)
    elif main_angle == 0:
        # Linee orizzontali
        for y in range(zy1, zy2, spacing):
            cv2.line(out, (zx1, y), (zx2, y), color, thickness,
                     cv2.LINE_AA)
    else:
        # Linee diagonali — metodo offset
        rad_main = math.radians(main_angle)
        cos_a    = math.cos(rad_main)
        sin_a    = math.sin(rad_main)

        # Calcola range di offset per coprire tutta la zona
        diag = int(math.sqrt((zx2-zx1)**2 + (zy2-zy1)**2))
        for offset in range(-diag, diag, spacing):
            # Punto sulla perpendicolare alla direzione
            px = int((zx1+zx2)/2 + offset * (-sin_a))
            py = int((zy1+zy2)/2 + offset * cos_a)

            # Aggiungi micro-jitter all'angolo
            jitter = random.uniform(-angle_spread, angle_spread)
            rad_j  = math.radians(main_angle + jitter)
            cos_j  = math.cos(rad_j)
            sin_j  = math.sin(rad_j)

            ext = diag
            x1_l = int(px - ext * cos_j)
            y1_l = int(py - ext * sin_j)
            x2_l = int(px + ext * cos_j)
            y2_l = int(py + ext * sin_j)

            cv2.line(out, (x1_l, y1_l), (x2_l, y2_l),
                     color, thickness, cv2.LINE_AA)

    # Blend con l'originale — le linee sono visibili ma non coprono tutto
    # Alpha alto = linee più marcate (come nella scansione reale)
    alpha = random.uniform(0.55, 0.85)
    out   = cv2.addWeighted(img, 1-alpha, out, alpha, 0)

    return out, boxes


# ─────────────────────────────────────────
# AUGMENTATION GEOMETRICHE
# ─────────────────────────────────────────

def aug_flip(img, boxes):
    img = cv2.flip(img, 1)
    return img, [[c, 1-cx, cy, w, h] for c,cx,cy,w,h in boxes]


def aug_flip_v(img, boxes):
    img = cv2.flip(img, 0)
    return img, [[c, cx, 1-cy, w, h] for c,cx,cy,w,h in boxes]


def aug_zoom(img, boxes):
    """
    Zoom aggressivo — ritagli ravvicinati sulle zone dense di quote.
    Scala da 0.4x (molto zoomato in) a 1.5x (zoomato fuori).
    I ritagli profondi (scale < 0.7) sono i più utili perché mostrano
    le quote grandi e ravvicinate come le vede il modello.
    """
    h, w   = img.shape[:2]
    # Distribuzione pesata verso zoom in aggressivo
    scale  = random.choices(
        [random.uniform(0.35, 0.55),   # zoom molto ravvicinato
         random.uniform(0.55, 0.75),   # zoom ravvicinato
         random.uniform(0.75, 1.0),    # zoom moderato
         random.uniform(1.0,  1.4)],   # zoom out
        weights=[0.35, 0.30, 0.25, 0.10]
    )[0]

    new_w   = int(w * scale)
    new_h   = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    if scale < 1.0:
        # Zoom in: ritaglia e reupscala — preferisci zone con quote
        quote_boxes = [b for b in boxes if CLASSES[b[0]] == "quote"]

        if quote_boxes and random.random() > 0.3:
            # Centra il crop su una quota casuale
            ref    = random.choice(quote_boxes)
            _, cx, cy, bw, bh = ref
            center_x = int(cx * w)
            center_y = int(cy * h)
            x_off = max(0, min(w - new_w, center_x - new_w//2))
            y_off = max(0, min(h - new_h, center_y - new_h//2))
        else:
            x_off = random.randint(0, max(0, w - new_w))
            y_off = random.randint(0, max(0, h - new_h))

        # Ritaglia dall'originale (non dal resized)
        crop    = img[y_off:y_off+new_h, x_off:x_off+new_w]
        out     = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

        new_boxes = []
        for cls, bcx, bcy, bw2, bh2 in boxes:
            # Aggiusta coordinate
            abs_cx = bcx * w
            abs_cy = bcy * h
            abs_bw = bw2 * w
            abs_bh = bh2 * h

            ncx = (abs_cx - x_off) / new_w
            ncy = (abs_cy - y_off) / new_h
            nw2 = abs_bw / new_w
            nh2 = abs_bh / new_h
            new_boxes.append([cls, ncx, ncy, nw2, nh2])

        return out, clip_boxes(new_boxes, w, h)

    else:
        # Zoom out: ridimensiona e padda con bianco carta
        paper_tone = random.randint(220, 248)
        out   = np.ones((h, w, 3), dtype=np.uint8) * paper_tone
        # Clamp per evitare che il resized superi il canvas
        new_w  = min(new_w, w)
        new_h  = min(new_h, h)
        resized = cv2.resize(img, (new_w, new_h))
        x_off = (w - new_w) // 2
        y_off = (h - new_h) // 2
        out[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        new_boxes = []
        for cls, bcx, bcy, bw2, bh2 in boxes:
            ncx = (bcx*new_w + x_off) / w
            ncy = (bcy*new_h + y_off) / h
            nw2 = bw2*new_w / w
            nh2 = bh2*new_h / h
            new_boxes.append([cls, ncx, ncy, nw2, nh2])

        return out, clip_boxes(new_boxes, w, h)


def aug_rotate(img, boxes):
    h, w  = img.shape[:2]
    angle = random.uniform(-10, 10)
    M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    # Sfondo colore carta (non bianco puro)
    paper = random.randint(220, 248)
    out   = cv2.warpAffine(img, M, (w, h),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(paper, paper, int(paper*0.95)))
    new_boxes = []
    for cls, cx, cy, bw, bh in boxes:
        x1,y1,x2,y2 = yolo_to_abs(cx,cy,bw,bh,w,h)
        corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
                           dtype=np.float32)
        rot = (M @ np.hstack([corners,
               np.ones((4,1), dtype=np.float32)]).T).T
        new_boxes.append([cls, *abs_to_yolo(
            rot[:,0].min(), rot[:,1].min(),
            rot[:,0].max(), rot[:,1].max(), w, h)])
    return out, clip_boxes(new_boxes, w, h)


# ─────────────────────────────────────────
# NUOVE AUGMENTATION
# ─────────────────────────────────────────

def aug_jpeg_compression(img, boxes):
    """
    Compressione JPEG aggressiva — simula scansione salvata con
    qualità bassa. Crea artefatti tipici sui bordi delle quote
    (blocchi 8x8, ringing attorno alle linee sottili).
    """
    quality = random.randint(40, 75)
    _, buf = cv2.imencode('.jpg', img,
                          [cv2.IMWRITE_JPEG_QUALITY, quality])
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out, boxes


def aug_erode_text(img, boxes):
    """
    Erosione morfologica sui pixel scuri — rompe le linee sottili
    e i numeri delle quote simulando inchiostro sbiadito,
    stampa consumata o scansione a bassa risoluzione.
    """
    kernel    = np.ones((2, 2), np.uint8)
    dark_mask = img < 80
    eroded    = cv2.erode(img, kernel, iterations=1)
    out       = img.copy()
    out[dark_mask] = eroded[dark_mask]
    return out, boxes


def aug_perspective(img, boxes):
    """
    Distorsione prospettica leggera — simula il foglio non
    perfettamente piatto sullo scanner o una foto leggermente
    angolata del disegno.
    Le label vengono trasformate con la stessa matrice prospettica.
    """
    h, w = img.shape[:2]
    margin = random.randint(10, 40)
    paper  = random.randint(220, 248)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, margin),   random.randint(0, margin)],
        [w - random.randint(0, margin), random.randint(0, margin)],
        [w - random.randint(0, margin), h - random.randint(0, margin)],
        [random.randint(0, margin),   h - random.randint(0, margin)],
    ])

    M   = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (w, h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(paper, paper, int(paper*0.95)))

    # Trasforma le bounding box con la stessa matrice
    new_boxes = []
    for cls, cx, cy, bw, bh in boxes:
        x1, y1, x2, y2 = yolo_to_abs(cx, cy, bw, bh, w, h)
        corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
                           dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
        rx1 = transformed[:, 0].min()
        ry1 = transformed[:, 1].min()
        rx2 = transformed[:, 0].max()
        ry2 = transformed[:, 1].max()
        new_boxes.append([cls, *abs_to_yolo(rx1, ry1, rx2, ry2, w, h)])

    return out, clip_boxes(new_boxes, w, h)


# ─────────────────────────────────────────
# AUGMENTATION QUOTE
# ─────────────────────────────────────────

def aug_erase_parts(img, boxes):
    """
    Cancella parti di quote con rettangoli colore carta —
    simula occlusioni, frecce tagliate, numeri illeggibili.
    """
    out = img.copy()
    h, w = out.shape[:2]
    paper = random.randint(220, 250)

    quote_boxes = [b for b in boxes if CLASSES[b[0]] == "quote"]
    if not quote_boxes:
        return out, boxes

    n_erase  = random.randint(1, min(5, len(quote_boxes)))
    selected = random.sample(quote_boxes, n_erase)

    for box in selected:
        _, cx, cy, bw, bh = box
        x1,y1,x2,y2 = yolo_to_abs(cx,cy,bw,bh,w,h)
        bw_px = x2-x1
        bh_px = y2-y1
        if bw_px < 4 or bh_px < 4:
            continue

        erase_ratio = random.uniform(0.15, 0.55)
        ew = max(2, int(bw_px * erase_ratio))
        eh = max(2, int(bh_px * erase_ratio))

        pos = random.choice(["left","right","top","bottom","center"])
        if pos == "left":
            ex,ey = x1, y1+(bh_px-eh)//2
        elif pos == "right":
            ex,ey = x2-ew, y1+(bh_px-eh)//2
        elif pos == "top":
            ex,ey = x1+(bw_px-ew)//2, y1
        elif pos == "bottom":
            ex,ey = x1+(bw_px-ew)//2, y2-eh
        else:
            ex = random.randint(x1, max(x1, x2-ew))
            ey = random.randint(y1, max(y1, y2-eh))

        cv2.rectangle(out, (ex, ey), (ex+ew, ey+eh),
                      (paper, paper, int(paper*0.95)), -1)

    return out, boxes


# ─────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────

def augment_image(img, boxes):
    # Geometriche prima — poi fotometriche — poi custom
    if random.random() < P_FLIP:
        img, boxes = aug_flip(img, boxes)
    if random.random() < P_FLIP_V:
        img, boxes = aug_flip_v(img, boxes)
    if random.random() < P_ZOOM:
        img, boxes = aug_zoom(img, boxes)
    if random.random() < P_ROTATE:
        img, boxes = aug_rotate(img, boxes)
    if random.random() < P_PERSPECTIVE:
        img, boxes = aug_perspective(img, boxes)

    # Qualità scansione
    if random.random() < P_SCAN_QUALITY:
        img, boxes = aug_scan_quality(img, boxes)
    if random.random() < P_SCAN_BANDS:
        img, boxes = aug_scan_bands(img, boxes)
    if random.random() < P_BLUR:
        img, boxes = aug_blur(img, boxes)
    if random.random() < P_JPEG:
        img, boxes = aug_jpeg_compression(img, boxes)
    if random.random() < P_ERODE:
        img, boxes = aug_erode_text(img, boxes)

    # Linee dense nere
    if random.random() < P_DENSE_LINES:
        img, boxes = aug_dense_lines(img, boxes)

    # Quote
    if random.random() < P_ERASE:
        img, boxes = aug_erase_parts(img, boxes)

    return img, boxes


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def process_split(split: str):
    input_img  = Path(INPUT_DIR) / split / "images"
    input_lbl  = Path(INPUT_DIR) / split / "labels"
    output_img = Path(OUTPUT_DIR) / split / "images"
    output_lbl = Path(OUTPUT_DIR) / split / "labels"
    output_img.mkdir(parents=True, exist_ok=True)
    output_lbl.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(list(input_img.glob("*.png")) +
                       list(input_img.glob("*.jpg")))

    print(f"\n{split}: {len(img_paths)} originali "
          f"→ {len(img_paths)*(AUGMENT_FACTOR+1)} attese")

    count = 0
    for img_path in img_paths:
        lbl_path = input_lbl / f"{img_path.stem}.txt"
        img      = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes = load_labels(lbl_path)

        # Copia originale
        shutil.copy(img_path, output_img / img_path.name)
        if lbl_path.exists():
            shutil.copy(lbl_path, output_lbl / f"{img_path.stem}.txt")
        count += 1

        if split != "train":
            continue

        for i in range(AUGMENT_FACTOR):
            aug_img, aug_boxes = augment_image(img.copy(), list(boxes))
            name = f"{img_path.stem}_aug{i:03d}{img_path.suffix}"
            cv2.imwrite(str(output_img / name), aug_img)
            save_labels(output_lbl / f"{img_path.stem}_aug{i:03d}.txt",
                        aug_boxes)
            count += 1

    print(f"  Salvate: {count} immagini")
    return count


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("SINTESI: Genesi — Augmentation v3 (fedele a scansione)")
    print("=" * 55)
    print(f"Input:   {INPUT_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Factor:  {AUGMENT_FACTOR}x")

    total = 0
    for split in ["train", "val", "test"]:
        if (Path(INPUT_DIR) / split / "images").exists():
            total += process_split(split)

    yaml_src = Path(INPUT_DIR) / "data.yaml"
    if yaml_src.exists():
        content = yaml_src.read_text().replace(
            str(Path(INPUT_DIR).resolve()),
            str(Path(OUTPUT_DIR).resolve())
        )
        (Path(OUTPUT_DIR) / "data.yaml").write_text(content)

    print(f"\n✓ Completato. Totale: {total} immagini in {OUTPUT_DIR}")