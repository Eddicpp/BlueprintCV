"""
<<<<<<< HEAD
SINTESI: Genesi — GUI Pipeline Detection + Classificazione Simboli
Interfaccia grafica Tkinter per la pipeline a due stadi:
  1. Carica modelli (detector + classificatore)
  2. Carica immagine blueprint
  3. Detection quote → crop → classificazione simbolo
  4. Visualizza risultati sull'immagine originale
  5. Opzione per vedere i singoli crop classificati
=======
BlueprintCV — GUI Pipeline M1 + M2 + M3
Tab 1: Quote trovate da M1 (azzurro con numero)
Tab 2: Immagine annotata completa (M1+M2+M3)
Tab 3: Simboli trovati (griglia crop)
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
"""

import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import threading
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────
<<<<<<< HEAD
# CONFIGURAZIONE DEFAULT
# ─────────────────────────────────────────

DEFAULT_DETECTOR    = r".\runs\detect\sintesi_genesi\yolo11s_run3_augmented\weights\best.pt"
DEFAULT_CLASSIFIER  = r".\runs\detect\sintesi_genesi\simboli_detector_run12\weights\best.pt"

CONF_DETECT  = 0.40
CONF_SYMBOLS = 0.25    # soglia più bassa per il detector simboli
IMG_SIZE_CLS = 640
QUOTE_CLS_ID = 2   # indice classe "quote" nel detector

SYMBOL_CLASSES = [
    "diameter", "radius", "angle", "surface_finish",
    "concentricity", "cylindricity", "position", "flatness",
    "perpendicularity", "total_runout", "circular_runout",
    "slope", "conical_taper", "symmetry", "surface_profile",
=======
# CONFIGURAZIONE
# ─────────────────────────────────────────

DEFAULT_M1 = r".\runs\detect\sintesi_genesi\yolo11s_run3_augmented\weights\best.pt"
DEFAULT_M2 = r".\runs\detect\sintesi_genesi\symbol_detector_run1\weights\best.pt"
DEFAULT_M3 = r".\runs\classify\runs\classify\sintesi_genesi\simboli_yolo11s_run1\weights\best.pt"

CONF_M1      = 0.40
CONF_M2      = 0.20
CONF_M3         = 0.50
IMG_SIZE_M1     = 1600
IMG_SIZE_M2     = 640
IMG_SIZE_M3     = 128
QUOTE_CLS_ID    = 2
CONTEXT_MARGIN  = 0.40

# ── Tiling ──
MAX_SIZE    = 2000   # se l'immagine supera questo lato si usa il tiling
TILE_SIZE   = 1600   # dimensione di ogni tile
TILE_OVERLAP= 200    # overlap tra tile per non perdere oggetti sul bordo
IOU_NMS     = 0.50   # soglia NMS per rimuovere duplicati


# ─────────────────────────────────────────
# TILING + NMS
# ─────────────────────────────────────────

def nms_boxes(boxes_list, iou_thresh=0.50):
    """
    boxes_list: lista di [x1,y1,x2,y2,conf,cls]
    Ritorna lista filtrata dopo NMS.
    """
    if not boxes_list:
        return []

    boxes = np.array(boxes_list, dtype=np.float32)
    x1 = boxes[:,0]; y1 = boxes[:,1]
    x2 = boxes[:,2]; y2 = boxes[:,3]
    scores = boxes[:,4]
    areas  = (x2-x1)*(y2-y1)

    order = scores.argsort()[::-1]
    keep  = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w   = np.maximum(0, xx2-xx1)
        h   = np.maximum(0, yy2-yy1)
        inter = w*h
        iou   = inter / (areas[i]+areas[order[1:]]-inter+1e-6)
        inds  = np.where(iou <= iou_thresh)[0]
        order = order[inds+1]

    return [boxes_list[i] for i in keep]


def tiled_predict(img_bgr, model, conf):
    """
    Se l'immagine supera MAX_SIZE la divide in tile sovrapposti,
    esegue la predizione su ognuno, riconverte le coordinate
    nel sistema originale e applica NMS per rimuovere i duplicati.
    Ritorna lista di [x1,y1,x2,y2,conf,cls].
    """
    ih, iw = img_bgr.shape[:2]

    # Immagine piccola — predizione diretta
    if max(ih, iw) <= MAX_SIZE:
        res   = model.predict(img_bgr, conf=conf,
                              imgsz=IMG_SIZE_M1, verbose=False)[0]
        boxes = []
        if res.boxes is not None:
            for b in res.boxes:
                x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
                boxes.append([x1,y1,x2,y2,float(b.conf[0]),int(b.cls[0])])
        return boxes

    # Tiling
    all_boxes = []
    step      = TILE_SIZE - TILE_OVERLAP

    y = 0
    while y < ih:
        x = 0
        while x < iw:
            x2t = min(x+TILE_SIZE, iw)
            y2t = min(y+TILE_SIZE, ih)
            x1t = max(0, x2t-TILE_SIZE)
            y1t = max(0, y2t-TILE_SIZE)

            tile = img_bgr[y1t:y2t, x1t:x2t]
            res  = model.predict(tile, conf=conf,
                                 imgsz=IMG_SIZE_M1, verbose=False)[0]

            if res.boxes is not None:
                for b in res.boxes:
                    bx1,by1,bx2,by2 = map(float, b.xyxy[0].tolist())
                    # Converti in coordinate originali
                    all_boxes.append([
                        bx1+x1t, by1+y1t,
                        bx2+x1t, by2+y1t,
                        float(b.conf[0]),
                        int(b.cls[0])
                    ])

            if x2t >= iw:
                break
            x += step
        if y2t >= ih:
            break
        y += step

    # NMS per rimuovere duplicati nelle zone di overlap
    return nms_boxes(all_boxes, IOU_NMS)

SYMBOL_CLASSES = [
    "diameter", "radius", "surface_finish",
    "concentricity", "cylindricity", "position", "flatness",
    "perpendicularity", "total_runout", "circular_runout",
    "slope", "conical_taper", "symmetry", "surface_profile",
    "linear",
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
]

CLASS_COLORS = {
    "diameter":        (255,  80,  80),
    "radius":          ( 80, 200,  80),
<<<<<<< HEAD
    "angle":           ( 80,  80, 255),
=======
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
    "surface_finish":  (220, 160,   0),
    "concentricity":   (  0, 180, 220),
    "cylindricity":    (180,   0, 220),
    "position":        (220, 120,   0),
    "flatness":        (  0, 200, 160),
    "perpendicularity":(160,   0, 220),
    "total_runout":    (220,   0, 120),
    "circular_runout": (120, 220,   0),
    "slope":           (  0, 120, 220),
    "conical_taper":   (220,  40, 160),
    "symmetry":        ( 40, 220, 220),
    "surface_profile": (180, 180,   0),
    "linear":          (140, 140, 140),
    "unknown":         (100, 100, 100),
}

<<<<<<< HEAD
=======
QUOTE_COLOR = (0, 160, 220)   # azzurro per le quote

>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)

# ─────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────

<<<<<<< HEAD
def run_pipeline(img_bgr, detector, sym_detector, conf_thresh):
    """
    Pipeline a due stadi:
      Stadio 1 — detector trova le quote (bounding box)
      Stadio 2 — detector simboli trova il simbolo nel crop
    """
    ih, iw = img_bgr.shape[:2]
    vis    = img_bgr.copy()
    font   = cv2.FONT_HERSHEY_SIMPLEX

    det_res = detector.predict(img_bgr, conf=conf_thresh, verbose=False)[0]
    boxes   = det_res.boxes
    crops_data = []

    if boxes is None or len(boxes) == 0:
        return vis, crops_data

    for box in boxes:
        if int(box.cls[0]) != QUOTE_CLS_ID:
            continue

        det_conf = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
=======
def run_pipeline(img_bgr, m1, m2, m3, conf_m1):
    """
    Esegue M1 → M2 → M3 in un unico passaggio.
    Ritorna:
      vis_quotes: immagine con sole quote (M1)
      vis_full:   immagine con quote + simboli
      crops_data: dati per tab simboli
    """
    ih, iw = img_bgr.shape[:2]
    vis_quotes = img_bgr.copy()
    vis_full   = img_bgr.copy()
    font       = cv2.FONT_HERSHEY_SIMPLEX
    crops_data = []

    det_boxes = tiled_predict(img_bgr, m1, conf_m1)
    if not det_boxes:
        return vis_quotes, vis_full, crops_data

    quote_idx = 0
    for det_box in det_boxes:
        bx1,by1,bx2,by2,det_conf,cls_id = det_box
        if int(cls_id) != QUOTE_CLS_ID:
            continue

        x1,y1,x2,y2 = int(bx1),int(by1),int(bx2),int(by2)
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
        x1=max(0,x1); y1=max(0,y1)
        x2=min(iw,x2); y2=min(ih,y2)
        if x2-x1 < 5 or y2-y1 < 5:
            continue

<<<<<<< HEAD
        crop = img_bgr[y1:y2, x1:x2].copy()

        # ── Stadio 2: detector simboli sul crop ──
        sym_res = sym_detector.predict(
            crop, imgsz=IMG_SIZE_CLS,
            conf=CONF_SYMBOLS, verbose=False)[0]

        sym_boxes = sym_res.boxes
        if sym_boxes is not None and len(sym_boxes) > 0:
            # Prendi il simbolo con confidence più alta
            best     = max(sym_boxes, key=lambda b: float(b.conf[0]))
            sym_id   = int(best.cls[0])
            sym_conf = float(best.conf[0])
            sym_name = sym_detector.names.get(sym_id,
                       SYMBOL_CLASSES[sym_id] if sym_id < len(SYMBOL_CLASSES)
                       else "unknown")
        else:
            sym_name = "linear"
            sym_conf = 0.0

        color     = CLASS_COLORS.get(sym_name, (100,100,100))
        color_rgb = (color[2], color[1], color[0])

        # Disegna box quota sull'immagine
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        label = f"{sym_name} {sym_conf:.2f}"
        (lw,lh),_ = cv2.getTextSize(label, font, 0.45, 1)
        ly = max(lh+4, y1-4)
        cv2.rectangle(vis, (x1,ly-lh-3), (x1+lw+6,ly+3), color, -1)
        cv2.putText(vis, label, (x1+3,ly),
                    font, 0.45, (255,255,255), 1, cv2.LINE_AA)

        crops_data.append({
            "crop":      crop,
            "sym_name":  sym_name,
            "sym_conf":  sym_conf,
            "det_conf":  det_conf,
            "box":       (x1,y1,x2,y2),
            "color_rgb": color_rgb,
        })

    return vis, crops_data
=======
        quote_idx += 1

        # ── Disegna quota su vis_quotes ──
        cv2.rectangle(vis_quotes,(x1,y1),(x2,y2),QUOTE_COLOR,2)
        ql = f"Q{quote_idx} {det_conf:.2f}"
        (lw,lh),_ = cv2.getTextSize(ql,font,0.38,1)
        ly = max(lh+3, y1-3)
        cv2.rectangle(vis_quotes,(x1,ly-lh-2),(x1+lw+4,ly+2),QUOTE_COLOR,-1)
        cv2.putText(vis_quotes,ql,(x1+2,ly),
                    font,0.38,(255,255,255),1,cv2.LINE_AA)

        # ── Disegna quota grigia su vis_full ──
        cv2.rectangle(vis_full,(x1,y1),(x2,y2),(180,180,180),1)

        # Box originale della quota (small)
        bw = x2 - x1
        bh = y2 - y1

        # Box allargato per dare contesto a M2 (large)
        mx = int(bw * CONTEXT_MARGIN)
        my = int(bh * CONTEXT_MARGIN)
        lx1 = max(0,  x1 - mx)
        ly1 = max(0,  y1 - my)
        lx2 = min(iw, x2 + mx)
        ly2 = min(ih, y2 + my)

        crop_large = img_bgr[ly1:ly2, lx1:lx2].copy()
        crop_small = img_bgr[y1:y2,   x1:x2].copy()

        # Coordinate del box small nel sistema del crop large
        small_in_large_x1 = x1 - lx1
        small_in_large_y1 = y1 - ly1
        small_in_large_x2 = x2 - lx1
        small_in_large_y2 = y2 - ly1

        # ── M2 sul crop LARGE ──
        sym_res   = m2.predict(crop_large, imgsz=IMG_SIZE_M2,
                               conf=CONF_M2, verbose=False)[0]
        sym_crops = []

        if sym_res.boxes is not None and len(sym_res.boxes) > 0:
            for sb in sym_res.boxes:
                sc = float(sb.conf[0])
                sx1,sy1,sx2,sy2 = map(int, sb.xyxy[0].tolist())
                sx1=max(0,sx1); sy1=max(0,sy1)
                sx2=min(crop_large.shape[1],sx2)
                sy2=min(crop_large.shape[0],sy2)
                if sx2-sx1 < 3 or sy2-sy1 < 3:
                    continue

                # Centro del simbolo nel crop large
                sym_cx = (sx1 + sx2) / 2
                sym_cy = (sy1 + sy2) / 2

                # Filtra: il centro del simbolo deve essere
                # dentro il box small (altrimenti è contesto esterno)
                if not (small_in_large_x1 <= sym_cx <= small_in_large_x2 and
                        small_in_large_y1 <= sym_cy <= small_in_large_y2):
                    continue   # simbolo appartenente ad altra quota

                sym_crop = crop_large[sy1:sy2, sx1:sx2].copy()

                # ── M3 ──
                if m3 is not None and sym_crop.size > 0:
                    cls_res = m3.predict(sym_crop, imgsz=IMG_SIZE_M3,
                                         verbose=False)
                    if cls_res and cls_res[0].probs is not None:
                        sym_name = cls_res[0].names[int(cls_res[0].probs.top1)]
                        sym_conf = float(cls_res[0].probs.top1conf)
                        if sym_name == "background" or sym_conf < CONF_M3:
                            continue
                    else:
                        sym_name = "unknown"; sym_conf = sc
                else:
                    sym_name = "symbol"; sym_conf = sc

                # Coordinate assolute nel sistema dell'immagine originale
                ax1 = lx1+sx1; ay1 = ly1+sy1
                ax2 = lx1+sx2; ay2 = ly1+sy2
                color = CLASS_COLORS.get(sym_name,(100,100,100))

                cv2.rectangle(vis_full,(ax1,ay1),(ax2,ay2),color,2)
                lbl = f"{sym_name} {sym_conf:.2f}"
                (lw2,lh2),_ = cv2.getTextSize(lbl,font,0.38,1)
                ly2b = max(lh2+3, ay1-3)
                cv2.rectangle(vis_full,(ax1,ly2b-lh2-2),(ax1+lw2+4,ly2b+2),color,-1)
                cv2.putText(vis_full,lbl,(ax1+2,ly2b),
                            font,0.38,(255,255,255),1,cv2.LINE_AA)

                sym_crops.append({
                    "sym_crop":  sym_crop,
                    "sym_name":  sym_name,
                    "sym_conf":  sym_conf,
                    "color_rgb": (color[2],color[1],color[0]),
                })

        if not sym_crops:
            cv2.rectangle(vis_full,(x1,y1),(x2,y2),CLASS_COLORS["linear"],1)

        crops_data.append({
            "crop":      crop_small,
            "box":       (x1,y1,x2,y2),
            "det_conf":  det_conf,
            "sym_crops": sym_crops,
            "idx":       quote_idx,
        })

    return vis_quotes, vis_full, crops_data
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)


# ─────────────────────────────────────────
# GUI
# ─────────────────────────────────────────

<<<<<<< HEAD
class PipelineGUI:
    def __init__(self, root):
        self.root     = root
        self.detector   = None
        self.classifier = None
        self.img_bgr    = None
        self.vis_bgr    = None
        self.crops_data = []
        self.zoom_scale = 1.0
        self.pan_x      = 0
        self.pan_y      = 0
        self._drag_start = None

        self._build_ui()

    # ─── UI ───────────────────────────────

    def _build_ui(self):
        self.root.title("SINTESI: Genesi — Pipeline Simboli")
        self.root.configure(bg="#1a1a2e")
        self.root.minsize(1100, 700)

        # Font
        title_font  = ("Courier New", 13, "bold")
        label_font  = ("Courier New", 10)
        button_font = ("Courier New", 10, "bold")

        # ── Sidebar sinistra ──
        sidebar = tk.Frame(self.root, bg="#16213e", width=280)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(10,0), pady=10)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="SINTESI GENESI",
                 bg="#16213e", fg="#e94560",
                 font=("Courier New", 14, "bold")).pack(pady=(20,2))
        tk.Label(sidebar, text="pipeline detector + classificatore",
                 bg="#16213e", fg="#a0a0b0",
                 font=("Courier New", 8)).pack(pady=(0,20))

        # ── Sezione modelli ──
        self._section(sidebar, "MODELLI")

        self.det_var = tk.StringVar(value=DEFAULT_DETECTOR)
        self._model_row(sidebar, "Detector:", self.det_var,
                        self._browse_detector)

        self.cls_var = tk.StringVar(value=DEFAULT_CLASSIFIER)
        self._model_row(sidebar, "Classificatore:", self.cls_var,
                        self._browse_classifier)

        self.load_btn = tk.Button(
            sidebar, text="⚡ CARICA MODELLI",
            bg="#e94560", fg="white", font=button_font,
            relief=tk.FLAT, cursor="hand2",
            command=self._load_models)
        self.load_btn.pack(fill=tk.X, padx=15, pady=(8,15))

        self.model_status = tk.Label(
            sidebar, text="● modelli non caricati",
            bg="#16213e", fg="#666680", font=label_font)
        self.model_status.pack(pady=(0,15))

        # ── Sezione immagine ──
        self._section(sidebar, "IMMAGINE")

        self.img_btn = tk.Button(
            sidebar, text="📂 CARICA IMMAGINE",
            bg="#0f3460", fg="white", font=button_font,
            relief=tk.FLAT, cursor="hand2",
            command=self._load_image)
        self.img_btn.pack(fill=tk.X, padx=15, pady=(4,8))

        self.img_label = tk.Label(
            sidebar, text="nessuna immagine",
            bg="#16213e", fg="#666680", font=("Courier New", 8),
            wraplength=240)
        self.img_label.pack(pady=(0,15))

        # ── Sezione parametri ──
        self._section(sidebar, "PARAMETRI")

        conf_frame = tk.Frame(sidebar, bg="#16213e")
        conf_frame.pack(fill=tk.X, padx=15, pady=4)
        tk.Label(conf_frame, text="Confidence:", bg="#16213e",
                 fg="#a0a0b0", font=label_font).pack(side=tk.LEFT)
        self.conf_var = tk.DoubleVar(value=CONF_DETECT)
        conf_spin = tk.Spinbox(conf_frame, from_=0.1, to=0.95,
                               increment=0.05,
                               textvariable=self.conf_var,
                               width=6, font=label_font,
                               bg="#0f3460", fg="white",
                               buttonbackground="#0f3460",
                               relief=tk.FLAT)
        conf_spin.pack(side=tk.RIGHT)

        # ── Run ──
        self.run_btn = tk.Button(
            sidebar, text="▶ ESEGUI PIPELINE",
            bg="#e94560", fg="white",
            font=("Courier New", 11, "bold"),
            relief=tk.FLAT, cursor="hand2",
            command=self._run, state=tk.DISABLED)
        self.run_btn.pack(fill=tk.X, padx=15, pady=(20,8))

        self.progress = ttk.Progressbar(sidebar, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=15, pady=(0,10))

        # ── Stats ──
        self._section(sidebar, "RISULTATI")
        self.stats_label = tk.Label(
            sidebar, text="—", bg="#16213e",
            fg="#a0a0b0", font=("Courier New", 9),
            justify=tk.LEFT, wraplength=250)
        self.stats_label.pack(padx=15, pady=4, anchor=tk.W)

        # ── Area principale ──
        main = tk.Frame(self.root, bg="#1a1a2e")
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                  padx=10, pady=10)

        # Tabs
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#1a1a2e",
                        borderwidth=0)
        style.configure("TNotebook.Tab",
                        background="#16213e", foreground="#a0a0b0",
                        font=("Courier New", 10, "bold"),
                        padding=(12,6))
=======
def make_tk_image(img_bgr, target_w, target_h, zoom=1.0, pan_x=0, pan_y=0):
    """Converte img_bgr in PhotoImage scalata per il canvas."""
    ih, iw = img_bgr.shape[:2]
    scale  = min(target_w/iw, target_h/ih) * zoom
    nw     = max(1, int(iw*scale))
    nh     = max(1, int(ih*scale))
    resized = cv2.resize(img_bgr,(nw,nh),interpolation=cv2.INTER_AREA)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb)), nw, nh


class CanvasView:
    """Wrapper canvas con zoom/pan."""
    def __init__(self, parent):
        self.canvas = tk.Canvas(parent, bg="#0f0f1a",
                                cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.zoom  = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._drag = None
        self._img  = None   # riferimento PhotoImage
        self._bgr  = None   # immagine BGR corrente
        self.canvas.bind("<MouseWheel>", self._scroll)
        self.canvas.bind("<ButtonPress-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>",  self._drag_move)
        self.canvas.bind("<Double-Button-1>", self.reset)

    def show(self, img_bgr):
        self._bgr = img_bgr
        self._render()

    def _render(self):
        if self._bgr is None:
            return
        cw = self.canvas.winfo_width()  or 800
        ch = self.canvas.winfo_height() or 600
        tk_img, nw, nh = make_tk_image(self._bgr, cw, ch, self.zoom)
        self._img = tk_img   # mantieni riferimento
        ox = max(0,(cw-nw)//2) + self.pan_x
        oy = max(0,(ch-nh)//2) + self.pan_y
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self._img)

    def _scroll(self, e):
        f = 1.15 if e.delta > 0 else 1/1.15
        self.zoom = max(0.1, min(15.0, self.zoom*f))
        self._render()

    def _drag_start(self, e):
        self._drag = (e.x, e.y)

    def _drag_move(self, e):
        if self._drag:
            self.pan_x += e.x-self._drag[0]
            self.pan_y += e.y-self._drag[1]
            self._drag = (e.x, e.y)
            self._render()

    def reset(self, e=None):
        self.zoom  = 1.0
        self.pan_x = self.pan_y = 0
        self._render()


class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.m1 = self.m2 = self.m3 = None
        self.img_bgr    = None
        self.crops_data = []
        self._build_ui()

    def _build_ui(self):
        self.root.title("BlueprintCV — M1 + M2 + M3")
        self.root.configure(bg="#1a1a2e")
        self.root.minsize(1200, 700)

        bf = ("Courier New", 10, "bold")
        lf = ("Courier New", 8)

        # ── Sidebar ──
        sb = tk.Frame(self.root, bg="#16213e", width=290)
        sb.pack(side=tk.LEFT, fill=tk.Y, padx=(10,0), pady=10)
        sb.pack_propagate(False)

        tk.Label(sb, text="BlueprintCV", bg="#16213e", fg="#e94560",
                 font=("Courier New", 14, "bold")).pack(pady=(20,2))
        tk.Label(sb, text="M1 → M2 → M3", bg="#16213e", fg="#a0a0b0",
                 font=("Courier New", 9)).pack(pady=(0,15))

        self._section(sb, "MODELLI")
        self.m1_var = tk.StringVar(value=DEFAULT_M1)
        self.m2_var = tk.StringVar(value=DEFAULT_M2)
        self.m3_var = tk.StringVar(value=DEFAULT_M3)
        self._model_row(sb, "M1 Quote Detector:", self.m1_var)
        self._model_row(sb, "M2 Symbol Detector:", self.m2_var)
        self._model_row(sb, "M3 Classifier:", self.m3_var)

        tk.Button(sb, text="⚡ CARICA MODELLI", bg="#e94560", fg="white",
                  font=bf, relief=tk.FLAT, cursor="hand2",
                  command=self._load_models).pack(fill=tk.X, padx=15, pady=(8,4))

        self.model_lbl = tk.Label(sb, text="● non caricati",
                                  bg="#16213e", fg="#666680", font=lf)
        self.model_lbl.pack(pady=(0,8))

        self._section(sb, "IMMAGINE")
        tk.Button(sb, text="📂 CARICA IMMAGINE", bg="#0f3460", fg="white",
                  font=bf, relief=tk.FLAT, cursor="hand2",
                  command=self._load_image).pack(fill=tk.X, padx=15, pady=(4,4))
        self.img_lbl = tk.Label(sb, text="nessuna immagine",
                                bg="#16213e", fg="#666680",
                                font=lf, wraplength=250)
        self.img_lbl.pack(pady=(0,8))

        self._section(sb, "PARAMETRI")
        cf = tk.Frame(sb, bg="#16213e")
        cf.pack(fill=tk.X, padx=15, pady=4)
        tk.Label(cf, text="Conf M1:", bg="#16213e",
                 fg="#a0a0b0", font=lf).pack(side=tk.LEFT)
        self.conf_var = tk.DoubleVar(value=CONF_M1)
        tk.Spinbox(cf, from_=0.1, to=0.95, increment=0.05,
                   textvariable=self.conf_var, width=5,
                   bg="#0f3460", fg="white", relief=tk.FLAT,
                   font=lf).pack(side=tk.RIGHT)

        self.run_btn = tk.Button(
            sb, text="▶ ESEGUI PIPELINE", bg="#e94560", fg="white",
            font=("Courier New", 11, "bold"), relief=tk.FLAT,
            cursor="hand2", command=self._run, state=tk.DISABLED)
        self.run_btn.pack(fill=tk.X, padx=15, pady=(15,6))

        self.progress = ttk.Progressbar(sb, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=15, pady=(0,8))

        self._section(sb, "RISULTATI")
        self.stats_lbl = tk.Label(sb, text="—", bg="#16213e",
                                  fg="#a0a0b0", font=lf,
                                  justify=tk.LEFT, wraplength=260)
        self.stats_lbl.pack(padx=15, pady=4, anchor=tk.W)

        # ── Area principale ──
        main = tk.Frame(self.root, bg="#1a1a2e")
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#1a1a2e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#16213e",
                        foreground="#a0a0b0",
                        font=("Courier New", 10, "bold"), padding=(12,6))
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
        style.map("TNotebook.Tab",
                  background=[("selected","#e94560")],
                  foreground=[("selected","white")])

<<<<<<< HEAD
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1 — immagine annotata
        self.tab_main = tk.Frame(self.notebook, bg="#0f0f1a")
        self.notebook.add(self.tab_main, text=" IMMAGINE ANNOTATA ")

        self.canvas = tk.Canvas(self.tab_main, bg="#0f0f1a",
                                cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Double-Button-1>", self._reset_view)

        # Tab 2 — crop individuali
        self.tab_crops = tk.Frame(self.notebook, bg="#0f0f1a")
        self.notebook.add(self.tab_crops, text=" CROP SINGOLI ")

        crops_container = tk.Frame(self.tab_crops, bg="#0f0f1a")
        crops_container.pack(fill=tk.BOTH, expand=True)

        self.crops_canvas = tk.Canvas(crops_container,
                                      bg="#0f0f1a",
                                      highlightthickness=0)
        crops_scroll = ttk.Scrollbar(crops_container,
                                     orient=tk.VERTICAL,
                                     command=self.crops_canvas.yview)
        self.crops_canvas.configure(yscrollcommand=crops_scroll.set)
        crops_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.crops_canvas.pack(fill=tk.BOTH, expand=True)

        self.crops_frame = tk.Frame(self.crops_canvas, bg="#0f0f1a")
        self.crops_canvas.create_window((0,0), window=self.crops_frame,
                                        anchor=tk.NW)
        self.crops_frame.bind("<Configure>",
            lambda e: self.crops_canvas.configure(
                scrollregion=self.crops_canvas.bbox("all")))

        # Status bar
        self.statusbar = tk.Label(
            self.root, text="pronto",
            bg="#0f3460", fg="#a0a0b0",
            font=("Courier New", 9), anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    def _section(self, parent, text):
        f = tk.Frame(parent, bg="#16213e")
        f.pack(fill=tk.X, padx=15, pady=(10,4))
=======
        nb = ttk.Notebook(main)
        nb.pack(fill=tk.BOTH, expand=True)
        self.notebook = nb
        nb.bind("<<NotebookTabChanged>>", self._on_tab_change)

        # Tab 0 — quote M1
        t0 = tk.Frame(nb, bg="#0f0f1a")
        nb.add(t0, text=" QUOTE (M1) ")
        self.view_quotes = CanvasView(t0)

        # Tab 1 — immagine annotata completa
        t1 = tk.Frame(nb, bg="#0f0f1a")
        nb.add(t1, text=" IMMAGINE ANNOTATA ")
        self.view_full = CanvasView(t1)

        # Tab 2 — simboli trovati
        t2 = tk.Frame(nb, bg="#0f0f1a")
        nb.add(t2, text=" SIMBOLI TROVATI ")
        sym_cont = tk.Frame(t2, bg="#0f0f1a")
        sym_cont.pack(fill=tk.BOTH, expand=True)
        self.sym_canvas = tk.Canvas(sym_cont, bg="#0f0f1a",
                                    highlightthickness=0)
        sym_scroll = ttk.Scrollbar(sym_cont, orient=tk.VERTICAL,
                                   command=self.sym_canvas.yview)
        self.sym_canvas.configure(yscrollcommand=sym_scroll.set)
        sym_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.sym_canvas.pack(fill=tk.BOTH, expand=True)
        self.sym_frame = tk.Frame(self.sym_canvas, bg="#0f0f1a")
        self.sym_canvas.create_window((0,0), window=self.sym_frame,
                                      anchor=tk.NW)
        self.sym_frame.bind("<Configure>",
            lambda e: self.sym_canvas.configure(
                scrollregion=self.sym_canvas.bbox("all")))

        # Status bar
        self.status = tk.Label(self.root, text="pronto",
                               bg="#0f3460", fg="#a0a0b0",
                               font=("Courier New", 9), anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _section(self, parent, text):
        f = tk.Frame(parent, bg="#16213e")
        f.pack(fill=tk.X, padx=15, pady=(8,3))
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
        tk.Label(f, text=text, bg="#16213e", fg="#e94560",
                 font=("Courier New", 9, "bold")).pack(side=tk.LEFT)
        tk.Frame(f, bg="#e94560", height=1).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8,0))

<<<<<<< HEAD
    def _model_row(self, parent, label, var, command):
        f = tk.Frame(parent, bg="#16213e")
        f.pack(fill=tk.X, padx=15, pady=2)
        tk.Label(f, text=label, bg="#16213e", fg="#a0a0b0",
                 font=("Courier New", 8)).pack(anchor=tk.W)
        row = tk.Frame(f, bg="#16213e")
        row.pack(fill=tk.X)
        entry = tk.Entry(row, textvariable=var, width=22,
                         bg="#0f3460", fg="#e0e0f0",
                         font=("Courier New", 7),
                         relief=tk.FLAT, insertbackground="white")
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(row, text="…", command=command,
                  bg="#e94560", fg="white",
                  font=("Courier New", 8), relief=tk.FLAT,
                  cursor="hand2", width=2).pack(side=tk.RIGHT)

    # ─── Browser file ──────────────────────

    def _browse_detector(self):
        p = filedialog.askopenfilename(
            title="Detector weights",
            filetypes=[("PyTorch", "*.pt"), ("Tutti", "*.*")])
        if p:
            self.det_var.set(p)

    def _browse_classifier(self):
        p = filedialog.askopenfilename(
            title="Classificatore weights",
            filetypes=[("PyTorch", "*.pt"), ("Tutti", "*.*")])
        if p:
            self.cls_var.set(p)

    def _load_image(self):
        p = filedialog.askopenfilename(
            title="Carica blueprint",
            filetypes=[("Immagini", "*.png *.jpg *.jpeg *.bmp"),
                       ("Tutti", "*.*")])
=======
    def _model_row(self, parent, label, var):
        f = tk.Frame(parent, bg="#16213e")
        f.pack(fill=tk.X, padx=15, pady=2)
        tk.Label(f, text=label, bg="#16213e", fg="#a0a0b0",
                 font=("Courier New", 7)).pack(anchor=tk.W)
        row = tk.Frame(f, bg="#16213e")
        row.pack(fill=tk.X)
        tk.Entry(row, textvariable=var, bg="#0f3460", fg="#e0e0f0",
                 font=("Courier New", 6), relief=tk.FLAT
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(row, text="…",
                  command=lambda v=var: v.set(
                      filedialog.askopenfilename(
                          filetypes=[("PyTorch","*.pt"),("Tutti","*.*")])
                      or v.get()),
                  bg="#e94560", fg="white", font=("Courier New",8),
                  relief=tk.FLAT, cursor="hand2", width=2
                  ).pack(side=tk.RIGHT)

    def _load_image(self):
        p = filedialog.askopenfilename(
            filetypes=[("Immagini","*.png *.jpg *.jpeg *.bmp"),
                       ("Tutti","*.*")])
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
        if not p:
            return
        img = cv2.imread(p)
        if img is None:
<<<<<<< HEAD
            messagebox.showerror("Errore", "Impossibile leggere l'immagine")
            return
        self.img_bgr   = img
        self.vis_bgr   = img.copy()
        self.crops_data = []
        self.img_label.config(text=Path(p).name)
        self._reset_view()
        self._show_image(img)
        self._set_status(f"Immagine caricata: {img.shape[1]}×{img.shape[0]}")
        if self.detector and self.classifier:
            self.run_btn.config(state=tk.NORMAL)

    # ─── Caricamento modelli ───────────────

    def _load_models(self):
        def _load():
            try:
                self._set_status("Caricamento detector...")
                self.detector = YOLO(self.det_var.get())
                self._set_status("Caricamento classificatore...")
                self.classifier = YOLO(self.cls_var.get())
                self.model_status.config(
                    text="● modelli caricati", fg="#00cc66")
=======
            messagebox.showerror("Errore","Impossibile leggere l'immagine")
            return
        self.img_bgr = img
        self.img_lbl.config(text=Path(p).name)
        # Mostra immagine originale in entrambi i tab
        self.view_quotes.show(img)
        self.view_full.show(img)
        self._set_status(f"Caricata: {img.shape[1]}×{img.shape[0]}")
        if self.m1 and self.m2:
            self.run_btn.config(state=tk.NORMAL)

    def _load_models(self):
        def _load():
            try:
                self._set_status("Caricamento M1...")
                self.m1 = YOLO(self.m1_var.get())
                self._set_status("Caricamento M2...")
                self.m2 = YOLO(self.m2_var.get())
                m3p = self.m3_var.get().strip()
                self.m3 = YOLO(m3p) if m3p else None
                self._set_status("Caricamento M3..." if self.m3 else "M3 non caricato")
                self.model_lbl.config(text="● modelli caricati", fg="#00cc66")
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
                self._set_status("Modelli pronti")
                if self.img_bgr is not None:
                    self.run_btn.config(state=tk.NORMAL)
            except Exception as e:
<<<<<<< HEAD
                messagebox.showerror("Errore modelli", str(e))
                self._set_status("Errore caricamento modelli")
        threading.Thread(target=_load, daemon=True).start()

    # ─── Pipeline ─────────────────────────

    def _run(self):
        if self.img_bgr is None or not self.detector or not self.classifier:
            return

        self.run_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        self._set_status("Esecuzione pipeline...")

        def _worker():
            try:
                vis, crops = run_pipeline(
                    self.img_bgr,
                    self.detector,
                    self.classifier,
                    self.conf_var.get()
                )
                self.vis_bgr    = vis
                self.crops_data = crops
                self.root.after(0, self._update_ui)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Errore pipeline", str(e)))
=======
                messagebox.showerror("Errore", str(e))
                self._set_status("Errore caricamento")
        threading.Thread(target=_load, daemon=True).start()

    def _run(self):
        if self.img_bgr is None or not self.m1 or not self.m2:
            return
        self.run_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        self._set_status("Pipeline in esecuzione...")

        def _worker():
            try:
                vis_q, vis_f, crops = run_pipeline(
                    self.img_bgr, self.m1, self.m2, self.m3,
                    self.conf_var.get())
                self.crops_data = crops
                self.root.after(0, lambda: self._update_ui(vis_q, vis_f))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Errore",str(e)))
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
                self.root.after(0, self._done)

        threading.Thread(target=_worker, daemon=True).start()

<<<<<<< HEAD
    def _update_ui(self):
        self._show_image(self.vis_bgr)
        self._show_crops()
        self._update_stats()
        self._done()

=======
    def _update_ui(self, vis_q, vis_f):
        self._vis_q = vis_q
        self._vis_f = vis_f
        # Mostra prima il tab quote, poi forza render con after
        self.notebook.select(0)
        self.root.after(100, self._render_results)
        self._show_symbols()
        self._update_stats()
        self._done()

    def _on_tab_change(self, event):
        tab = self.notebook.index(self.notebook.select())
        self.root.after(50, lambda: self._render_tab(tab))

    def _render_tab(self, tab):
        if tab == 0 and hasattr(self,"_vis_q") and self._vis_q is not None:
            self.view_quotes._render()
        elif tab == 1 and hasattr(self,"_vis_f") and self._vis_f is not None:
            self.view_full._render()

    def _render_results(self):
        if hasattr(self, "_vis_q") and self._vis_q is not None:
            self.view_quotes.show(self._vis_q)
        if hasattr(self, "_vis_f") and self._vis_f is not None:
            self.view_full.show(self._vis_f)

>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
    def _done(self):
        self.progress.stop()
        self.run_btn.config(state=tk.NORMAL)

<<<<<<< HEAD
    # ─── Visualizzazione immagine ──────────

    def _show_image(self, img_bgr):
        self._img_bgr_display = img_bgr
        self._render_canvas()

    def _render_canvas(self):
        if not hasattr(self, "_img_bgr_display"):
            return
        img = self._img_bgr_display
        ih, iw = img.shape[:2]
        cw = self.canvas.winfo_width()  or 800
        ch = self.canvas.winfo_height() or 600

        # Scala base per fit
        base_scale = min(cw/iw, ch/ih)
        scale = base_scale * self.zoom_scale
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))

        resized = cv2.resize(img, (nw,nh), interpolation=cv2.INTER_AREA)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._tk_img = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        ox = max(0, (cw-nw)//2) + self.pan_x
        oy = max(0, (ch-nh)//2) + self.pan_y
        self.canvas.create_image(ox, oy, anchor=tk.NW,
                                 image=self._tk_img)

    def _on_scroll(self, event):
        factor = 1.15 if event.delta > 0 else 1/1.15
        self.zoom_scale = max(0.2, min(10.0,
                                       self.zoom_scale * factor))
        self._render_canvas()

    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if self._drag_start:
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            self.pan_x += dx
            self.pan_y += dy
            self._drag_start = (event.x, event.y)
            self._render_canvas()

    def _reset_view(self, event=None):
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._render_canvas()

    # ─── Visualizzazione crop ──────────────

    def _show_crops(self):
        for w in self.crops_frame.winfo_children():
            w.destroy()

        if not self.crops_data:
            tk.Label(self.crops_frame,
                     text="nessun crop rilevato",
=======
    def _show_symbols(self):
        for w in self.sym_frame.winfo_children():
            w.destroy()

        all_syms = [s for d in self.crops_data for s in d["sym_crops"]]

        if not all_syms:
            tk.Label(self.sym_frame, text="nessun simbolo trovato",
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
                     bg="#0f0f1a", fg="#666680",
                     font=("Courier New", 10)).pack(pady=20)
            return

<<<<<<< HEAD
        # Griglia 4 colonne
        cols = 4
        for i, d in enumerate(self.crops_data):
            row = i // cols
            col = i % cols

            cell = tk.Frame(self.crops_frame, bg="#16213e",
                            relief=tk.FLAT, bd=1)
            cell.grid(row=row, column=col,
                      padx=5, pady=5, sticky=tk.NSEW)

            # Crop ridimensionato
            crop = d["crop"]
            ch, cw = crop.shape[:2]
            target = 120
            sc     = target / max(cw, ch)
            nw2    = max(1, int(cw*sc))
            nh2    = max(1, int(ch*sc))
            small  = cv2.resize(crop, (nw2,nh2))
            rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pil    = Image.fromarray(rgb)
            # Padding su canvas 130x130
            bg_pil = Image.new("RGB", (130,130),
                               (15,15,26))
            ox2 = (130-nw2)//2; oy2 = (130-nh2)//2
            bg_pil.paste(pil, (ox2, oy2))
            tk_img = ImageTk.PhotoImage(bg_pil)

            lbl_img = tk.Label(cell, image=tk_img,
                               bg="#16213e", cursor="hand2")
            lbl_img.image = tk_img
            lbl_img.pack()

            # Colore bordo
            r,g,b = d["color_rgb"]
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            cell.config(highlightbackground=hex_color,
                        highlightthickness=2,
                        highlightcolor=hex_color)

            # Label simbolo
            sym_text = f"{d['sym_name']}\n{d['sym_conf']:.0%}"
            tk.Label(cell, text=sym_text,
                     bg="#16213e", fg=hex_color,
                     font=("Courier New", 8, "bold"),
                     justify=tk.CENTER).pack(pady=(2,4))

    # ─── Stats ────────────────────────────

    def _update_stats(self):
        if not self.crops_data:
            self.stats_label.config(text="0 quote rilevate")
            return

        n = len(self.crops_data)
        counts = {}
        for d in self.crops_data:
            counts[d["sym_name"]] = counts.get(d["sym_name"], 0) + 1

        lines = [f"Quote rilevate: {n}", ""]
        for sym, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {sym}: {cnt}")

        self.stats_label.config(text="\n".join(lines))
        self._set_status(f"Pipeline completata — {n} quote rilevate")

    def _set_status(self, text):
        self.statusbar.config(text=f"  {text}")
=======
        cols = 5
        for i, s in enumerate(all_syms):
            cell = tk.Frame(self.sym_frame, bg="#16213e")
            cell.grid(row=i//cols, column=i%cols, padx=5, pady=5)

            crop = s["sym_crop"]
            ch2, cw2 = crop.shape[:2]
            sc  = 100 / max(cw2, ch2, 1)
            nw2 = max(1, int(cw2*sc)); nh2 = max(1, int(ch2*sc))
            small  = cv2.resize(crop,(nw2,nh2))
            bg_pil = Image.new("RGB",(110,110),(15,15,26))
            bg_pil.paste(Image.fromarray(cv2.cvtColor(small,cv2.COLOR_BGR2RGB)),
                         ((110-nw2)//2,(110-nh2)//2))
            tk_img = ImageTk.PhotoImage(bg_pil)

            lbl = tk.Label(cell, image=tk_img, bg="#16213e")
            lbl.image = tk_img
            lbl.pack()

            r,g,b = s["color_rgb"]
            hc = f"#{r:02x}{g:02x}{b:02x}"
            cell.config(highlightbackground=hc,
                        highlightthickness=2, highlightcolor=hc)
            tk.Label(cell, text=f"{s['sym_name']}\n{s['sym_conf']:.0%}",
                     bg="#16213e", fg=hc,
                     font=("Courier New", 7, "bold"),
                     justify=tk.CENTER).pack(pady=(2,3))

    def _update_stats(self):
        nq = len(self.crops_data)
        ns = sum(len(d["sym_crops"]) for d in self.crops_data)
        counts = {}
        for d in self.crops_data:
            for s in d["sym_crops"]:
                counts[s["sym_name"]] = counts.get(s["sym_name"],0)+1
        lines = [f"Quote M1:  {nq}", f"Simboli:   {ns}", ""]
        for sym,cnt in sorted(counts.items(), key=lambda x:-x[1]):
            lines.append(f"  {sym}: {cnt}")
        self.stats_lbl.config(text="\n".join(lines))
        self._set_status(f"Completato — {nq} quote, {ns} simboli")

    def _set_status(self, text):
        self.status.config(text=f"  {text}")
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
<<<<<<< HEAD
    app  = PipelineGUI(root)
=======
    PipelineGUI(root)
>>>>>>> 24e961a (BlueprintCV: pipeline M1+M2+M3, tiling, augmentation aggiornata)
    root.mainloop()