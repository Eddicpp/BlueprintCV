"""
SINTESI: Genesi — GUI Pipeline Detection + Classificazione Simboli
Interfaccia grafica Tkinter per la pipeline a due stadi:
  1. Carica modelli (detector + classificatore)
  2. Carica immagine blueprint
  3. Detection quote → crop → classificazione simbolo
  4. Visualizza risultati sull'immagine originale
  5. Opzione per vedere i singoli crop classificati
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
]

CLASS_COLORS = {
    "diameter":        (255,  80,  80),
    "radius":          ( 80, 200,  80),
    "angle":           ( 80,  80, 255),
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


# ─────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────

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
        x1=max(0,x1); y1=max(0,y1)
        x2=min(iw,x2); y2=min(ih,y2)
        if x2-x1 < 5 or y2-y1 < 5:
            continue

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


# ─────────────────────────────────────────
# GUI
# ─────────────────────────────────────────

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
        style.map("TNotebook.Tab",
                  background=[("selected","#e94560")],
                  foreground=[("selected","white")])

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
        tk.Label(f, text=text, bg="#16213e", fg="#e94560",
                 font=("Courier New", 9, "bold")).pack(side=tk.LEFT)
        tk.Frame(f, bg="#e94560", height=1).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8,0))

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
        if not p:
            return
        img = cv2.imread(p)
        if img is None:
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
                self._set_status("Modelli pronti")
                if self.img_bgr is not None:
                    self.run_btn.config(state=tk.NORMAL)
            except Exception as e:
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
                self.root.after(0, self._done)

        threading.Thread(target=_worker, daemon=True).start()

    def _update_ui(self):
        self._show_image(self.vis_bgr)
        self._show_crops()
        self._update_stats()
        self._done()

    def _done(self):
        self.progress.stop()
        self.run_btn.config(state=tk.NORMAL)

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
                     bg="#0f0f1a", fg="#666680",
                     font=("Courier New", 10)).pack(pady=20)
            return

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


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app  = PipelineGUI(root)
    root.mainloop()