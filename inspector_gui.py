"""
SINTESI: Genesi — GUI inferenza live
Interfaccia grafica per analizzare blueprint con il modello YOLO.
Permette di caricare immagini singole o cartelle e vedere le predizioni in tempo reale.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
import threading
import time

# ─────────────────────────────────────────
# CONFIGURAZIONE DEFAULT
# ─────────────────────────────────────────

DEFAULT_WEIGHTS = "./runs/detect/sintesi_genesi/yolo11s_run2_augmented/weights/best.pt"
CLASSES         = ["border", "table", "quote"]
IMG_SIZE        = 1024

COLORS_BGR = {
    "border": (209, 144, 74),
    "table":  (56,  168, 232),
    "quote":  (122, 184, 93),
}
COLORS_HEX = {
    "border": "#4A90D9",
    "table":  "#E8A838",
    "quote":  "#5DB87A",
}

# ─────────────────────────────────────────
# MOTORE INFERENZA
# ─────────────────────────────────────────

class InferenceEngine:
    def __init__(self):
        self.model   = None
        self.weights = None

    def load(self, weights_path: str):
        from ultralytics import YOLO
        self.model   = YOLO(weights_path)
        self.weights = weights_path

    def is_loaded(self):
        return self.model is not None

    def predict(self, img_path: str, conf: float, iou: float):
        if not self.is_loaded():
            return None, []
        results  = self.model.predict(
            source  = img_path,
            imgsz   = IMG_SIZE,
            conf    = conf,
            iou     = iou,
            device  = 0,
            verbose = False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                conf_v = float(box.conf)
                xyxy   = box.xyxy[0].tolist()
                name   = CLASSES[cls_id] if cls_id < len(CLASSES) else "unknown"
                detections.append({
                    "class": name,
                    "conf":  conf_v,
                    "xyxy":  [int(v) for v in xyxy],
                })
        return results[0].orig_img if results else None, detections


def draw_detections(img, detections, show_conf=True):
    out   = img.copy()
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(1.0, img.shape[1] / 2000))
    thick = max(1, int(img.shape[1] / 1500))

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        color = COLORS_BGR.get(det["class"], (128, 128, 128))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick + 1)

        label = f"{det['class']} {det['conf']:.2f}" if show_conf else det["class"]
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────
# APPLICAZIONE GUI
# ─────────────────────────────────────────

class SintesiGUI:
    def __init__(self, root):
        self.root        = root
        self.engine      = InferenceEngine()
        self.img_paths   = []
        self.current_idx = 0
        self.current_img = None
        self.detections  = []
        # Stato zoom
        self.zoom_level  = 1.0
        self.zoom_min    = 0.1
        self.zoom_max    = 10.0
        self.pan_x       = 0
        self.pan_y       = 0
        self._pan_start  = None
        self._setup_ui()
        self._try_load_default_weights()

    # ── SETUP UI ──────────────────────────

    def _setup_ui(self):
        self.root.title("SINTESI: Genesi — Inspector")
        self.root.configure(bg="#1C1C1E")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)

        # Stile
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame",       background="#1C1C1E")
        style.configure("TLabel",       background="#1C1C1E", foreground="#E5E5EA",
                        font=("Helvetica", 11))
        style.configure("TButton",      background="#2C2C2E", foreground="#E5E5EA",
                        font=("Helvetica", 11), borderwidth=0, padding=6)
        style.map("TButton",
                  background=[("active", "#3A3A3C"), ("pressed", "#48484A")])
        style.configure("Accent.TButton", background="#0A84FF", foreground="white",
                        font=("Helvetica", 11, "bold"), padding=8)
        style.map("Accent.TButton",
                  background=[("active", "#0071E3")])
        style.configure("TScale",       background="#1C1C1E", troughcolor="#3A3A3C",
                        sliderthickness=16)
        style.configure("TEntry",       fieldbackground="#2C2C2E", foreground="#E5E5EA",
                        insertcolor="#E5E5EA")

        # Layout principale
        self._build_sidebar()
        self._build_canvas_area()
        self._build_bottom_bar()

    def _build_sidebar(self):
        sidebar = tk.Frame(self.root, bg="#2C2C2E", width=260)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0,0))
        sidebar.pack_propagate(False)

        # Titolo
        tk.Label(sidebar, text="SINTESI", bg="#2C2C2E", fg="#0A84FF",
                 font=("Helvetica", 18, "bold")).pack(pady=(20,2))
        tk.Label(sidebar, text="Genesi Inspector", bg="#2C2C2E", fg="#8E8E93",
                 font=("Helvetica", 11)).pack(pady=(0,20))

        tk.Frame(sidebar, bg="#3A3A3C", height=1).pack(fill=tk.X, padx=16)

        # Sezione modello
        self._section_label(sidebar, "MODELLO")
        self.lbl_model = tk.Label(sidebar, text="Nessun modello caricato",
                                  bg="#2C2C2E", fg="#FF453A",
                                  font=("Helvetica", 10), wraplength=220)
        self.lbl_model.pack(padx=16, anchor="w")

        btn_frame = tk.Frame(sidebar, bg="#2C2C2E")
        btn_frame.pack(fill=tk.X, padx=16, pady=8)
        ttk.Button(btn_frame, text="Carica modello",
                   command=self._load_weights).pack(fill=tk.X)

        tk.Frame(sidebar, bg="#3A3A3C", height=1).pack(fill=tk.X, padx=16, pady=8)

        # Sezione immagini
        self._section_label(sidebar, "IMMAGINI")
        btn_frame2 = tk.Frame(sidebar, bg="#2C2C2E")
        btn_frame2.pack(fill=tk.X, padx=16, pady=(0,8))
        ttk.Button(btn_frame2, text="Apri immagine",
                   command=self._open_image).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame2, text="Apri cartella",
                   command=self._open_folder).pack(fill=tk.X, pady=2)

        self.lbl_files = tk.Label(sidebar, text="0 immagini caricate",
                                  bg="#2C2C2E", fg="#8E8E93",
                                  font=("Helvetica", 10))
        self.lbl_files.pack(padx=16, anchor="w")

        tk.Frame(sidebar, bg="#3A3A3C", height=1).pack(fill=tk.X, padx=16, pady=8)

        # Soglie
        self._section_label(sidebar, "PARAMETRI")

        self._slider_row(sidebar, "Confidence", 0.05, 1.0, 0.618, "conf_var")
        self._slider_row(sidebar, "IoU (NMS)",  0.1,  0.9, 0.45,  "iou_var")

        # Opzioni
        self.show_conf_var = tk.BooleanVar(value=True)
        tk.Checkbutton(sidebar, text="Mostra confidence",
                       variable=self.show_conf_var,
                       bg="#2C2C2E", fg="#E5E5EA",
                       selectcolor="#3A3A3C",
                       activebackground="#2C2C2E",
                       activeforeground="#E5E5EA",
                       command=self._refresh_display).pack(padx=16, anchor="w")

        tk.Frame(sidebar, bg="#3A3A3C", height=1).pack(fill=tk.X, padx=16, pady=8)

        # Legenda classi
        self._section_label(sidebar, "CLASSI")
        for cls, color in COLORS_HEX.items():
            row = tk.Frame(sidebar, bg="#2C2C2E")
            row.pack(fill=tk.X, padx=16, pady=2)
            tk.Canvas(row, width=14, height=14, bg="#2C2C2E",
                      highlightthickness=0).pack(side=tk.LEFT)
            c = tk.Canvas(row, width=14, height=14, bg=color,
                          highlightthickness=0)
            c.pack(side=tk.LEFT)
            tk.Label(row, text=f"  {cls}", bg="#2C2C2E", fg="#E5E5EA",
                     font=("Helvetica", 11)).pack(side=tk.LEFT)

        # Pulsante analizza
        tk.Frame(sidebar, bg="#2C2C2E").pack(fill=tk.BOTH, expand=True)
        ttk.Button(sidebar, text="Analizza",
                   style="Accent.TButton",
                   command=self._run_inference).pack(
                       fill=tk.X, padx=16, pady=16)

    def _build_canvas_area(self):
        self.canvas_frame = tk.Frame(self.root, bg="#1C1C1E")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Toolbar navigazione
        nav = tk.Frame(self.canvas_frame, bg="#2C2C2E", height=44)
        nav.pack(fill=tk.X)
        nav.pack_propagate(False)

        ttk.Button(nav, text="◀", width=3,
                   command=self._prev_image).pack(side=tk.LEFT, padx=8, pady=6)
        ttk.Button(nav, text="▶", width=3,
                   command=self._next_image).pack(side=tk.LEFT, pady=6)

        self.lbl_filename = tk.Label(nav, text="Nessuna immagine",
                                     bg="#2C2C2E", fg="#E5E5EA",
                                     font=("Helvetica", 11))
        self.lbl_filename.pack(side=tk.LEFT, padx=16)

        # Controlli zoom (destra)
        self.lbl_counter = tk.Label(nav, text="",
                                    bg="#2C2C2E", fg="#8E8E93",
                                    font=("Helvetica", 10))
        self.lbl_counter.pack(side=tk.RIGHT, padx=16)

        ttk.Button(nav, text="Salva",
                   command=self._save_result).pack(side=tk.RIGHT, padx=4, pady=6)

        ttk.Button(nav, text="Reset zoom",
                   command=self._zoom_reset).pack(side=tk.RIGHT, padx=4, pady=6)

        ttk.Button(nav, text="−", width=2,
                   command=self._zoom_out).pack(side=tk.RIGHT, padx=2, pady=6)

        self.lbl_zoom = tk.Label(nav, text="100%", bg="#2C2C2E", fg="#0A84FF",
                                 font=("Helvetica", 10), width=5)
        self.lbl_zoom.pack(side=tk.RIGHT, padx=2)

        ttk.Button(nav, text="+", width=2,
                   command=self._zoom_in).pack(side=tk.RIGHT, padx=2, pady=6)

        # Canvas immagine
        self.canvas = tk.Canvas(self.canvas_frame, bg="#141414",
                                highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>",    self._on_resize)
        # Zoom con rotella del mouse
        self.canvas.bind("<MouseWheel>",   self._on_mousewheel)        # Windows
        self.canvas.bind("<Button-4>",     self._on_mousewheel)        # Linux scroll up
        self.canvas.bind("<Button-5>",     self._on_mousewheel)        # Linux scroll down
        # Pan con click sinistro + trascina
        self.canvas.bind("<ButtonPress-1>",   self._on_pan_start)
        self.canvas.bind("<B1-Motion>",       self._on_pan_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_pan_end)
        # Doppio click reset zoom
        self.canvas.bind("<Double-Button-1>", lambda e: self._zoom_reset())

        # Pannello detections
        self.det_frame = tk.Frame(self.canvas_frame, bg="#2C2C2E", height=120)
        self.det_frame.pack(fill=tk.X)
        self.det_frame.pack_propagate(False)

        tk.Label(self.det_frame, text="RILEVAZIONI",
                 bg="#2C2C2E", fg="#8E8E93",
                 font=("Helvetica", 9)).pack(anchor="w", padx=12, pady=(6,2))

        self.det_text = tk.Text(self.det_frame, bg="#2C2C2E", fg="#E5E5EA",
                                font=("Courier", 10), height=4,
                                state=tk.DISABLED, relief=tk.FLAT,
                                insertbackground="#E5E5EA")
        self.det_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0,8))

    def _build_bottom_bar(self):
        self.status_bar = tk.Label(self.root, text="Pronto",
                                   bg="#0A84FF", fg="white",
                                   font=("Helvetica", 10),
                                   anchor="w", padx=12)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _section_label(self, parent, text):
        tk.Label(parent, text=text, bg="#2C2C2E", fg="#8E8E93",
                 font=("Helvetica", 9)).pack(padx=16, anchor="w", pady=(8,2))

    def _slider_row(self, parent, label, from_, to, default, var_name):
        var = tk.DoubleVar(value=default)
        setattr(self, var_name, var)

        row = tk.Frame(parent, bg="#2C2C2E")
        row.pack(fill=tk.X, padx=16, pady=2)
        tk.Label(row, text=label, bg="#2C2C2E", fg="#E5E5EA",
                 font=("Helvetica", 10), width=12, anchor="w").pack(side=tk.LEFT)
        val_lbl = tk.Label(row, text=f"{default:.2f}",
                           bg="#2C2C2E", fg="#0A84FF",
                           font=("Helvetica", 10), width=4)
        val_lbl.pack(side=tk.RIGHT)

        def on_change(v):
            val_lbl.config(text=f"{float(v):.2f}")
            self._refresh_display()

        ttk.Scale(parent, from_=from_, to=to, variable=var,
                  orient=tk.HORIZONTAL,
                  command=on_change).pack(fill=tk.X, padx=16, pady=(0,4))

    # ── LOGICA ────────────────────────────

    def _try_load_default_weights(self):
        path = Path(DEFAULT_WEIGHTS)
        if path.exists():
            self._load_weights_path(str(path))

    def _load_weights(self):
        path = filedialog.askopenfilename(
            title="Seleziona pesi modello",
            filetypes=[("PyTorch weights", "*.pt"), ("Tutti i file", "*.*")]
        )
        if path:
            self._load_weights_path(path)

    def _load_weights_path(self, path):
        self._set_status("Caricamento modello...", "#FF9F0A")
        self.root.update()
        try:
            self.engine.load(path)
            name = Path(path).parent.parent.name
            self.lbl_model.config(text=name, fg="#30D158")
            self._set_status(f"Modello caricato: {name}", "#30D158")
        except Exception as e:
            self.lbl_model.config(text=f"Errore: {e}", fg="#FF453A")
            self._set_status("Errore caricamento modello", "#FF453A")

    def _open_image(self):
        paths = filedialog.askopenfilenames(
            title="Seleziona immagini",
            filetypes=[("Immagini", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                       ("Tutti", "*.*")]
        )
        if paths:
            self.img_paths   = list(paths)
            self.current_idx = 0
            self.lbl_files.config(text=f"{len(self.img_paths)} immagini caricate")
            self._load_current_image()

    def _open_folder(self):
        folder = filedialog.askdirectory(title="Seleziona cartella immagini")
        if folder:
            exts = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
            self.img_paths = sorted([
                str(p) for p in Path(folder).iterdir()
                if p.suffix.lower() in exts
            ])
            self.current_idx = 0
            self.lbl_files.config(text=f"{len(self.img_paths)} immagini caricate")
            if self.img_paths:
                self._load_current_image()
            else:
                messagebox.showinfo("Nessuna immagine",
                                    "Nessuna immagine trovata nella cartella.")

    def _load_current_image(self):
        if not self.img_paths:
            return
        path = self.img_paths[self.current_idx]
        self.current_img = cv2.imread(path)
        self.detections  = []
        # Reset zoom ad ogni nuova immagine
        self.zoom_level  = 1.0
        self.pan_x       = 0
        self.pan_y       = 0
        self.lbl_zoom.config(text="100%")
        self.lbl_filename.config(text=Path(path).name)
        self.lbl_counter.config(
            text=f"{self.current_idx + 1} / {len(self.img_paths)}")
        self._update_det_panel([])
        self._display_image(self.current_img)

        # Avvia inferenza automatica se il modello è caricato
        if self.engine.is_loaded():
            self._run_inference()

    def _run_inference(self):
        if not self.img_paths:
            messagebox.showinfo("Nessuna immagine", "Carica prima un'immagine.")
            return
        if not self.engine.is_loaded():
            messagebox.showinfo("Nessun modello", "Carica prima il modello.")
            return

        path = self.img_paths[self.current_idx]
        self._set_status("Analisi in corso...", "#FF9F0A")
        self.root.update()

        def worker():
            t0 = time.time()
            _, dets = self.engine.predict(
                path,
                conf=self.conf_var.get(),
                iou=self.iou_var.get()
            )
            elapsed = time.time() - t0
            self.detections = dets
            self.root.after(0, lambda: self._on_inference_done(dets, elapsed))

        threading.Thread(target=worker, daemon=True).start()

    def _on_inference_done(self, dets, elapsed):
        self._refresh_display()
        n = len(dets)
        counts = {cls: sum(1 for d in dets if d["class"] == cls)
                  for cls in CLASSES}
        summary = "  |  ".join(
            f"{cls}: {counts[cls]}" for cls in CLASSES if counts[cls] > 0
        ) or "nessuna rilevazione"
        self._set_status(
            f"{n} rilevazioni  ({elapsed*1000:.0f}ms)  —  {summary}",
            "#30D158"
        )
        self._update_det_panel(dets)

    def _refresh_display(self):
        if self.current_img is None:
            return
        annotated = draw_detections(
            self.current_img,
            self.detections,
            show_conf=self.show_conf_var.get()
        )
        self._display_image(annotated)

    def _display_image(self, img):
        if img is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        h, w = img.shape[:2]

        # Scala base per far entrare l'immagine nel canvas
        base_scale = min(cw / w, ch / h)
        # Scala totale = base × zoom
        total_scale = base_scale * self.zoom_level
        nw = int(w * total_scale)
        nh = int(h * total_scale)

        rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        tk_img  = ImageTk.PhotoImage(pil_img)

        # Posizione centrata + offset pan
        cx = cw // 2 + self.pan_x
        cy = ch // 2 + self.pan_y

        self.canvas.delete("all")
        self.canvas.create_image(cx, cy, anchor=tk.CENTER, image=tk_img)
        self.canvas._tk_img = tk_img

    # ── ZOOM E PAN ────────────────────────

    def _zoom_in(self):
        self._apply_zoom(1.25)

    def _zoom_out(self):
        self._apply_zoom(0.8)

    def _zoom_reset(self):
        self.zoom_level = 1.0
        self.pan_x      = 0
        self.pan_y      = 0
        self.lbl_zoom.config(text="100%")
        self._refresh_display()

    def _apply_zoom(self, factor, cx=None, cy=None):
        old_zoom = self.zoom_level
        new_zoom = max(self.zoom_min, min(self.zoom_max, self.zoom_level * factor))
        if new_zoom == old_zoom:
            return

        # Se zoom centrato su un punto del canvas (rotella mouse)
        if cx is not None and cy is not None:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            # Offset del cursore rispetto al centro del canvas
            dx = cx - (cw // 2 + self.pan_x)
            dy = cy - (ch // 2 + self.pan_y)
            # Aggiusta pan in modo che il punto sotto il cursore resti fisso
            scale_ratio = new_zoom / old_zoom
            self.pan_x = int(cx - cw // 2 - dx * scale_ratio)
            self.pan_y = int(cy - ch // 2 - dy * scale_ratio)

        self.zoom_level = new_zoom
        self.lbl_zoom.config(text=f"{int(new_zoom * 100)}%")
        self._refresh_display()

    def _on_mousewheel(self, event):
        # Determina direzione scroll
        if event.num == 4:
            delta = 1
        elif event.num == 5:
            delta = -1
        else:
            delta = 1 if event.delta > 0 else -1

        factor = 1.15 if delta > 0 else (1 / 1.15)
        self._apply_zoom(factor, cx=event.x, cy=event.y)

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y)
        self.canvas.config(cursor="fleur")

    def _on_pan_move(self, event):
        if self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self.pan_x      += dx
        self.pan_y      += dy
        self._pan_start  = (event.x, event.y)
        self._refresh_display()

    def _on_pan_end(self, event):
        self._pan_start = None
        self.canvas.config(cursor="")

    def _update_det_panel(self, dets):
        self.det_text.config(state=tk.NORMAL)
        self.det_text.delete("1.0", tk.END)
        if not dets:
            self.det_text.insert(tk.END, "  Nessuna rilevazione\n")
        else:
            for i, d in enumerate(sorted(dets, key=lambda x: -x["conf"])):
                x1,y1,x2,y2 = d["xyxy"]
                line = (f"  [{i+1:2d}] {d['class']:8s}  "
                        f"conf={d['conf']:.3f}  "
                        f"bbox=({x1},{y1}) → ({x2},{y2})\n")
                self.det_text.insert(tk.END, line)
        self.det_text.config(state=tk.DISABLED)

    def _prev_image(self):
        if self.img_paths and self.current_idx > 0:
            self.current_idx -= 1
            self._load_current_image()

    def _next_image(self):
        if self.img_paths and self.current_idx < len(self.img_paths) - 1:
            self.current_idx += 1
            self._load_current_image()

    def _save_result(self):
        if self.current_img is None:
            return
        annotated = draw_detections(
            self.current_img, self.detections,
            show_conf=self.show_conf_var.get()
        )
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if path:
            cv2.imwrite(path, annotated)
            self._set_status(f"Salvato: {Path(path).name}", "#30D158")

    def _on_resize(self, event):
        self._refresh_display()

    def _set_status(self, text, color="#0A84FF"):
        self.status_bar.config(text=f"  {text}", bg=color)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app  = SintesiGUI(root)

    # Navigazione da tastiera
    root.bind("<Left>",  lambda e: app._prev_image())
    root.bind("<Right>", lambda e: app._next_image())
    root.bind("<Return>",lambda e: app._run_inference())
    # Zoom da tastiera
    root.bind("<plus>",        lambda e: app._zoom_in())
    root.bind("<equal>",       lambda e: app._zoom_in())   # tasto = senza shift
    root.bind("<minus>",       lambda e: app._zoom_out())
    root.bind("<0>",           lambda e: app._zoom_reset())

    root.mainloop()