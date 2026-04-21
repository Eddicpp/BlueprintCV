"""
SINTESI: Genesi — Test Symbol Detector
Testa il symbol detector (M2) in tre modalità:

    # Su crop singolo di quota
    python test_symbol_detector.py --img path/crop.jpg

    # Su cartella di crop
    python test_symbol_detector.py --dir path/cartella/

    # Pipeline completa su blueprint (M1 + M2)
    python test_symbol_detector.py --blueprint path/blueprint.jpg

    # Senza argomenti → test su 10 immagini random dal test set
    python test_symbol_detector.py
"""

import cv2
import argparse
import random
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

SYMBOL_DETECTOR = r".\runs\detect\sintesi_genesi\symbol_detector_run1\weights\best.pt"
QUOTE_DETECTOR  = r".\runs\detect\sintesi_genesi\yolo11s_run3_augmented\weights\best.pt"
OUTPUT_DIR      = "./test_symbol_output"
CONF            = 0.20
IMG_SIZE        = 640

SYMBOL_CLASSES = [
    "diameter", "radius", "surface_finish",
    "concentricity", "cylindricity", "position", "flatness",
    "perpendicularity", "total_runout", "circular_runout",
    "slope", "conical_taper", "symmetry", "surface_profile",
]

CLASS_COLORS = [
    (255,80,80),(80,200,80),(220,160,0),(0,180,220),
    (180,0,220),(220,120,0),(0,200,160),(160,0,220),
    (220,0,120),(120,220,0),(0,120,220),(220,40,160),
    (40,220,220),(180,180,0),
]

# ─────────────────────────────────────────
# FUNZIONI
# ─────────────────────────────────────────

def draw_detections(img, results, offset_x=0, offset_y=0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    ih, iw = img.shape[:2]
    found = []

    if results.boxes is None or len(results.boxes) == 0:
        return img, found

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        x1=max(0,x1+offset_x); y1=max(0,y1+offset_y)
        x2=min(iw,x2+offset_x); y2=min(ih,y2+offset_y)

        color    = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        cls_name = results.names.get(cls_id,
                   SYMBOL_CLASSES[cls_id] if cls_id < len(SYMBOL_CLASSES)
                   else "unknown")

        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        label = f"{cls_name} {conf:.2f}"
        (lw,lh),_ = cv2.getTextSize(label,font,0.4,1)
        cv2.rectangle(img,(x1,max(0,y1-lh-4)),(x1+lw+4,y1),color,-1)
        cv2.putText(img,label,(x1+2,y1-2),
                    font,0.4,(255,255,255),1,cv2.LINE_AA)
        found.append((cls_name, conf, (x1,y1,x2,y2)))

    return img, found


def test_single(img_path: Path, model, out_dir: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Impossibile leggere: {img_path}")
        return []

    res = model.predict(img, imgsz=IMG_SIZE, conf=CONF, verbose=False)[0]
    vis, found = draw_detections(img.copy(), res)
    cv2.imwrite(str(out_dir / img_path.name), vis)

    if found:
        for cls_name, conf, box in found:
            print(f"  ✓ {cls_name:20s} conf={conf:.2f}  box={box}")
    else:
        print(f"  — nessun simbolo trovato")
    return found


def test_blueprint(img_path: Path, det_quotes, det_symbols, out_dir: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Impossibile leggere: {img_path}")
        return

    ih, iw = img.shape[:2]
    vis = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    det_res = det_quotes.predict(img, conf=0.40, verbose=False)[0]
    if det_res.boxes is None:
        print("  Nessuna quota trovata")
        return

    n_quotes = 0; n_symbols = 0

    for box in det_res.boxes:
        if int(box.cls[0]) != 2:
            continue
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        x1=max(0,x1); y1=max(0,y1)
        x2=min(iw,x2); y2=min(ih,y2)
        if x2-x1 < 5 or y2-y1 < 5:
            continue
        n_quotes += 1

        cv2.rectangle(vis,(x1,y1),(x2,y2),(160,160,160),1)
        crop    = img[y1:y2, x1:x2]
        sym_res = det_symbols.predict(
            crop, imgsz=IMG_SIZE, conf=CONF, verbose=False)[0]
        vis, found = draw_detections(vis, sym_res, offset_x=x1, offset_y=y1)
        n_symbols += len(found)

        if found:
            syms = ", ".join(f"{s}({c:.2f})" for s,c,_ in found)
            print(f"  [{x1},{y1},{x2},{y2}] → {syms}")

    cv2.imwrite(str(out_dir / img_path.name), vis)
    print(f"\n  Quote: {n_quotes}  |  Simboli: {n_symbols}")
    print(f"  Salvato: {out_dir / img_path.name}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",       help="Crop singolo di quota")
    parser.add_argument("--dir",       help="Cartella di blueprint — cerca simboli in ognuno")
    parser.add_argument("--blueprint", help="Blueprint singolo (M1+M2)")
    parser.add_argument("--conf", type=float, default=CONF)
    args = parser.parse_args()

    CONF = args.conf
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    print("=" * 55)
    print("SINTESI: Genesi — Test Symbol Detector")
    print("=" * 55)

    det_symbols = YOLO(SYMBOL_DETECTOR)

    if args.blueprint:
        det_quotes = YOLO(QUOTE_DETECTOR)
        print(f"\nBlueprint: {args.blueprint}\n")
        test_blueprint(Path(args.blueprint), det_quotes, det_symbols, out_dir)

    elif args.img:
        print(f"\nCrop singolo: {args.img}\n")
        test_single(Path(args.img), det_symbols, out_dir)

    elif args.dir:
        # Cartella di blueprint — pipeline completa M1+M2 su ogni immagine
        det_quotes = YOLO(QUOTE_DETECTOR)
        imgs = sorted(list(Path(args.dir).glob("*.jpg")) +
                      list(Path(args.dir).glob("*.JPG")) +
                      list(Path(args.dir).glob("*.jpeg")) +
                      list(Path(args.dir).glob("*.JPEG")) +
                      list(Path(args.dir).glob("*.png")) +
                      list(Path(args.dir).glob("*.PNG")))

        if not imgs:
            print(f"Nessuna immagine trovata in {args.dir}")
            exit(1)

        print(f"\nCartella: {args.dir}")
        print(f"Immagini trovate: {len(imgs)}\n")

        total_quotes  = 0
        total_symbols = 0

        for i, img_path in enumerate(imgs, 1):
            print(f"[{i}/{len(imgs)}] {img_path.name}")
            img = cv2.imread(str(img_path))
            if img is None:
                print("  Impossibile leggere, skip")
                continue

            ih, iw = img.shape[:2]
            vis = img.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX

            det_res = det_quotes.predict(img, conf=0.40, verbose=False)[0]
            if det_res.boxes is None:
                print("  Nessuna quota trovata")
                cv2.imwrite(str(out_dir / img_path.name), vis)
                continue

            n_quotes = 0; n_symbols = 0
            for box in det_res.boxes:
                if int(box.cls[0]) != 2:
                    continue
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                x1=max(0,x1); y1=max(0,y1)
                x2=min(iw,x2); y2=min(ih,y2)
                if x2-x1 < 5 or y2-y1 < 5:
                    continue
                n_quotes += 1

                cv2.rectangle(vis,(x1,y1),(x2,y2),(160,160,160),1)
                crop    = img[y1:y2, x1:x2]
                sym_res = det_symbols.predict(
                    crop, imgsz=IMG_SIZE, conf=CONF, verbose=False)[0]
                vis, found = draw_detections(vis, sym_res,
                                             offset_x=x1, offset_y=y1)
                n_symbols += len(found)
                if found:
                    syms = ", ".join(f"{s}({c:.2f})" for s,c,_ in found)
                    print(f"  [{x1},{y1},{x2},{y2}] → {syms}")

            cv2.imwrite(str(out_dir / img_path.name), vis)
            print(f"  Quote: {n_quotes}  |  Simboli: {n_symbols}")
            total_quotes  += n_quotes
            total_symbols += n_symbols

        print(f"\n── Riepilogo ──")
        print(f"  Immagini elaborate: {len(imgs)}")
        print(f"  Quote totali:       {total_quotes}")
        print(f"  Simboli trovati:    {total_symbols}")

    else:
        # Default — 10 immagini random dal test set
        test_dir = Path("./dataset_simboli_detection/test/images")
        imgs = sorted(list(test_dir.glob("*.jpg")) +
                      list(test_dir.glob("*.png")))
        random.shuffle(imgs)
        imgs = imgs[:10]
        print(f"\nTest set random ({len(imgs)} immagini)\n")
        for p in imgs:
            print(f"  {p.name}")
            test_single(p, det_symbols, out_dir)

    print(f"\n✓ Output in: {out_dir}")