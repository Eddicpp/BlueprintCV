"""
SINTESI: Genesi — YOLOv8 Training + Valutazione con grafici per classe
GPU: NVIDIA locale
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")           # backend non-interattivo, funziona senza display
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

DATASET_YAML = "./dataset_finale/data.yaml"
CLASSES      = ["border", "table", "quote"]
MODEL_SIZE = "rtdetr-l"
EPOCHS       = 100
IMG_SIZE     = 640
BATCH_SIZE   = 8
PATIENCE     = 20
PROJECT      = "sintesi_genesi"
RUN_NAME   = "rtdetr_run1"

COLORS = {
    "border": "#4A90D9",
    "table":  "#E8A838",
    "quote":  "#5DB87A",
}

# ─────────────────────────────────────────
# 1. TRAINING
# ─────────────────────────────────────────

def train():
    print("=" * 55)
    print("SINTESI: Genesi — Training YOLOv8n")
    print("=" * 55)

    model = YOLO(f"{MODEL_SIZE}")

    model.train(
        data        = DATASET_YAML,
        epochs      = EPOCHS,
        imgsz       = IMG_SIZE,
        batch       = BATCH_SIZE,
        patience    = PATIENCE,
        device      = 0,
        project     = PROJECT,
        name        = RUN_NAME,
        exist_ok    = True,
        plots       = True,
        save        = True,
        save_period = 10,
        verbose     = True,
        # Augmentation per blueprint B&N
        hsv_h   = 0.0,
        hsv_s   = 0.0,
        hsv_v   = 0.2,
        fliplr  = 0.5,
        flipud  = 0.0,
        degrees = 0.0,
        scale   = 0.5,
        mosaic  = 1.0,
    )

    run_dir = Path(PROJECT) / RUN_NAME
    print(f"\n✓ Training completato → {run_dir}")
    return run_dir


# ─────────────────────────────────────────
# 2. VALUTAZIONE PER CLASSE
# ─────────────────────────────────────────

def evaluate(run_dir: Path):
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        print(f"Pesi non trovati: {weights}")
        return None

    print("\n" + "=" * 55)
    print("Valutazione sul test set")
    print("=" * 55)

    model   = YOLO(str(weights))
    metrics = model.val(
        data     = DATASET_YAML,
        split    = "test",
        imgsz    = IMG_SIZE,
        device   = 0,
        plots    = True,
        project  = PROJECT,
        name     = RUN_NAME + "_eval",
        exist_ok = True,
    )

    names     = metrics.names
    per_class = {}
    for i, name in names.items():
        per_class[name] = {
            "precision": float(metrics.box.p[i]),
            "recall":    float(metrics.box.r[i]),
            "ap50":      float(metrics.box.ap50[i]),
            "ap50_95":   float(metrics.box.ap[i]),
        }

    overall = {
        "mAP50":     float(metrics.box.map50),
        "mAP50_95":  float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
    }

    print(f"\n{'Classe':10s}  {'P':>6}  {'R':>6}  {'AP50':>6}  {'AP50-95':>8}")
    print("-" * 45)
    for cls, m in per_class.items():
        print(f"{cls:10s}  {m['precision']:6.3f}  {m['recall']:6.3f}  "
              f"{m['ap50']:6.3f}  {m['ap50_95']:8.3f}")
    print("-" * 45)
    print(f"{'Overall':10s}  {overall['precision']:6.3f}  {overall['recall']:6.3f}  "
          f"{overall['mAP50']:6.3f}  {overall['mAP50_95']:8.3f}")

    metrics_path = run_dir / "metrics_per_class.json"
    with open(metrics_path, "w") as f:
        json.dump({"per_class": per_class, "overall": overall}, f, indent=2)
    print(f"\nMetriche salvate: {metrics_path}")

    return per_class, overall


# ─────────────────────────────────────────
# 3. GRAFICI
# ─────────────────────────────────────────

def plot_per_class_metrics(per_class: dict, overall: dict, run_dir: Path):
    """Barchart Precision / Recall / AP50 / AP50-95 per classe"""

    metrics_keys = ["precision", "recall", "ap50", "ap50_95"]
    labels_map   = {
        "precision": "Precision",
        "recall":    "Recall",
        "ap50":      "AP@50",
        "ap50_95":   "AP@50-95",
    }
    overall_map = {
        "precision": "precision",
        "recall":    "recall",
        "ap50":      "mAP50",
        "ap50_95":   "mAP50_95",
    }

    classes = list(per_class.keys())
    x       = np.arange(len(classes))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Metriche per classe — YOLOv8n Baseline",
                 fontsize=14, fontweight="bold")

    for ax, key in zip(axes, metrics_keys):
        values = [per_class[c][key] for c in classes]
        bars   = ax.bar(x, values, 0.6,
                        color=[COLORS.get(c, "#888") for c in classes],
                        edgecolor="white", linewidth=0.8)

        ov_val = overall.get(overall_map[key])
        if ov_val is not None:
            ax.axhline(ov_val, color="#333", linestyle="--",
                       linewidth=1.2, label=f"Overall {ov_val:.3f}")
            ax.legend(fontsize=8)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(labels_map[key], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Valore", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    out = run_dir / "grafici_per_classe.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Barchart per classe → {out}")


def plot_training_curves(run_dir: Path):
    """Loss e mAP per epoca da results.csv"""
    import csv
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"results.csv non trovato — salto le curve di training")
        return

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {}
            for k, v in row.items():
                try:
                    clean[k.strip()] = float(v.strip())
                except ValueError:
                    pass
            if clean:
                rows.append(clean)

    if not rows:
        return

    epochs     = [r["epoch"] for r in rows]
    train_loss = [r.get("train/box_loss", 0) + r.get("train/cls_loss", 0) for r in rows]
    val_loss   = [r.get("val/box_loss", 0)   + r.get("val/cls_loss", 0)   for r in rows]
    map50      = [r.get("metrics/mAP50(B)", 0)    for r in rows]
    map50_95   = [r.get("metrics/mAP50-95(B)", 0) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Curve di training — YOLOv8n", fontsize=14, fontweight="bold")

    ax1.plot(epochs, train_loss, color="#4A90D9", linewidth=1.8, label="Train loss")
    ax1.plot(epochs, val_loss,   color="#E8A838", linewidth=1.8, label="Val loss")
    ax1.set_title("Loss (box + cls)")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(epochs, map50,    color="#5DB87A", linewidth=1.8, label="mAP@50")
    ax2.plot(epochs, map50_95, color="#9B59B6", linewidth=1.8, label="mAP@50-95")
    ax2.set_title("mAP per epoca")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("mAP")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out = run_dir / "grafici_training.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Curve di training → {out}")


def plot_radar(run_dir: Path, per_class: dict):
    """Radar chart Precision / Recall / AP50 per classe"""
    classes   = list(per_class.keys())
    keys      = ["precision", "recall", "ap50"]
    labels    = ["Precision", "Recall", "AP@50"]
    n         = len(keys)
    angles    = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles   += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.suptitle("Radar — Metriche per classe", fontsize=13, fontweight="bold")

    for cls in classes:
        vals  = [per_class[cls][k] for k in keys] + [per_class[cls][keys[0]]]
        color = COLORS.get(cls, "#888")
        ax.plot(angles, vals, linewidth=2, color=color, label=cls)
        ax.fill(angles, vals, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(color="gray", alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    out = run_dir / "grafici_radar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Radar chart → {out}")


# ─────────────────────────────────────────
# 4. INFERENZA RAPIDA
# ─────────────────────────────────────────

def quick_inference(image_path: str, run_dir: Path):
    weights = run_dir / "weights" / "best.pt"
    model   = YOLO(str(weights))
    results = model.predict(
        source   = image_path,
        imgsz    = IMG_SIZE,
        conf     = 0.25,
        iou      = 0.45,
        device   = 0,
        save     = True,
        save_txt = True,
        project  = PROJECT,
        name     = "inference",
        exist_ok = True,
    )
    for r in results:
        for box in r.boxes:
            cls  = int(box.cls)
            conf = float(box.conf)
            xyxy = box.xyxy[0].tolist()
            print(f"  {model.names[cls]:10s}  conf={conf:.2f}  "
                  f"bbox=[{', '.join(f'{v:.0f}' for v in xyxy)}]")
    return results


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":

    # 1. Training
    run_dir = train()

    # 2. Curve di training
    print("\nGenerazione grafici training...")
    plot_training_curves(run_dir)

    # 3. Valutazione sul test set
    result = evaluate(run_dir)

    if result:
        per_class, overall = result

        # 4. Grafici per classe
        print("\nGenerazione grafici per classe...")
        plot_per_class_metrics(per_class, overall, run_dir)
        plot_radar(run_dir, per_class)

    print("\n" + "=" * 55)
    print(f"Output salvati in: {run_dir}/")
    print("  grafici_training.png   — loss e mAP per epoca")
    print("  grafici_per_classe.png — P / R / AP50 / AP50-95")
    print("  grafici_radar.png      — radar per classe")
    print("  metrics_per_class.json — metriche in JSON")
    print("  weights/best.pt        — pesi migliori")
    print("=" * 55)
