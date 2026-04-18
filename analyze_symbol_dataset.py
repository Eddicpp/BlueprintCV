"""
SINTESI: Genesi — Analisi dataset simboli quote
Legge i JSON di LabelMe dalla cartella di labeling
e mostra la distribuzione delle classi.
"""

import json
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

LABELED_DIR = "./quote_per_labeling"

SYMBOL_CLASSES = [
    "diameter",
    "radius",
    "angle",
    "surface_finish",
    "concentricity",
    "cylindricity",
    "position",
    "flatness",
    "perpendicularity",
    "total_runout",
    "circular_runout",
    "slope",
    "conical_taper",
    "symmetry",
    "surface_profile",
    "linear",         # nessun simbolo
]

# ─────────────────────────────────────────
# ANALISI
# ─────────────────────────────────────────

if __name__ == "__main__":
    labeled_dir = Path(LABELED_DIR)
    json_files  = sorted(labeled_dir.glob("*.json"))

    print("=" * 55)
    print("SINTESI: Genesi — Analisi dataset simboli")
    print("=" * 55)
    print(f"JSON trovati: {len(json_files)}\n")

    counts      = defaultdict(int)
    unlabeled   = 0
    multi_label = 0
    errors      = 0

    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            errors += 1
            continue

        shapes = data.get("shapes", [])

        if not shapes:
            unlabeled += 1
            continue

        if len(shapes) > 1:
            multi_label += 1

        for shape in shapes:
            label = shape.get("label", "").strip().lower().replace(" ", "_")
            counts[label] += 1

    # Totale annotazioni
    total = sum(counts.values())

    print(f"{'Classe':25s}  {'N':>6}  {'%':>6}  {'Barra'}")
    print("─" * 60)

    # Ordina per frequenza decrescente
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        pct  = n / total * 100 if total > 0 else 0
        bar  = "█" * int(pct / 2)
        flag = " ← RARA" if n < 30 else ""
        print(f"  {cls:23s}  {n:>6}  {pct:>5.1f}%  {bar}{flag}")

    print("─" * 60)
    print(f"  {'TOTALE':23s}  {total:>6}")
    print(f"\nImmagini senza label:   {unlabeled}")
    print(f"Immagini multi-label:   {multi_label}")
    print(f"Errori lettura:         {errors}")

    # Classi mancanti
    found    = set(counts.keys())
    expected = set(c.replace(" ","_") for c in SYMBOL_CLASSES)
    missing  = expected - found
    unknown  = found - expected

    if missing:
        print(f"\nClassi attese ma non trovate: {sorted(missing)}")
    if unknown:
        print(f"Classi trovate non attese:    {sorted(unknown)}")

    # Suggerimento bilanciamento
    if counts:
        max_n   = max(counts.values())
        target  = max_n
        print(f"\n── Augmentation necessaria per bilanciare a {target} ──")
        for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
            needed = max(0, target - n)
            factor = target / n if n > 0 else float("inf")
            print(f"  {cls:23s}  hanno {n:>4}  "
                  f"→ generare {needed:>4}  (x{factor:.1f})")
