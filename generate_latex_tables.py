"""
SINTESI: Genesi — Generatore tabelle LaTeX
Raccoglie automaticamente le metriche da tutti i run YOLO e produce
un file .tex con due tabelle pronte per la tesi:
  1. Confronto modelli (una riga per modello)
  2. Dettaglio per classe (una riga per classe per modello)
"""

import json
import csv
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

RUNS_DIR    = "./runs/detect/sintesi_genesi"   # cartella con tutti i run
OUTPUT_FILE = "./tabelle_tesi.tex"
CLASSES     = ["border", "table", "quote"]

# Mappa nome cartella → nome leggibile per la tabella
MODEL_NAMES = {
    "yolov8n_run1": "YOLOv8n",
    "yolov8s_run1": "YOLOv8s",
    "yolo11s_run1": "YOLO11s",
    "yolov9s_run1": "YOLOv9s",
    "yolov10s_run1": "YOLOv10s",
}

# ─────────────────────────────────────────
# LETTURA METRICHE
# ─────────────────────────────────────────

def read_overall_metrics(run_dir: Path):
    """
    Legge mAP, P, R da results.csv (ultima riga = epoch migliore).
    Legge F1 ottimale dal nome del grafico non disponibile direttamente,
    quindi lo calcola come 2*P*R/(P+R).
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None

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
        return None

    # Prendi la riga con mAP50 massimo
    best = max(rows, key=lambda r: r.get("metrics/mAP50(B)", 0))

    p  = best.get("metrics/precision(B)", 0)
    r  = best.get("metrics/recall(B)", 0)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    return {
        "mAP50":    best.get("metrics/mAP50(B)", 0),
        "mAP5095":  best.get("metrics/mAP50-95(B)", 0),
        "precision": p,
        "recall":    r,
        "f1":        f1,
    }


def read_per_class_metrics(run_dir: Path):
    """
    Legge le metriche per classe da metrics_per_class.json
    (generato da train_yolov8.py dopo la valutazione).
    """
    json_path = run_dir / "metrics_per_class.json"
    if not json_path.exists():
        return None

    with open(json_path) as f:
        data = json.load(f)
    return data.get("per_class", None)


def collect_all_runs(runs_dir: Path):
    """
    Scansiona tutti i run disponibili e raccoglie le metriche.
    """
    results = {}
    runs_path = Path(runs_dir)

    if not runs_path.exists():
        print(f"Cartella runs non trovata: {runs_path.resolve()}")
        return results

    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.is_dir():
            continue

        run_name   = run_dir.name
        model_name = MODEL_NAMES.get(run_name, run_name)

        overall   = read_overall_metrics(run_dir)
        per_class = read_per_class_metrics(run_dir)

        if overall is None:
            print(f"  ✗ {run_name}: results.csv non trovato, skip")
            continue

        results[model_name] = {
            "overall":   overall,
            "per_class": per_class,
        }
        print(f"  ✓ {model_name}: mAP@50={overall['mAP50']:.3f}")

    return results


# ─────────────────────────────────────────
# GENERAZIONE LATEX
# ─────────────────────────────────────────

def bold_best(values: list, fmt=".3f"):
    """
    Dato una lista di valori float, restituisce lista di stringhe
    dove il valore più alto è in \\textbf{}.
    """
    if not values or all(v is None for v in values):
        return ["—"] * len(values)
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return ["—"] * len(values)
    best_i = max(valid, key=lambda x: x[1])[0]
    out = []
    for i, v in enumerate(values):
        if v is None:
            out.append("—")
        elif i == best_i:
            out.append(f"\\textbf{{{v:{fmt}}}}")
        else:
            out.append(f"{v:{fmt}}")
    return out


def table_confronto_modelli(results: dict) -> str:
    """
    Tabella 1: confronto modelli — una riga per modello.
    Colonne: Modello | P | R | F1 | mAP@50 | mAP@50-95
    """
    models = list(results.keys())
    metrics_keys = ["precision", "recall", "f1", "mAP50", "mAP5095"]
    headers = ["Modello", "Precision", "Recall", "F1", "mAP@50", "mAP@50-95"]

    # Raccogli valori per colonna (per bold_best)
    col_values = {k: [results[m]["overall"].get(k) for m in models]
                  for k in metrics_keys}

    # Bold per colonna
    col_bold = {k: bold_best(col_values[k]) for k in metrics_keys}

    lines = []
    lines.append("% ─────────────────────────────────────────────────────")
    lines.append("% Tabella 1: Confronto modelli YOLO — SINTESI: Genesi")
    lines.append("% ─────────────────────────────────────────────────────")
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append("  \\caption{Confronto delle metriche di detection tra le varianti YOLO "
                 "sul dataset di blueprint industriali. "
                 "In grassetto il valore migliore per ogni metrica.}")
    lines.append("  \\label{tab:confronto_modelli}")
    lines.append("  \\begin{tabular}{lcccccc}")
    lines.append("    \\toprule")
    lines.append("    " + " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\")
    lines.append("    \\midrule")

    for i, model in enumerate(models):
        row = [model]
        for k in metrics_keys:
            row.append(col_bold[k][i])
        lines.append("    " + " & ".join(row) + " \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def table_dettaglio_per_classe(results: dict) -> str:
    """
    Tabella 2: dettaglio per classe — una riga per (modello, classe).
    Colonne: Modello | Classe | Precision | Recall | AP@50 | AP@50-95
    """
    lines = []
    lines.append("% ─────────────────────────────────────────────────────")
    lines.append("% Tabella 2: Metriche per classe — SINTESI: Genesi")
    lines.append("% ─────────────────────────────────────────────────────")
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append("  \\caption{Metriche di detection per classe e modello. "
                 "In grassetto il valore migliore per ogni classe e metrica.}")
    lines.append("  \\label{tab:dettaglio_classi}")
    lines.append("  \\begin{tabular}{llcccc}")
    lines.append("    \\toprule")
    lines.append("    \\textbf{Modello} & \\textbf{Classe} & "
                 "\\textbf{Precision} & \\textbf{Recall} & "
                 "\\textbf{AP@50} & \\textbf{AP@50-95} \\\\")
    lines.append("    \\midrule")

    # Per ogni classe, trova il valore migliore tra i modelli
    for cls in CLASSES:
        p_vals    = []
        r_vals    = []
        ap50_vals = []
        ap_vals   = []
        model_list = []

        for model, data in results.items():
            pc = data.get("per_class")
            if pc and cls in pc:
                model_list.append(model)
                p_vals.append(pc[cls].get("precision"))
                r_vals.append(pc[cls].get("recall"))
                ap50_vals.append(pc[cls].get("ap50"))
                ap_vals.append(pc[cls].get("ap50_95"))
            else:
                model_list.append(model)
                p_vals.append(None)
                r_vals.append(None)
                ap50_vals.append(None)
                ap_vals.append(None)

        p_bold    = bold_best(p_vals)
        r_bold    = bold_best(r_vals)
        ap50_bold = bold_best(ap50_vals)
        ap_bold   = bold_best(ap_vals)

        for i, model in enumerate(model_list):
            model_cell = f"\\multirow{{{len(model_list)}}}{{*}}{{{model}}}" if i == 0 else ""
            lines.append(
                f"    {model_cell} & {cls} & "
                f"{p_bold[i]} & {r_bold[i]} & "
                f"{ap50_bold[i]} & {ap_bold[i]} \\\\"
            )

        lines.append("    \\midrule")

    # Rimuovi l'ultimo midrule e metti bottomrule
    lines[-1] = "    \\bottomrule"
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("SINTESI: Genesi — Generatore tabelle LaTeX")
    print("=" * 55)

    # Raccogli metriche
    print(f"\nRun trovati in {RUNS_DIR}:")
    results = collect_all_runs(Path(RUNS_DIR))

    if not results:
        print("\nNessun run trovato. Controlla RUNS_DIR.")
        exit(1)

    print(f"\nModelli raccolti: {list(results.keys())}")

    # Genera LaTeX
    header = "\n".join([
        "% ═══════════════════════════════════════════════════",
        "% SINTESI: Genesi — Tabelle per tesi",
        "% Generato automaticamente da generate_latex_tables.py",
        "% ═══════════════════════════════════════════════════",
        "",
        "% Pacchetti necessari nel preambolo:",
        "% \\usepackage{booktabs}",
        "% \\usepackage{multirow}",
        "",
    ])

    tab1 = table_confronto_modelli(results)
    tab2 = table_dettaglio_per_classe(results)

    output = header + tab1 + "\n" + tab2

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\n✓ File LaTeX salvato: {OUTPUT_FILE}")
    print("\nPer usarlo nella tesi aggiungi nel preambolo:")
    print("  \\usepackage{booktabs}")
    print("  \\usepackage{multirow}")
    print("\nPoi includi il file con:")
    print("  \\input{tabelle_tesi}")
