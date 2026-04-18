"""
SINTESI: Genesi — Merge leggero dataset
NON copia i file — crea solo file .txt con i percorsi delle immagini.
YOLO supporta nativamente i dataset definiti tramite file di testo con path.
"""

from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────

DATASETS = [
    "./dataset_yolo_aug",
    "./dataset_blueprint_strutturato", 
]

OUTPUT_DIR = "./dataset_finale"
CLASSES    = ["border", "table", "quote"]

# ─────────────────────────────────────────
# MERGE LEGGERO
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("SINTESI: Genesi — Merge leggero (solo path)")
    print("=" * 55)

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}

    for split in ["train", "val", "test"]:
        all_paths = []

        for dataset in DATASETS:
            src_img = Path(dataset) / split / "images"
            if not src_img.exists():
                print(f"  ✗ {dataset}/{split} non trovato, skip")
                continue
            imgs = sorted(list(src_img.glob("*.png")) +
                          list(src_img.glob("*.jpg")))
            all_paths.extend([str(p.resolve()) for p in imgs])
            counts[split] += len(imgs)

        # Scrivi il file txt con tutti i path
        txt_path = Path(OUTPUT_DIR) / f"{split}.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(all_paths))

        print(f"  {split:5s}: {counts[split]} immagini → {txt_path}")

    # data.yaml — punta ai .txt invece che alle cartelle
    yaml = f"""path: {Path(OUTPUT_DIR).resolve()}
train: train.txt
val:   val.txt
test:  test.txt

nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path = Path(OUTPUT_DIR) / "data.yaml"
    yaml_path.write_text(yaml)

    print(f"\n✓ Completato in pochi secondi — nessun file copiato")
    print(f"  Totale: {sum(counts.values())} immagini")
    print(f"  data.yaml: {yaml_path}")
    print(f"\nPer allenare:")
    print(f'  DATASET_YAML = "./{OUTPUT_DIR}/data.yaml"')