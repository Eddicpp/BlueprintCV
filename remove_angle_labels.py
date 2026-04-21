"""
Elimina tutti i JSON da quote_per_labeling/ che contengono
almeno una shape con label "angle".
L'immagine corrispondente viene mantenuta.
"""

import json
from pathlib import Path

LABELED_DIR = "./quote_per_labeling"

removed = 0
kept    = 0

for json_path in sorted(Path(LABELED_DIR).glob("*.json")):
    try:
        data   = json.load(open(json_path, encoding="utf-8"))
        shapes = data.get("shapes", [])
        labels = [s.get("label","").strip().lower() for s in shapes]

        if "angle" in labels:
            json_path.unlink()
            removed += 1
        else:
            kept += 1
    except Exception:
        continue

print(f"JSON con angle eliminati: {removed}")
print(f"JSON mantenuti:           {kept}")
