from pathlib import Path

lines = open("dataset_finale/train.txt").readlines()[:20]
for l in lines:
    p = Path(l.strip())
    print("OK" if p.exists() else "MANCANTE", p)