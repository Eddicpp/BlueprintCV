from pathlib import Path

for split in ["train", "val", "test"]:
    img_dir = Path(f"./dataset_simboli_detection/{split}/images")
    lbl_dir = Path(f"./dataset_simboli_detection/{split}/labels")
    
    if not img_dir.exists():
        continue
        
    img_stems = set(p.stem for p in img_dir.glob("*.jpg")) | \
                set(p.stem for p in img_dir.glob("*.png"))
    
    removed = 0
    for lbl in lbl_dir.glob("*.txt"):
        if lbl.stem not in img_stems:
            lbl.unlink()
            removed += 1
    
    print(f"{split}: {removed} label orfane rimosse")