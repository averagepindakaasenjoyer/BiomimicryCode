from pathlib import Path
labels_list = [Path("../Strawberry_Flower_Dataset/labels/train"),
          Path("../Strawberry_Flower_Dataset2/train/labels"),
          Path("../Strawberry_Flower_Dataset3/train/labels"),
          Path("../Combined_Dataset/labels/train")]
counts = {}
for labels in labels_list:
    counts = {}
    print(f"Counting labels in: {labels}")
    for p in labels.glob("*.txt"):
        for line in p.read_text(encoding="utf-8").splitlines():
            s=line.strip()
            if not s: continue
            parts=s.split()
            try:
                cid = int(float(parts[0]))
            except:
                continue
            counts[cid] = counts.get(cid, 0) + 1
    print(counts)