from pathlib import Path
labels = Path("../Combined_Dataset/labels")
counts = {}
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