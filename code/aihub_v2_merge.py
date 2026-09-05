"""Merge shard dirs into the main v2 dir (move arrays/validity, append manifest/excluded), then leave audit to the chain."""
import json, shutil; from pathlib import Path
OUT=Path("/home/work/data/olmoearth/aihub/s2_12band_v2"); n=0
for i in range(4):
    S=OUT.parent/f"s2_12band_v2_shard{i}"
    if not S.exists(): continue
    for sub in ("arrays","validity"):
        for p in sorted((S/sub).glob("*")):
            dst=OUT/sub/p.name
            if not dst.exists(): shutil.move(str(p),str(dst)); n+=1
    for f in ("manifest.jsonl","excluded.jsonl"):
        if (S/f).exists():
            with open(OUT/f,"a") as o: o.write((S/f).read_text())
print("moved",n)
