"""Write 4 disjoint key files (sorted inventory keys minus keys already in the main v2 manifest/excluded) for parallel materialization into shard dirs."""
import json; from pathlib import Path
ROOT=Path("/home/work/data/olmoearth/aihub"); OUT=ROOT/"s2_12band_v2"
keys=sorted(json.loads(l)["key"] for l in open(ROOT/"inventory/inventory.jsonl") if l.strip())
done=set()
for f in ("manifest.jsonl","excluded.jsonl"):
    if (OUT/f).exists(): done|={json.loads(l)["key"] for l in open(OUT/f) if l.strip()}
todo=[k for k in keys if k not in done]; n=4; sz=(len(todo)+n-1)//n
for i in range(n): (OUT/f"shard{i}_keys.txt").write_text("\n".join(todo[i*sz:(i+1)*sz])+"\n")
print("todo",len(todo),"per shard",sz,"done",len(done))
