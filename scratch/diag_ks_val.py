import numpy as np, json
from pathlib import Path
C=Path("/home/work/data/olmoearth/kurosiwo_s1_cache"); meta=[json.loads(l) for l in open(C/"meta.jsonl") if l.strip()]
bad={}
for m in meta:
    if m["split"]=="train": continue
    for k in ("stale2","single","teacher3"):
        a=np.load(C/f"{k}_fp16"/f"{m['id']}.npy")
        if not np.isfinite(a.astype("float32")).all(): bad.setdefault(m["split"],[]).append((m["id"],k,int((~np.isfinite(a.astype("float32"))).sum())))
print({k:len(v) for k,v in bad.items()}); print(list(bad.items())[:2])
json.dump(bad,open("/home/work/data/olmoearth/artifacts/kurosiwo_nonfinite_tiles.json","w"))
