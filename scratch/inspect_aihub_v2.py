import numpy as np, glob, json, os, collections
R="/home/work/data/olmoearth/aihub"
fs=sorted(glob.glob(f"{R}/s2_12band_v2/arrays/*.npy")); print("arrays",len(fs))
a=np.load(fs[0],mmap_mode="r"); print("shape",a.shape,a.dtype)
tiles=collections.Counter(os.path.basename(f).split("_")[0] for f in fs); print("tiles",len(tiles),"dates/tile min/med/max",min(tiles.values()),sorted(tiles.values())[len(tiles)//2],max(tiles.values()))
man=[json.loads(l) for l in open(f"{R}/s2_12band_v2/manifest.jsonl") if l.strip()]; print("manifest rows",len(man)); print("manifest keys",list(man[0].keys())); print(json.dumps(man[0],ensure_ascii=False)[:500])
au=json.load(open(f"{R}/s2_12band_v2/audit_full.json")); print("audit keys",list(au.keys())[:20])
for k in ("gate_pass","n_files","n_valid","status","summary"):
    if k in au: print(k, str(au[k])[:300])
print("--- label sources");
for d in ("raw","inventory","contract","splits","extract"):
    p=f"{R}/{d}";
    if os.path.isdir(p): print(d, os.listdir(p)[:8])
print("--- excluded", sum(1 for _ in open(f"{R}/s2_12band_v2/excluded.jsonl")))
