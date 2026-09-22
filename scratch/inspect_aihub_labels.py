import json, os, glob, collections
R="/home/work/data/olmoearth/aihub"
print("--- extract/meta:", os.listdir(f"{R}/extract/meta")[:10])
lt=f"{R}/extract/label_tif_train"; fs=sorted(glob.glob(f"{lt}/**/*",recursive=True)); print("label_tif_train files",len(fs)); print([os.path.relpath(f,lt) for f in fs[:6]])
ex=json.load(open(f"{R}/s2_12band_v2/exclusion_bias.json")); print("--- exclusion_bias keys",list(ex.keys())[:12]); print(json.dumps(ex,ensure_ascii=False)[:900])
ms=json.load(open(f"{R}/s2_12band_v2/materialize_summary.json")); print("--- materialize_summary",json.dumps(ms,ensure_ascii=False)[:600])
inv=[json.loads(l) for l in open(f"{R}/inventory/inventory.jsonl") if l.strip()]; print("--- inventory rows",len(inv),"keys",list(inv[0].keys())[:25]); print(json.dumps(inv[0],ensure_ascii=False)[:600])
ta=[json.loads(l) for l in open(f"{R}/splits/tile_assignment.jsonl") if l.strip()]; print("--- tile_assignment",len(ta),list(ta[0].keys())); print(collections.Counter(t.get("split") or t.get("role") for t in ta))
raw=f"{R}/raw/71363"; print("--- raw/71363:", os.listdir(raw)[:10])
for d in os.listdir(raw)[:3]:
    p=os.path.join(raw,d)
    if os.path.isdir(p): print(d, os.listdir(p)[:6])
