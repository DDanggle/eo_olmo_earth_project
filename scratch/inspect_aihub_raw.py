import os, json, glob, collections
R="/home/work/data/olmoearth/aihub/raw/71363/310.AI기반_국립공원_변화탐지_모니터링_플랫폼_구축/01-1.정식개방데이터"
def walk(p,depth=0,maxd=3):
    if depth>maxd: return
    try: items=sorted(os.listdir(p))
    except Exception as e: print("  "*depth,"ERR",e); return
    print("  "*depth, os.path.basename(p), f"({len(items)} items)", items[:6])
    for it in items[:3]:
        q=os.path.join(p,it)
        if os.path.isdir(q): walk(q,depth+1,maxd)
walk(R)
# find a label file example
lab=[f for f in glob.glob(R+"/**/*.json",recursive=True) if "META" not in f][:3]
print("--- label json examples",len(lab));
for f in lab[:1]:
    d=json.load(open(f)); print(os.path.basename(f), list(d.keys())[:10]); print(json.dumps(d,ensure_ascii=False)[:700])
tifs=glob.glob(R+"/**/*.tif",recursive=True); print("--- tif count",len(tifs), [os.path.basename(t) for t in tifs[:4]])
ta=[json.loads(l) for l in open("/home/work/data/olmoearth/aihub/splits/tile_assignment.jsonl") if l.strip()]
print("--- SB13 tiles:", [(t["tile_id"],t["split"],t["cluster"]) for t in ta if t["tile_id"].startswith("SB13")])
