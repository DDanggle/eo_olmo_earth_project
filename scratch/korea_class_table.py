import zipfile, json, collections
Z="/home/work/data/olmoearth/aihub/raw/71363/310.AI기반_국립공원_변화탐지_모니터링_플랫폼_구축/01-1.정식개방데이터/Training/02.라벨링데이터/TL_02.JSON_03._Sentinel2.zip"
zf=zipfile.ZipFile(Z); names=[n for n in zf.namelist() if n.endswith(".json")]; print("json files",len(names))
d=json.loads(zf.read(names[0])); print(json.dumps(d,ensure_ascii=False)[:1200])
cnt=collections.Counter(); cls_keys=set()
for n in names[:400]:
    d=json.loads(zf.read(n)); feats=d.get("features") or d.get("annotations") or d
    if isinstance(feats,dict): feats=feats.get("features",[])
    for f in feats:
        p=f.get("properties",f)
        for k in p:
            if any(s in k.lower() for s in ("class","label","code","type","name","ann")): cls_keys.add(k); cnt[(k,str(p[k]))]+=1
print("keys",cls_keys); print(sorted(cnt.items(),key=lambda x:-x[1])[:40])
