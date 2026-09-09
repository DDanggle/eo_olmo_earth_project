#!/usr/bin/env python3
"""ONE-TIME label opening for Korea (AI-Hub 71363), run only after the readiness gate is recorded in MEASURED_FINDINGS.md.
1. Unzips the LABEL rasters (Training TL_01 + Validation VL_01) to aihub/labels_v1/tif/ (class-code raster per tile-date, 1024x1024).
2. Cuts them into 128-px chips aligned to aihub/korea_chip_manifest.jsonl -> aihub/labels_v1/mask_u8/<chip_id>__<date>.npy (uint8 class codes).
3. Writes labels_v1/label_inventory.json: unique codes, per-split per-class pixel counts, per-date availability — and an OPENED_AT_UTC marker.
No model output or performance metric is computed here."""
import zipfile, json, os, glob, numpy as np, collections, time, hashlib
from pathlib import Path
import rasterio
R=Path("/home/work/data/olmoearth/aihub"); RAW=R/"raw/71363/310.AI기반_국립공원_변화탐지_모니터링_플랫폼_구축/01-1.정식개방데이터"; OUT=R/"labels_v1"; (OUT/"tif").mkdir(parents=True,exist_ok=True); (OUT/"mask_u8").mkdir(exist_ok=True)
if (OUT/"OPENED_AT_UTC.txt").exists(): raise SystemExit("labels already opened: "+(OUT/"OPENED_AT_UTC.txt").read_text())
t0=time.time(); zips=[RAW/"Training/02.라벨링데이터/TL_01.LABEL_03._Sentinel2.zip", RAW/"Validation/02.라벨링데이터/VL_01.LABEL_03._Sentinel2.zip"]
for z in zips:
    if z.exists():
        with zipfile.ZipFile(z) as zf: zf.extractall(OUT/"tif")
tifs={Path(p).stem:p for p in glob.glob(str(OUT/"tif")+"/**/*.tif",recursive=True)}; print("label tifs",len(tifs))
chips=[json.loads(l) for l in (R/"korea_chip_manifest.jsonl").read_text().splitlines() if l.strip()]
codes=collections.Counter(); per_split=collections.defaultdict(collections.Counter); avail=collections.Counter(); missing=collections.Counter(); shapes=collections.Counter()
by_key={}
for c in chips:
    for k in c["keys"]:
        if k not in by_key: by_key[k]=[]
        by_key[k].append(c)
for k,cs in sorted(by_key.items()):
    p=tifs.get(k)
    if p is None: missing[cs[0]["split"]]+=len(cs); continue
    with rasterio.open(p) as src: lab=src.read(1); shapes[(lab.shape,str(lab.dtype))]+=1
    u,n=np.unique(lab,return_counts=True); codes.update(dict(zip(u.tolist(),n.tolist())))
    for c in cs:
        m=lab[c["y0"]:c["y0"]+128,c["x0"]:c["x0"]+128].astype(np.uint8 if lab.max()<256 else np.uint16)
        np.save(OUT/"mask_u8"/f"{c['chip_id']}__{k.split('_')[1]}.npy",m); avail[c["split"]]+=1
        uu,nn=np.unique(m,return_counts=True); per_split[c["split"]].update(dict(zip(uu.tolist(),nn.tolist())))
inv={"n_label_tifs":len(tifs),"raster_shapes":{str(k):v for k,v in shapes.items()},"codes_pixel_counts":{str(k):v for k,v in sorted(codes.items())},"per_split_codes":{s:{str(k):v for k,v in sorted(d.items())} for s,d in per_split.items()},"chip_dates_with_label":dict(avail),"chip_dates_missing_label":dict(missing),"note":"codes follow AI-Hub 71363 class table (e.g., 10 building, 20 river, 30 road, 40 paddy, 50 field, 60 forest, 70 logged, 100 non-target); landslide code to be confirmed from META"}
(OUT/"label_inventory.json").write_text(json.dumps(inv,ensure_ascii=False,indent=1)); (OUT/"OPENED_AT_UTC.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
print(json.dumps({k:inv[k] for k in ("n_label_tifs","raster_shapes","codes_pixel_counts","chip_dates_with_label","chip_dates_missing_label")},ensure_ascii=False)); print("LABELS OPENED",round(time.time()-t0),"s")
