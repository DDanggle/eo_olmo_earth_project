#!/usr/bin/env python3
"""임의 fold의 test 타일 중심좌표를 추출한다. 확증 지역 CI에 필요."""
import json, pathlib, sys
import xarray as xr
FOLD = sys.argv[1]
SRC = pathlib.Path(sys.argv[2])          # per-sample jsonl
ROOT = pathlib.Path("/home/work/data/sen12landslides/extracted")
OUT = pathlib.Path(f"/home/work/data/olmoearth/gp_official_bundle/tile_coords_{FOLD}.json")
sids = sorted({json.loads(l)["sample_id"] for l in SRC.read_text().splitlines() if l})
out = {}
for s in sids:
    with xr.open_dataset(ROOT / f"{s}.nc", decode_times=False, cache=False) as ds:
        out[s] = [float(ds["x"].values.mean()), float(ds["y"].values.mean())]
OUT.write_text(json.dumps(out), encoding="utf-8")
print(len(out), "tiles →", OUT)
