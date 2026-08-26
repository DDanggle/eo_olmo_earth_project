#!/usr/bin/env python3
"""test 타일 중심좌표(EPSG:32736)를 한 번 추출해 JSON으로 봉인한다. 로컬 분석용."""
import json, pathlib
import xarray as xr
import sys
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
B = pathlib.Path(f"/home/work/data/olmoearth/gp_official_bundle/per_sample/P2_{SPLIT}.jsonl")
ROOT = pathlib.Path("/home/work/data/sen12landslides/extracted")
sids = sorted({json.loads(l)["sample_id"] for l in B.read_text().splitlines() if l})
out = {}
for s in sids:
    with xr.open_dataset(ROOT / f"{s}.nc", decode_times=False, cache=False) as ds:
        out[s] = [float(ds["x"].values.mean()), float(ds["y"].values.mean())]
p = pathlib.Path(f"/home/work/data/olmoearth/gp_official_bundle/tile_coords_{SPLIT}.json")
p.write_text(json.dumps(out), encoding="utf-8")
print(len(out), "tiles")
