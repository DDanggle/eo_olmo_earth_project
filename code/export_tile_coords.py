#!/usr/bin/env python3
"""test 타일 중심좌표(EPSG:32736)를 한 번 추출해 JSON으로 봉인한다. 로컬 분석용."""
import json, pathlib
import xarray as xr
B = pathlib.Path("/home/work/data/olmoearth/gp_official_bundle/per_sample/P2_test.jsonl")
ROOT = pathlib.Path("/home/work/data/sen12landslides/extracted")
sids = sorted({json.loads(l)["sample_id"] for l in B.read_text().splitlines() if l})
out = {}
for s in sids:
    with xr.open_dataset(ROOT / f"{s}.nc", decode_times=False, cache=False) as ds:
        out[s] = [float(ds["x"].values.mean()), float(ds["y"].values.mean())]
p = pathlib.Path("/home/work/data/olmoearth/gp_official_bundle/tile_coords.json")
p.write_text(json.dumps(out), encoding="utf-8")
print(len(out), "tiles")
