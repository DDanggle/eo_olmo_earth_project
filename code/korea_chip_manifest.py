#!/usr/bin/env python3
"""Freeze the Korea (AI-Hub 71363) chip grid and cache manifest BEFORE any label is opened (korea_shared_cache_3task_prereg_v1_amendment readiness gate).
Input : aihub/s2_12band_v2/manifest.jsonl (materialized cubes, status), aihub/splits/tile_assignment.jsonl (M10 spatial holdout: split, cluster)
Output: aihub/korea_chip_manifest.jsonl  one row per chip: chip_id, tile_id, row, col, split, cluster, dates (sorted), n_dates, keys
        aihub/korea_chip_manifest_summary.json with sha256 of the jsonl and counts
Grid  : 1024 px tile -> 8x8 chips of 128 px (same geometry as Sen12/Solar; 32x32 tokens at patch 4). No label file is read here."""
import json, hashlib, collections
from pathlib import Path
R=Path("/home/work/data/olmoearth/aihub")
man=[json.loads(l) for l in (R/"s2_12band_v2/manifest.jsonl").read_text().splitlines() if l.strip()]
ok=[m for m in man if m.get("status")=="coverage_valid"]   # v2 manifest status value (2,541 rows, all coverage_valid on 2026-09-10)
assign={t["tile_id"]:t for t in (json.loads(l) for l in (R/"splits/tile_assignment.jsonl").read_text().splitlines() if l.strip())}
by_tile=collections.defaultdict(list)
for m in ok: by_tile[m["key"].split("_")[0]].append(m)
rows=[]; skipped=collections.Counter()
for tile,ms in sorted(by_tile.items()):
    a=assign.get(tile)
    if a is None or a["split"]=="excluded": skipped["no_assignment_or_excluded"]+=1; continue
    ms=sorted(ms,key=lambda m:m["date"]); dates=[m["date"] for m in ms]
    for r in range(8):
        for c in range(8):
            rows.append({"chip_id":f"{tile}_r{r}c{c}","tile_id":tile,"row":r,"col":c,"y0":r*128,"x0":c*128,"split":a["split"],"cluster":a["cluster"],"dates":dates,"n_dates":len(dates),"keys":[m["key"] for m in ms]})
out=R/"korea_chip_manifest.jsonl"; out.write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in rows)+"\n")
sha=hashlib.sha256(out.read_bytes()).hexdigest()
summ={"n_chips":len(rows),"n_tiles":len({x["tile_id"] for x in rows}),"n_cubes_used":sum(len(v) for t,v in by_tile.items() if t in assign and assign[t]["split"]!="excluded"),"by_split":dict(collections.Counter(x["split"] for x in rows)),"by_cluster":dict(collections.Counter(x["cluster"] for x in rows)),"dates_per_tile":dict(collections.Counter(len(v) for t,v in by_tile.items())),"skipped":dict(skipped),"manifest_sha256":sha,"grid":"8x8 chips of 128 px per 1024 px tile","labels_read":False}
(R/"korea_chip_manifest_summary.json").write_text(json.dumps(summ,ensure_ascii=False,indent=1)); print(json.dumps(summ,ensure_ascii=False))
