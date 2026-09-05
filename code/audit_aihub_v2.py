"""AIHUB_CUBE_V2_CONTRACT audit of a materialized set (pilot or full): per key platform match, 12 assets, shape/dtype, band coverage, common coverage>=0.999, plus exclusion reasons. Prints gate verdict."""
import json, sys, numpy as np
from pathlib import Path
OUT=Path("/home/work/data/olmoearth/aihub/s2_12band_v2"); keys=set(Path(sys.argv[1]).read_text().split()) if len(sys.argv)>1 else None
inv={json.loads(l)["key"]:json.loads(l) for l in open("/home/work/data/olmoearth/aihub/inventory/inventory.jsonl") if l.strip()}
man=[json.loads(l) for l in open(OUT/"manifest.jsonl") if l.strip()]; exc=[json.loads(l) for l in open(OUT/"excluded.jsonl") if l.strip()]
if keys: man=[m for m in man if m["key"] in keys]; exc=[e for e in exc if e["key"] in keys]
res={"n_manifest":len(man),"n_excluded":len(exc),"excluded_reasons":{}}
for e in exc: res["excluded_reasons"][e["reason"]]=res["excluded_reasons"].get(e["reason"],0)+1
fails=[]; covs=[]
for m in man:
    k=m["key"]; a=np.load(OUT/"arrays"/f"{k}.npy",mmap_mode="r"); ok=True; why=[]
    if len(m["bands"])!=12: ok=False; why.append("bands!=12")
    if tuple(a.shape)!=(12,1024,1024) or str(a.dtype)!="uint16": ok=False; why.append(f"shape/dtype {a.shape} {a.dtype}")
    cc=m.get("common_coverage", min(m["band_coverage"].values())); covs.append(cc)
    if cc<0.999: ok=False; why.append(f"coverage {cc:.4f}")
    plat=inv.get(k,{}).get("platform"); cands=m.get("candidate_ids",[])
    if plat and cands and not all(c.startswith(plat.replace("S2","S2")) for c in cands): ok=False; why.append(f"platform {plat} vs {cands[0][:3]}")
    if not ok: fails.append({"key":k,"why":why})
res.update({"n_fail":len(fails),"fails":fails[:20],"coverage_min":min(covs) if covs else None,"coverage_p05":float(np.percentile(covs,5)) if covs else None})
res["gate_pass"]=(len(fails)==0 and (keys is None or len(man)+len(exc)==len(keys)))
res["note"]="excluded keys are deterministic exclusions (no_stac_item / coverage<0.999 / cloud>60), not failures; they count toward selection-bias reporting (gate 6)"
name="pilot40" if keys else "full"; (OUT/f"audit_{name}.json").write_text(json.dumps(res,indent=1)); print(json.dumps({k:v for k,v in res.items() if k!="fails"},indent=1)); print("FAILS",fails[:5])
