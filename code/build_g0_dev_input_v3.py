"""G0 development input v3 — episodes widened beyond 'task': (task x support condition x cache contract). All numbers already measured (MS-97/99, C0-dev replay, MS-102/108). Fixed anchors as v2.
Episode types: (1) few-shot K=5 stratified vs random support (per seed, 8-region macro FP-matched IoU) with actions CACHED_HEAD / HEAD_ADAPT / RAW_FINETUNE;
(2) cache-contract episodes on Sen12 full-label: the deployed cache is Galileo-base / Clay-in256 / OlmoEarth-nano / OlmoEarth-base; actions CACHED_HEAD (train head on that cache) vs RAW_FINETUNE (raw UNet3D full-label P2). Single seed for the cache side (flagged)."""
import json, numpy as np
from collections import defaultdict
from pathlib import Path
rows=[]; L={"sen12":0.196558375,"solar":0.33252233333333336}; U={"sen12":0.27216575,"solar":0.5931661666666667}
rep=json.load(open("artifacts/c0_policy_replay/report.json"))
for task in ("sen12","task2"):
    t="sen12" if task=="sen12" else "solar"
    for sup in ("random","stratified"):
        d=rep["tasks"][task][f"{sup}_K5"]; per=defaultdict(lambda:defaultdict(list))
        for r in d["runs"]:
            for act,key in (("CACHED_HEAD","A0"),("HEAD_ADAPT","A1"),("RAW_FINETUNE","A4w")):
                if r.get(key) is not None: per[(act,r["seed"])][r["region"]].append(r[key])
        for (act,seed),regs in per.items():
            rows.append({"episode_id":f"{t}_{sup}_K5","task":t,"action":act,"seed":f"s{seed}","score":float(np.mean([np.mean(v) for v in regs.values()])),"higher_is_better":True,"gpu_seconds":{"CACHED_HEAD":0,"HEAD_ADAPT":40,"RAW_FINETUNE":200}[act],"raw_bytes":{"CACHED_HEAD":0,"HEAD_ADAPT":0,"RAW_FINETUNE":11e9}[act],"cache_bytes":0,"support_label_count":0 if act=="CACHED_HEAD" else 40,"lower_anchor":L[t],"upper_anchor":U[t],"support_condition":sup})
bv=json.load(open("artifacts/bv1_diagnostics_summary.json")); ref={x["fold"]:x["primary_mean"] for x in json.load(open("artifacts/confirmatory_8region_summary.json"))["regions"]}
fam={"galileo_base":"galileo_cache","clay_in256":"clay_cache_in256","olmo_nano":"olmo_nano","olmo_tiny":"olmo_tiny"}
for name,c in fam.items():
    v=bv["verdict"].get(c)
    if not v: continue
    rows.append({"episode_id":f"sen12_full_cache={name}","task":"sen12","action":"CACHED_HEAD","seed":"s1","score":v["macro"],"higher_is_better":True,"gpu_seconds":600,"raw_bytes":0,"cache_bytes":0,"support_label_count":5971,"lower_anchor":L["sen12"],"upper_anchor":U["sen12"],"cache_contract":name,"single_seed":True})
    rows.append({"episode_id":f"sen12_full_cache={name}","task":"sen12","action":"RAW_FINETUNE","seed":"s1","score":float(np.mean([ref[f]["raw_strong"] for f in ref])),"higher_is_better":True,"gpu_seconds":3000,"raw_bytes":1.1e10,"cache_bytes":0,"support_label_count":5971,"lower_anchor":L["sen12"],"upper_anchor":U["sen12"],"cache_contract":name,"single_seed":True})
rows.append({"episode_id":"sen12_full_cache=olmo_base","task":"sen12","action":"CACHED_HEAD","seed":"s1","score":U["sen12"],"higher_is_better":True,"gpu_seconds":600,"raw_bytes":0,"cache_bytes":0,"support_label_count":5971,"lower_anchor":L["sen12"],"upper_anchor":U["sen12"],"cache_contract":"olmo_base"})
rows.append({"episode_id":"sen12_full_cache=olmo_base","task":"sen12","action":"RAW_FINETUNE","seed":"s1","score":L["sen12"],"higher_is_better":True,"gpu_seconds":3000,"raw_bytes":1.1e10,"cache_bytes":0,"support_label_count":5971,"lower_anchor":L["sen12"],"upper_anchor":U["sen12"],"cache_contract":"olmo_base"})
out={"_note":"dev input v3: episodes = task x support condition (K=5 stratified/random, per seed) + cache-contract episodes (full label, cache family/scale; single seed on the cache side, raw side is the sealed 3-seed mean). Costs for few-shot rows are rounded from measured v2 values; cache-contract costs are estimates. Anchors fixed as v2.","rows":rows,"budgets":[{"name":"unlimited"},{"name":"no_raw_read","max_raw_bytes":0}]}
Path("artifacts/g0_dev/g0_dev_input_v3.json").write_text(json.dumps(out,indent=1)); print("rows",len(rows))
for r in rows: print(r["episode_id"],r["action"],r["seed"],round(r["score"],4))
