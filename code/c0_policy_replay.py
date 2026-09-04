"""C0 deterministic safe-action policy — offline replay on existing few-shot reports (development screen, no GPU).
Policy input = support labels only (n positive tiles in the K support set). Actions: n_pos==0 -> A0_REUSE ; else A1_HEAD_ADAPT.
Support ids re-drawn with the runner's RNG (random: Random(100+seed+K*7).sample(pool,K); stratified: manifest support[K]).
Outputs per-(task, support, K): region-mean FP-matched IoU of always-A0 / always-A1 / always-A4w / C0 policy, wins vs each, and oracle (per-run max of A0,A1)."""
import json, random, sys
from pathlib import Path
import numpy as np
ROOT=Path("/home/work/data/olmoearth")
TASKS={"sen12":{"reports":{"random":"artifacts/fewshot_confirmatory/fu_random/report_fixed_update.json","stratified":"artifacts/fewshot_confirmatory/fu/report_fixed_update.json"},
                "mandir":"artifacts/fewshot_confirmatory_manifests","mask":"sen12_pilot/holdout_chimanimani/mask_u8"},
       "task2":{"reports":{"random":"artifacts/task2_fewshot/fu_random/report_fixed_update.json","stratified":"artifacts/task2_fewshot/fu/report_fixed_update.json"},
                "mandir":"artifacts/task2_fewshot_manifests","mask":"task2_cache/mask_u8"}}
def npos(ids,maskdir): return int(sum(np.load(maskdir/f"{s}.npy").max()>0 for s in ids))
out={"schema":"c0-policy-replay-v1","policy":"n_pos(support)==0 -> A0 else A1","status":"development_screen_offline","tasks":{}}
for task,cfg in TASKS.items():
    maskdir=ROOT/cfg["mask"]; res={}
    for sup,rp in cfg["reports"].items():
        rep=json.loads((ROOT/rp).read_text()); runs=rep["runs"]
        a0={(r["region"],r["seed"]):r["eval"]["iou_fp_matched"] for r in runs if r["arm"]=="A0"}
        byk={}
        for r in runs:
            if r["arm"]=="A0": continue
            byk.setdefault((r["region"],r["K"],r["seed"]),{})[r["arm"]]=r["eval"]["iou_fp_matched"]
        for K in sorted({k[1] for k in byk}):
            rows=[]
            for (region,k,seed),arms in byk.items():
                if k!=K: continue
                man=json.loads((ROOT/cfg["mandir"]/f"{region}_manifest.json").read_text())
                if sup=="random": sids=sorted(random.Random(100+seed+K*7).sample(man["support_pool"]["ids"],K))
                else: sids=next(x for x in man["draws"][str(K)]["draws"] if x["seed"]==seed)["support_ids"]
                n=npos(sids,maskdir); A0=a0[(region,seed)]; A1=arms.get("A1"); A4=arms.get("A4w")
                pol=A0 if n==0 else A1
                rows.append({"region":region,"seed":seed,"n_pos":n,"A0":A0,"A1":A1,"A4w":A4,"C0":pol,"oracle":max(A0,A1)})
            regs=sorted({r["region"] for r in rows}); rm=lambda key:{g:float(np.mean([r[key] for r in rows if r["region"]==g and r[key] is not None])) for g in regs}
            M={k:rm(k) for k in ["A0","A1","A4w","C0","oracle"]}
            wins=lambda a,b:sum(M[a][g]>M[b][g]+1e-9 for g in regs)
            res[f"{sup}_K{K}"]={"n_runs":len(rows),"n_pos_zero_runs":sum(r["n_pos"]==0 for r in rows),
                "macro":{k:float(np.mean(list(v.values()))) for k,v in M.items()},
                "C0_wins_vs_A1":wins("C0","A1"),"C0_ties_A1":sum(abs(M["C0"][g]-M["A1"][g])<1e-9 for g in regs),"C0_wins_vs_A0":wins("C0","A0"),"C0_ties_A0":sum(abs(M["C0"][g]-M["A0"][g])<1e-9 for g in regs),
                "C0_wins_vs_A4w":wins("C0","A4w"),"n_regions":len(regs),"per_region":{g:{k:round(M[k][g],4) for k in M} for g in regs},"runs":rows}
    out["tasks"][task]=res
o=ROOT/"artifacts/c0_policy_replay/report.json"; o.parent.mkdir(parents=True,exist_ok=True); o.write_text(json.dumps(out,indent=1))
for task,res in out["tasks"].items():
    for k,v in res.items(): print(task,k,"n_pos0=%d/%d"%(v["n_pos_zero_runs"],v["n_runs"]),{a:round(b,4) for a,b in v["macro"].items()},"C0>A1 %d/%d (tie %d)"%(v["C0_wins_vs_A1"],v["n_regions"],v["C0_ties_A1"]),"C0>A0 %d (tie %d)"%(v["C0_wins_vs_A0"],v["C0_ties_A0"]),"C0>A4w %d"%v["C0_wins_vs_A4w"])
