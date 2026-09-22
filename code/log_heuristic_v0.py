#!/usr/bin/env python3
"""LOG_heuristic control (vlm_memory_comparison_prereg_v0 amendment_3): model-free reading of event_log_v1 with the region's own 10% validation threshold. Same subsamples and scorer as stage 1."""
import json, numpy as np
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth"); OUT=ROOT/"vlm_memory_stage1_v0"; seed=1
THR=json.loads((ROOT/"event_log_v1/thresholds.json").read_text())["regions"]
LOG={reg:{r["tile"]:r for r in (json.loads(l) for l in (ROOT/f"event_log_v1/{reg}.jsonl").read_text().splitlines() if l)} for reg in ("hiroshima","thrissur","itogon","hokkaido")}
def items(reg,splits): return [x for x in (json.loads(l) for l in (ROOT/f"sequential_qa_v1/{reg}/items.jsonl").read_text().splitlines() if l) if x["split"] in splits]
tests={"hiroshima_test":items("hiroshima",("test",)),"thrissur":items("thrissur",("test_external",)),"itogon":items("itogon",("test_external",)),"hokkaido":items("hokkaido",("test_external",))}
rng=np.random.default_rng(1000+seed)
for k,v in tests.items():
    tl=sorted({x["tile"] for x in v}); keep=set(rng.choice(tl,min(150,len(tl)),replace=False).tolist()); tests[k]=[x for x in v if x["tile"] in keep]
def gold(it,q):
    if q=="Q_state": return it["answer_state"]
    if q=="Q_when": return it["evidence_date"] if it["label_state"] in ("confirmed","keep") else "none"
    if q=="Q_where": return ", ".join(it["evidence_quadrants"]) if it["label_state"] in ("confirmed","keep") and it["evidence_quadrants"] else "none"
    return it["update_action"]
def pred(it,q):
    r=LOG[it["region"]][it["tile"]]; thr=THR[it["region"]]["own_val_thr"]["far10"]; alerts=[i for i in range(it["step"]+1) if r["area"][i]>=thr]
    if q=="Q_state": return "yes" if alerts else "no"
    if q=="Q_when": return r["dates"][alerts[0]] if alerts else "none"
    if q=="Q_where": return ", ".join(r["quadrants"][alerts[0]]) if alerts and r["quadrants"][alerts[0]] else "none"
    return "strengthen" if (alerts and alerts[0]==it["step"]) else "keep"
res={"arm":"LOG_heuristic","evals":{}}
for name,T in tests.items():
    rows=[{"tile":it["tile"],"step":it["step"],"q":q,"pred":pred(it,q),"gold":gold(it,q)} for it in T for q in ("Q_state","Q_when","Q_where","Q_update")]
    sc={q:{"n":sum(1 for r in rows if r["q"]==q),"acc":sum(1 for r in rows if r["q"]==q and r["pred"]==r["gold"])/max(1,sum(1 for r in rows if r["q"]==q))} for q in ("Q_state","Q_when","Q_where","Q_update")}
    st={(r["tile"],r["step"]):r["pred"] for r in rows if r["q"]=="Q_state"}; ka=[r for r in rows if r["q"]=="Q_update" and r["gold"] in ("keep","abstain") and r["step"]>0]
    sc["cloud_stability"]={"n":len(ka),"frac_state_unchanged":sum(1 for r in ka if st.get((r["tile"],r["step"]))==st.get((r["tile"],r["step"]-1)))/max(len(ka),1)}
    res["evals"][name]=sc; print(name,json.dumps(sc))
(OUT/"scores_LOG_heuristic_seed1.json").write_text(json.dumps(res,indent=1)); print("LOG DONE")
