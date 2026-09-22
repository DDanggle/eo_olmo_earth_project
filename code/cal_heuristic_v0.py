#!/usr/bin/env python3
"""CAL_heuristic control (vlm_memory_comparison_prereg_v0 amendment_2): text-only calendar rule memorising the hiroshima event date. Same 150-tile subsamples and scorer as stage 1."""
import json, sys, numpy as np
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth"); OUT=ROOT/"vlm_memory_stage1_v0"; EV="2018-06-28"; seed=1
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
def pred(it,q,prev_yes):
    post=[d for d in it["seen_dates"] if d>=EV]; yes=bool(post)
    if q=="Q_state": return "yes" if yes else "no"
    if q=="Q_when": return post[0] if yes else "none"
    if q=="Q_where": return "none"
    return "strengthen" if (yes and not prev_yes) else "keep"
res={"arm":"CAL_heuristic","evals":{}}
for name,T in tests.items():
    rows=[]; state={}
    for it in sorted(T,key=lambda x:(x["tile"],x["step"])):
        prev_yes=state.get(it["tile"],False)
        for q in ("Q_state","Q_when","Q_where","Q_update"): rows.append({"tile":it["tile"],"step":it["step"],"q":q,"pred":pred(it,q,prev_yes),"gold":gold(it,q),"label_state":it["label_state"]})
        state[it["tile"]]=any(d>=EV for d in it["seen_dates"])
    sc={q:{"n":sum(1 for r in rows if r["q"]==q),"acc":sum(1 for r in rows if r["q"]==q and r["pred"]==r["gold"])/max(1,sum(1 for r in rows if r["q"]==q))} for q in ("Q_state","Q_when","Q_where","Q_update")}
    st={(r["tile"],r["step"]):r["pred"] for r in rows if r["q"]=="Q_state"}; ka=[r for r in rows if r["q"]=="Q_update" and r["gold"] in ("keep","abstain") and r["step"]>0]
    sc["cloud_stability"]={"n":len(ka),"frac_state_unchanged":sum(1 for r in ka if st.get((r["tile"],r["step"]))==st.get((r["tile"],r["step"]-1)))/max(len(ka),1)}
    res["evals"][name]=sc; print(name,json.dumps(sc))
(OUT/"scores_CAL_heuristic_seed1.json").write_text(json.dumps(res,indent=1)); print("CAL DONE")
