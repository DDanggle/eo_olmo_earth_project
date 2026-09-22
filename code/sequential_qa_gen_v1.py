#!/usr/bin/env python3
"""Sequential evidence QA v1 (config/sequential_evidence_qa_prereg_v1.json): per-arrival-step labels derived only from observations seen so far."""
import json, hashlib, sys, numpy as np
from pathlib import Path
from collections import Counter
ROOT=Path("/home/work/data/olmoearth"); OUT=ROOT/"sequential_qa_v1"; REGIONS=sys.argv[1].split(",") if len(sys.argv)>1 else ["hiroshima","thrissur","itogon","hokkaido"]
def quadrants(m):
    h,w=m.shape; tot=m.sum(); out=[]
    for name,sl in (("NW",(slice(0,h//2),slice(0,w//2))),("NE",(slice(0,h//2),slice(w//2,w))),("SW",(slice(h//2,h),slice(0,w//2))),("SE",(slice(h//2,h),slice(w//2,w)))):
        if tot and m[sl].sum()/tot>=0.25: out.append(name)
    return out
blk=lambda s:int(hashlib.sha256(s.encode()).hexdigest(),16)%5
allstats={}
for reg in REGIONS:
    D=ROOT/"arrival_v0"/reg; MD=ROOT/f"sen12_pilot/holdout_{reg}/mask_u8"; (OUT/reg).mkdir(parents=True,exist_ok=True)
    ids=sorted(p.stem for p in (D/"meta").glob("*.json") if (MD/f"{p.stem}.npy").exists())
    items=[]; cnt=Counter(); per_step=Counter()
    for s in ids:
        m=json.loads((D/"meta"/f"{s}.json").read_text()); mask=np.load(MD/f"{s}.npy")>0; positive=bool(mask.any()); post=m["post_index"]; clear=[float(v) for v in m["clear"]]; dates=[d[:10] for d in m["dates"]]
        split=("train" if blk(s) in (0,1,2) else "val" if blk(s)==3 else "test") if reg=="hiroshima" else "test_external"
        prev="no_change"; conf_step=None
        for i in range(len(dates)):
            if not positive or post is None or i<post: state="no_change"
            elif conf_step is not None: state="keep"
            elif clear[i]>=0.5: state="confirmed"; conf_step=i
            else: state="abstain"
            upd={("no_change","no_change"):"keep",("no_change","abstain"):"abstain",("abstain","abstain"):"abstain",("abstain","confirmed"):"strengthen",("no_change","confirmed"):"strengthen",("confirmed","keep"):"keep",("keep","keep"):"keep"}[(prev,state)]
            ans_state={"no_change":"no","abstain":"cannot tell yet","confirmed":"yes","keep":"yes"}[state]
            items.append({"id":f"{s}_t{i}","tile":s,"region":reg,"split":split,"step":i,"date":dates[i],"clear":clear[i],"seen_dates":dates[:i+1],"label_state":state,"answer_state":ans_state,"evidence_step":conf_step,"evidence_date":dates[conf_step] if conf_step is not None else None,"evidence_quadrants":quadrants(mask) if conf_step is not None else None,"update_action":upd,"positive":positive})
            cnt[state]+=1; per_step[(i,state)]+=1; prev=state
    (OUT/reg/"items.jsonl").write_text("\n".join(json.dumps(x) for x in items)+"\n")
    st={"n_tiles":len(ids),"n_positive":sum(1 for s in ids if (np.load(MD/f"{s}.npy")>0).any()),"n_items":len(items),"label_counts":dict(cnt),"update_counts":dict(Counter(x["update_action"] for x in items)),"abstain_tiles":sum(1 for s in ids if any(x["tile"]==s and x["label_state"]=="abstain" for x in items)),"confirm_step_hist":dict(Counter(x["evidence_step"] for x in items if x["label_state"]=="confirmed"))}
    (OUT/reg/"stats.json").write_text(json.dumps(st,indent=1)); allstats[reg]=st; print(reg,json.dumps({k:st[k] for k in ("n_tiles","n_positive","n_items","label_counts","update_counts","abstain_tiles")}),flush=True)
(OUT/"stats_all.json").write_text(json.dumps(allstats,indent=1)); print("SEQ QA V1 DONE")
