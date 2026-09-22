#!/usr/bin/env python3
"""VLM memory comparison stage 1 (config/vlm_memory_comparison_prereg_v0.json): arms A (latest-only) and C (latest + rule-based event log text). Projector-only, Olmo-3-7B-Instruct frozen. Sequential-evidence QA v1 items."""
import argparse, json, re, time, sys, math, hashlib
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
ap=argparse.ArgumentParser(); ap.add_argument("--arm",required=True,choices=["A_latest_only","C_rule_log"]); ap.add_argument("--out",required=True); ap.add_argument("--epochs",type=int,default=2); ap.add_argument("--seed",type=int,default=1); ap.add_argument("--probe",action="store_true"); ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--max-log-tokens",type=int,default=256); ap.add_argument("--eval-tiles",type=int,default=150,help="seeded uniform subsample of tiles per eval set (all 15 steps kept); 0 = all tiles"); a=ap.parse_args()
ROOT=Path("/home/work/data/olmoearth"); OUT=ROOT/a.out; OUT.mkdir(parents=True,exist_ok=True); dev=torch.device("cuda"); torch.manual_seed(a.seed); np.random.seed(a.seed)
sys.path.insert(0,str(ROOT/"code")); from cache_decoder_train_lib import emb_stats_from_cache
mean,sd=emb_stats_from_cache(ROOT/"sen12_pilot/holdout_hiroshima","holdout_hiroshima"); mean,sd=mean.to(dev),sd.to(dev)
tok=AutoTokenizer.from_pretrained(str(ROOT/"olmo_llm/Olmo-3-7B-Instruct")); llm=AutoModelForCausalLM.from_pretrained(str(ROOT/"olmo_llm/Olmo-3-7B-Instruct"),dtype=torch.bfloat16).to(dev).eval()
for p in llm.parameters(): p.requires_grad_(False)
EMB=llm.get_input_embeddings(); H=EMB.weight.shape[1]; EMB_RMS=float(EMB.weight.detach().float().pow(2).mean().sqrt())
class Projector(nn.Module):
    def __init__(s):
        super().__init__(); s.mlp=nn.Sequential(nn.Linear(768,2048),nn.GELU(),nn.Linear(2048,H)); s.ttype=nn.Embedding(2,H); s.norm=nn.LayerNorm(768); s.out=nn.LayerNorm(H); s.gain=nn.Parameter(torch.tensor(1.0)); nn.init.normal_(s.ttype.weight,std=0.02)
    def forward(s,tokens,types): return s.out(s.mlp(s.norm(tokens)))*(EMB_RMS*s.gain)+s.ttype(types)*EMB_RMS
proj=Projector().to(dev); opt=torch.optim.AdamW(proj.parameters(),a.lr,weight_decay=0.01)
THR=json.loads((ROOT/"event_log_v1/thresholds.json").read_text())["regions"]
LOG={}
for reg in ("hiroshima","thrissur","itogon","hokkaido"):
    LOG[reg]={r["tile"]:r for r in (json.loads(l) for l in (ROOT/f"event_log_v1/{reg}.jsonl").read_text().splitlines() if l)}
def items(reg,splits):
    return [x for x in (json.loads(l) for l in (ROOT/f"sequential_qa_v1/{reg}/items.jsonl").read_text().splitlines() if l) if x["split"] in splits]
QTEXT={"Q_state":"As of the latest observation, has a landslide occurred in this tile? Answer exactly one of: no / cannot tell yet / yes.","Q_when":"If a landslide has been confirmed, which observation date is the earliest evidence? Answer with the date (YYYY-MM-DD) or 'none'.","Q_where":"If confirmed, which quadrant(s) contain it? Answer with one or more of NW, NE, SW, SE, or 'none'.","Q_update":"Compared with the previous observation, should the assessment be kept, strengthened, revised, or abstained? Answer exactly one of: keep / strengthen / revise / abstain."}
def gold(it,q):
    if q=="Q_state": return it["answer_state"]
    if q=="Q_when": return it["evidence_date"] if it["label_state"] in ("confirmed","keep") else "none"
    if q=="Q_where": return ", ".join(it["evidence_quadrants"]) if it["label_state"] in ("confirmed","keep") and it["evidence_quadrants"] else "none"
    return it["update_action"]
def log_text(it):
    r=LOG[it["region"]][it["tile"]]; thr=THR[it["region"]]["own_val_thr"]["far10"]; lines=[]
    for i in range(it["step"]+1):
        flag="ALERT" if r["area"][i]>=thr else "quiet"; q=",".join(r["quadrants"][i]) or "-"; lines.append(f"{r['dates'][i]} clear={r['clear'][i]:.2f} {flag} area={r['area'][i]} quad={q}")
    text="Event log (arrival order, detector alerts):\n"+"\n".join(lines)
    while len(tok(text,add_special_tokens=False).input_ids)>a.max_log_tokens and len(lines)>1: lines=lines[1:]; text="Event log (arrival order, most recent entries):\n"+"\n".join(lines)
    return text
def eo_tokens(it):
    S=np.load(ROOT/f"arrival_v0/{it['region']}/single_fp16/{it['tile']}.npy",mmap_mode="r")[it["step"]].astype("float32"); T=torch.from_numpy(S)[None]; T=F.avg_pool2d(T,4).flatten(2).permute(0,2,1)[0]
    return T.to(dev),torch.zeros(64,dtype=torch.long,device=dev)
def build(it,q,answer=None):
    user=f"Sentinel-2 observations of one tile have arrived in order on {', '.join(it['seen_dates'])}. The latest observation's representation: <EO> "+(log_text(it)+"\n" if a.arm=="C_rule_log" else "")+QTEXT[q]
    text=tok.apply_chat_template([{"role":"user","content":user}],tokenize=False,add_generation_prompt=True); pre,post=text.split("<EO>")
    ids_pre=tok(pre,add_special_tokens=False,return_tensors="pt").input_ids[0].to(dev); ids_post=tok(post,add_special_tokens=False,return_tensors="pt").input_ids[0].to(dev)
    t,ty=eo_tokens(it); e=proj(t,ty).to(torch.bfloat16); embs=[EMB(ids_pre),e,EMB(ids_post)]; labels=[torch.full((len(ids_pre),),-100,device=dev),torch.full((64,),-100,device=dev),torch.full((len(ids_post),),-100,device=dev)]
    if answer is not None:
        ids_ans=tok(answer+tok.eos_token,add_special_tokens=False,return_tensors="pt").input_ids[0].to(dev); embs.append(EMB(ids_ans)); labels.append(ids_ans)
    return torch.cat(embs),torch.cat(labels)
train=items("hiroshima",("train",)); tests={"hiroshima_test":items("hiroshima",("test",)),"thrissur":items("thrissur",("test_external",)),"itogon":items("itogon",("test_external",)),"hokkaido":items("hokkaido",("test_external",))}
if a.eval_tiles:
    rng=np.random.default_rng(1000+a.seed)
    for k,v in tests.items():
        tl=sorted({x["tile"] for x in v}); keep=set(rng.choice(tl,min(a.eval_tiles,len(tl)),replace=False).tolist()); tests[k]=[x for x in v if x["tile"] in keep]
if a.probe: train=train[:48]; tests={k:v[:16] for k,v in tests.items()}; a.epochs=1
QS=["Q_state","Q_when","Q_where","Q_update"]; print("arm",a.arm,"train items",len(train),{k:len(v) for k,v in tests.items()},flush=True)
t0=time.perf_counter(); step=0
for ep in range(a.epochs):
    order=np.random.permutation(len(train)); tot=0; n=0
    for i in range(0,len(order),8):
        batch=[train[j] for j in order[i:i+8]]; seqs=[]
        for it in batch:
            q=QS[np.random.randint(4)]; seqs.append(build(it,q,gold(it,q)))
        L=max(s[0].shape[0] for s in seqs); E=torch.zeros(len(seqs),L,H,dtype=torch.bfloat16,device=dev); Y=torch.full((len(seqs),L),-100,device=dev); M=torch.zeros(len(seqs),L,dtype=torch.long,device=dev)
        for k,(e,y) in enumerate(seqs): E[k,:len(e)]=e; Y[k,:len(y)]=y; M[k,:len(e)]=1
        out=llm(inputs_embeds=E,attention_mask=M,labels=Y); opt.zero_grad(); out.loss.backward(); opt.step(); tot+=float(out.loss); n+=1; step+=1
        if step%50==0: print(f"ep {ep} step {step} loss {out.loss.item():.4f} {time.perf_counter()-t0:.0f}s",flush=True)
    print(f"epoch {ep} mean loss {tot/max(n,1):.4f}",flush=True)
torch.save(proj.state_dict(),OUT/f"projector_{a.arm}_seed{a.seed}.pt")
def parse(q,ans):
    s=ans.strip().lower()
    if q=="Q_state": m=re.search(r"cannot tell yet|\byes\b|\bno\b",s); return m.group(0) if m else None
    if q=="Q_when": m=re.search(r"\d{4}-\d{2}-\d{2}|none",s); return m.group(0) if m else None
    if q=="Q_where": qs=sorted(set(re.findall(r"\b(nw|ne|sw|se)\b",s))); return ", ".join(x.upper() for x in qs) if qs else ("none" if "none" in s else None)
    m=re.search(r"keep|strengthen|revise|abstain",s); return m.group(0) if m else None
proj.eval(); res={"arm":a.arm,"n_train":len(train),"train_s":time.perf_counter()-t0,"evals":{}}
with torch.no_grad():
    for name,T in tests.items():
        rows=[]
        for it in T:
            for q in QS:
                e,_=build(it,q); out=llm.generate(inputs_embeds=e[None],attention_mask=torch.ones(1,e.shape[0],dtype=torch.long,device=dev),max_new_tokens=16,do_sample=False,pad_token_id=tok.eos_token_id)
                ans=tok.decode(out[0],skip_special_tokens=True); rows.append({"id":it["id"],"tile":it["tile"],"step":it["step"],"q":q,"pred":parse(q,ans),"gold":gold(it,q),"label_state":it["label_state"],"raw":ans[:60]})
        (OUT/f"answers_{a.arm}_{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
        sc={}
        for q in QS:
            R=[r for r in rows if r["q"]==q]; sc[q]={"n":len(R),"acc":sum(1 for r in R if r["pred"]==r["gold"])/max(len(R),1),"parse_fail":sum(1 for r in R if r["pred"] is None)/max(len(R),1)}
        # cloud-stability: among keep/abstain steps, fraction where Q_state prediction equals the previous step's prediction for the same tile
        st={(r["tile"],r["step"]):r["pred"] for r in rows if r["q"]=="Q_state"}; ka=[r for r in rows if r["q"]=="Q_update" and r["gold"] in ("keep","abstain") and r["step"]>0]
        sc["cloud_stability"]={"n":len(ka),"frac_state_unchanged":sum(1 for r in ka if st.get((r["tile"],r["step"]))==st.get((r["tile"],r["step"]-1)))/max(len(ka),1)}
        res["evals"][name]=sc; print(name,json.dumps(sc),flush=True)
(OUT/f"scores_{a.arm}_seed{a.seed}.json").write_text(json.dumps(res,indent=1)); print("EARTHTALK SEQ DONE")
