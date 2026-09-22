#!/usr/bin/env python3
"""G-L1 EarthTalk projector-only alignment (config/earthtalk_projector_prereg_v0.json): OlmoEarth single-observation tokens -> Olmo-3-7B-Instruct (frozen)."""
import argparse, json, re, time, sys, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
ap=argparse.ArgumentParser(); ap.add_argument("--train",default="sentinel_qa_train_v0"); ap.add_argument("--test",default="sentinel_qa_v0_1"); ap.add_argument("--out",required=True); ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--seed",type=int,default=1); ap.add_argument("--text-only",action="store_true"); ap.add_argument("--probe",action="store_true"); ap.add_argument("--lr",type=float,default=1e-4); a=ap.parse_args()
ROOT=Path("/home/work/data/olmoearth"); SRC=ROOT/"olmo_streaming_dev/single_fp16"; OUT=ROOT/a.out; OUT.mkdir(parents=True,exist_ok=True); dev=torch.device("cuda"); torch.manual_seed(a.seed); np.random.seed(a.seed)
REC={json.loads(l)["sample_id"]:json.loads(l) for l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl") if l.strip()}
def kept_dates(tile): r=REC[tile]; q=r["scl_clear_fraction"]; k=sorted(sorted(range(15),key=lambda i:(-float(q[i]),i))[:12]); return [str(r["times"][i])[:10] for i in k]
tok=AutoTokenizer.from_pretrained(str(ROOT/"olmo_llm/Olmo-3-7B-Instruct")); llm=AutoModelForCausalLM.from_pretrained(str(ROOT/"olmo_llm/Olmo-3-7B-Instruct"),dtype=torch.bfloat16).to(dev).eval()
for p in llm.parameters(): p.requires_grad_(False)
H=llm.get_input_embeddings().weight.shape[1]; EMB=llm.get_input_embeddings()
EMB_RMS=float(EMB.weight.detach().float().pow(2).mean().sqrt())
class Projector(nn.Module):
    """Output is layer-normalised and rescaled to the LLM token-embedding RMS so untrained EO tokens do not derail the frozen LLM (probe 2026-09-19: raw-scale outputs produced word salad)."""
    def __init__(s):
        super().__init__(); s.mlp=nn.Sequential(nn.Linear(768,2048),nn.GELU(),nn.Linear(2048,H)); s.ttype=nn.Embedding(4,H); s.norm=nn.LayerNorm(768); s.out=nn.LayerNorm(H); s.gain=nn.Parameter(torch.tensor(1.0))
        nn.init.normal_(s.ttype.weight,std=0.02)
    def forward(s,tokens,types): return s.out(s.mlp(s.norm(tokens)))*(EMB_RMS*s.gain)+s.ttype(types)*EMB_RMS
proj=Projector().to(dev); opt=torch.optim.AdamW(proj.parameters(),a.lr,weight_decay=0.01)
QTEXT={"Q1":"Did a landslide occur between the two observations? Answer with yes or no.","Q2":"A landslide occurred between exactly one adjacent pair of these observations. Between which pair did it occur? Answer with 1, 2, or 3.","Q3":"A landslide occurred between the two observations. In which quadrant(s) is the landslide located? Answer with one or more of NW, NE, SW, SE.","Q4":"How many observations are shown? Answer with a number."}
def eo_tokens(tile,dates):
    S=np.load(SRC/f"{tile}.npy").astype("float32"); kd=kept_dates(tile); idx=[kd.index(d) for d in dates]
    T=torch.from_numpy(S[idx]); T=F.avg_pool2d(T,4).flatten(2).permute(0,2,1)  # n,64,768
    parts=[T[i] for i in range(len(idx))]; types=[torch.full((64,),min(i,2),dtype=torch.long) for i in range(len(idx))]
    if len(idx)>=2: parts.append(T[-1]-T[0]); types.append(torch.full((64,),3,dtype=torch.long))
    return torch.cat(parts).to(dev),torch.cat(types).to(dev)
def build(it,answer=None):
    n=len(it["dates"]); user=f"These are {n} Sentinel-2 observations of the same area in chronological order, taken on {', '.join(it['dates'])}: <EO> {QTEXT[it['type']]}"
    text=tok.apply_chat_template([{"role":"user","content":user}],tokenize=False,add_generation_prompt=True); pre,post=text.split("<EO>")
    ids_pre=tok(pre,add_special_tokens=False,return_tensors="pt").input_ids[0].to(dev); ids_post=tok(post,add_special_tokens=False,return_tensors="pt").input_ids[0].to(dev)
    embs=[EMB(ids_pre)]; labels=[torch.full((len(ids_pre),),-100,device=dev)]
    if not a.text_only:
        t,ty=eo_tokens(it["tile"],it["dates"]); e=proj(t,ty).to(torch.bfloat16); embs.append(e); labels.append(torch.full((len(t),),-100,device=dev))
    embs.append(EMB(ids_post)); labels.append(torch.full((len(ids_post),),-100,device=dev))
    if answer is not None:
        ids_ans=tok(answer+tok.eos_token,add_special_tokens=False,return_tensors="pt").input_ids[0].to(dev); embs.append(EMB(ids_ans)); labels.append(ids_ans)
    return torch.cat(embs),torch.cat(labels)
def gold_text(it): return ", ".join(it["answer"]) if isinstance(it["answer"],list) else it["answer"]
def load_items(d): return [x for x in (json.loads(l) for l in (ROOT/d/"items.jsonl").read_text().splitlines() if l) if (SRC/f"{x['tile']}.npy").exists() and x["type"] in ("Q1","Q3","Q4")]
train=load_items(a.train); test=load_items(a.test)
if a.probe: train=train[:96]; test=test[:12]; a.epochs=2
print("train",len(train),"test",len(test),"hidden",H,"emb_rms",round(EMB_RMS,5),flush=True)
t0=time.perf_counter(); step=0
if not a.text_only:
    for ep in range(a.epochs):
        order=np.random.permutation(len(train)); tot=0
        for i in range(0,len(order),8):
            batch=[train[j] for j in order[i:i+8]]; seqs=[build(it,gold_text(it)) for it in batch]; L=max(s[0].shape[0] for s in seqs)
            E=torch.zeros(len(seqs),L,H,dtype=torch.bfloat16,device=dev); Y=torch.full((len(seqs),L),-100,device=dev); M=torch.zeros(len(seqs),L,dtype=torch.long,device=dev)
            for k,(e,y) in enumerate(seqs): E[k,:len(e)]=e; Y[k,:len(y)]=y; M[k,:len(e)]=1
            out=llm(inputs_embeds=E,attention_mask=M,labels=Y); opt.zero_grad(); out.loss.backward(); opt.step(); tot+=float(out.loss); step+=1
            if step%25==0: print(f"ep {ep} step {step} loss {out.loss.item():.4f} {time.perf_counter()-t0:.0f}s",flush=True)
        print(f"epoch {ep} mean loss {tot/max(1,math.ceil(len(order)/8)):.4f}",flush=True)
    torch.save(proj.state_dict(),OUT/f"projector_seed{a.seed}.pt")
def parse(t,ans):
    ans=ans.strip().lower()
    if t=="Q1": m=re.search(r"\b(yes|no)\b",ans); return m.group(1) if m else None
    if t=="Q3": q=sorted(set(re.findall(r"\b(nw|ne|sw|se)\b",ans))); return [x.upper() for x in q] or None
    if t=="Q4": m=re.search(r"\b([2-9]|two|three|four)\b",ans); return {"two":"2","three":"3","four":"4"}.get(m.group(1),m.group(1)) if m else None
rows=[]; proj.eval()
with torch.no_grad():
    for it in test:
        e,_=build(it); out=llm.generate(inputs_embeds=e[None],attention_mask=torch.ones(1,e.shape[0],dtype=torch.long,device=dev),max_new_tokens=16,do_sample=False,pad_token_id=tok.eos_token_id)
        ans=tok.decode(out[0],skip_special_tokens=True); rows.append({"id":it["id"],"type":it["type"],"answer_raw":ans,"parsed":parse(it["type"],ans),"gold":it["answer"]})
tag="textonly" if a.text_only else f"seed{a.seed}"; (OUT/f"answers_{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
sc={"n_train":len(train),"n_test":len(test),"text_only":a.text_only,"train_s":time.perf_counter()-t0}
for t in ("Q1","Q3","Q4"):
    R=[r for r in rows if r["type"]==t]
    if not R: continue
    pf=sum(1 for r in R if r["parsed"] is None)/len(R)
    if t=="Q3": sc[t]={"n":len(R),"exact":sum(1 for r in R if r["parsed"]==r["gold"])/len(R),"jaccard":sum(len(set(r["parsed"] or [])&set(r["gold"]))/len(set(r["parsed"] or [])|set(r["gold"])) for r in R)/len(R),"parse_fail":pf}
    else:
        sc[t]={"n":len(R),"acc":sum(1 for r in R if r["parsed"]==r["gold"])/len(R),"parse_fail":pf}
        if t=="Q1": neg=[r for r in R if r["gold"]=="no"]; pos=[r for r in R if r["gold"]=="yes"]; sc[t]["fpr_on_negatives"]=sum(1 for r in neg if r["parsed"]=="yes")/max(len(neg),1); sc[t]["recall_on_positives"]=sum(1 for r in pos if r["parsed"]=="yes")/max(len(pos),1); sc[t]["balanced_acc"]=0.5*(sc[t]["recall_on_positives"]+1-sc[t]["fpr_on_negatives"])
(OUT/f"scores_{tag}.json").write_text(json.dumps(sc,indent=1)); print(json.dumps(sc,indent=1)); print("EARTHTALK DONE")
