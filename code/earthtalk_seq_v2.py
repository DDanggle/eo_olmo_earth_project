#!/usr/bin/env python3
"""VLM memory comparison stage 1 (config/vlm_memory_comparison_prereg_v0.json): arms A (latest-only) and C (latest + rule-based event log text). Projector-only, Olmo-3-7B-Instruct frozen. Sequential-evidence QA v1 items."""
import argparse, json, re, time, sys, math, hashlib
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
ap=argparse.ArgumentParser(); ap.add_argument("--arm",required=True,choices=["B_t1_cache","C_lora","D_learned_memory","E_trust_z","E_z_only","E_trust_only"]); ap.add_argument("--lora-r",type=int,default=16); ap.add_argument("--lora-lr",type=float,default=5e-5); ap.add_argument("--out",required=True); ap.add_argument("--epochs",type=int,default=2); ap.add_argument("--seed",type=int,default=1); ap.add_argument("--probe",action="store_true"); ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--max-log-tokens",type=int,default=256); ap.add_argument("--eval-tiles",type=int,default=150,help="seeded uniform subsample of tiles per eval set (all 15 steps kept); 0 = all tiles"); a=ap.parse_args()
ROOT=Path("/home/work/data/olmoearth"); OUT=ROOT/a.out; OUT.mkdir(parents=True,exist_ok=True); dev=torch.device("cuda"); torch.manual_seed(a.seed); np.random.seed(a.seed)
sys.path.insert(0,str(ROOT/"code")); from cache_decoder_train_lib import emb_stats_from_cache
mean,sd=emb_stats_from_cache(ROOT/"sen12_pilot/holdout_hiroshima","holdout_hiroshima"); mean,sd=mean.to(dev),sd.to(dev)
tok=AutoTokenizer.from_pretrained(str(ROOT/"olmo_llm/Olmo-3-7B-Instruct")); llm=AutoModelForCausalLM.from_pretrained(str(ROOT/"olmo_llm/Olmo-3-7B-Instruct"),dtype=torch.bfloat16).to(dev).eval()
for p in llm.parameters(): p.requires_grad_(False)
from peft import LoraConfig, get_peft_model
llm=get_peft_model(llm,LoraConfig(r=a.lora_r,lora_alpha=2*a.lora_r,lora_dropout=0.05,target_modules=["q_proj","k_proj","v_proj","o_proj"],bias="none",task_type="CAUSAL_LM")); llm.print_trainable_parameters()
EMB=llm.get_input_embeddings(); H=EMB.weight.shape[1]; EMB_RMS=float(EMB.weight.detach().float().pow(2).mean().sqrt())
class Projector(nn.Module):
    def __init__(s):
        super().__init__(); s.mlp=nn.Sequential(nn.Linear(770,2048),nn.GELU(),nn.Linear(2048,H)); s.ttype=nn.Embedding(6,H); s.norm=nn.LayerNorm(768); s.sig=nn.Linear(2,2); nn.init.eye_(s.sig.weight); nn.init.zeros_(s.sig.bias); s.out=nn.LayerNorm(H); s.gain=nn.Parameter(torch.tensor(1.0)); s.date=nn.Linear(3,H); nn.init.normal_(s.ttype.weight,std=0.02); nn.init.zeros_(s.date.weight); nn.init.zeros_(s.date.bias)
    def forward(s,tokens,types,datef=None,sig=None):
        x=s.norm(tokens); x=torch.cat([x,(s.sig(sig) if sig is not None else torch.zeros(x.shape[0],2,device=x.device))],1)
        e=s.out(s.mlp(x))*(EMB_RMS*s.gain)+s.ttype(types)*EMB_RMS
        if datef is not None: e=e+s.date(datef)*EMB_RMS
        return e
class Writer(nn.Module):
    def __init__(s): super().__init__(); s.w=nn.Linear(4,1); nn.init.zeros_(s.w.weight); nn.init.constant_(s.w.bias,0.0)
    def forward(s,feat): return torch.sigmoid(s.w(feat))
proj=Projector().to(dev); writer=Writer().to(dev)
params=[{"params":proj.parameters(),"lr":a.lr},{"params":writer.parameters(),"lr":a.lr},{"params":[p for p in llm.parameters() if p.requires_grad],"lr":a.lora_lr}]; opt=torch.optim.AdamW(params,weight_decay=0.01)
# T1 updater (hiroshima-trained Δt-GRU), frozen
class GRU(nn.Module):
    def __init__(s,c=768,extra=0): super().__init__(); s.zr=nn.Conv2d(2*c+extra,2*c,1); s.h=nn.Conv2d(2*c+extra,c,1); s.extra=extra
    def forward(s,m,u,dt,q,x=None):
        inp=torch.cat([m,u]+([x] if x is not None else []),1); z,r=torch.sigmoid(s.zr(inp)).chunk(2,1); n=torch.tanh(s.h(torch.cat([r*m,u]+([x] if x is not None else []),1))); return (1-z)*m+z*n
class GRUdt(GRU):
    def __init__(s,c=768): super().__init__(c,16); s.emb=nn.Linear(1,16)
    def forward(s,m,u,dt,q): x=s.emb(dt.view(-1,1)).view(-1,16,1,1).expand(-1,16,m.shape[2],m.shape[3]); return super().forward(m,u,dt,q,x)
UPD=GRUdt(); UPD.load_state_dict(torch.load(ROOT/"p1_causal_v0_3/gru_dt_seed1.pt",map_location="cpu")); UPD=UPD.to(dev).eval()
from datetime import datetime as _dt
T1CACHE={}
@torch.no_grad()
def t1_state_tokens(it):
    key=(it["region"],it["tile"])
    if key not in T1CACHE:
        U=torch.from_numpy(np.load(ROOT/f"arrival_v0/{it['region']}/single_fp16/{it['tile']}.npy").astype("float32")).to(dev); d=[_dt.fromisoformat(x) for x in json.loads((ROOT/f"arrival_v0/{it['region']}/meta/{it['tile']}.json").read_text())["dates"]]
        dt=torch.tensor([0.0]+[(d[i]-d[i-1]).days/30.0 for i in range(1,len(d))],device=dev); m=U[0:1]; out=[m]
        for i in range(1,U.shape[0]): m=UPD(m,U[i:i+1],dt[i:i+1],torch.zeros(1,device=dev)); out.append(m)
        S=torch.cat(out); T1CACHE[key]=F.avg_pool2d(S,4).flatten(2).permute(0,2,1).cpu()  # 15,64,768
        if len(T1CACHE)>400: T1CACHE.pop(next(iter(T1CACHE)))
    return T1CACHE[key][it["step"]].to(dev)
SINGLES={}
def single_tokens(region,tile):
    key=(region,tile)
    if key not in SINGLES:
        S=torch.from_numpy(np.load(ROOT/f"arrival_v0/{region}/single_fp16/{tile}.npy").astype("float32")); SINGLES[key]=F.avg_pool2d(S,4).flatten(2).permute(0,2,1)  # 15,64,768
        if len(SINGLES)>400: SINGLES.pop(next(iter(SINGLES)))
    return SINGLES[key]
SIGS={}
def signal_tokens(region,tile):
    key=(region,tile)
    if key not in SIGS:
        A=torch.from_numpy(np.load(ROOT/f"arrival_v0/{region}/signal_fp16/{tile}.npy").astype("float32"))  # 15,2,32,32
        P=F.avg_pool2d(A,4).flatten(2).permute(0,2,1)  # 15,64,2
        P[:,:,0]=P[:,:,0]/5.0; SIGS[key]=P
        if len(SIGS)>400: SIGS.pop(next(iter(SIGS)))
    return SIGS[key]
def datefeat(d):
    import math; doy=(d.timetuple().tm_yday-1)/365.0*2*math.pi; return [math.sin(doy),math.cos(doy),(d.year-2018)/2.0]
def memory_tokens(it):
    """K=4 LRU slots written with a learned soft gate; differentiable through the gate and projector."""
    toks=single_tokens(it["region"],it["tile"]).to(dev); meta=json.loads((ROOT/f"arrival_v0/{it['region']}/meta/{it['tile']}.json").read_text()); d=[_dt.fromisoformat(x) for x in meta["dates"]]; clear=meta["clear"]
    K=4; slots=[None]*K; sdate=[None]*K; last=[-1]*K
    for i in range(it["step"]+1):
        cand=toks[i]; ch=float((toks[i]-toks[i-1]).abs().mean()) if i>0 else 0.0; dt=((d[i]-d[i-1]).days/365.0) if i>0 else 0.0
        g=writer(torch.tensor([[float(clear[i]),ch,dt,1.0]],device=dev))[0,0]; j=int(np.argmin(last))
        slots[j]=g*cand+((1-g)*slots[j] if slots[j] is not None else 0); sdate[j]=d[i]; last[j]=i
    parts=[]; types=[]; dfs=[]
    for j in range(K):
        if slots[j] is None: continue
        parts.append(slots[j]); types.append(torch.full((64,),2+j,dtype=torch.long,device=dev)); dfs.append(torch.tensor(datefeat(sdate[j]),device=dev).expand(64,3))
    return torch.cat(parts),torch.cat(types),torch.cat(dfs)
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
def eo_embeds(it):
    if a.arm=="B_t1_cache": t=t1_state_tokens(it); return proj(t,torch.ones(64,dtype=torch.long,device=dev))
    lat=single_tokens(it["region"],it["tile"])[it["step"]].to(dev); sig=None
    if a.arm.startswith("E_"):
        sg=signal_tokens(it["region"],it["tile"])[it["step"]].to(dev).clone()
        if a.arm=="E_z_only": sg[:,1]=0
        if a.arm=="E_trust_only": sg[:,0]=0
        sig=sg
    e=proj(lat,torch.zeros(64,dtype=torch.long,device=dev),sig=sig)
    if a.arm=="D_learned_memory":
        mt,ty,df=memory_tokens(it); e=torch.cat([e,proj(mt,ty,df)])
    return e
def build(it,q,answer=None):
    user=f"Sentinel-2 observations of one tile have arrived in order on {', '.join(it['seen_dates'])}. The latest observation's representation: <EO> "+(log_text(it)+"\n" if a.arm in ("C_lora","E_trust_z","E_z_only","E_trust_only") else "")+QTEXT[q]
    text=tok.apply_chat_template([{"role":"user","content":user}],tokenize=False,add_generation_prompt=True); pre,post=text.split("<EO>")
    ids_pre=tok(pre,add_special_tokens=False,return_tensors="pt").input_ids[0].to(dev); ids_post=tok(post,add_special_tokens=False,return_tensors="pt").input_ids[0].to(dev)
    e=eo_embeds(it).to(torch.bfloat16); embs=[EMB(ids_pre),e,EMB(ids_post)]; labels=[torch.full((len(ids_pre),),-100,device=dev),torch.full((e.shape[0],),-100,device=dev),torch.full((len(ids_post),),-100,device=dev)]
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
torch.save({"proj":proj.state_dict(),"writer":writer.state_dict()},OUT/f"projector_{a.arm}_seed{a.seed}.pt"); llm.save_pretrained(str(OUT/f"lora_{a.arm}_seed{a.seed}"))
def parse(q,ans):
    s=ans.strip().lower()
    if q=="Q_state": m=re.search(r"cannot tell yet|\byes\b|\bno\b",s); return m.group(0) if m else None
    if q=="Q_when": m=re.search(r"\d{4}-\d{2}-\d{2}|none",s); return m.group(0) if m else None
    if q=="Q_where": qs=sorted(set(re.findall(r"\b(nw|ne|sw|se)\b",s))); return ", ".join(x.upper() for x in qs) if qs else ("none" if "none" in s else None)
    m=re.search(r"keep|strengthen|revise|abstain",s); return m.group(0) if m else None
proj.eval(); writer.eval(); llm.eval(); res={"arm":a.arm,"n_train":len(train),"train_s":time.perf_counter()-t0,"evals":{}}
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
