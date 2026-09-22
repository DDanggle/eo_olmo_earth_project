#!/usr/bin/env python3
"""G1-lite v0.1: same items/conditions as sn7_diag_lite_v0 (items.jsonl), reader = Qwen3-VL-8B-Instruct zero-shot (config/bottleneck_diag_lite_prereg_v0_1.json). Writes answers_qwen.jsonl; score via sn7_diag_lite_v0.py logic on that file."""
import json, sys, time, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent)); 
ROOT=Path("/home/work/data/olmoearth"); OUT=ROOT/"spacenet7/diag_lite_v0"; MODEL=ROOT/"models/Qwen3-VL-8B-Instruct"
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import importlib.util
spec=importlib.util.spec_from_file_location("d0",str(OUT/"sn7_diag_lite_v0.py"))
# reuse prompt()/parse() text from v0 without executing its CLI dispatch
src=(OUT/"sn7_diag_lite_v0.py").read_text().split('{"build":build')[0]; ns={}; exec(src,ns); prompt,parse=ns["prompt"],ns["parse"]
proc=AutoProcessor.from_pretrained(MODEL); model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda").eval()
items=[json.loads(l) for l in (OUT/"items.jsonl").read_text().splitlines() if l]; ap=OUT/"answers_qwen.jsonl"
done={(r["id"],r["cond"]) for r in map(json.loads,ap.read_text().splitlines())} if ap.exists() else set(); f=open(ap,"a"); t0=time.perf_counter(); n=0
for it in items:
    for cond,frames in it["conds"].items():
        if frames is None or (it["id"],cond) in done: continue
        text=prompt(it,frames).replace("<video> ","")
        content=[]
        for m in frames: content+= [{"type":"text","text":f"[{m}]"},{"type":"image","image":it["png"][m]}]
        content.append({"type":"text","text":text})
        msgs=[{"role":"user","content":content}]
        try:
            imgs=[Image.open(it["png"][m]).convert("RGB") for m in frames]
            chat=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
            inp=proc(text=[chat],images=imgs,return_tensors="pt").to("cuda")
            with torch.no_grad(): out=model.generate(**inp,max_new_tokens=32,do_sample=False)
            ans=proc.batch_decode(out[:,inp["input_ids"].shape[1]:],skip_special_tokens=True)[0]
        except Exception as e: ans=f"ERROR {str(e)[:200]}"
        f.write(json.dumps({"id":it["id"],"aoi":it["aoi"],"q":it["q"],"cond":cond,"n_frames":len(frames),"raw":ans[:200],"pred":parse(it["q"],ans),"gold":it["gold"]})+"\n"); f.flush(); n+=1
        if n%20==0: print(n,f"{time.perf_counter()-t0:.0f}s",flush=True)
f.close(); print("RUN DONE")
