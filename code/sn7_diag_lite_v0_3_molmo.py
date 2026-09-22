#!/usr/bin/env python3
"""G1-lite v0.3 amendment 1: second reader Molmo2-O-7B on the identical diag_lite_v0_3 items/conditions. Writes answers_molmo.jsonl; score with sn7_diag_lite_v0_3.py logic (see score_molmo())."""
import json, sys, time, torch
from pathlib import Path
from PIL import Image
ROOT=Path("/home/work/data/olmoearth"); OUT=ROOT/"spacenet7/diag_lite_v0_3"; MODEL=ROOT/"models/Molmo2-O-7B"
src=(OUT/"sn7_diag_lite_v0_3.py").read_text().split('{"build":build')[0]; ns={}; exec(src.replace("import json, re, sys, time, numpy as np","import json, re, sys, time\nimport numpy as np"),ns); prompt,parse=ns["prompt"],ns["parse"]
def run():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc=AutoProcessor.from_pretrained(MODEL,trust_remote_code=True); model=AutoModelForImageTextToText.from_pretrained(MODEL,trust_remote_code=True,dtype=torch.bfloat16,device_map="cuda").eval()
    items=[json.loads(l) for l in (OUT/"items.jsonl").read_text().splitlines() if l]; ap=OUT/"answers_molmo.jsonl"
    done={(r["id"],r["cond"]) for r in map(json.loads,ap.read_text().splitlines())} if ap.exists() else set(); f=open(ap,"a"); t0=time.perf_counter(); n=0
    for it in items:
        for cond,frames in it["conds"].items():
            if frames is None or (it["id"],cond) in done: continue
            paths=[it["png"][m][it["region"]] for m in frames]; text=prompt(it,frames).replace("<video> ","")
            content=[]
            for m,pth in zip(frames,paths): content+=[{"type":"text","text":f"[{m}]"},{"type":"image","image":Image.open(pth).convert("RGB")}]
            content.append({"type":"text","text":text})
            try:
                inp=proc.apply_chat_template([{"role":"user","content":content}],tokenize=True,add_generation_prompt=True,return_tensors="pt",return_dict=True)
                inp={k:(v.to("cuda") if hasattr(v,"to") else v) for k,v in inp.items()}
                with torch.no_grad(): out=model.generate(**inp,max_new_tokens=32,do_sample=False)
                ans=proc.tokenizer.decode(out[0,inp["input_ids"].shape[1]:],skip_special_tokens=True)
            except Exception as e: ans=f"ERROR {str(e)[:200]}"
            f.write(json.dumps({"id":it["id"],"aoi":it["aoi"],"q":it["q"],"cond":cond,"n_frames":len(frames),"raw":ans[:200],"pred":parse(it["q"],ans),"gold":it["gold"]})+"\n"); f.flush(); n+=1
            if n%20==0: print(n,f"{time.perf_counter()-t0:.0f}s",flush=True)
    f.close(); print("RUN DONE")
def score_molmo():
    import shutil
    a,b=OUT/"answers.jsonl",OUT/"answers_qwen_v0_3.jsonl"
    if not b.exists(): shutil.copy(a,b)
    shutil.copy(OUT/"answers_molmo.jsonl",a); ns2={}; exec((OUT/"sn7_diag_lite_v0_3.py").read_text().split('{"build":build')[0].replace("import json, re, sys, time, numpy as np","import json, re, sys, time\nimport numpy as np"),ns2); ns2["score"]()
    (OUT/"scores.json").rename(OUT/"scores_molmo.json"); shutil.copy(b,a); print("SCORE MOLMO DONE")
{"run":run,"score":score_molmo}[sys.argv[1]]()
