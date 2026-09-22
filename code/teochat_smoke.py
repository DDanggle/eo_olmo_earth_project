#!/usr/bin/env python3
"""G-L0 smoke: TEOChat 8-bit zero-shot on one Sen12 tile (pre/post S2 RGB composites from raw_u16). No metric; checks the pipeline runs."""
import json, sys, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0,"/home/work/data/olmoearth/teochat/code")
ROOT=Path("/home/work/data/olmoearth"); OUT=ROOT/"teochat/smoke"; OUT.mkdir(parents=True,exist_ok=True)
REC={json.loads(l)["sample_id"]:json.loads(l) for l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl") if l.strip()}
fold="holdout_hiroshima"; SRC=ROOT/"sen12_pilot"/fold
sid=next(s for s in sorted(p.stem for p in (SRC/"mask_u8").glob("*.npy")) if REC[s]["region"]=="hiroshima" and REC[s].get("post_index") is not None and np.load(SRC/"mask_u8"/f"{s}.npy").sum()>500)
r=REC[sid]; q=r["scl_clear_fraction"]; kept=sorted(sorted(range(15),key=lambda i:(-float(q[i]),i))[:12]); post=r["post_index"]
pre_slot=max(j for j,i in enumerate(kept) if i<post); post_slot=min(j for j,i in enumerate(kept) if i>=post)
raw=np.load(SRC/"raw_u16"/f"{sid}.npy").astype("float32")  # 10,12,128,128 bands B02,B03,B04,B08,...
def rgb(t):
    x=np.stack([raw[2,t],raw[1,t],raw[0,t]],-1); x=np.clip(x/3000.0,0,1); return Image.fromarray((x*255).astype("uint8")).resize((512,512),Image.BICUBIC)
paths=[]; dates=[]
for tag,slot in (("pre",pre_slot),("post",post_slot)):
    p=OUT/f"{sid}_{tag}.png"; rgb(slot).save(p); paths.append(str(p)); dates.append(str(r["times"][kept[slot]])[:10])
print("tile",sid,"dates",dates,"mask_px",int(np.load(SRC/"mask_u8"/f"{sid}.npy").sum()),flush=True)
from videollava.eval.eval import load_model
from videollava.eval.inference import run_inference_single
tokenizer, model, processor = load_model(model_path=str(ROOT/"teochat/TEOChat"), model_base=None, load_8bit=False, device="cuda")
qs=["These are two Sentinel-2 satellite images of the same area in chronological order: <video> Did a landslide occur between the two images? Answer yes or no and explain briefly.",
    "These are two Sentinel-2 satellite images of the same area in chronological order: <video> Describe the main change between the images."]
res=[]
for inp in qs:
    out=run_inference_single(model, processor, tokenizer, inp, paths, timestamps=dates); print("Q:",inp[-90:]); print("A:",out,flush=True); res.append({"q":inp,"a":out})
(OUT/"smoke_result.json").write_text(json.dumps({"tile":sid,"dates":dates,"results":res},indent=1)); print("TEOCHAT SMOKE DONE")
