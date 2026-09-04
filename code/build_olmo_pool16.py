"""B-v1 control: OlmoEarth 768x32x32 cache -> 2x2 arithmetic mean -> 768x16x16 (same physical support as Clay native 16x16 on 128 px)."""
import json, os, numpy as np
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth"); SRC=ROOT/"sen12_pilot/holdout_chimanimani"; OUT=ROOT/"olmo_cache_pool16"; (OUT/"emb_fp16").mkdir(parents=True,exist_ok=True)
for d in ("raw_u16","mask_u8"):
    if not (OUT/d).exists(): os.symlink(SRC/d,OUT/d)
for f in ("months.jsonl","cache_audit.json"):
    if not (OUT/f).exists(): os.symlink(SRC/f,OUT/f)
fs=sorted((SRC/"emb_fp16").glob("*.npy")); n=0
for f in fs:
    o=OUT/"emb_fp16"/f.name
    if o.exists(): n+=1; continue
    x=np.load(f).astype("float32"); np.save(o,x.reshape(x.shape[0],16,2,16,2).mean(axis=(2,4)).astype("float16")); n+=1
print("pool16 done",n); (OUT/"pool16_audit.json").write_text(json.dumps({"n":n,"shape":[768,16,16],"rule":"2x2 arithmetic mean of the sealed 32x32 cache"}))
