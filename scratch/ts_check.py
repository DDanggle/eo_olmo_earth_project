import json,sys,numpy as np,torch
from pathlib import Path
from datetime import datetime,timedelta
sys.path.insert(0,"/home/work/data/olmoearth/code")
from olmoearth_pretrain_minimal import ModelID
from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage
ROOT=Path("/home/work/data/olmoearth"); SRC=ROOT/"sen12_pilot/holdout_chimanimani"; dev=torch.device("cuda")
sid="chimanimani_s2_1000"; rec=next(json.loads(l) for l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl") if sid in l)
print("times",rec["times"][:3],len(rec["times"]))
w=OlmoEarth(patch_size=4, model_id=ModelID.OLMOEARTH_V1_BASE, token_pooling=True, use_legacy_timestamps=False, normalize=True, autocast_dtype="bfloat16").to(dev).eval()
months={json.loads(l)["sample_id"]:json.loads(l)["months_0_11"] for l in open(SRC/"months.jsonl") if l.strip()}
raw=np.load(SRC/"raw_u16"/f"{sid}.npy").astype("float32"); T=raw.shape[1]; cube=np.zeros((12,T,128,128),dtype="float32"); cube[:10]=raw
def emb(ts):
    feat=torch.empty((768,32,32))
    for y0,x0 in ((0,0),(0,64),(64,0),(64,64)):
        image=torch.from_numpy(np.ascontiguousarray(cube[:,:,y0:y0+64,x0:x0+64])).to(dev); inp={"sentinel2_l2a":RasterImage(image=image,timestamps=[(t,t) for t in ts])}; w.normalizer(inp,{})
        sample,_,_=w._prepare_modality_inputs(ModelContext(inputs=[inp],metadatas=[])); sample.sentinel2_l2a_mask[...,2]=MaskValue.MISSING.value
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            tm=w.model(sample,fast_pass=False,patch_size=4)["tokens_and_masks"]; m=(tm.sentinel2_l2a_mask!=MaskValue.MISSING.value).unsqueeze(-1)
            feat[:,y0//4:(y0+64)//4,x0//4:(x0+64)//4]=((tm.sentinel2_l2a*m).sum(dim=(3,4))/m.sum(dim=(3,4)).clamp(min=1))[0].permute(2,0,1).float().cpu()
    return feat
ref=torch.from_numpy(np.load(SRC/"emb_fp16"/f"{sid}.npy").astype("float32"))
# which timesteps did the sealed extractor keep? select_timestep_indices keeps 12 of 15 by SCL clear fraction
q=rec["scl_clear_fraction"]; idx=sorted(sorted(range(len(q)),key=lambda i:(-float(q[i]),i))[:12])
real=[datetime.fromisoformat(str(rec["times"][i])[:19]) for i in idx]
syn=[datetime(2020,int(m)+1,1)+timedelta(days=1+i) for i,m in enumerate(months[sid][:T])]
for name,ts in (("real",real),("synthetic",syn)):
    e=emb(ts); print(name,"max|diff|",float((e-ref).abs().max()),"cos",float(torch.nn.functional.cosine_similarity(e.flatten(),ref.flatten(),dim=0)),"ref max",float(ref.abs().max()))
