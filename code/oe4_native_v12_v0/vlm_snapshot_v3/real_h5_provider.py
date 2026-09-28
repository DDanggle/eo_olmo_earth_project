#!/usr/bin/env python3
"""Real official H5 input for an OlmoEarth -> Qwen interface engineering test."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()

def read_example(config):
 import numpy as np
 import h5py
 import hdf5plugin  # noqa: F401
 manifest=json.loads(Path(config['manifest']).read_text())
 row=manifest['selected'][config.get('example_index',0)]
 y,x,h,w=row['crop_yxhw'];compact=row['compact_timesteps'];indices=row['timestamp_indices']
 with h5py.File(row['file'],'r') as f:
  raw=np.asarray(f['sentinel2_l2a'][y:y+h,x:x+w,:len(compact),:],dtype=np.float32)
  wc=np.asarray(f['worldcover'][y:y+h,x:x+w],dtype=np.float32)
  if wc.ndim==3:wc=wc[:,:,None,:]
  ts=np.asarray(f['timestamps'][()],dtype=np.int64)[indices]
  if not np.array_equal(compact,np.arange(len(compact))):raise ValueError('Only first compact timesteps supported')
  if 'missing_timesteps_masks/sentinel2_l2a' in f:
   actual=np.flatnonzero(np.asarray(f['missing_timesteps_masks/sentinel2_l2a'][()]).reshape(-1))[:len(indices)]
   if not np.array_equal(actual,indices):raise ValueError('Presence/time provenance mismatch')
 actual_sha=hashlib.sha256(raw.tobytes()+wc.tobytes()+ts.tobytes()).hexdigest()
 if actual_sha!=row['raw_crop_sha256']:raise ValueError('Real crop differs from frozen data manifest')
 if not np.isfinite(raw).all() or np.any(raw==-99999):raise ValueError('Invalid raw EO input')
 return raw,ts,row

def make_provider(config,device,dtype):
 sys.path[:0]=[config['source_root'],config['deps_root']]
 import torch
 from olmoearth_pretrain.model_loader import load_model_from_path
 from olmoearth_pretrain.data.constants import Modality
 from olmoearth_pretrain.data.normalize import Normalizer,Strategy
 from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample,MaskValue
 raw,ts,row=read_example(config)
 norm=Normalizer(Strategy.COMPUTED).normalize(Modality.SENTINEL2_L2A,raw)
 checkpoint_hashes={name:sha(Path(config['checkpoint_dir'])/name) for name in ['config.json','weights.pth']}
 if checkpoint_hashes!={'config.json':'0d531a67ad3e477e7011efabcceb01ed80f430aa0a0a3d344fe18cec0f229b8a','weights.pth':'57f7b66faf206db1307670673839e639d3a19c305f6ad968c62392ad3e88deec'}:raise ValueError('Not the pinned original v1.2 Base checkpoint')
 full=load_model_from_path(config['checkpoint_dir'])
 encoder=full.encoder
 del full
 encoder.to(device=device,dtype=torch.float32)
 encoder.eval()
 for module in encoder.modules():
  if getattr(module,'use_flash_attn',False):module.use_flash_attn=False
 if encoder.tokenization_config.get_num_bandsets('sentinel2_l2a')!=1:raise ValueError('Require v1.2 single bandset')
 data=torch.as_tensor(norm,dtype=torch.float32,device=device).unsqueeze(0)
 mask=torch.full((*data.shape[:-1],1),MaskValue.ONLINE_ENCODER.value,dtype=torch.int32,device=device)
 sample=MaskedOlmoEarthSample(sentinel2_l2a=data,sentinel2_l2a_mask=mask,timestamps=torch.as_tensor(ts,dtype=torch.int64,device=device).unsqueeze(0))
 def encode():
  result=encoder(sample,patch_size=4,fast_pass=True)['tokens_and_masks'].sentinel2_l2a
  if result.ndim!=6 or result.shape[0]!=1:raise ValueError('Unexpected official EO token shape')
  return result.reshape(1,-1,result.shape[-1])
 return {'model':encoder,'encode':encode,'metadata':{'source':'official_v1_2_real_H5','checkpoint_hashes':checkpoint_hashes,'manifest':config['manifest'],'row':row,'encoder_precision':'float32','normalization':'official COMPUTED once','token_layout':'flattened Hpatch,Wpatch,T,bandset(1); no band-specific output claim','mode':'eval_with_gradients','scope':'one-step interface; no semantic result','vlm_requested_dtype':str(dtype)}}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--out-dir',type=Path,required=True);ap.add_argument('--source-root',required=True);ap.add_argument('--deps-root',required=True);ap.add_argument('--checkpoint-dir',required=True);a=ap.parse_args()
 sys.path[:0]=[a.source_root,a.deps_root]
 import numpy as np
 from PIL import Image
 a.out_dir.mkdir(parents=True,exist_ok=False)
 config={'manifest':a.manifest,'example_index':0,'source_root':a.source_root,'deps_root':a.deps_root,'checkpoint_dir':a.checkpoint_dir}
 raw,ts,row=read_example(config)
 # Official S2 order starts B02 B03 B04 B08; display stretch is not model normalization.
 rgb=np.clip(raw[:,:,0,[2,1,0]]/3000,0,1)
 Image.fromarray(np.round(rgb*255).astype(np.uint8),'RGB').save(a.out_dir/'rgb.png')
 (a.out_dir/'provider_config.json').write_text(json.dumps(config,indent=2)+'\n')
 (a.out_dir/'provenance.json').write_text(json.dumps({'row':row,'rgb_bands':['B04','B03','B02'],'rgb_timestamp':ts[0].tolist(),'rgb_display_stretch':'fixed raw DN 0..3000, clipped; not calibrated reflectance or cloud interpretation','scope':'engineering interface input'},indent=2)+'\n')
 print(json.dumps({'prepared':str(a.out_dir),'raw_crop_sha256':row['raw_crop_sha256']}))

if __name__=='__main__':main()
