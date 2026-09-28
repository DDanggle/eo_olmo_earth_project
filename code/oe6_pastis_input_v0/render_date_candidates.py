"""Render existing raw PASTIS dates for input QA, not a learned cloud detector."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw

p=Path('/private/tmp/oe6_crossdomain_20260927/server_prepared/prepared_v1')
c=json.loads((p/'contract.json').read_text())
canvas=Image.new('RGB',(128*8,156*2),'white')
d=ImageDraw.Draw(canvas)
notes=[]
for row,r in enumerate(c['records']):
    z=np.load(p/r['path'])
    raw=z['raw_s2']
    indices=np.linspace(0,len(raw)-1,8).astype(int).tolist()
    notes.append({'patch_id':r['patch_id'],'indices':indices,'dates':[r['all_dates_yyyymmdd'][i] for i in indices]})
    for col,i in enumerate(indices):
        rgb=np.transpose(raw[i,[2,1,0]],(1,2,0))
        canvas.paste(Image.fromarray(np.round(np.clip(rgb/3000,0,1)*255).astype('uint8')),(col*128,row*156+23))
        d.text((col*128+2,row*156+5),f'{i}: {r["all_dates_yyyymmdd"][i]}',fill='black')
canvas.save(p/'date_candidates.png')
(p/'date_candidates.json').write_text(json.dumps({'status':'visual_input_QA_only','display':'fixed DN 0..3000','cases':notes},indent=2)+'\n')
print(json.dumps(notes))
