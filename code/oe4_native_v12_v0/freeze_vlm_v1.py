from pathlib import Path
import hashlib
import json
import shutil

stage=Path('/private/tmp/oe4_execute_20260927')
repo=Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project')
dest=repo/'code/oe4_native_v12_v0/vlm_snapshot_v1'
dest.mkdir(exist_ok=False)
names=['vlm_interface_smoke.py','real_h5_provider.py','run_vlm_bounded.py']
for name in names:
 data=(stage/name).read_text()
 if name=='run_vlm_bounded.py':
  data=data.replace('code_snapshot/vlm_v0','code_snapshot/vlm_v1').replace('runs/vlm_interface_v0','runs/vlm_interface_v1').replace('vlm_interface_v0.log','vlm_interface_v1.log').replace('vlm_sequence_status.json','vlm_sequence_v1_status.json').replace('vlm_sequence.lock','vlm_sequence_v1.lock')
 (dest/name).write_text(data)
manifest={'scope':'v1 packaging repair after v0 failed on unlisted macOS AppleDouble sidecar sources; no worker math changes', 'files':{n:hashlib.sha256((dest/n).read_bytes()).hexdigest() for n in names}}
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
