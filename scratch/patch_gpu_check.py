import re,sys
def patch(p, old, new):
    s=open(p).read(); assert old in s, p; open(p,"w").write(s.replace(old,new)); print("patched",p)
OWN='extract_olmo_|temporal_readout_train|cache_decoder_train|extract_sen12_fold_cache|streaming_update_train'
patch("code/resolution_chain.sh",
'''require_gpu1_free () {
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | grep -Fxq "$GPU1_UUID"; then''',
'''require_gpu1_free () {
  # 2026-09-07: only FOREIGN processes block; our own queue may share GPU1.
  local pids p foreign=0
  pids=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F', ' -v u="$GPU1_UUID" '$1==u {print $2}')
  for p in $pids; do ps -o cmd= -p "$p" | grep -Eq "'''+OWN+'''" || foreign=1; done
  if [[ $foreign -eq 1 ]]; then''')
patch("code/resolution_resume_loop.sh",
'''  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | grep -Fxq "$GPU1_UUID"; then echo "$(date -u +%FT%TZ) GPU1 busy, wait 120s" >> "$LOG"; sleep 120; continue; fi''',
'''  foreign=0; for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F', ' -v u="$GPU1_UUID" '$1==u {print $2}'); do ps -o cmd= -p "$p" | grep -Eq "'''+OWN+'''" || foreign=1; done
  if [[ $foreign -eq 1 ]]; then echo "$(date -u +%FT%TZ) GPU1 foreign busy, wait 120s" >> "$LOG"; sleep 120; continue; fi''')
patch("code/evening_queue_0907.sh",
'''foreign_busy(){ nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | grep -Fxq "$GPU1_UUID" && ! ours; }''',
'''foreign_busy(){ local p f=0; for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F', ' -v u="$GPU1_UUID" '$1==u {print $2}'); do ps -o cmd= -p "$p" | grep -Eq "'''+OWN+'''" || f=1; done; [[ $f -eq 1 ]]; }''')
