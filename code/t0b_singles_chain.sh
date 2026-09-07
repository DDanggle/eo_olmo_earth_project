#!/usr/bin/env bash
# T0b: rerun the temporal-evidence screen on SINGLE-ACQUISITION embeddings (true observed change), after the T1 extraction finishes.
set -uo pipefail; ROOT=/home/work/data/olmoearth; cd "$ROOT"; PY="$ROOT/.venv-master/bin/python"; L="$ROOT/logs"; log(){ echo "$(date -u +%FT%TZ) $*" >> "$L/t0b_chain.log"; }
until grep -q "OLMO STREAMING DEV DONE" "$L/x_olmo_streaming.log" 2>/dev/null; do sleep 60; done; log "T1 extraction done; singles extraction start"
mkdir -p olmo_single_p4; for d in raw_u16 mask_u8 emb_fp16; do [[ -e olmo_single_p4/$d ]] || ln -s "$ROOT/sen12_pilot/holdout_chimanimani/$d" olmo_single_p4/$d; done
ls sen12_pilot/holdout_chimanimani/emb_fp16 | sed 's/.npy$//' > sen12_gp_contract/all_ids.txt
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/extract_olmo_streaming.py --out olmo_single_p4 --ids-file sen12_gp_contract/all_ids.txt --singles-only > "$L/x_olmo_single.log" 2>&1; log "singles extraction rc=$?"
$PY -c 'import json,sys; a=json.load(open("olmo_single_p4/olmo_single_audit.json")); sys.exit(0 if a["all_gates_pass"] else 1)' || { log "singles audit FAILED"; touch "$L/t0b_FAILED.txt"; exit 5; }
env -u PYTHONPATH $PY code/t0_precompute.py olmo_single_p4 holdout_hiroshima holdout_chimanimani > "$L/t0b_precompute.log" 2>&1; log "precompute rc=$?"
for fold in holdout_hiroshima holdout_chimanimani; do for ro in diffpca sketch full; do for s in 1 2 3; do
  [[ -f artifacts/temporal_t0b/${fold}_${ro}_seed${s}.json ]] && continue
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/temporal_readout_train.py --cache olmo_single_p4 --fold $fold --readout $ro --seed $s --out artifacts/temporal_t0b > "$L/t0b_${fold}_${ro}_s${s}.log" 2>&1; log "t0b $fold $ro s$s rc=$?"
done; done; done; log "T0b DONE"; touch "$L/t0b_DONE.txt"
