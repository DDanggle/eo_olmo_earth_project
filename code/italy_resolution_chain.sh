#!/usr/bin/env bash
# Italy resolution decomposition (config/italy_resolution_decomposition_prereg_v0.json). GPU1.
set -uo pipefail; ROOT=/home/work/data/olmoearth; cd "$ROOT"; PY="$ROOT/.venv-master/bin/python"; L="$ROOT/logs"; log(){ echo "$(date -u +%FT%TZ) $*" >> "$L/italy_res_chain.log"; }
OUT=olmo_italy_p2_rt; mkdir -p $OUT/emb_fp16
for d in raw_u16 mask_u8 months.jsonl; do [[ -e $OUT/$d ]] || ln -s "$ROOT/sen12_pilot/holdout_italy/$d" $OUT/$d; done
# reuse the real-date 20 m embeddings of the confirmatory tiles
for f in olmo_base_p2_rt/emb_fp16/*.npy; do b=$(basename "$f"); [[ -e $OUT/emb_fp16/$b ]] || ln -s "$ROOT/$f" $OUT/emb_fp16/$b; done
log "start; prelinked $(ls $OUT/emb_fp16 | wc -l)"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/extract_olmo_variants.py --size base --patch 2 --real-times --src sen12_pilot/holdout_italy --out $OUT > "$L/x_italy_p2_rt.log" 2>&1; log "extraction rc=$?"
$PY -c 'import json,sys; a=json.load(open("olmo_italy_p2_rt/olmo_variant_audit.json")); print(a.get("n_valid"),a.get("all_gates_pass")); sys.exit(0 if a.get("all_gates_pass") else 1)' >> "$L/italy_res_chain.log" 2>&1 || { log "audit FAILED"; exit 5; }
for s in 1 2 3; do
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/cache_decoder_train.py --cache sen12_pilot/holdout_italy --grid-transform upsample2 --folds sen12_gp_contract/loco_folds_italy.json --fold holdout_italy --seed $s --out artifacts/italy_sealed/p4_upsample2 > "$L/italy_up_s$s.log" 2>&1; log "upsample2 s$s rc=$?"
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/cache_decoder_train.py --cache $OUT --grid-transform native --folds sen12_gp_contract/loco_folds_italy.json --fold holdout_italy --seed $s --out artifacts/italy_sealed/p2_native > "$L/italy_p2_s$s.log" 2>&1; log "p2_native s$s rc=$?"
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY code/cache_decoder_train.py --cache $OUT --grid-transform avgpool2 --folds sen12_gp_contract/loco_folds_italy.json --fold holdout_italy --seed $s --out artifacts/italy_sealed/p2_avgpool2 > "$L/italy_pool_s$s.log" 2>&1; log "p2_avgpool2 s$s rc=$?"
done; log "ITALY_RES DONE"; touch "$L/italy_res_DONE.txt"
