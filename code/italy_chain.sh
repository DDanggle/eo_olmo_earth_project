#!/usr/bin/env bash
# Italy sealed-region chain: build fold -> prepopulate reused tiles by symlink -> extract italy with a snapshot of the sealed extractor -> 3-seed decoders.
set -euo pipefail; ROOT=/home/work/data/olmoearth; cd "$ROOT"; PY="$ROOT/.venv-master/bin/python"; OUT="$ROOT/sen12_pilot/holdout_italy"; SRC="$ROOT/sen12_pilot/holdout_chimanimani"; SNAP="$ROOT/italy_snapshot"
GPU1_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', ' '$1==1 {print $2}')
busy(){ nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | grep -Fxq "$GPU1_UUID"; }
busy && { echo "GPU1 occupied; not starting" >&2; exit 4; }
$PY code/build_italy_fold.py | tee logs/italy_fold.json
mkdir -p "$OUT/emb_fp16" "$OUT/raw_u16" "$OUT/mask_u8" "$SNAP"; cp -n code/extract_sen12_fold_cache.py code/cache_decoder_train.py code/cache_grid_controls.py "$SNAP/"; (cd "$SNAP" && sha256sum ./* > SHA256SUMS)
for d in emb_fp16 raw_u16 mask_u8; do for f in "$SRC/$d"/*.npy; do b=$(basename "$f"); [[ -e "$OUT/$d/$b" ]] || ln -s "$f" "$OUT/$d/$b"; done; done
[[ -e "$OUT/months.jsonl" ]] || ln -s "$SRC/months.jsonl" "$OUT/months.jsonl"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY "$SNAP/extract_sen12_fold_cache.py" --folds sen12_gp_contract/loco_folds_italy.json --fold holdout_italy --out sen12_pilot > logs/x_italy.log 2>&1
for s in 1 2 3; do busy && exit 4; env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 $PY "$SNAP/cache_decoder_train.py" --cache sen12_pilot/holdout_italy --folds sen12_gp_contract/loco_folds_italy.json --fold holdout_italy --seed $s --out artifacts/italy_sealed/decoder > logs/italy_decoder_seed$s.log 2>&1; done
echo ITALY_DONE | tee logs/italy_DONE.txt
