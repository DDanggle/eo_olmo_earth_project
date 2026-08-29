#!/usr/bin/env bash
# M75 복구: 모든 S1 포함 모드의 임베딩을 dB 정규화 레시피(model_s1db.yaml)로 재계산함.
# 기존 선형판 산출물은 layers/embeddings_linear_s1 로 보존(L3), 매니페스트는 embedding_manifest.linear_s1.json 으로 보존.
set -u
cd /home/work/data/olmoearth
export RSLEARN_BIN=/home/work/data/olmoearth/.venv-master/bin/rslearn PYTHON_BIN=/home/work/data/olmoearth/.venv-master/bin/python OLMO_GPU=1 HF_HOME=/home/work/data/.cache/huggingface MODEL_CONFIG=/home/work/data/olmoearth/code/model_s1db.yaml
run_mode() {  # $1=MATERIALIZED_DIR $2=mode
  local root="artifacts/external_data/nepal_olmo_live_v1/$1/$2"
  [[ -f "$root/materialization_manifest.json" ]] || { echo "SKIP $1/$2 (no manifest)"; return; }
  for w in "$root"/dataset/windows/nepal/*; do
    if [[ -d "$w/layers/embeddings" && ! -d "$w/layers/embeddings_linear_s1" ]]; then mv "$w/layers/embeddings" "$w/layers/embeddings_linear_s1"; fi
    rm -rf "$w/layers/embeddings"
  done
  [[ -f "$root/embedding_manifest.json" && ! -f "$root/embedding_manifest.linear_s1.json" ]] && mv "$root/embedding_manifest.json" "$root/embedding_manifest.linear_s1.json"
  rm -f "$root/embedding_manifest.json"
  MATERIALIZED_DIR=$1 bash code/run_nepal_olmo_embeddings.sh "$2" > "logs/s1db_${1}_${2}.log" 2>&1
  if [[ -f "$root/embedding_manifest.json" ]]; then echo "OK $1/$2 $(date +%H:%M)"; else echo "FAIL $1/$2"; tail -3 "logs/s1db_${1}_${2}.log"; fi
}
for m in baseline placebo_a placebo_b placebo_20260617 placebo_20260624 placebo_20260701 placebo_20260708 placebo_20260715 placebo_20260722 placebo_20260729 placebo_20260805 s1_live; do run_mode materialized "$m"; done
for m in baseline placebo_a s1_live; do run_mode materialized_corridor "$m"; done
echo S1DB_RECOMPUTE_DONE
