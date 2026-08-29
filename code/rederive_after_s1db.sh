#!/usr/bin/env bash
# M75 복구 2단계: dB 임베딩으로 5앵커·회랑 판정을 재도출하고, S1-only(dB) 분해까지 돌림 (서버에서 실행).
set -u
cd /home/work/data/olmoearth
PY=/home/work/data/olmoearth/.venv-master/bin/python
echo "== naive delta (10 placebo)"; env -u PYTHONPATH $PY code/analyze_nepal_delta.py --live-mode s1_live 2>&1 | grep -E "live=|report" | cut -c1-160
echo "== matched pairs + token-level"; env -u PYTHONPATH $PY code/analyze_nepal_delta_matched.py 2>&1 | grep -E "mean-rank|report" | cut -c1-220
echo "== corridor sealed (borrowed)"; env -u PYTHONPATH $PY code/analyze_corridor_sealed.py 2>&1 | grep -E "^thr|^[0-9]+ w" | head -8
echo "== corridor matched (own)"; env -u PYTHONPATH $PY code/analyze_corridor_matched.py 2>&1 | grep -E "own thr|^[0-9]+ w|unobservable" | head -9
echo "== S1-only dB corridor"
for m in baseline s1_live placebo_a; do
  ds=artifacts/external_data/nepal_olmo_live_v1/materialized_corridor/$m/dataset
  for w in $ds/windows/nepal/*; do rm -rf "$w/layers/embeddings_s1"; done
  DATASET_PATH=$ds CUDA_VISIBLE_DEVICES=1 HF_HOME=/home/work/data/.cache/huggingface env -u PYTHONPATH .venv-master/bin/rslearn model predict --config code/model_s1db_only.yaml --data.init_args.num_workers=2 --data.init_args.batch_size=4 > logs/s1db_only_corridor_$m.log 2>&1; echo "s1only $m exit $?"
done
EMB_LAYER=embeddings_s1 OUT_NAME=corridor_sealed_s1only env -u PYTHONPATH $PY code/analyze_corridor_sealed.py 2>&1 | grep -E "^thr|^[0-9]+ w" | head -6
echo REDERIVE_DONE
