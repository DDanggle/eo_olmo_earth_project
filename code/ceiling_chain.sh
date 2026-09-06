#!/usr/bin/env bash
# In-region full-label ceilings for the few-shot method headroom: A1 on the whole support pool (cache), then A4w on the whole pool (raw). Sen12 8 confirmatory regions + Solar 8 folds.
set -uo pipefail; cd /home/work/data/olmoearth; LOG=logs/ceiling_chain.log; PY=./.venv-master/bin/python
run(){ env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY code/fewshot_a1_a4.py "$@"; echo "$(date -u +%FT%TZ) rc=$? $*" >> $LOG; }
run --confirmatory --arms A0,A1 --support pool --exposure fixed_exposure --out artifacts/fewshot_ceiling/sen12_a1 > logs/ceiling_sen12_a1.log 2>&1
run --task2 --arms A0,A1 --support pool --exposure fixed_exposure --out artifacts/fewshot_ceiling/solar_a1 > logs/ceiling_solar_a1.log 2>&1
run --confirmatory --arms A4w --support pool --exposure fixed_exposure --out artifacts/fewshot_ceiling/sen12_a4w > logs/ceiling_sen12_a4w.log 2>&1
run --task2 --arms A4w --support pool --exposure fixed_exposure --out artifacts/fewshot_ceiling/solar_a4w > logs/ceiling_solar_a4w.log 2>&1
echo "$(date -u +%FT%TZ) CEILING_DONE" >> $LOG
