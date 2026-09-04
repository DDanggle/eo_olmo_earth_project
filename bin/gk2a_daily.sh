#!/bin/zsh
# GK2A 경량화 endpoint 일일 보충 수집 (docs/DAILY_OPS.md). launchd가 매일 09:30 실행. 키는 ../.env 에서만 읽음.
cd /Users/dgyi/dong/ai_projects/olmoearth_projects || exit 1
set -a; . ./.env; set +a
echo "== $(date '+%F %T') gaps" >> ~/Library/Logs/gk2a/daily.log
/usr/bin/python3 _work/code/gk2a_snapshot.py --gaps >> ~/Library/Logs/gk2a/daily.log 2>&1
/usr/bin/python3 _work/code/gk2a_snapshot.py --status 2>&1 | tail -3 >> ~/Library/Logs/gk2a/daily.log
