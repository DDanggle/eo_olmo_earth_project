#!/usr/bin/env python3
"""GK2A 일일 스냅샷 — 보존기간 2일 자료를 영구 축적함.

왜 지금 시작하는가: 이 프로그램에서 **되돌릴 수 없는 것은 이것 하나**임.
GK2A 경량화 API는 "최대 조회 기간은 오늘 기준으로 2일 전까지"임(M-기록 참조).
즉 오늘 받지 않은 날짜는 **영구히 사라짐.** 실험 설계·모델 학습은 나중에 해도 되지만
이 수집은 미루면 그만큼 자산이 없어짐.

실행 위치: **로컬**. 서버(kt cloud)에서는 apis.data.go.kr 연결이 차단됨(실측).

수집 대상 (10개 오퍼레이션 전부 검증됨):
  한반도(All) 2km 격자   CLD 구름탐지 / AOD 에어로졸 / FOG 안개 / COT 주간구름광학두께 / CT 구름분석
  행정구역(Area)          같은 5종 + dongCode

저장: /home/work/data/olmoearth/gk2a/YYYY/MM/DD/<op>_<resultType>_<HHMM>.json.gz
      원본 응답을 그대로 보존함. 파생 가공은 나중에 함.
매니페스트: 같은 디렉터리의 manifest.jsonl 에 (시각, op, bytes, sha256, http상태)를 적음.

키는 인자로 받지 않음. 환경변수 DATA_GO_KR_SERVICE_KEY 만 읽음.
Encoding 형태를 쿼리스트링에 그대로 넣어야 함 (재인코딩하면 실패함).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://apis.data.go.kr/1360000/CloudSatlitInfoService"
KST = timezone(timedelta(hours=9))
# 서버(kt cloud)에서는 apis.data.go.kr이 차단됨 — DNS는 풀리는데 연결이 타임아웃함
# (같은 서버에서 huggingface.co는 0.15s). 따라서 **로컬에서 수집**하고 나중에 서버로 동기함.
# 저장 위치는 GK2A_ROOT 환경변수로 바꿀 수 있음.
ROOT = Path(os.environ.get("GK2A_ROOT",
            str(Path.home() / "dong/ai_projects/data/gk2a")))

# M-기록에서 확정한 (오퍼레이션, resultType) 조합
PRODUCTS = [
    ("getGk2acldAll", "CLD"),
    ("getGk2aappsAll", "AOD"),
    ("getGk2afogAll", "FOG"),
    ("getGk2adcoewAll", "COT"),
    ("getGk2aclaAll", "CT"),
]
# 하루에 받을 관측 슬롯 (KST 시각). 2일 보존이므로 어제 것을 안전하게 받음.
SLOT_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]
SLEEP_S = 4          # 동시호출 제한(code 99 "이미 호출중") 회피
# 한반도(All) 응답은 item 1개에 격자 전체가 들어 있음
# ("gridKm":"2.0","xdim":"320","ydim":"397","value":"0,0,0,...") — 실측 확인.
# 크게 잡으면 게이트웨이가 타임아웃함. 1이 맞음.
NUM_ROWS = 1


def fetch(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=90) as r:
            return r.status, r.read()
    except Exception as exc:  # noqa: BLE001
        return -1, f"EXC {type(exc).__name__}: {exc}".encode()


def main() -> None:
    key = os.environ.get("DATA_GO_KR_SERVICE_KEY", "")
    if not key:
        print("DATA_GO_KR_SERVICE_KEY 없음", file=sys.stderr)
        raise SystemExit(2)

    # 보존 2일이므로 '어제'를 받음. 인자로 날짜를 주면 그 날짜를 받음(재시도용).
    target = (datetime.now(KST) - timedelta(days=1)).date()
    if len(sys.argv) > 1:
        target = datetime.strptime(sys.argv[1], "%Y%m%d").date()

    outdir = ROOT / f"{target:%Y/%m/%d}"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = outdir / "manifest.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                if r.get("ok"):
                    done.add(r["file"])

    ok = fail = skip = 0
    with manifest.open("a", encoding="utf-8") as mf:
        for hh in SLOT_HOURS:
            dt = f"{target:%Y%m%d}{hh:02d}00"
            for op, rt in PRODUCTS:
                fname = f"{op}_{rt}_{hh:02d}00.json.gz"
                if fname in done:
                    skip += 1
                    continue
                url = (f"{BASE}/{op}?ServiceKey={key}&pageNo=1&numOfRows={NUM_ROWS}"
                       f"&dataType=JSON&dateTime={dt}&resultType={rt}")
                status, body = fetch(url)
                time.sleep(SLEEP_S)
                # 서비스 자체 오류 코드도 판정에 씀
                code = None
                try:
                    code = json.loads(body.decode("utf-8"))["response"]["header"]["resultCode"]
                except Exception:  # noqa: BLE001
                    pass
                good = status == 200 and code == "00"
                if good:
                    p = outdir / fname
                    p.write_bytes(gzip.compress(body))
                    rec = {"file": fname, "op": op, "resultType": rt, "dateTime": dt,
                           "bytes_raw": len(body), "bytes_gz": p.stat().st_size,
                           "sha256": hashlib.sha256(body).hexdigest(),
                           "http": status, "resultCode": code, "ok": True}
                    ok += 1
                else:
                    rec = {"file": fname, "op": op, "resultType": rt, "dateTime": dt,
                           "http": status, "resultCode": code, "ok": False,
                           "head": body[:200].decode("utf-8", "replace")}
                    fail += 1
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mf.flush()

    total_gz = sum(p.stat().st_size for p in outdir.glob("*.json.gz"))
    print(f"[{datetime.now(KST):%Y-%m-%d %H:%M}] target={target} "
          f"ok={ok} fail={fail} skip={skip} dir_gz={total_gz/1e6:.1f}MB")


if __name__ == "__main__":
    main()
