#!/usr/bin/env python3
"""GK2A 일일 스냅샷 — 보존기간 2일 자료를 영구 축적함.

왜 지금 시작하는가: 이 프로그램에서 **되돌릴 수 없는 것은 이것 하나**임.
GK2A 경량화 API는 "최대 조회 기간은 오늘 기준으로 2일 전까지"임(M-기록 참조).
즉 오늘 받지 않은 날짜는 **영구히 사라짐.** 실험 설계·모델 학습은 나중에 해도 되지만
이 수집은 미루면 그만큼 자산이 없어짐.

실행 위치: 로컬·서버 어디서든 됨. (2026-08-25 정정: 처음에 "서버에서 차단됨"이라고 적었는데
틀렸음. `numOfRows=200000`으로 요청해 API의 동시호출 락이 걸려 있던 것이었음 —
`resultCode 99 "이미 호출중에 있습니다"`와 같은 증상임. 고친 뒤 서버에서 3/3 정상 응답함.)

**얼마나 자주 돌려야 하는가**: 보존이 2일이므로 **2일에 한 번**이면 충분함. 매일 아닐도 됨.
인자 없이 돌리면 어제분을 받고, `--gaps`를 주면 아직 받을 수 있는 범위(D-1, D-2)에서
빠진 것만 보충함.

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

# M-기록에서 확정한 (오퍼레이션, resultType) 조합과 **유효 시간대**.
#
# 2026-08-24분을 시각별로 검사한 결과 (유효 픽셀 비율):
#            00   03   06   09   12   18   21
#   CLD     100  100  100  100  100  100  100   ← 전천후
#   FOG     100  100  100  100  100  100  100   ← 전천후
#   AOD       0    0    0   33   27   20    0   ← 주간 전용
#   COT       0    0    0   52   60   63    0   ← 주간 전용
#   CT        0    0    0   57   64   66    0   ← 주간 전용
#
# AOD·COT·CT는 태양광 반사가 필요해 야간에는 격자 전체가 -9999임. 3시간 격자로는
# 주간 산출물을 8슬롯 중 3개만 잡았음. 산출물별로 시간대를 따로 잡음.
DAY_HOURS   = list(range(8, 19))          # 08~18 KST. 06시는 태양고도가 낮아 실패함
ALLDAY_HOURS = list(range(0, 24, 2))      # 2시간 간격
PRODUCTS = [
    ("getGk2acldAll",   "CLD", ALLDAY_HOURS),
    ("getGk2afogAll",   "FOG", ALLDAY_HOURS),
    ("getGk2aappsAll",  "AOD", DAY_HOURS),
    ("getGk2adcoewAll", "COT", DAY_HOURS),
    ("getGk2aclaAll",   "CT",  DAY_HOURS),
]
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
    if not os.environ.get("DATA_GO_KR_SERVICE_KEY"):
        print("DATA_GO_KR_SERVICE_KEY 없음", file=sys.stderr)
        raise SystemExit(2)

    today = datetime.now(KST).date()
    # 보존 2일이므로 받을 수 있는 날짜는 D-1, D-2뿐임.
    if len(sys.argv) > 1 and sys.argv[1] == "--gaps":
        targets = [today - timedelta(days=d) for d in (1, 2)]
    elif len(sys.argv) > 1 and sys.argv[1] == "--status":
        report_status()
        return
    elif len(sys.argv) > 1:
        targets = [datetime.strptime(sys.argv[1], "%Y%m%d").date()]
    else:
        targets = [today - timedelta(days=1)]

    for target in targets:
        collect(target)
        collect_anchors(target)


def report_status() -> None:
    """쌓인 상태를 보여줌. 빠진 날짜와 총량을 냄."""
    days = sorted(p for p in ROOT.glob("*/*/*") if p.is_dir())
    if not days:
        print(f"수집 없음. ROOT={ROOT}")
        return
    total_files = total_bytes = 0
    rows = []
    for d in days:
        gz = list(d.glob("*.json.gz"))
        b = sum(f.stat().st_size for f in gz)
        total_files += len(gz); total_bytes += b
        rows.append((f"{d.parent.parent.name}-{d.parent.name}-{d.name}", len(gz), b))
    first = datetime.strptime(rows[0][0], "%Y-%m-%d").date()
    last = datetime.strptime(rows[-1][0], "%Y-%m-%d").date()
    have = {r[0] for r in rows}
    missing = [str(first + timedelta(days=i)) for i in range((last - first).days + 1)
               if str(first + timedelta(days=i)) not in have]
    print(f"ROOT      {ROOT}")
    print(f"기간      {first} ~ {last}  ({len(rows)}일 수집, 빠진 날 {len(missing)}일)")
    print(f"총량      파일 {total_files}개 / {total_bytes/1e6:.1f} MB")
    print(f"하루 평균 {total_bytes/max(1,len(rows))/1e6:.2f} MB  → 1년 추정 {total_bytes/max(1,len(rows))*365/1e9:.2f} GB")
    if missing:
        print(f"빠진 날   {', '.join(missing[:12])}{' …' if len(missing) > 12 else ''}")
        print("          (D-1/D-2 안에 있는 것만 --gaps 로 보충 가능. 그보다 오래된 것은 영구 소실)")
    for name, n, b in rows[-7:]:
        print(f"  {name}  파일{n:3d}  {b/1e6:5.2f} MB")


# Area(행정구역) 앵커. 격자 좌표계 확정과 즉시 사용 가능한 residual 둘 다에 쓰임.
# 목록 파일이 있으면 그것을 씀 — 행정동코드 한 줄씩. 없으면 검증된 4개만 씀.
ANCHOR_FILE = Path(os.environ.get("GK2A_ANCHORS", str(ROOT / "_crs" / "dongcodes.txt")))
DEFAULT_ANCHORS = ["1111051500", "1114052000", "2611051000", "3111051000"]


def anchors() -> list[str]:
    if ANCHOR_FILE.exists():
        codes = [l.strip() for l in ANCHOR_FILE.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
        if codes:
            return codes
    return DEFAULT_ANCHORS


def collect_anchors(target) -> None:
    """행정구역(Area) 계열 수집. lon/lat이 함께 오므로 좌표계 없이도 바로 쓸 수 있음.

    이 자료도 2일 보존이므로 격자와 같은 날 받아야 짝이 맞음.
    """
    key = os.environ["DATA_GO_KR_SERVICE_KEY"]
    codes = anchors()
    outdir = ROOT / "_crs"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "area_anchors.jsonl"
    seen = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                seen.add((r["dateTime"], r["dong"], r.get("resultType", "CLD")))
    ok = skip = fail = 0
    with path.open("a", encoding="utf-8") as f:
        for hh in ALLDAY_HOURS:
            dt = f"{target:%Y%m%d}{hh:02d}00"
            for op, rt in (("getGk2acldArea", "CLD"), ("getGk2afogArea", "FOG")):
                for dong in codes:
                    if (dt, dong, rt) in seen:
                        skip += 1
                        continue
                    u = (f"{BASE}/{op}?ServiceKey={key}&pageNo=1&numOfRows=1"
                         f"&dataType=JSON&dateTime={dt}&resultType={rt}&dongCode={dong}")
                    status, body = fetch(u)
                    time.sleep(SLEEP_S)
                    try:
                        d = json.loads(body.decode())
                        if d["response"]["header"]["resultCode"] != "00":
                            fail += 1
                            continue
                        it = ((d["response"].get("body") or {}).get("items")
                              or {}).get("item") or [{}]
                        it = it[0]
                        f.write(json.dumps({"dateTime": dt, "dong": dong,
                                            "resultType": rt, "lon": float(it["lon"]),
                                            "lat": float(it["lat"]),
                                            "value": it.get("value")},
                                           ensure_ascii=False) + "\n")
                        f.flush()
                        ok += 1
                    except Exception:  # noqa: BLE001
                        fail += 1
    print(f"    앵커 {len(codes)}개 · ok={ok} skip={skip} fail={fail}")


def collect(target) -> None:
    key = os.environ.get("DATA_GO_KR_SERVICE_KEY", "")
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
    jobs = [(hh, op, rt) for op, rt, hours in PRODUCTS for hh in hours]
    jobs.sort()
    with manifest.open("a", encoding="utf-8") as mf:
        for hh, op, rt in jobs:
            dt = f"{target:%Y%m%d}{hh:02d}00"
            if True:
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
