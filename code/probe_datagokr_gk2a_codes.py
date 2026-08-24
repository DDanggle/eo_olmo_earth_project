#!/usr/bin/env python3
"""GK2A 산출물별 resultType 코드 탐색 — 1단계에서 CLD가 통한 것을 일반화한다.

확정된 사실 (probe_datagokr_gk2a.py):
  dateTime  = YYYYMMDDHHMM (10분 간격, 최근 시각)
  resultType = 출력 형식이 아니라 **산출물 변수 코드**다. getGk2acldAll + CLD -> code 00.
  키는 Encoding 형태를 그대로 쿼리스트링에 넣는다 (재인코딩하면 안 된다).

이 스크립트는 나머지 9개 오퍼레이션의 변수 코드를 찾는다. 한반도 계열(...All)과
행정구역 계열(...Area, dongCode 추가 필요)을 나눠 시도한다.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://apis.data.go.kr/1360000/CloudSatlitInfoService"
KST = timezone(timedelta(hours=9))
OUT = Path("artifacts/datagokr_gk2a_codes.json")

# 오퍼레이션별 변수 코드 후보. GK2A L2 산출물 약어를 넓게 잡는다.
CANDIDATES = {
    "cld":   ["CLD", "CF", "CLDMASK", "CLOUD"],
    "apps":  ["AOD", "APPS", "AE", "ADP", "AOT", "AI", "AEROSOL", "SSA", "FMF"],
    "fog":   ["FOG", "FF", "FOGMASK", "FOGD"],
    "dcoew": ["DCOEW", "COT", "CER", "CWP", "CLD", "DCW", "CTT"],
    "cla":   ["CLA", "CTT", "CTH", "CTP", "CLP", "CTYPE", "CLDTYPE", "CT"],
}
# 서울 종로구 청운효자동 등 실제 행정동 코드 후보 (dongCode)
DONG_CODES = ["1111051500", "1111000000", "1100000000", "4113510900", "1168010100"]


def now_slot(hours_back: int = 3) -> str:
    t = datetime.now(KST).replace(second=0, microsecond=0)
    t = t.replace(minute=(t.minute // 10) * 10) - timedelta(hours=hours_back)
    return t.strftime("%Y%m%d%H%M")


def call(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f'{{"response":{{"header":{{"resultCode":"EXC","resultMsg":"{type(exc).__name__}"}}}}}}'


def code_msg(body: str) -> tuple[str, str]:
    c = re.search(r'"resultCode"\s*:\s*"?([A-Z0-9]+)', body)
    m = re.search(r'"resultMsg"\s*:\s*"([^"]*)', body)
    return (c.group(1) if c else "?", (m.group(1) if m else body[:60]))


def main() -> None:
    key = os.environ.get("DATA_GO_KR_SERVICE_KEY", "")
    if not key:
        raise SystemExit("DATA_GO_KR_SERVICE_KEY가 비어 있다.")
    dt = now_slot()
    result: dict = {"schema": "datagokr-gk2a-codes-v1", "dateTime_used": dt,
                   "confirmed": {}, "attempts": []}

    for prod, codes in CANDIDATES.items():
        # --- 한반도 계열 ---
        op = f"getGk2a{prod}All"
        found = None
        for rt in codes:
            body = call(f"{BASE}/{op}?ServiceKey={key}&pageNo=1&numOfRows=1"
                        f"&dataType=JSON&dateTime={dt}&resultType={rt}")
            c, m = code_msg(body)
            result["attempts"].append({"op": op, "resultType": rt, "code": c, "msg": m})
            print(f"  {op:18s} resultType={rt:8s} code={c} {m[:40]}", flush=True)
            if c == "00":
                found = rt
                result["confirmed"][op] = {"resultType": rt, "sample": body[:400]}
                break
        if found is None:
            print(f"  {op:18s} -> 코드 미발견 (후보 {len(codes)}개 소진)", flush=True)

        # --- 행정구역 계열: 확정된 코드로 dongCode를 찾는다 ---
        area_op = f"getGk2a{prod}Area"
        rt = found or codes[0]
        for dc in DONG_CODES:
            body = call(f"{BASE}/{area_op}?ServiceKey={key}&pageNo=1&numOfRows=1"
                        f"&dataType=JSON&dateTime={dt}&resultType={rt}&dongCode={dc}")
            c, m = code_msg(body)
            result["attempts"].append({"op": area_op, "resultType": rt, "dongCode": dc,
                                       "code": c, "msg": m})
            if c == "00":
                print(f"  {area_op:18s} resultType={rt} dongCode={dc} -> code=00", flush=True)
                result["confirmed"][area_op] = {"resultType": rt, "dongCode": dc,
                                                "sample": body[:400]}
                break
        else:
            print(f"  {area_op:18s} -> dongCode 미발견", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"\n확정 {len(result['confirmed'])}/10 → {OUT}")
    for op, v in sorted(result["confirmed"].items()):
        print(f"  {op:18s} resultType={v['resultType']}"
              + (f" dongCode={v['dongCode']}" if "dongCode" in v else ""))
    print("DONE")


if __name__ == "__main__":
    main()
