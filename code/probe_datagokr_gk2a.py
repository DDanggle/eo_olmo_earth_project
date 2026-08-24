#!/usr/bin/env python3
"""data.go.kr 기상청 CloudSatlitInfoService(천리안2A 기상산출물) 파라미터 조합 탐색.

배경: 오퍼레이션 10개는 확보했고 인증도 통과했다 (resultCode 11 = 서비스 자체 응답,
게이트웨이의 code 30/12가 아니다). 남은 것은 `dateTime` 형식과 `resultType` 허용값이며
포털 명세가 둘 다 "참고 참조"로만 적어두었다.

포털 안내: "Encoding/Decoding 된 인증키를 적용하면서 구동되는 키를 사용" — 두 형태를 모두 시도한다.

성공 판정: resultCode가 11(필수파라미터 누락)이 아닌 응답. 00이면 정상, 그 외 코드는
무엇이 잘못됐는지에 대한 새 정보다.

키는 인자로 받지 않는다. 환경변수 DATA_GO_KR_SERVICE_KEY만 읽는다.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

BASE = "https://apis.data.go.kr/1360000/CloudSatlitInfoService"
OPS = [
    "getGk2acldAll", "getGk2aappsAll", "getGk2afogAll", "getGk2adcoewAll", "getGk2aclaAll",
    "getGk2acldArea", "getGk2aappsArea", "getGk2afogArea", "getGk2adcoewArea", "getGk2aclaArea",
]
# 천리안2A 산출물별로 다른 출력변수 이름이 있을 수 있으므로 후보를 넓게 잡는다.
RESULT_TYPES = ["", "json", "xml", "img", "txt", "CLD", "cld", "CF", "AOD", "apps",
                "fog", "dcoew", "cla", "1", "0", "all", "A", "R"]
KST = timezone(timedelta(hours=9))


def datetime_candidates() -> list[str]:
    """관측시간 후보. GK2A는 10분 간격 산출물이 흔하므로 최근 시각을 내려가며 만든다."""
    now = datetime.now(KST).replace(second=0, microsecond=0)
    now = now.replace(minute=(now.minute // 10) * 10)
    out = []
    for hours in (2, 3, 6, 12, 24):
        t = now - timedelta(hours=hours)
        out += [t.strftime("%Y%m%d%H%M"), t.strftime("%Y%m%d%H"), t.strftime("%Y%m%d")]
    # 중복 제거, 순서 유지
    return list(dict.fromkeys(out))


def call(url: str, timeout: int = 20) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"<<EXC {type(exc).__name__}: {exc}>>"


def code_of(body: str) -> str:
    m = re.search(r'"?resultCode"?\s*[:>]\s*"?(\d+)', body) or \
        re.search(r"returnReasonCode\"?\s*[:>]\s*\"?(\d+)", body)
    return m.group(1) if m else "?"


def msg_of(body: str) -> str:
    m = re.search(r'"?(?:resultMsg|errMsg)"?\s*[:>]\s*"?([A-Z_ ]+)', body)
    return m.group(1).strip() if m else body[:80].replace("\n", " ")


def main() -> None:
    enc = os.environ.get("DATA_GO_KR_SERVICE_KEY", "")
    if not enc:
        raise SystemExit("DATA_GO_KR_SERVICE_KEY가 비어 있다.")
    dec = urllib.parse.unquote(enc)
    key_forms = [("encoded", enc)] + ([("decoded", dec)] if dec != enc else [])
    dts = datetime_candidates()

    findings: list[dict] = []
    print(f"오퍼레이션 {len(OPS)} × 키형태 {len(key_forms)} × dateTime {len(dts)} × resultType {len(RESULT_TYPES)}")

    # 1단계: 대표 오퍼레이션 하나로 (키형태 × dateTime × resultType)을 훑어 유효 조합을 찾는다.
    probe_op = OPS[0]
    hit = None
    for (kname, k), dt, rt in itertools.product(key_forms, dts, RESULT_TYPES):
        q = f"ServiceKey={k}&pageNo=1&numOfRows=1&dataType=JSON&dateTime={dt}"
        if rt:
            q += f"&resultType={rt}"
        body = call(f"{BASE}/{probe_op}?{q}")
        c = code_of(body)
        if c != "11":
            rec = {"op": probe_op, "key_form": kname, "dateTime": dt, "resultType": rt,
                   "resultCode": c, "resultMsg": msg_of(body), "body_head": body[:300]}
            findings.append(rec)
            print(f"  [변화] dt={dt} rt={rt!r} key={kname} -> code={c} {rec['resultMsg']}")
            if c == "00" and hit is None:
                hit = (kname, k, dt, rt)
                break

    if hit is None:
        print("\n유효 조합을 찾지 못했다. 11 이외의 코드가 나온 조합만 위에 기록했다.")
        print("참고문서의 dateTime 형식·resultType 허용값 코드표가 필요하다.")
    else:
        kname, k, dt, rt = hit
        print(f"\n유효 조합: key={kname} dateTime={dt} resultType={rt!r}")
        # 2단계: 같은 조합으로 10개 오퍼레이션 전부 확인한다.
        for op in OPS:
            q = f"ServiceKey={k}&pageNo=1&numOfRows=1&dataType=JSON&dateTime={dt}"
            if rt:
                q += f"&resultType={rt}"
            body = call(f"{BASE}/{op}?{q}")
            c = code_of(body)
            print(f"  {op:20s} code={c} {msg_of(body)}")
            findings.append({"op": op, "key_form": kname, "dateTime": dt, "resultType": rt,
                             "resultCode": c, "resultMsg": msg_of(body), "body_head": body[:300]})

    print("\n" + json.dumps({"n_findings": len(findings), "findings": findings[:40]},
                            ensure_ascii=False, indent=1)[:4000])
    print("DONE")


if __name__ == "__main__":
    main()
