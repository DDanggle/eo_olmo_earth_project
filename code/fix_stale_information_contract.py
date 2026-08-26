#!/usr/bin/env python3
"""이미 봉인된 pilot JSON의 stale `information_contract`를 **덮어쓰지 않고 주석 추가**한다.

봉인 산출물을 사후 수정하면 그것이 곧 재현성 위반이다. 그래서 원본 필드를 지우지 않고
`information_contract_correction_2026_08_26` 키를 나란히 추가한다. 원본이 무엇이었는지와
왜 틀렸는지가 같은 파일 안에 남는다.

근거: M39(인코더 월 해상도 양자화, 날짜 이동 시 5/5 비트 동일),
      forward()의 timestamp parity 처리, 로그의 "months 로드 6834개".
"""
from __future__ import annotations
import json, pathlib, sys

CORRECTION = {
    "corrected_at": "2026-08-26",
    "what_was_wrong": [
        "known_mismatch='P4 encoder received acquisition timestamps; P1/P2 only receive order'",
        "claim_status='not timestamp-matched'",
    ],
    "why_wrong": [
        "P1/P2/P3는 order가 아니라 월(month/11) 1채널을 받는다 (forward() timestamp parity).",
        "M39 실측: 월을 보존한 채 날짜를 ±1~3일 옮기면 임베딩이 5/5 비트 단위 동일하다. "
        "인코더는 시간을 월 해상도로 양자화하므로 P4도 날짜를 쓰지 않는다.",
    ],
    "corrected_statement": {
        "information_parity": True,
        "encoder_time_resolution": "month",
        "raw_arm_time_input": "month-of-year scalar channel (month/11)",
        "residual_asymmetry": "encoding form only (sinusoidal PE vs broadcast scalar)",
        "claim_status": "information-matched at encoder time resolution",
    },
    "policy": "원본 필드는 삭제하지 않는다. 봉인 산출물의 사후 수정은 재현성 위반이므로 "
              "정정을 나란히 기록한다.",
    "evidence": ["MEASURED_FINDINGS.md M39", "code/probe_timestamp_asymmetry.py",
                 "evidence/timestamp_asymmetry.json"],
}
KEY = "information_contract_correction_2026_08_26"


def main() -> None:
    roots = [pathlib.Path(a) for a in sys.argv[1:]] or [pathlib.Path("evidence")]
    hit = 0
    for root in roots:
        for p in sorted(root.rglob("*_pilot.json")):
            txt = p.read_text(encoding="utf-8")
            if "only receive order" not in txt:
                continue
            d = json.loads(txt)
            if KEY in d:
                print(f"  이미 정정됨: {p}")
                continue
            d[KEY] = CORRECTION
            p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"  정정 추가: {p}")
            hit += 1
    print(f"총 {hit}개 파일에 정정 주석 추가 (원본 필드 보존)")


if __name__ == "__main__":
    main()
