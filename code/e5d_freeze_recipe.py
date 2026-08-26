#!/usr/bin/env python3
"""E5d — 확증 실험용 recipe를 **동결**한다. 미열람 지역을 열기 전 마지막 관문.

이 파일이 곧 사전등록서다. 여기 적힌 것 외의 어떤 선택도 확증 실행에서 하지 않는다.
동결 근거는 전부 M-기록이며, 각 항목에 "왜 이 값인가"를 붙인다.

동결 후 금지 사항:
  - 미열람 지역 결과를 본 뒤 arm·decoder·epoch·임계값·지표를 바꾸는 것
  - 지역별로 다른 설정을 쓰는 것
  - 결과가 나쁜 지역을 제외하는 것
"""
from __future__ import annotations
import hashlib, json, pathlib, subprocess

OUT = pathlib.Path("evidence/recipe_frozen_v1.json")

RECIPE = {
    "schema": "confirmatory-recipe-frozen-v1",
    "frozen_at_commit": None,          # 실행 시 채움
    "status": "PREREGISTERED",

    "arms": {
        "reuse": {
            "id": "P4",
            "desc": "frozen OlmoEarth v1 캐시(tiled 4x64) + 작은 판독기 237,537",
            "why": "M51: china val에서 3/3 seed·두 지표·세 블록 크기 전부 우위, CI 0 제외",
        },
        "raw_strong": {
            "id": "P2",
            "desc": "공식 UNet3D 2,693,121 (raw 학습)",
            "why": "M27 공식 구조 이식. 가장 강한 raw baseline",
        },
        "raw_efficient": {
            "id": "P3",
            "desc": "공식 U-TAE 1,165,409 (raw 학습)",
            "why": "M38: P2보다 7배 싸고 IoU 76%. 비용 축의 진짜 경쟁자",
        },
    },
    "excluded_arms": {
        "P4c": "M43: seed 폭 0.033, micro 우위가 seed 1의 운. M46: 블록 우위도 잡음 안",
        "P4_full / P4c_full": "M37: 문맥 복원이 음의 효과. M41: 고주파 16.4% 손실",
        "P1": "대조 하한. 확증 표에 넣지 않음",
    },

    "protocol": {
        "seeds": [1, 2, 3],
        "why_seeds": "M43: 단일 seed 비교가 결론을 뒤집었다. 3 seed 미만 금지",
        "epochs": 40,
        "batch": 16,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 0.0,
        "why_no_clip": "M49: clip 1.0 + lr 절반이 분산을 26%만 줄이고 평균을 떨어뜨렸다",
        "model_selection": "val IoU 최고 epoch. test는 선택에 사용 금지",
        "timestep": "S12q — SCL clear 상위 12개를 라벨 무관 선택 후 시간순 복원",
        "why_s12q": "v1 time table 12개 한계. 모든 arm이 동일 index",
        "missing_bands": "B01/B09 zeros + band-set 2를 MaskValue.MISSING",
        "time_encoding": "월(month/11) 1채널을 raw arm에 부여",
        "why_month": "M39: 인코더가 월 해상도로 양자화. 날짜 정렬은 없는 문제",
        "threshold": 0.5,
        "why_threshold": "고정. M44에서 FP율 정합 비교를 별도 보조 분석으로 보고",
        "save_probs": True,
        "why_probs": "M44·M45의 전제. 임계값 스윕·불일치 특징에 필수",
    },

    "metrics": {
        "primary": "양성 타일 macro IoU",
        "why_primary": "M40: test 타일의 62.7%가 양성 0. 전체 micro는 사실상 오경보 지표",
        "secondary": ["빈 타일 FP 화소 수", "ECE(15bin)", "AUPRC", "micro IoU"],
        "cost": "샘플당 forward FLOPs + 학습 배율 (M38)",
        "why_cost": "M38: 벽시계는 GPU 경합으로 같은 계산에서 2.14배 차이",
        "reporting": "지표마다 3-seed 평균 ± 폭을 병기. 단일 seed 값 단독 보고 금지",
    },

    "statistics": {
        "ci": "공간 블록 부트스트랩 10,000회, 블록 2.56/5.12/10.24 km 전부 보고",
        "why_blocks": "M33: 타일 i.i.d.는 공간 상관을 무시. M43: seed도 함께 필요",
        "seed_handling": "seed 평균 격차에 대한 CI. seed 재표집은 n=3으로 검정력 없음",
        "noise_floor": "같은 구성 seed쌍의 per-tile oracle gain을 차감 (M41)",
        "forbidden": ["in-sample 임계값 탐색 (M40)", "지표 사후 선택 (M40)",
                      "단일 seed 대 단일 seed 비교 (M43)"],
    },

    "region_release_plan": {
        "development": "chimanimani (test, 다회 노출) + china (val, epoch 선택에 사용)",
        "confirmatory": "미열람 LOCO 지역 — 한 번에 하나씩 순차 공개",
        "rule": "지역을 연 뒤에는 그 지역 수치를 보고 어떤 설정도 바꾸지 않는다",
        "stopping": "3지역 연속으로 reuse가 raw_strong 대비 primary에서 열세면 중단하고 강등",
    },

    "predictions_registered_before_unsealing": {
        "P4_beats_P2_on_primary": "3지역 중 2지역 이상에서 성립할 것으로 예측",
        "why": "M51 china 결과의 외삽. 틀리면 기록하고 주장을 지역 특수로 강등",
        "P4_spread_smaller_than_P2": "고정 임계값 지표에서 성립할 것으로 예측 (M49)",
        "AUPRC_spread_not_smaller": "성립하지 않을 것으로 예측 (M47)",
    },

    "known_limitations_at_freeze_time": [
        "china val은 epoch 선택에 쓰였으므로 완전한 held-out이 아님",
        "P2 불안정이 다른 튜닝 조합으로 고쳐질 가능성은 배제되지 않음 (M49)",
        "task 이질성은 미증명 (M42 kill gate, M46 블록 이질성 잡음)",
        "라벨 없는 승자 예측은 등록 기준 미달 (M45, lift +5.9%p이나 fold 1개 미달)",
    ],
}


def main():
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, check=True).stdout.strip()
        RECIPE["frozen_at_commit"] = c
    except Exception as e:
        RECIPE["frozen_at_commit"] = f"unavailable: {e!r}"
    body = json.dumps(RECIPE, ensure_ascii=False, indent=2, sort_keys=True)
    RECIPE["self_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(RECIPE, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({"frozen_at_commit": RECIPE["frozen_at_commit"],
                      "self_sha256": RECIPE["self_sha256"],
                      "arms": list(RECIPE["arms"]),
                      "primary": RECIPE["metrics"]["primary"],
                      "seeds": RECIPE["protocol"]["seeds"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
