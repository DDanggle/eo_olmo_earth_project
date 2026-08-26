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

OUT = pathlib.Path("evidence/recipe_frozen_v2.json")

RECIPE = {
    "schema": "confirmatory-recipe-frozen-v2",
    "frozen_at_commit": None,          # 실행 시 채움
    "status": "PREREGISTERED",
    "supersedes": "recipe_frozen_v1.json (commit e214218)",
    "why_v2": (
        "v1은 (a) 확증 지역 순서, (b) '승리' 정의, (c) 집계 방식, (d) 중단 규칙, "
        "(e) freeze 경계의 의미를 명시하지 않았다. 외부 감사가 이를 지적했고 사실이었다. "
        "v2가 그 다섯 개를 못 박는다."),
    "preregistration_boundary_statement": (
        "thrissur 실행은 v1 하에서 **시작**됐고 v2 커밋 시점에 9실행 중 1개(P4 seed 1)가 "
        "완료돼 있었다. 그러나 **어떤 결과 수치도 읽지 않은 상태**에서 v2를 커밋했다. "
        "사전등록의 기준은 데이터 수집 이전이 아니라 **결과 관찰 이전**이므로, "
        "v2의 승리 정의·집계·중단 규칙은 thrissur에 대해서도 사전등록으로 유효하다. "
        "이 문장 자체가 그 주장의 증거이며, git 커밋 시각이 검증 수단이다."),

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
        # 순서는 HEADLINE_REGIONS 알파벳 순서에서 개발용 2개를 제외한 것이다.
        # 결과를 보고 순서를 바꾸지 않기 위해 **여기에 전부 못 박는다**.
        "release_order": ["holdout_thrissur", "holdout_hiroshima", "holdout_hokkaido",
                          "holdout_indonesia", "holdout_itogon", "holdout_kyrgyzstan1",
                          "holdout_kyrgyzstan2", "holdout_newzealand"],
        "first_three": ["holdout_thrissur", "holdout_hiroshima", "holdout_hokkaido"],
        "why_this_order": "thrissur는 이미 개봉했으므로 첫 자리. 나머지는 알파벳 순서 "
                          "고정으로 선택 여지를 없앤다",
        "one_at_a_time": True,
        "rule": "지역을 연 뒤에는 그 지역 수치를 보고 어떤 설정도 바꾸지 않는다",
    },

    "win_definition": {
        # 감사 지적 반영: '승리'가 평균 양수인지 CI 제외인지 불명확했다. 여기서 고정한다.
        "per_region_win": "seed-mean primary(양성 macro)에서 reuse > raw_strong "
                          "**그리고** 3 seed 전부에서 reuse > raw_strong",
        "why_both": "M52: C_large·상호작용이 평균은 음수인데 seed 3에서 부호가 뒤집혔다. "
                    "평균만으로는 재현 가능한 효과를 식별할 수 없다",
        "per_region_ci": "공간 블록 부트스트랩 2.56/5.12/10.24 km 전부 보고. "
                         "CI 0 제외는 '강한 승리'로 별도 표기하되 승리 판정의 필수조건은 아님",
        "headline_aggregate": "8개 held-out 지역의 **region-macro 평균**(지역마다 동일 가중). "
                              "타일 수로 가중하지 않는다 — 큰 지역이 결론을 지배하지 않도록",
        "no_exclusion": "중단하더라도 **이미 본 지역은 최종 평균에서 제외하지 않는다**. "
                        "나쁜 지역을 빼는 것이 optional stopping의 가장 흔한 형태다",
    },

    "stopping_rule": {
        "condition": "첫 3지역 중 reuse가 raw_strong에 대해 per_region_win을 "
                     "**1지역 이하**에서만 달성하면 중단",
        "on_stop": "reuse 우위 주장을 '지역 특수'로 강등하고 논문 축을 계약·감사로 이동. "
                   "이미 개봉한 지역 결과는 전부 보고한다",
        "on_continue": "2지역 이상이면 남은 5지역을 순서대로 개봉",
    },

    "predictions_registered_before_unsealing": {
        "P4_beats_P2_on_primary": "first_three 중 2지역 이상에서 per_region_win 달성 예측",
        "why": "M51 china 결과의 외삽. 틀리면 기록하고 주장을 지역 특수로 강등",
        "P4_spread_smaller_than_P2": "고정 임계값 지표에서 성립할 것으로 예측 (M49)",
        "AUPRC_spread_not_smaller": "성립하지 않을 것으로 예측 (M47) — 일부러 넣은 실패 예측",
        "block_heterogeneity_stays_noise": "확증 지역에서도 블록 단위 결정적 action은 "
                                           "10% 미만일 것으로 예측 (M46·M52 재계산: 69블록 중 1개)",
    },

    "freeze_boundary": {
        # 감사 지적 반영: frozen_at_commit은 파일 생성 시 HEAD일 뿐이며
        # 당시 worktree가 clean했음을 증명하지 않는다. 공식 경계는 git blob이다.
        "note": "공식 freeze 경계는 이 파일이 처음 git에 들어간 커밋의 blob이다",
        "first_committed_at": "e214218",
        "blob_hash_command": "git rev-parse HEAD:evidence/recipe_frozen_v1.json",
        "frozen_at_commit_meaning": "파일 생성 시점의 HEAD (worktree 상태 증명 아님)",
    },

    "known_limitations_at_freeze_time": [
        "china val은 epoch 선택에 쓰였으므로 완전한 held-out이 아님",
        "P2 불안정이 다른 튜닝 조합으로 고쳐질 가능성은 배제되지 않음 (M49)",
        "task 이질성은 미증명 (M42 kill gate, M46 블록 이질성 잡음)",
        "라벨 없는 승자 예측은 등록 기준 미달 (M45, lift +5.9%p이나 fold 1개 미달)",
        "P4 > P3 격차 0.0132는 양쪽 seed 폭(0.0397/0.0484) 안이며 paired CI 미측정. "
        "'P4가 P3보다 우수'로 승격 금지 — 현재는 '관측 평균 1위'까지만",
        "reuse가 '가장 싸다'는 것은 **warm cache 기준**임. cold start 단일 task에서는 "
        "인코더 비용 때문에 raw_efficient(U-TAE)가 더 쌀 수 있음 (M38: 손익분기 8~12 task)",
        "블록 단위 action 이질성은 3-seed 재계산 후에도 69블록 중 결정적 1개뿐 (M46 후속)",
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
