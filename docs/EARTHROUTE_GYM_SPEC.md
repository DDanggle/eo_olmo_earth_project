# EarthRoute Gym v0 — 검증 가능한 Earth tool-use 환경 명세

작성 2026-08-26. **설계 문서임. 이번 사이클에서 LLM 학습은 하지 않음**(사용자 결정).
목적: cache maintenance action 선택을 수학·코드처럼 **자동 검증 가능한 환경**으로 만드는
명세를 고정하는 것. 나중에 SFT ranker(E9)·contextual bandit·offline RL이 같은 환경 위에
올라감.

## 설계 원칙

1. **시나리오는 전부 실측 실패에서 도출함.** 상상한 고장이 아니라 이 저장소의 M-기록이
   잡은 고장만 씀. 합성이되 근거가 있음.
2. **reward는 verifier가 계산함.** 사람 판정·LLM 판정 없음. 전 항목이 기존 감사 코드의
   재사용임.
3. **utility label은 E5 통과본만 자격 있음.** M40의 oracle(지표 불일치·잡음 바닥 미차감)을
   label로 쓰면 오염된 정답을 학습하게 됨(M40·M41).

## 상태(state) 스키마

```json
{
  "task": "landslide | landcover | deforestation | retrieval",
  "region": "chimanimani | korea_cluster_k | nepal_koshi | ...",
  "contract": {
    "bands_present": 10,
    "band_set_mask": [1, 1, 0],
    "cloud_fraction": 0.21,
    "coverage_fraction": 0.999,
    "model_release": "v1 | v1.2",
    "cache_age_days": 45,
    "serving_crop": "tiled_4x64 | full_1x128",
    "scl_clear_top12_mean": 0.91
  },
  "cache_stats": {
    "hf_energy": 2.14,
    "effective_rank": 356.2,
    "token_norm_mean": 9.59
  },
  "available_actions": ["reuse", "adapter", "reembed", "raw", "abstain"],
  "budget": {"gflops": 500000, "storage_gb": 12, "latency_s": 60}
}
```

`cache_stats` 세 값은 `code/diagnose_cache_geometry.py`가 이미 계산함(M41).
`serving_crop`을 상태에 넣는 이유: M37이 **동일 캐시 버전·동일 task에서도 crop 계약에 따라
최적 decoder가 반전됨**(상호작용 −0.0823)을 보였기 때문임. 계약 없는 캐시는 미지정 산출물임.

## 시나리오 카탈로그 — 실측 근거 대응표

| # | 시나리오 | 실측 근거 | 올바른 판정 |
|---|---|---|---|
| S1 | band-set 부재가 **조용히** 무시됨 | M8: v1.2는 `mask[...,0]`만 읽음 | contract violation 검출 → `abstain` 또는 re-embed with v1 |
| S2 | 원천이 반사도가 아니라 8-bit RGB | M28: AI-Hub TS_03 3밴드 uint8 | 입력 거부 → 12밴드 재물질화 요구 |
| S3 | 해당 날짜 STAC item 없음 | M31: 2,699 중 149건 (5.5%) | `abstain` 또는 인접일 탐색(사전 등록된 창 안에서만) |
| S4 | coverage 미달을 성공으로 집계 | M35: 24.6%가 격자 밖 0-채움, 100% 빈 큐브 포함 | coverage 검사 실패 → 모자이크 재시도 |
| S5 | 구름 초과 | cc>60 11건 (M31) | 기록 후 제외 (사전 등록 임계값) |
| S6 | release 변경 v1→v1.2 | M1: R@1=0, M7: 로딩 환경 분리 | stale cache 사용 금지 → re-embed 또는 raw |
| S7 | serving crop 계약 변경 | M34: seam 비율 1.49, M37: 성능 반전 | 계약 불일치 검출 → 같은 계약으로 재계산 |
| S8 | timestamp 중복/월 경계 | M39: 월 양자화, 중복 시 wrapper 거부 | 시점 계약 정규화 |
| S9 | API 일시 실패·타임아웃 | 세션 실측: SSH/터널 단절, numOfRows 타임아웃 오판 사례 | 크기 축소 재시도, "차단" 단정 금지 |
| S10 | 라벨 출처가 평가 모델과 같은 계열 | Nepal Koshi: U-Net 자동탐지+수동보정 | silver label 표기, gold 주장 금지 |

## Verifier 체크리스트 — 전 항목 기존 코드 재사용

| 검사 | 자동화 근거 (기존 코드) |
|---|---|
| 12밴드 존재·순서 | `materialize_aihub_s2_12band.py`의 BANDS 계약, `inspect_materialized_s2.py` |
| coverage ≥ 사전 등록 임계값 | `audit_aihub_cubes.py` (all-band-zero fraction) |
| CRS·transform·checksum | `seal_aihub_data_contract.py`, manifest SHA-256 |
| event cutoff 이전 관측 | `sample_contract.jsonl`의 `pre_post_valid`·`s_cutoff_positive_eligible` |
| split 누수 없음 | `audit_sen12_fold_cache.py` (해시 3/3), `tests/test_aihub_split_invariants.py` |
| 선택 action의 utility 개선 | action matrix(E6)의 사전 등록 지표 Δ − λ·FLOPs |
| 연산·저장·지연 예산 | `measure_flops_cost.py` (경합 불변), 저장은 파일 크기 실측 |
| 결정성 | replay diff 0.0 계약 (M26·M38 보강: 타 세션 비트 일치 재현) |

## reward

```
reward = Δ(사전등록 지표; action vs reuse) − λ_c·FLOPs(action) − λ_s·storage − 위반 페널티
```

- λ는 실험 전 문서로 고정함 (E6에서 등록)
- 위반 페널티: verifier 실패 항목당 고정 감점. **위반을 감추고 성능을 얻는 경로를 차단**함
- `abstain`의 reward = 0 − 소액 기회비용. 잘못된 action보다 abstain이 낫다는 부호 관계를 보존함

## 이 환경이 기존 벤치와 다른 점 (기여 후보)

- EarthShift는 shift에서의 **성능 저하 측정**까지임. 여기서는 저하가 아니라 **행동 선택**이
  평가 대상임.
- 수학·코드 RLVR처럼 정답이 검증 가능하지만, 정답의 원천이 **실측된 운영 고장 목록**이라는
  점이 다름. 시나리오마다 M-기록 번호가 붙음.

## 아직 정하지 않은 것

- λ 값들 (E6에서 utility 분포를 본 뒤 등록함 — 단 결과를 보고 고르는 것을 막기 위해
  후보 3개를 먼저 적고 그중에서 고름)
- retrieval task(E7a)의 reward 지표 (Recall@K vs nDCG)
- multi-step 확장(cache 나이가 시간에 따라 증가) — v0는 단일 단계로 고정함
