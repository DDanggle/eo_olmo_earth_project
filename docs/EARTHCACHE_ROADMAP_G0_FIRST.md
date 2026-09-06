# EarthCache 로드맵 — G0 먼저, 그다음 GEO-Bench (2026-09-06)

이 문서는 "G0부터 빡세게 해서 GEO-Bench를 올리는 큰 그림"이다. 상위 SSOT는
`config/geobench_cache_action_prereg_v0.json`(기계 판독 계약)과
`docs/EARTHCACHE_GEOBENCH_UPGRADE_2026_09_06.md`(novelty·설계).

## 0. 한 문장

> 이미 materialize된 EO 캐시의 모델·release·입력계약과 실측 비용을 근거로, held-out
> 과업/family에서 **재사용(CACHED_HEAD) / 적응(HEAD_ADAPT) / 재계산(REEMBED) / 라벨요청**의
> 상대 이득을 test label 없이 예측하고, best-static 정책보다 GEO-Bench score–cost frontier를
> 개선한다. 개선이 없으면 정직하게 EarthCacheBench 특성화 논문으로 간다.

## 1. 지금 어디 서 있나 — G0를 실제로 돌린 결과

개발 과업 2개(Sen12 MS-97, Solar MS-99)의 **실측 region-macro 평균**을 G0에 넣었다.
`artifacts/g0_dev/g0_dev_report.json`.

| 과업 | CACHED_HEAD | HEAD_ADAPT | REEMBED | 최고 action |
|---|---:|---:|---:|---|
| Sen12 (K=5) | .258 | **.294** | .179 | HEAD_ADAPT |
| Solar (K=5) | **.591** | .582 | .240 | CACHED_HEAD |

G0 판정:

| 예산 | best_static | oracle headroom | headroom≥.02 | 역전군≥2 | 종합 |
|---|---|---:|:--:|:--:|:--:|
| unlimited | HEAD_ADAPT | **+0.013** | ✗ | ✗ | fail |
| no_reembed (≤300s) | CACHED_HEAD | +0.500 | ✓ | ✗ | fail |
| reuse_only (≤75s) | CACHED_HEAD | 0.000 | ✗ | ✗ | fail |

**`budget_static_winner_crossover = True`**: 예산이 바뀌면 최고 정적 action이 HEAD_ADAPT→CACHED_HEAD로 바뀐다.

### 읽기 (정직하게)

1. **점수만 보면 selector는 아직 정당화되지 않는다.** unlimited에서 HEAD_ADAPT가 거의 oracle
   (headroom .013 < .02). "항상 적응"이 매우 강한 정적 정책이다. 이게 냉정한 사실이다.
2. **그러나 selector 가치의 자리가 드러났다 — 비용 축이다.** REEMBED를 예산으로 빼면 답이
   CACHED_HEAD로 뒤집히고, Sen12(적응)↔Solar(재사용) 역전이 실재한다. HEAD_ADAPT는 라벨과
   학습이 더 비싸므로, 빡빡한 예산에서는 CACHED_HEAD(라벨 0, warm read)가 frontier를 이길 수 있다.
3. **2 과업으로는 검정력이 없다.** G0는 독립 역전군 ≥2를 요구하는데 우리에겐 역전 1개뿐이다.
   → GEO-Bench가 풀어야 할 것이 정확히 이것이다: **더 많은 독립 과업에서 (a) 재사용↔적응 역전이
   체계적인가, (b) 실측 비용이 Pareto 교차를 만드는가.**

## 2. 3단계 큰 그림

```
[G0] 필요조건 게이트        ── 지금~데이터 준비. "고를 이유가 있나"를 먼저 측정.
      ↓ 통과(역전≥2 또는 비용 Pareto 교차 + headroom≥.02)   ↓ 불통과
[S] 선택기 구축·검증        [A] EarthCacheBench 특성화 논문
      ↓                        (지금 있는 걸로 이미 성립)
[E] 외부 검증 (한국 3-task = 추가 사례, 유일한 시험 아님)
```

## 3. G0 완결 계획 — action matrix를 채운다 (임계 경로)

### 3.1 과업 (Core-6 + 개발 2)

| 역할 | 과업 | 유형 | 상태 |
|---|---|---|---|
| 개발 | Sen12 Landslide | dense seg | ✅ matrix 있음 |
| 개발 | Solar Farm | dense seg | ✅ matrix 있음 |
| 외부 | PASTIS | 시계열 seg (S2 10밴드, 우리 계약) | 데이터 수신 |
| 외부 | DynamicEarthNet | 멀티모달 seg (S2+Planet) | ✅ 확보·검증 |
| 외부 | Fields of the World | 계약이동 seg (4밴드·2시점) | ✅ 확보·검증 |
| 외부 | **BigEarthNet-v2** | **classification** (편향 해소) | 미확보 |
| 외부 | **BioMassters** | **regression** (편향 해소) | 미확보 |
| 외부 | CloudSEN12 | seg (S1+S2) | 미확보 |

segmentation 편향을 깨려면 classification·regression을 반드시 하나씩 넣는다. 단 유형이 다르면
**head 종류가 다르므로**(dense mask head vs 전역 분류 head vs 회귀 head), 각 episode 메타에
`head_type`을 명시한다. G0의 episode-내 정규화가 유형 차이를 흡수하지만 기록은 남겨야 한다.

### 3.2 각 과업에서 잴 것 (action × 예산 × 시드)

action = {CACHED_HEAD, HEAD_ADAPT, REEMBED} (PEFT_REEMBED는 천장 arm, novelty 아님).
예산 = {Z0/K5/K20} × BUDGET envelope. 시드 = 3 (승격 셀만).

**비용 절감 규칙**: 개발 2과업에서 REEMBED(raw 재학습)가 **둘 다 최하위**였다. 이 지배가 외부
과업에서도 유지되면, REEMBED는 과업당 1회만 확증하고 selector의 실질 결정을
**CACHED_HEAD vs HEAD_ADAPT**로 좁힌다 → GPU 부하 절반. (REEMBED가 어느 과업에서 살아나면
그 자체가 중요한 발견 — release 전환·계약 이동 과업에서 그럴 수 있다.)

### 3.3 실측 비용 벡터 (prereg 정의, 추정 아님)

cold encoder GPU초 · warm cache-read초 · head/adapt GPU초 · raw bytes read · cache bytes written
· cache invalidation bytes. §1의 G0는 **추정 비용**을 썼다 — 여기서 실측으로 대체한다.
이게 G3(비용 게이트)의 근거이자 selector 가치의 핵심 축이다.

### 3.4 G0 판정 (사전 고정, `config` 기준)

- 통과: 독립 과업/배포군 ≥2에서 최고 action 역전(시드 불확실성 초과) **또는** 실측 예산이
  Pareto 교차, **그리고** oracle이 best-static보다 정규화 점수 ≥+0.02 (1개 이상 예산 트랙).
- 불통과: learned selector·MoE·VLM로 억지 확장 금지 → 트랙 A로 간다.

## 4. 트랙 S — 선택기 (G0 통과 시에만)

- **입력**: 계약(모델·release·밴드·GSD·pooling) + 실측 비용 + 값싼 라벨없는 신호.
  단 MS-110에서 effective_rank 단독은 family 판별기로 강등됐다 → selector 입력으로 쓰되
  **support 기반 신호**(K개 support의 head 적합 이득 추정)와 결합한다.
- **베이스라인**: always-OlmoEarth · always-cheapest · Task2Vec/LogME(라벨 허용) ·
  Capabilities-Encoding식 성능예측 · contract-only 규칙 · oracle(평가용).
- **평가**: leave-one-task-out + leave-one-family-out, task/family/지역 cluster bootstrap.
  주지표 = episode-내 oracle-정규화 action regret.
- **승격 게이트 G1–G5**: 이미 prereg에 고정(regret −30%, score +0.02, cost −25%, 외부 방향일치, 3시드+해시봉인).

## 5. 트랙 A — 특성화 (G0 불통과 시)

지금 있는 걸로 성립: 캐시 가치는 family·scale·release·계약에 따라 raw 위아래로 교차한다
(MS-102/105/108/109) + 공개 벤치마크 + 라벨없는 진단(탐색적, MS-110) + G0 headroom 지도.
CVPR main은 아니어도 정직한 기여.

## 6. 선행연구 경계 (novelty 방어선)

- 일반 model selection: Task2Vec, LogME — **베이스라인으로 인용**, 우리 기여 아님.
- RS 성능 예측: Capabilities Encoding — 인용.
- feature-store refresh: RALF — downstream feedback 있음. 우리는 **즉시 feedback 없는 EO
  cold-start + 센서·시간창 계약 + 통제 불가한 third-party release**가 차이.
- LLM 라우팅/캐스케이드: regret 틀 성숙 — **방법이 아니라 EO 라이프사이클 이식이 novelty**.
- 확인된 갭: reuse/adapt/re-embed를 라벨없이 regret+비용으로 고르는 EO 논문은 없음.

## 7. 마감·리스크

- CVPR 2027 본문 11/16 AoE. 오늘 9/6.
- 최대 리스크: (a) GPU 경합(GPU0/1 타 사용자 점유 빈번) → action matrix 채우기가 병목.
  (b) BigEarthNet/BioMassters 미확보. (c) 실측 비용 계측 미착수.
- **G0가 GPU 없이 대부분 준비 가능**: 개발 matrix는 이미 있고, 스키마·계산기·테스트 완비.
  GPU가 비는 즉시 외부 과업 셀부터 CACHED_HEAD/HEAD_ADAPT(값쌈)로 채운다.

## 8. 다음 3수 (순서)

1. **BigEarthNet-v2 + BioMassters 확보** (classification·regression 편향 해소). preflight → HF 다운로드.
2. **chipping·head_type·시간선택을 prereg에 확정** (256/512px GEO-Bench ↔ 128px 칩 계약).
   GPU 실험 전에 기계 판독 파일로 고정 (L4).
3. **GPU 비면 외부 과업 action matrix 채우기** — CACHED_HEAD/HEAD_ADAPT 먼저, REEMBED 1회 확증.
   3~4 과업 채워지면 G0 재실행 → 트랙 S/A 분기 확정.
