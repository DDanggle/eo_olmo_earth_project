# OlmoEarth 스터디 — 작업과 함께 자라는 학습 로그

> 규약: 실제 작업에서 **부딪힌 개념만** 카드로 만든다 (교과서 순서 금지).
> 매 작업 세션 종료 시 ① 새 개념 카드 추가 ② 커리큘럼 체크 ③ 확인 질문에 스스로 답해보기.
> 확인 질문은 그대로 면접 예상 질문이 된다.

## 커리큘럼 맵 (루프 연동)

### 모델 트랙
- [x] EO 데이터 기초: 재방문 주기, 광학 vs SAR, 모자이크 (루프 1에서 체득)
- [x] OlmoEarth 입력 표현: 멀티센서 시계열 큐브, 밴드 구성 (lfmc/forest_loss 설정 해부)
- [x] 태스크 헤드 패턴: UNet 분할 / Pooling 분류 / 회귀 head (두 model.yaml 비교)
- [x] OlmoEarth v1 논문 정독 (arXiv 2511.13655) — 카드 #6~#8 (2026-08-14)
- [ ] v1 vs v1.1/v1.2 아키텍처 차이 — 무엇을 줄여서 연산 1/3이 됐나 (벤치마크 A 설계와 동시)
- [ ] ViT 패치 임베딩과 patch_size=4의 의미 — 해상도/메모리/속도 트레이드오프 (A 측정하며)
- [ ] 파인튜닝 전략 비교: freeze-unfreeze vs full FT vs linear probe (lfmc 학습 재현하며 실측)
- [ ] FoldRefresh를 rslearn 출력에 맞게 재정식화 (루프 3)

### 엔지니어링 트랙
- [x] rslearn 데이터 모델: dataset → layers → windows → groups (sample/forest_loss로 체득)
- [x] materialize vs ingest 경로 차이와 각각의 함정 (NotImplementedError로 체득)
- [x] class_path 플러그인 아키텍처: jsonargparse/Lightning CLI 패턴 (5차 디버깅으로 체득)
- [x] GPU 서버 운영: 세션 수명, 영구/휘발 스토리지, 백그라운드 job, 터널 (nexus로 체득)
- [ ] rslearn 추론 경로 프로파일링: 병목이 IO인가 GPU인가 — py-spy/torch profiler (루프 2)
- [ ] 슬라이딩 윈도우 추론 최적화: patch_size/overlap/batch/precision 스윕 (루프 2 = 산출물 A)
- [ ] bf16/DDP on H200: precision이 EO 회귀 정확도에 주는 영향 실측 (루프 2)
- [ ] 추론 결과의 버전 태깅 설계 — 4-튜플 레지스트리 최소 구현 (루프 3 = 산출물 B)

## 개념 카드

### #49 현재 위험 추정과 action utility 추정은 다른 문제다 (2026-08-26, M37 claim 확장)

새 target batch의 정확도를 label 없이 추정하는 GdScore·ODD·agreement·IUPM이 이미 있다. 하지만
EarthRoute가 필요한 값은 현재 오류 하나가 아니라 `reuse / repair / re-embed / task-raw`를 각각
했을 때 **reuse 대비 얼마나 나아지는가**인 counterfactual action-gain matrix다. 같은 shift score가
커도 모든 action이 비슷하게 나쁘면 갱신할 이유가 없고, 현재 위험이 작아도 저비용 repair의 gain이
크면 행동할 수 있다. 여러 task가 같은 re-embedding을 쓰면 representation 비용은 한 번만 내므로
task별 독립 argmax가 아니라 공동 비용 최적화가 된다. 단 source/development label은 gain predictor
학습에 쓰며, label-free는 새 target의 선택 시점만 뜻한다. support 밖에서는 보편 보장을 주장하지
말고 abstain/audit-label action으로 보내야 한다.

**확인 질문**: target error estimator가 완벽해도 최적 refresh action을 못 고를 수 있는 예를 들고,
task 수 `K`가 늘 때 shared re-embedding과 task-specific raw model의 비용식이 어떻게 달라지는지
설명하라.

### #48 모델의 최대 time embedding 길이도 downstream 비교 계약이다 (2026-08-25, G-P smoke)

Sen12는 15시점이지만 OLMoEarth v1의 learned time embedding table은 12개여서 실제 forward가
`12 != 15`로 실패했다. OLMo만 12개, task-specific baseline은 15개를 쓰면 표현력 차이와 입력정보량
차이가 섞인다. 그래서 라벨·event date를 보지 않고 SCL clear fraction 상위 12개를 고른 뒤 시간순으로
복원하는 S12q를 만들고 모든 G-P arm에 같은 index를 준다. 논문의 15시점 성능은 S15-ref로 따로
재현해야 하며 G-P 분모로 직접 쓰지 않는다.

**확인 질문**: foundation model의 입력 시점 수가 baseline보다 작을 때 “baseline의 95%” gate를
유효하게 만들려면 어떤 matched-input arm과 reference arm이 필요한가?

### #47 시계열 cube의 반복 MASK는 시점별 독립 라벨이 아니다 (2026-08-25, Sen12 C0)

Sen12 S2 표본은 `time=15`이고 `MASK`도 time 차원을 갖지만, 실물 smoke에서 한 표본의 mask는
모든 시점에 동일했다. 이것은 각 시점에 새로 관측한 산사태 상태가 아니라 한 event polygon을
모든 영상 옆에 반복 저장한 것이다. 이를 15개 label로 세면 표본 수와 독립성을 15배 부풀리고,
event 전 영상에도 사후 polygon을 붙인 채 prospective detection이라고 오해하게 된다. 따라서
retrospective S15 segmentation과 cutoff-valid S≤t를 분리하고, 통계 단위는 canonical event/region으로
둔다. 음성 patch에는 event date가 없으므로 S≤t의 pseudo-cutoff도 계절·지역을 맞춰 미리 동결한다.

**확인 질문**: `MASK(time)`의 배열 shape만 보고 time-varying label이라고 판단하면 왜 leakage와
pseudoreplication이 동시에 생기는가? S15와 S≤t는 baseline 입력과 주장 문구가 어떻게 달라야 하는가?

### #46 느린 EO cache와 near-real-time residual은 다른 시계다 (2026-08-25)

Sentinel-1/2 기반 Earth embedding을 `실시간`이라고 부르며 매 alert마다 전체 gallery를 다시
임베딩하면 센서 재방문·구름·I/O 비용을 숨기게 된다. 느린 `z_global`은 새 유효 acquisition이나
encoder release 때 갱신하고, 강우·적설·경보처럼 자주 바뀌는 `r_t`는
`observed_at/published_at/retrieved_at/freshness`를 보존해 요청 시 갱신한다. 미래 inventory나 사후
피해조사는 prospective input이 아니라 label/evidence다. `r_t`가 있을 때만 좋아지면 fusion 개선이고,
EO-only student나 region-static representation이 좋아져야 embedding 개선이라고 말할 수 있다.

**확인 질문**: MountainShift가 “근실시간 embedding 개선”을 주장하려면 어떤 cutoff 계약과
ablation이 필요한가? FoldRefresh가 갱신하는 대상과 live residual이 갱신하는 대상은 어떻게 다른가?

### #1 재방문 주기와 모자이크 (2026-08-14)
Sentinel-2는 쌍둥이 위성이 같은 지점을 ~5일마다 재촬영(한국 ~2-3일). 광학은 구름에 죽으므로
`period_duration: 14d` + `PER_PERIOD_MOSAIC`으로 구간당 최선의 합성 1장을 만든다.
**확인 질문**: 몬순 지역에서 12타임스텝 중 5개가 결측이면 모델 입력은 어떻게 되고,
어떤 대응이 가능한가? (힌트: S1 비중, period 길이, 결측 마스킹 — 우리 E2 프로브 수치로 답할 것)

### #2 materialize vs ingest (2026-08-14)
rslearn은 원격 데이터를 (a) `ingest: true` — 타일스토어로 복사 후 조립, (b) `ingest: false` —
원격에서 직접 창 단위로 읽기(직접 materialize)의 두 경로로 가져온다. (b)가 빠르지만 데이터소스
구현이 불완전할 수 있다 — 우리는 PC Sentinel2 + `CONTAINS`에서 `get_item_by_name`
미구현(NotImplementedError)을 밟았다.
**확인 질문**: 어떤 조건에서 (a)를 강제해야 하며, 비용 차이는 어디서 나는가?

### #3 버전 삼각 스큐 (2026-08-14) — 이번 주 최대 교훈
공개 체크포인트·main의 설정·PyPI 패키지가 각각 다른 rslearn 버전을 전제 →
어떤 공식 조합으로도 재현 불가. 5차 디버깅 끝에 runner 0.1.14 + rslearn 0.0.27 +
설정 패치 2건으로 정착. 아티팩트에 (백본, 헤드, 전처리, 설정) 버전 4-튜플이 기록돼야
하는 이유의 실증이자 olmoearth-migrate의 존재 근거.

2026-08-22의 제주 v5에서도 같은 문제가 재현됐다. rslearn master는
`PER_PERIOD_MOSAIC`를 폐기 예정으로 두고 `MOSAIC + period_duration`을 권고했으며,
기간 안 장면을 최신순으로 반환하는 기본값도 시간순으로 바뀔 예정이라고 경고했다. 즉 YAML
문자열이 같아도 코드 버전에 따라 합성 후보의 순서와 결과 픽셀이 바뀔 수 있다.
**확인 질문**: 이 문제를 팀 입장에서 재발 방지하려면 CI/배포 어디에 무엇을 넣어야 하는가?

### #4 태스크 헤드 교체 패턴 (2026-08-13)
같은 백본(OLMOEARTH_V1_BASE, patch 4)에 lfmc는 UNetDecoder(1/4해상도 768ch → 픽셀 회귀),
forest_loss는 SimpleTimeSeries(pre/post 그룹 concat, 1536ch) + PoolingDecoder(10클래스).
태스크 전환 = decoder/head + task 정의 + 라벨 레이어 교체.
**확인 질문**: 맹그로브(분할)를 메콩 델타로 옮길 때 설정 3종에서 바꿔야 할 최소 항목은?

### #5 EO 추론의 진짜 병목 (2026-08-14)
100 윈도우 추론: GPU 2.3초 vs 전체 수십 분 — **materialize(다운로드+조립)가 지배**.
GPU-초/km²만 재면 반쪽 벤치마크. 산출물 A는 (다운로드, 조립, 추론, 후처리) 단계별
분해 측정이어야 하고, 데이터 캐시 재사용 시나리오(릴리스 교체 시 영상은 그대로!)가
마이그레이션 비용 절감의 최대 지렛대다.
**확인 질문**: v1→v1.2 교체 시 재사용 가능한 산출물과 재계산 필수 산출물을 구분하라.

### #6 Latent MIM Lite — 동결 랜덤 프로젝션 타깃 (2026-08-14, 논문 §2.4)
MAE(픽셀 복원, 안정하나 얕음)와 Latent MIM(잠재 복원, 좋으나 붕괴) 사이에서, 타깃 인코더를
**랜덤 초기화 후 영원히 동결된 선형 프로젝션**으로 대체해 붕괴를 원천 차단. 맵(라벨)도 같은
프로젝션에 통과시켜 지도/자기지도를 단일 loss로 통일 — 단 맵은 decode-only(인코딩 금지,
추론은 관측만 쓰므로). Ablation: Full Latent MIM 32.2(붕괴)→Lite 42.2→마스킹+loss+맵 62.4.
**확인 질문**: 타깃이 학습되지 않는데도 표현이 붕괴하지 않고 유용한 이유는? (랜덤 프로젝션의
거리 보존 + 타깃 불변성) EMA teacher 방식과의 트레이드오프는?

### #7 밴드셋 단위 모달리티 인식 마스킹 (2026-08-14, 논문 §2.3)
EO는 이웃 시공간/모달리티에 정답이 널려 랜덤 마스킹이 너무 쉬움(90% 마스킹 강요됨).
대신 밴드셋(해상도별 밴드 그룹; S2=3개, Landsat=2개)마다 {미선택/encode-only/decode-only/
둘 다}를 배정 — "다른 밴드셋의 부분 관측으로 빠진 밴드셋 통째 복원"으로 재구성. 패치 판별
loss도 **같은 밴드셋 안에서만** 대조 (쉬운 cross-modal negative 제거).
**확인 질문**: 이 구조가 우리 H3(SAR 태스크의 릴리스 드리프트) 측정 단위로 적합한 이유는?

### #8 가변 패치 크기와 토큰 예산 (2026-08-14, 논문 §2.2/3.1)
FlexiViT식 가변 패치(1~8) + 랜덤 크롭(1~12토큰) + 3~12타임스텝으로 사전학습 →
파인튜닝 시 patch_size가 자유 파라미터 (우리 설정의 4). 패치 크기는 토큰 수를 제곱으로
바꾸므로 (MACs, 메모리, 정확도)의 1차 조절 손잡이 = 벤치마크 A의 핵심 축.
Fig 1이 MACs-성능 파레토 주장; Large가 Base보다 항상 낫지 않음(EO 스케일링 미해결 자인).
사이즈: Nano 1.4M/Tiny 6.2M/Base 90M/Large 300M; Base 사전학습 = H100 2,989 GPU시간.
**확인 질문**: patch 4→8이면 토큰 수·MACs·경계 정밀도는 각각 어떻게 변하나?
32px 윈도우에서 patch 8이 만드는 토큰 그리드는?

### #9 임베딩 이방성과 mean-centering (2026-08-20, Korea Earth Search에서 체득)
파운데이션 모델의 원시 임베딩은 소수 지배 차원 탓에 모든 벡터 쌍의 cosine이 한 값(~0.7)에
몰린다(이방성). 평균 벡터를 빼고 재정규화하면 판별력이 살아난다 — 완도에서 교정 전
쿼리 구분 불가 → 교정 후 built precision@2000 ×26. 검색·클러스터링 전 필수 후처리.
**확인 질문**: 왜 사전학습 목적함수(패치 판별 + 인스턴스 대조)를 썼는데도 이방성이
남는가? mean-centering과 PCA 화이트닝의 차이는?

### #10 임베딩 스토어의 스키마 일관성 (2026-08-20)
모달리티 구성이 다르면(S1+S2 vs S2-only) 같은 모델이라도 임베딩 공간이 갈라져
교차 지역 검색이 무의미해진다. 스토어 설계 시 입력 스키마(모달리티·타임스텝 수·
정규화)를 전 지역 통일해야 한다. 실측 계기: PC의 S1이 제주 2024에 0장(완도는 정상).
**확인 질문**: 릴리스 마이그레이션(v1→v1.2)에서 임베딩 스토어가 stale이 되는 이유를
이 카드의 논리로 설명하라 — 4-튜플 중 무엇이 바뀌는 것인가?

### #11 모델 릴리스 변화와 세계 변화의 2×2 분해 (2026-08-21)

연도 `t0/t1`과 모델 `v1/v1.2`를 교차하면 같은 입력에서 생긴 **릴리스 효과**, 같은 모델에서
생긴 **세계 변화**, 둘의 **상호작용**을 분리할 수 있다. 단, 서로 다른 모델의 latent 좌표는
직접 뺄 수 없으므로 raw vector 차이만 주장하면 안 된다. 공통 anchor로 Orthogonal Procrustes를
맞춘 결과와, 좌표계에 불변인 neighbor overlap·Top-k Jaccard·Kendall τ·행정구역 집계 차이를
함께 보고한다. cloud/nodata와 입력 모달리티는 네 셀에서 동일하게 통제한다.
**확인 질문**: `v1(t1)-v1(t0)`와 `v1.2(t1)-v1.2(t0)`가 다를 때 이를 실제 변화량 차이라고
바로 해석할 수 없는 이유는? 어떤 통제와 불변 지표가 필요한가?

### #12 OlmoEarth v1.2와 릴리스 인지형 manifest (2026-08-21)

v1.2는 modality별 여러 band-set token을 하나로 합쳐 Base 추론 MACs를 v1보다 2.9배 줄이고,
RoPE로 tiled embedding의 grid-aligned striping artifact를 줄인다. 평균 benchmark 성능은 Base
65.2→65.2로 유지되지만 m-EuroSAT·CropHarvest처럼 후퇴하는 태스크도 있어 "drop-in"이
"downstream 결과 불변"을 뜻하지는 않는다. 임베딩 제품에는 최소한 `model_id`, weight hash,
code commit, modalities/bands/timesteps, AOI/time, CRS/GSD, normalization/centering,
cloud/nodata mask, output schema를 manifest로 고정해야 재현과 selective refresh가 가능하다.
**확인 질문**: v1.2가 더 빠르고 평균 성능이 같아도 기존 검색 인덱스를 전량 stale로 봐야 하는
이유는 무엇이며, 어떤 downstream 안정성 검사를 통과하면 부분 갱신을 허용할 수 있는가?

### #13 임베딩 변화탐지의 두 함정: 요동 vs 오염 (2026-08-21, 제주 4개년에서 체득)

**함정 1 — 요동:** "연도 간 코사인 거리"로 변화를 재면 **바다가 상위권을 독식**한다.
파도·햇빛 반사 때문에 실제 변화 없이도 지문이 매년 달라진다. 실측: 바다 클래스의
점수 표준편차 0.094 vs 육지 클래스 0.042~0.060 (약 2배).
→ 대책 (a) 계단형 검출: 4개년을 앞/뒤로 나눠 `그룹간 거리 − 그룹내 거리`를 최대화 —
한 번 바뀌고 유지되는 변화만 살아남고 왕복 요동은 상쇄. 부수 효과로 변화 시점도 나옴.
(b) 토지피복 클래스별 z점수 층화 — "같은 유형 중 얼마나 이례적인가"로 비교 기준 통일.

**함정 2 — 오염:** 위 두 대책 후 Top-30이 전부 육지로 바뀌었지만, 육안 검증에서
**5곳 중 4곳의 특정 연도 모자이크가 구름으로 덮여 있었다** (30곳 중 26곳이 같은
연도 전환에 몰린 것이 단서). 즉 탐지기는 정상 작동했고 입력이 오염돼 있었다.
Earth Embeddings 서베이가 지적한 *"단일 결정론적 벡터가 구름 오염을 숨긴다,
어떤 제품도 품질 마스크를 포함하지 않는다"*의 구체적 실증.
→ 대책 순서: ① 밝기 기반 구름 비율 마스크로 오염 픽셀 제외(즉시) ② 연중 모자이크를
여러 조로 나눠 조별 일치 여부로 검증(불확실성 정량화) ③ SCL 밴드를 받아 애초에
오염 시점을 모델 입력에서 배제(근본).

**확인 질문**: 계단형 점수가 "왕복 요동"을 억제하는 원리를 수식으로 설명하라.
그리고 구름 오염이 남아 있을 때 이 점수가 왜 높게 나오는가? (힌트: 오염은 특정
연도에만 있으므로 그룹 내 일관성을 해치지 않고 그룹 간 차이만 키운다)

### #14 사후 마스킹의 한계와 합성 레시피 (2026-08-21, 제주 실측)

구름 오염을 **사후에 픽셀 단위로 제외**하려면 "얼마나 오염됐나"의 정의가 필요한데,
- 12장 모자이크의 **평균** 구름 비율 → 1장이 완전 구름이어도 1/12≈0.08로 희석돼 안 걸림
  (v3: 기준 0.20 통과 → 육안 검증 5곳 중 3곳에 구름 잔존)
- 12장 중 **최악의 한 장** → 제주에서는 거의 모든 픽셀이 탈락
  (연도별 최악-모자이크 평균 구름 0.53~0.84 → 기준 0.35에서 생존 픽셀 1.2%, 전부 바다)

결론: **구름이 많은 지역에서 사후 마스킹은 원리적으로 불가능**하다. 입력 단계에서
해결해야 한다. 원인은 합성 레시피였다 — rslearn 임베딩 가이드 기본값
`space_mode: MOSAIC`(기간당 장면 1개)은 흐린 달을 그대로 통과시킨다. Ai2의 실전 모델
설정(lfmc)은 `PER_PERIOD_MOSAIC`(기간당 여러 장면을 겹쳐 합성)으로 구름 구멍을 메운다.
→ **합성 레시피는 4-튜플 manifest의 필수 항목**이다. 같은 모델·같은 AOI·같은 기간이라도
레시피가 다르면 임베딩이 달라지고, 하류 변화탐지 결론이 뒤집힌다.
**확인 질문**: 기간당 장면 수를 늘리면 구름은 줄지만 무엇이 나빠지는가?
(힌트: 시간 해상도 희석, 계절 신호 평활, 다운로드 비용) 태스크별 최적점은 어떻게 정하나?

### #15 지도에서 통계적으로 valid한 결론 뽑기 — PPI (2026-08-21, Wang et al.)

우리 변화 지도는 오차가 있는 예측이므로, 여기서 "제주 연안에서 X ha가 바뀌었다"를
그냥 세면 **편향된 추정**이다. Wang et al.(RSE 2025, arXiv:2407.13659)의
**Prediction-Powered Inference(PPI)**: 지도 전체 + **소량의 무작위 표본 정답**을 결합해
편향을 보정하면, ① 지도를 100% 정확하다고 가정하는 것보다 신뢰할 수 있고
② 정답 표본만 쓰는 것보다 불확실성이 작은 추정을 얻는다.
우리 적용 설계: 변화 점수로 층(strata)을 나눠 무작위 표본 추출 → 시계열 RGB 칩으로
사람이 변화/비변화 판정 → PPI로 전체 변화 면적과 신뢰구간 산출. Top-k 목록은
"조사 우선순위"로, 면적 추정은 "PPI + CI"로 분리해 보고한다.
**확인 질문**: 왜 Top-k 리스트의 정밀도(precision)만 보고하면 면적 추정에 부적절한가?
(힌트: 선택 편향 — 고득점 표본만 검증하면 저득점 구간의 누락을 모른다)

### #16 FIRST_VALID 합성과 nodata sentinel (2026-08-21, 제주 v5 실행 경고)

`PER_PERIOD_MOSAIC` materialize 중 rslearn이 `FIRST_VALID` 합성에 쓸 nodata 메타데이터를
찾지 못해 0을 기본값으로 쓴다는 경고가 반복됐다. FIRST_VALID는 장면 순서대로 "유효한"
첫 픽셀을 고르므로, 어떤 값이 nodata인지 잘못 정의하면 실제 결측을 관측값으로 채택하거나
반대로 유효한 0을 건너뛸 수 있다. 따라서 파이프라인이 완주했다는 사실만으로 구름 강건 입력이
만들어졌다고 결론 내릴 수 없다. 원본 밴드의 0값 의미, 래스터 마스크, 합성 결과의 0/nodata
비율과 RGB 칩을 함께 확인해야 한다.

**확인 질문**: FIRST_VALID 합성에서 nodata sentinel을 잘못 지정하면 최종 모자이크와
임베딩 변화 점수에 각각 어떤 방향의 편향이 생길 수 있는가?

### #17 관측 가능한 proxy와 생태 target 분리 (2026-08-22, MARC 적용 설계)

40m Sentinel-2/OlmoEarth 임베딩이 관측하는 것은 돌고래가 아니라 토지·수면 피복,
해안 인프라, 양식장, 광학적 수색 같은 **서식지 압력의 proxy(대리변수)**다. MARC의 target은
개체 출현·행동·서식지 이용·인간 영향이다. proxy와 target을 같은 것으로 취급하면 생태학적
오류와 과장된 인과 주장이 생긴다. 위성 레이어는 현장조사의 대체물이 아니라 조사 후보와
환경 맥락을 제공하고, target과의 관계는 현장자료·표본설계·별도 인과 가정으로 검증해야 한다.

**확인 질문**: 연안 양식장 변화와 돌고래 먹이행동이 같은 시기에 관측됐을 때, 왜 위성
상관만으로 인간 활동의 인과효과를 주장할 수 없으며 어떤 추가 설계가 필요한가?

### #18 설정 표면과 실행 의미의 분리 (2026-08-22, 제주 v5 폐기 경고)

설정 파일의 필드 이름은 실험의 의미가 아니다. 제주 v5 로그에서 rslearn은
`PER_PERIOD_MOSAIC` → `MOSAIC + period_duration`, 최신순 → 시간순 기본값, legacy timestep
지원 종료를 각각 예고했다. 같은 YAML을 나중에 다시 돌렸을 때 장면 그룹·순서·모델 입력이
달라질 수 있다는 뜻이다. 재현 manifest에는 패키지 버전/commit뿐 아니라 **정규화된 최종 설정,
선택된 STAC item ID와 순서의 hash, 대표 window의 합성 결과 checksum**이 필요하다.

**확인 질문**: 설정 파일 diff가 0인데 결과가 바뀌었을 때, golden-window 통합 테스트는
어느 단계의 hash를 비교해야 원격 카탈로그 변화와 라이브러리 의미 변화를 구분할 수 있는가?

### #19 materialize한 시간축과 모델이 실제 소비한 시간축 (2026-08-22, v5 감사)

제주 데이터셋은 한 해를 30일 간격 **12개 레이어**로 materialize했지만,
`model_s2.yaml`의 입력은 `sentinel2_l2a`부터 `.3`까지 **앞 4개 레이어만** 명시한다.
따라서 152 GiB를 만들고 12기간 품질을 검사해도 임베딩의 실제 관측 범위는 별개다.
더 심각하게 2023~2025 윈도우는 1월 시작, rolling-2026은 7월 시작이므로, 첫 4기간만 쓰면
연도 변화와 계절 변화가 섞일 수 있다. 감사는 `all_12`와 `model_used_4`를 분리하고,
변화탐지 재실행 전 실제 레이어 날짜·순서와 계절 정렬을 고정해야 한다.

**확인 질문**: 같은 “4개년 임베딩”이라도 각 연도의 시작월과 선택 timestep이 다르면
왜 world shift 추정이 무효가 되며, 어떤 시간축 manifest와 통제군이 필요한가?

### #20 설정 개입의 의미적 등가성 (2026-08-22, v1↔v5 전수 감사)

실험에서 바꾼 문자열이 실제 개입(intervention)이라는 보장은 없다. v1의
`MOSAIC + period_duration`과 v5의 `PER_PERIOD_MOSAIC + period_duration`은 이름이 다르지만,
rslearn 0.1.13에서는 같은 `match_with_space_mode_mosaic` handler를 쓴다. 그 결과 ordered
source group 2,592개가 전부 같고, cloud/zero 전수 지표와 원본·임베딩 표본도 같았다.
실험 요인을 선언하기 전 **정규화 설정 → 실행 함수 → 선택 item hash → 출력 pixel** 순으로
개입이 실제 데이터를 바꿨는지 확인해야 한다. 이 검사는 대규모 계산 전 golden window에서 한다.

**확인 질문**: 두 설정 파일의 diff가 있는데 결과가 같을 때, 어느 네 단계의 증거를 확인해야
“모델이 강건했다”가 아니라 “실험 개입이 존재하지 않았다”고 판정할 수 있는가?

### #21 보조 품질 밴드의 의존성·보간 분리 (2026-08-22, v7 SCL smoke)

SCL(Scene Classification Layer)처럼 **장면 선택에는 필요하지만 모델 입력에는 필요하지 않은
보조 밴드**도 데이터소스가 읽을 수 있게 명시적으로 등록해야 한다. rslearn의 Planetary
Computer Sentinel-2 구현은 layer `band_sets`와 교차하는 자산만 tile store에 등록한다.
따라서 item metadata에 SCL URL이 있어도 반사도 12밴드만 설정하면
`Sentinel2SCLBestClear`는 `missing scoring bands ['SCL']`로 실패한다.

또 반사도는 연속값이라 bilinear resampling이 자연스럽지만 SCL class ID는 범주형이다.
SCL=4/5/6 같은 equality로 clear pixel을 세면서 bilinear 보간을 쓰면 존재하지 않는 중간 class가
생겨 점수가 왜곡된다. v7은 SCL을 별도 band set으로 등록하고, **점수 read만 nearest**, 선택된
장면의 반사도 materialize는 bilinear로 유지했다. 이 분리는 golden window에서 실제 source
group·pixel을 바꾸고 bad proxy를 95.64% 줄인 첫 유효 입력 개입이 됐다.

**확인 질문**: 모델에 넣지 않는 SCL이 왜 `band_sets`에 필요하며, 같은 레이어에서 SCL 점수와
반사도 출력에 서로 다른 resampling을 써야 하는가?

### #22 사람 검수에서 계절·stretch·중복을 통제하는 법 (2026-08-22, 제주 14후보 감사)

변화 후보 RGB를 나란히 놓는 것만으로는 검증이 되지 않는다. 기존 `verify_candidates.py`는
연도별로 서로 다른 계절을 보여주고 각 칩의 2–98 percentile을 따로 stretch해, 농경지
phenology와 실제 밝기 차이를 숨길 수 있었다. 이번 감사는 후보를 먼저 고정한 뒤 각 연도의
5월 15일에 가장 가까운 관측을 골라, 모든 칩에 같은 0–3000 DN stretch를 적용하고 1.28 km
context와 400 m detail을 함께 보였다.

또 record 수와 고유 사건 수를 분리해야 한다. v3 rank 2와 v6 rank 16은 약 60 m 떨어진 같은
대면적 절개지를 잡아 고확신 record는 5개지만 고유 site는 4개였다. 알고리즘끼리 같은 사건을
반복 검출하면 consensus 증거가 될 수는 있어도 precision 분모에서 두 건으로 세면 안 된다.
동부 중산간 좌표 집단도 실제로는 지속 토지전환, 경작/피복지, 구름이 섞여 있었으므로
“오름 군집 변화”라는 지역 라벨을 개별 원인 판정으로 전파해서도 안 된다.

**확인 질문**: 변화 후보의 사람 검수에서 계절 정렬, 고정 stretch, 두 공간 scale, 고유-site
deduplication을 각각 빼면 어떤 종류의 false positive나 과대 계수가 생기는가?

### #23 공간 결합에서 현재 지도와 행정 데이터의 음성 증거 (2026-08-22, 제주 공공데이터 결합)

좌표를 지도에 얹는 것과 변화의 원인을 검증하는 것은 다르다. OSM 현재 스냅샷에서 `r11`
주변 419–951 m에 태양광 발전소 6개가 보여도, 후보 폴리곤과 겹치거나 2024~2025에 조성됐다는
증거가 없으면 “태양광 개발”로 확정할 수 없다. 제주 공식 오름현황도 이름·주소·면적은 있지만
좌표와 경계가 없어서 OSM peak 이름을 통해 위치를 보조했을 뿐, 후보가 오름 경계 안이라는
판정은 아니다.

행정 데이터의 0건은 더 위험하다. 국토부 최신 개발행위허가 파일의 제주 240행에는 2023·2024
허가가 하나도 없었다. 따라서 후보 지역과 일치하는 행이 0이어도 “허가 없음”이나 “무허가”의
음성 증거가 아니다. 실제로 초기 규칙은 포함 경계 `삼양동`에 행이 없자 근처 `화북이동` 3건을
연결하는 false join을 만들었다. 공간 결합은 `exact parcel / containing boundary / nearest
feature`의 증거 수준을 분리하고, 누락 가능한 데이터의 `no match`는 `unknown`으로 닫아야 한다.

**확인 질문**: 현재 OSM 시설이 후보에서 400 m 떨어져 있고 허가 CSV 일치가 0건일 때,
어떤 추가 자료가 있어야 “시설이 변화 원인이다” 또는 “허가가 없다”는 주장으로 승격할 수 있는가?

### #24 고정 분모·record linkage·선택적 보류 (2026-08-22, 오름 368 전수 레지스트리)

“전수 조사”는 대상 목록을 전부 데이터베이스에 넣는 것, 위성 점수를 전부 계산하는 것,
원인을 전부 검증하는 것이 서로 다르다. 제주 공식 오름 368건을 분모로 고정한 뒤
`목록/위치/모델/행정근거/사람검수` 상태를 분리하니 첫 단계는 368/368이지만 offline OSM
peak 위치는 243/368뿐이고, 필지·사업구역 경계와 시점이 맞는 공식 원인 근거는 0/368이었다.
따라서 후자를 숨긴 채 “오름 368개 조사 완료”라고 쓰면 coverage inflation이다.

record linkage도 같은 문제를 만든다. 사용자가 준 제주시 표의 번호 1–210을 공식 368개 파일의
연번으로 간주하자 첫 4건 뒤 206건이 가짜 충돌이 됐다. 첨부 번호는 제주시 부분집합의 내부
순번이었다. 연번 결합을 폐기하고 이름·주소·면적 복합키로 바꾸자 209건이 연결됐고, 188건은
핵심 필드 일치, 21건은 주로 주소 변경/차이, `빈내오름` 1건은 최신 공식 목록에서 미연결로
남았다. 키 이름이 같아 보이는 것보다 **키의 모집단과 생성 규칙**을 먼저 확인해야 한다.

공식 원인 근거가 10% 미만이면 시스템은 억지 분류 대신 선택적 변화탐지로 전환한다. 모델이
높은 점수를 내도 A/B급 경계·시점 근거가 없으면 `조사 우선` 또는 `보류`이며, 침묵률 자체를
evidence coverage–risk 곡선으로 평가한다. 이때 abstention은 실패가 아니라 불완전 기록 아래의
정직한 출력이다.

**확인 질문**: 공식 목록 368/368, OSM 위치 243/368, 모델 screen 243/368, A/B급 원인 근거
0/368일 때 “전수조사 완료율”을 하나의 숫자로 보고하면 왜 잘못이며, 어떤 상태별 분모와
보류 규칙을 따로 보고해야 하는가?

### #25 모델 합의의 공통오류 — common-mode failure (2026-08-22, 오름 RGB 감사)

서로 다른 점수 두 개가 같은 후보를 높게 평가해도 두 증거가 독립적이라는 뜻은 아니다.
오름 점별 screen에서 4기간·12기간 percentile이 모두 90 이상이고 변화 split도 같은
`high_stable` 8건을 만들었지만, 동일 월·고정 stretch RGB로 확인하자 **8/8이 2023년의
구름·연무를 공유한 거짓 양성**이었다. 두 계산이 같은 원시 장면과 전처리를 소비했기 때문에
합의가 오류를 상쇄하지 않고 오히려 같은 오염을 반복한 것이다.

따라서 ensemble agreement를 강화 증거로 쓰려면 장면·센서·품질마스크·전처리·모델 등
오류 경로의 독립성을 먼저 설명해야 한다. 현재 방법은 SCL 품질 게이트, 같은 계절의 복수
장면, 두 공간 축척 RGB, 독립 사람 판정이 통과될 때까지 M급 모델 합의를 `조사 우선`으로만
사용한다. 높은 일치율이 높은 진실성보다 먼저 보고되어서는 안 된다.

**확인 질문**: 4기간 모델과 12기간 모델이 같은 2023년 구름 장면을 포함할 때 두 점수의
합의를 독립 반복실험으로 간주할 수 없는 이유는 무엇이며, 어떤 입력·평가 설계가 공통오류를
줄이는가?

### #26 PNU는 상호운용 spine이지 경계·인과 정답이 아니다 (2026-08-22, 공공데이터 카탈로그)

한국 공공데이터를 연결할 때 필지고유번호(PNU)는 연속지적도, 건축인허가, 개발행위허가,
사유림사업, 팜맵을 잇는 가장 강한 공통키다. 그러나 키가 같다는 사실은 변화 원인이 같다는 뜻이
아니다. 공식 오름현황의 지번은 대표 필지일 수 있고, 오름은 여러 필지에 걸칠 수 있으며, 허가일과
실제 착공·관측 변화일도 다르다. 따라서 `주소→PNU`는 후보 생성이고, `변화 footprint∩공식 polygon`
과 사전 고정한 시간창의 부합을 통과해야 B급 근거가 된다.

이번 검색에서는 행정 시스템의 모집단 차이도 실제로 부딪혔다. 보유 개발행위허가 snapshot은
제주 2023·2024행이 0이지만, 별도 개념인 제주시 산지이용지정현황은 2023년 714건·230.6 ha,
2024년 542건·74.2 ha를 보고한다. 두 수치를 직접 같은 허가로 비교할 수는 없지만, 한 시스템의
0건을 “행정활동 없음”으로 일반화해서는 안 된다는 경보다. no-match를 음성 증거로 쓰려면 위치·
기간·행위유형 모집단·수집완전성·join 필드를 모두 감사해야 한다.

**확인 질문**: 오름 공식 주소에서 만든 PNU와 건축인허가 PNU가 같을 때도 원인 판정을 바로
내릴 수 없는 이유는 무엇이며, 어떤 geometry·시간·coverage 조건을 추가로 충족해야 하는가?

### #27 데이터셋 기준일·갱신일·관측일은 서로 다르다 (2026-08-22, FarmMap 실제 ingest)

“2025 팜맵”이라는 제품명이나 파일 기준일은 polygon 속 지표를 2025년에 관측했다는 뜻이 아니다.
실제 제주 FarmMap에는 항공 촬영일(`FLIGHT_YMD`)과 행정 갱신일(`UPDT_YMD`)이 따로 있고,
`oreum_v6_r08`은 촬영 2022-12-30·갱신 2023-12-08이었다. 변화 후보의 전후 Sentinel 관측은
2024-05-16→2025-05-13이므로 이 polygon은 변화 전 영상을 기준으로 한 상태다. 초기 구현이
갱신일을 우선하자 통합 테스트가 날짜 의미의 모호성을 드러냈고, 실제 지표 관측에 가까운 촬영일을
우선하며 `state_date_basis`와 503일 gap을 edge에 저장하도록 고쳤다.

공식 polygon과 좌표가 정확히 맞아도 이 결과는 “당시 밭이었다”는 B급 상태근거일 뿐, 이후 변화의
원인이나 허가를 증명하지 않는다. evidence chain의 최종 주장은 가장 약한 고리로 제한된다.
`공간 B + 시간 B + 상태 B`가 있어도 사건 레코드가 없으면 `cause_supported`로 승격하지 않는다.
반대로 OSM 오름 point가 FarmMap polygon에 든 7건은 입력 위치 자체가 C이므로 결과도 C다.

**확인 질문**: 제품명이 2025이고 polygon·PNU가 정확히 일치해도 `r08`을 2024–2025 변화의
원인근거로 쓸 수 없는 이유는 무엇이며, `FLIGHT_YMD`, `UPDT_YMD`, 변화 관측구간 중 어떤 값을
어떤 주장에 사용해야 하는가?

### #28 HTTP 성공과 데이터 성공은 다르다 (2026-08-22, 공공 API 실수집)

공공 API는 HTTP 200을 반환해도 본문 안의 업무 오류코드로 실패할 수 있다. 이번 bounded
snapshot은 207개 요청이 전부 HTTP 성공이었지만 의미상 성공은 200개였다. VWorld는 로컬과
H100 VM에서 모두 `INCORRECT_KEY`, GK2A는 OlmoEarth의 과거 6관측일에 대해 최근 2일 조회 제한을
반환했다. 이를 HTTP 성공 207/207로만 보고하면 필지와 역사 구름자료를 확보한 것처럼 보이는
coverage inflation이 생긴다. transport/HTTP/API-semantic/schema/item-count를 별도 상태로 저장해야 한다.

pagination도 응답이 말하는 계약을 따라야 한다. BuildingHUB 요청에는 1,000행을 넣었지만 서버는
page size 100을 반환했고 첫 법정동은 182행이었다. 첫 페이지만 보존했다면 45%를 조용히 잃었다.
실제 `numOfRows`와 `totalCount`로 후속 page를 생성해 45개 법정동 111페이지·8,794행을 소진했다.
반대로 철거·멸실 0행은 pagination이 완전해도, 이 endpoint와 날짜 필터의 모집단 밖 사건까지
없다는 뜻은 아니다.

**확인 질문**: HTTP 207/207 성공, API-semantic 200/207 성공, BuildingHUB 111페이지 완료라는
세 수치를 하나로 합치면 왜 안 되며, VWorld/GK2A/철거 0행 각각의 no-match를 어떤 상태로 닫아야 하는가?

### #29 전이 효과는 평균 점수가 아니라 matched counterfactual이다 (2026-08-22, K-EvidenceShift 설계)

“OlmoEarth가 한국에서 좋다”는 말은 pretrained 모델의 점수만으로 성립하지 않는다. 같은 입력
밴드·시간축, 같은 decoder, 같은 라벨 예산, 같은 augmentation·search budget과 compute를 쓴 scratch
및 일반 vision/EO baseline이 반사실 비교군이어야 한다. 지역 `g`와 라벨 예산 `b`별 전이효과를
`score(pretrained+adapted)-score(matched scratch)`로 두면 전체 평균이 양수여도 제주 high-cloud,
미래 연도, 희귀 class에서는 음수일 수 있다. 이 subgroup 손해를 숨기면 label efficiency가 아니라
평균으로 negative transfer를 상쇄한 것이다.

픽셀을 독립 표본으로 간주하면 신뢰구간도 과도하게 좁아진다. 같은 사건·필지·인접 tile을 묶고
site/event 단위 paired spatial bootstrap CI가 0 아래일 때만 confirmed negative transfer로 부른다.
모델의 native modality를 더 준 실험은 operational ceiling으로 유용하지만, 공통 S2 paired-input
track과 같은 표에서 representation 효과로 해석하면 안 된다.

**확인 질문**: OlmoEarth가 전국 평균 AUPRC는 scratch보다 3%p 높지만 제주 high-cloud에서는 4%p
낮을 때, 어떤 입력·decoder·compute 통제와 표본단위 CI가 있어야 이를 전이 이득과 negative
transfer로 각각 말할 수 있는가?

### #30 능동 라벨과 확률표본은 목적이 다르다 (2026-08-22, active transfer 설계)

active learning은 현재 모델을 가장 빨리 개선할 표본을 의도적으로 편향되게 고른다. 반면 한국
전체 변화율·오류율과 유효한 신뢰구간을 추정하려면 target population에서 알려진 포함확률을 가진
표본이 필요하다. uncertainty나 model disagreement가 큰 곳만 판독하면 학습에는 유용할 수 있지만,
그 비율을 전국 prevalence로 해석할 수 없다. 따라서 active query pool, 봉인 spatial-temporal test,
층화 확률표본을 세 자산으로 분리한다.

이번 오름 감사에서 두 모델 입력의 high-stable 후보 8/8이 구름 false positive였던 것은 disagreement/
agreement도 품질 게이트 없이 정보량이 아니라 공통오염을 선택할 수 있음을 보여준다. PDE의 `beta`,
advection–diffusion, 모호성선, D-opt도 자동으로 옮기지 않는다. Earth에서는 그룹별 전이효과,
경험적 disagreement region, embedding-diversity baseline으로 새로 정의하며, D-opt/log-det는 deep
shift 아래 식별성 보장이 아니라 비교 방법 하나일 뿐이다.

**확인 질문**: cross-model disagreement가 큰 high-cloud 필지 100개를 능동 판독해 성능이 올랐을 때,
왜 그 100개로 한국 전체 변화율을 추정할 수 없으며, 별도 확률표본과 어떤 cloud·공통오류 gate가
필요한가?

### #31 요청 hash는 응답 snapshot의 정체성이 아니다 (2026-08-22, VWorld 재승인 결합)

VWorld key 설정 전후의 대표점 요청은 endpoint·공개 parameter가 같아 동일한 request hash를
가졌지만, 첫 응답은 `INCORRECT_KEY`, 두 번째 응답은 `status=OK`였다. request hash는 “무엇을
물었는가”를 식별할 뿐 “언제 어떤 답을 받았는가”를 식별하지 않는다. 응답 lineage에는 반드시
`snapshot/retrieved_at/raw_sha256`이 함께 있어야 하고, 새 성공 응답으로 과거 실패 raw를
덮어쓰면 key·coverage 변화 자체를 잃는다.

같은 점의 필지도 단일한 영구 정답으로 가정할 수 없었다. `oreum_v6_r08`은 과거 항공 기반
FarmMap polygon의 PNU와 현재 VWorld point parcel PNU가 달랐다. 이는 즉시 어느 source가 틀렸다는
뜻이 아니라 source 기준일·경계 버전·점의 경계 위치를 다시 봐야 한다는 신호다. 따라서 현재
VWorld 필지와 dated FarmMap 필지를 둘 다 저장하고 `parcel_pnu_relation=conflict`로 보류했다.

**확인 질문**: 공개 parameter가 같은 두 API 호출의 request hash가 같아도 왜 하나의 응답으로
deduplicate하면 안 되며, FarmMap과 VWorld PNU가 다를 때 어떤 시점·geometry·source-version
정보가 있어야 충돌을 해소할 수 있는가?

### #32 Hugging Face snapshot 경로와 blob 심링크는 다른 정체성이다 (2026-08-23)

OlmoEarth 체크포인트를 immutable commit으로 고정하는 첫 구현에서 `Path.resolve()`를 파일까지
적용하자, `snapshots/<commit>/weights.pth` 심링크가 실제 `blobs/<sha>`로 풀리며 경로에서 commit을
읽을 수 없게 됐다. blob hash는 파일 내용의 정체성이고 snapshot commit은 repo revision의
정체성이므로 둘 중 하나로 다른 하나를 대신할 수 없다. 고친 resolver는 snapshot 경로의 raw
parts에서 commit을 보존하고, config·weights 각각의 byte 수와 SHA-256을 별도로 검증한다.

**확인 질문**: 같은 weight blob을 두 commit이 공유할 수 있을 때, 왜 weight SHA만으로 재현 가능한
모델 릴리스를 고정했다고 할 수 없으며 manifest에 repo·commit·config SHA·weight SHA를 모두
넣어야 하는가?

### #33 retrospective audit pool은 prospective benchmark가 아니다 (2026-08-23)

현재 14후보는 과거 모델 순위로 선택됐고, assistant 판독에는 일부 후보의 `t1` 뒤 EO frame이
포함되며, 한국 공공 API snapshot도 마지막 관측 뒤 수집됐다. 이 자료는 실패 원인을 찾는 audit에는
유용하지만 미래 시점에서 사용할 수 있었던 입력도, 알려진 포함확률의 모집단 표본도 아니다.
따라서 모든 레코드를 `pilot_audit_pool`로 묶고 후시점 EO는
`future_after_t1_review_only`, 공공근거는 prospective input 불가로 표시했다. 성능표는 별도의 sealed
확률표본과 frozen input/evidence cutoff가 생긴 뒤에만 연다.

**확인 질문**: 후보의 실제 변화가 사후 RGB와 행정사건으로 명확해 보여도, 그것을 그대로 test
label/feature로 쓰면 어떤 selection·temporal leakage가 생기며 prospective 평가에서는 어떤 cutoff와
split을 먼저 고정해야 하는가?

### #34 릴리스 표현 연속성과 task 성능은 다른 estimand다 (2026-08-23)

v1/v1.2의 embedding 차원이 둘 다 768이어도 각 축이 같은 의미라는 보장은 없다. 따라서 대응
벡터끼리 raw cosine을 계산하거나 8개 smoke 전부로 Procrustes를 맞춘 뒤 같은 8개에 평가하면
coordinate assumption 또는 rank≤7 과적합을 결과로 오인한다. P0에서는 같은 공간 token의
linear CKA, 각 릴리스 내부의 pooled distance rank와 이웃 보존을 본다. 공간 자기상관만으로 CKA가
높아지는지 toroidal block shift를 null로 두고, 8 site-years가 7 spatial clusters임을 LOO 범위에
반영한다. 이 지표들도 표현 구조 감사이지 정확도·negative transfer 증거는 아니다.
실측에서는 pooled site-year CKA 0.981·거리 순위상관 0.889와 달리, 창 내부 spatial CKA가
평균 0.427(0.133–0.828)이었다. 전역 이웃 구조의 보존과 국소 token geometry의 이동은 동시에
성립할 수 있으므로 두 값을 하나의 “호환성 점수”로 합치지 않는다.

**확인 질문**: 두 릴리스의 동일 위치 embedding raw cosine이 낮지만 CKA와 within-release neighbor
overlap이 높다면 무엇을 말할 수 있고, downstream head 성능·cache backward compatibility를
말하려면 어떤 sealed gallery와 spatial calibration/test split이 추가로 필요한가?

### #35 다중 GPU 안전 gate는 선택 장치에만 걸어야 한다 (2026-08-23)

GPU0이 비어 있고 GPU1에서 다른 프로젝트가 실행 중인 상태에서, 최초 실행기는 `nvidia-smi`의
모든 compute process를 하나로 모아 GPU0 실행까지 거부했다. 전역 process 존재 여부는 서버가
바쁜지는 말하지만 선택 장치를 선점해도 되는지는 말하지 않는다. index→GPU UUID를 먼저 고정하고
그 UUID의 process만 gate하도록 고친 뒤 GPU1을 건드리지 않고 GPU0 smoke를 완주했다. 반대로
메모리 사용량 0만 확인하는 것도 초기화 중인 process를 놓칠 수 있으므로 process UUID와 output
staleness를 함께 검사한다.

**확인 질문**: GPU1에 다른 학습이 있을 때 GPU0 작업을 안전하게 시작하려면 왜 전체 process 수나
메모리 0 하나만으로 부족하며, device UUID·PID·output mtime 중 무엇을 실행 전후에 고정해야 하는가?

### #36 결과 hash와 실행 전 gate도 하나의 증거 사슬이어야 한다 (2026-08-23)

release 결과·COMPLETE·분석 marker는 모두 맞았지만 로컬 `preflight.json`만 과거
`ready=false` 보류본으로 남아 있었다. 결과 hash가 맞는 것과 어떤 GPU·입력·checkpoint 조건을
통과해 그 결과가 시작됐는지는 다른 증거다. launcher 첫 JSON과 실제 실행 preflight를 동일 SHA로
묶고, exact input 208·checkpoint 4·output 16을 다시 해시한 뒤에야 758/758 check가 닫혔다.
로컬처럼 raw가 없는 곳에서는 같은 verifier가 `PARTIAL_VERIFIED`만 내고, 서버 raw를 요구하면
누락 하나도 `FAILED`로 만든다. 분석 재실행의 byte identity는 실행 무결성을 강화하지만 새 task
성능 근거를 만들지는 않는다.

**확인 질문**: run summary와 COMPLETE hash가 맞아도 stale preflight가 남아 있으면 왜 실행 조건을
재현했다고 할 수 없으며, `PARTIAL_VERIFIED`와 `FULL_EVIDENCE_VERIFIED`를 나누는 raw evidence는
정확히 무엇인가?

### #37 높은 CKA는 cache backward compatibility가 아니다 (2026-08-23)

full 216 audit의 sealed 64 site-years에서 두 릴리스의 pooled embedding geometry는 linear CKA
0.9786, pairwise-distance Spearman 0.9525로 높았다. 그러나 동일 공간 token의 raw cosine 평균은
−0.0086이고, v1.2 query를 v1 cache에서 찾거나 그 반대로 찾은 R@1은 양방향 0.0이었다.
calibration-only Procrustes는 0.491/0.436, affine ridge도 0.697/0.609까지만 회복해 사전 0.95
gate를 실패했다. CKA는 두 표현 집합의 관계적 구조를 보지만 동일 좌표계와 개별 identity 보존을
요구하지 않는다. 따라서 “패널 내부의 pooled geometry가 비슷하다”와 “old gallery를 new query가 그대로
사용할 수 있다”는 서로 다른 estimand다.

**확인 질문**: pooled CKA와 거리 순위상관이 0.95 이상인데 exact-token R@1이 0인 결과가 왜
모순이 아니며, 운영 cache 호환성을 주장하려면 representation proxy 외에 어떤 downstream task
gate와 migration 조건이 필요한가?

### #38 privileged supervision과 inference fusion은 다른 estimand다 (2026-08-23)

GeoLink·MMEarth·Galileo·SatMIP를 비교하면서 “공공데이터를 결합해 좋아졌다”는 문장이 서로 다른
효과를 섞는다는 마찰을 만났다. 추론 때 EO와 context를 같이 넣어 좋아지는 `E_fusion`은 추가 입력의
정보 이득이다. 반면 train 때 context를 auxiliary/teacher signal로만 쓰고 test에서는 EO만 넣는
student가 좋아지는 `E_repr`만 EO embedding 강화의 직접 증거다. 영상 예측 뒤 행정근거로 보류가
좋아지는 것은 다시 `E_decision`이다. 세 track은 같은 split과 target으로 비교하되 표와 claim을
분리해야 한다.

**확인 질문**: EO+행정 context teacher의 성능이 영상-only보다 높아도 왜 EO embedding이 강화됐다고
말할 수 없으며, 그 주장을 하려면 train/test 입력을 어떻게 구성한 student 실험이 필요한가?

### #39 공공데이터 누락은 modality dropout과 같지 않다 (2026-08-23)

GeoLink의 무작위 OSM 객체 삭제와 한국 API snapshot을 대조하면서 자연 누락은 무작위 결측이
아님을 다시 확인했다. 행정자료의 존재 여부는 지역, 사건 종류, 규모, 신고·공개 지연, API 모집단과
연관된다. 따라서 random source dropout에서 강건하다는 결과는 실제 `missing/error/out-of-window/
conflict`에 강하다는 증거가 아니다. context token과 평가 strata에 event/observed/published/retrieved
time과 coverage 상태를 보존하고, 자연 누락과 synthetic dropout을 별도 곡선으로 보고해야 한다.

**확인 질문**: public-record no-match가 missing not at random일 때 이를 0 또는 무작위 modality
dropout으로 처리하면 어떤 shortcut·selection bias가 생기며, 어떤 provenance 필드와 control이 이를
드러내는가?

### #40 feature distillation과 embedding compatibility는 다른 목표다 (2026-08-23)

AM-RADIO와 Theia를 현재 full-216 실패와 대조하면서, 다른 teacher의 feature를 student가 잘
회귀하는 것과 old/new query–gallery가 같은 좌표계에서 작동하는 것은 다른 estimand임을 만났다.
teacher별 projector가 낮은 MSE를 내도 projector를 거치지 않은 stable bus의 cross-model retrieval이
0일 수 있고, 반대로 retrieval 호환성을 강제하면서 downstream task 정보가 줄 수 있다. 따라서
student task utility, teacher reconstruction/relational loss, old/new/family compatibility, 효율을
각각 별도 표로 둔다.

**확인 질문**: multi-teacher student의 feature MSE가 낮다는 결과가 왜 기존 EO gallery 재사용을
보장하지 않으며, compatible representation을 주장하려면 어떤 네 query/gallery 조합과 task
upper/lower bound가 필요한가?

### #41 paired cross-view data와 embodied trajectory는 같은 자산이 아니다 (2026-08-23)

GeoBridge·UniGeoRS·PAUL과 DINO-WM을 함께 보니 satellite–drone/ground pair는 위치 검색·pose를
평가할 수 있지만 action에 따른 다음 관측과 성공/충돌을 평가할 수 없다. 이미지 pair에 EO
embedding을 넣은 결과는 cross-view localization이고, robot navigation/world model로 승격하려면
pose·action·시간순서가 있는 trajectory와 predict→plan→act 평가가 필요하다. synthetic trajectory만
쓸 경우에는 real 또는 별도 simulator family의 OOD test가 없으면 sim-to-real 주장을 하지 않는다.

**확인 질문**: satellite–drone Recall@1이 올랐어도 왜 로봇 navigation이 좋아졌다고 말할 수 없으며,
navigation claim에 필요한 observation/action/pose split과 primary metric은 무엇인가?

### #42 simulation fidelity는 pixel realism이 아니라 task fidelity다 (2026-08-23)

Sat2GroundScape·Vid2Sim·Sky2Ground를 비교하면서 위성 조건 ground image가 사실적으로 보이는 것과
그 장면의 도로·높이·장애물·통행가능성이 맞는 것은 다름을 확인했다. FID/LPIPS·사람 선호만 좋고
geometry나 semantics가 틀리면 그 simulator에서 학습한 policy가 real 환경에서 실패할 수 있다.
따라서 생성 지표는 보조로 두고, geometry/semantic consistency와 held-out/real Success·SPL·collision을
promotion gate로 둔다.

**확인 질문**: satellite-to-ground 생성 모델의 FID가 좋아졌는데 real navigation success가
떨어질 수 있는 이유는 무엇이며, task-faithful simulation을 검증할 최소 real control은 무엇인가?

### #43 stable cache와 dynamic context는 다른 시계로 갱신된다 (2026-08-23)

한국 public alignment와 model compatibility를 한 연구로 합치면서, EO model release·새 위성관측·
행정 record publication이 서로 다른 시각에 발생한다는 설계 마찰을 만났다. 모든 정보를 하나의
embedding에 섞으면 건축 record 한 건이 추가될 때 전국 EO gallery를 backfill해야 하고, 무엇이
성능을 바꿨는지도 분리할 수 없다. 따라서 EO-only `z_stable`은 compatible bus에 오래 보존하고,
cutoff-valid public context는 provenance gate가 있는 `r_context`로 따로 갱신한다. 이 분해는
`E_repr / E_compat / E_fusion / E_refresh`를 각각 평가할 때만 의미가 있다.

**확인 질문**: 새 BuildingHUB record가 공개됐을 때 왜 EO stable cache까지 전부 다시 계산할 필요가
없으며, residual-only refresh가 안전하다고 주장하려면 어떤 task·compatibility·staleness gate가
필요한가?

### #44 embedding의 시간계약은 파일 차원으로 복구할 수 없다 (2026-08-23)

제주 변화 후보를 다시 추적하니 2025 연간창과 rolling-2026 창이 184일 겹쳤고, 4기간 입력은
2023–2025의 9–12월과 rolling-2026의 3–6월을 비교했다. 두 경로의 출력은 모두 768채널이지만
이 채널은 월이 아니라 인코더가 시간축을 융합한 feature다. 따라서 저장된 embedding에서 겹친
월이나 특정 계절만 사후 제거할 수 없다. candidate score 전에 실제 acquisition dates·window overlap·
month coverage가 통과해야 하며 실패하면 입력을 다시 구성해 encoder를 재실행해야 한다. 기존
14후보 중 9건이 이 두 계약결함에 노출됐지만, 이는 9건 모두 시각적 false positive라는 뜻이 아니라
annual-change claim의 lineage가 무효라는 뜻이다.

**확인 질문**: 출력 shape가 두 실행 모두 `768×H×W`로 같아도 왜 시간축 호환성을 의미하지 않으며,
월별 재슬라이싱 대신 재추론이 필요한 이유와 후보 생성 전 반드시 검사할 세 시간 필드는 무엇인가?

### #45 version metadata와 operational compatibility는 다른 계약이다 (2026-08-24)

EarthKV 통합안을 최신 제품 문서와 대조하면서 `기존 Earth embedding 제품에는 버전 의미가 없다`는
전제가 깨졌다. AlphaEarth는 model/process/data version을, TESSERA는 dataset/model/build version을
제공하고 다른 model version의 store를 섞지 말라고 명시한다. 그러나 version tag가 있다고 old head,
old query/gallery, 변화 집계가 새 release에서도 유지되는지는 알 수 없다. metadata contract는
**무엇이 달라졌는지 식별**하고, compatibility experiment는 **그 차이가 task에 허용 가능한지 판정**한다.
EarthKV는 이 판정 뒤의 reuse/repair/recompute lifecycle이고, EarthEmbedContract는 첫 논문의 좁은
estimand다. 둘을 같은 기여로 쓰면 구현하지 않은 paging·eviction까지 주장하게 된다.

**확인 질문**: 두 embedding store가 모두 `model_version`을 기록해도 왜 backward compatible하다고
말할 수 없으며, old frozen head·dual index·full re-embed를 어떤 risk–cost 표에서 비교해야 하는가?

### #19 미세조정은 인코더 교체 가능성을 파괴한다 (2026-08-24, M6 실패에서 체득)

"head를 고정하고 인코더만 릴리스 업그레이드"는 **head만 학습된 모델에서만** 성립하는 조작이다.
lfmc 레시피는 `FreezeUnfreeze(unfreeze_at_epoch=20)`로 백본을 해동하므로 ep33은 인코더까지
미세조정된 상태이고, 실측으로 **231개 인코더 텐서 중 206개가 릴리스 원본과 다르다.**
릴리스 원본을 끼우면 co-adaptation이 깨져 test MSE가 558.79 → 1006.91(1.80배)로 붕괴한다.

운영적 함의가 더 크다: **end-to-end 미세조정은 파트너를 그 릴리스에 비가역적으로 결합시킨다.**
새 릴리스가 나와도 백본만 갈아끼울 수 없고, ① 구버전 유지 ② 전면 재미세조정 ③ 캐시 임베딩 +
bridge 중에서 골라야 한다. 아카이브 임베딩 재사용 연구가 필요한 이유가 여기서 나온다.

설계 교훈: 통제 실험은 **대조군이 원본을 재현하는지 먼저 확인**해야 한다(그들의 dose-0
byte-identical 검사와 같은 논리). 이 게이트가 없었다면 v1.2 arm의 1.8배 손해를
"릴리스 드리프트"로 잘못 보고했을 것이다.
**확인 질문**: 인코더를 동결한 채 head만 학습하면 왜 릴리스 교체 실험이 통제되는가?
그 설정이 실제 배포 시나리오를 어떤 부분에서 여전히 대표하지 못하는가?

### #20 릴리스 간 구조 차이가 pooling으로 흡수되는 경우 (2026-08-24, C1)

v1은 S2를 해상도별 **3 band_set**으로, v1.2는 **12밴드 단일 그룹**으로 토큰화한다
(로드된 모델에서 확인: 88.96M vs 113.99M). rslearn은 mask를 항상 3-set으로 만든다.
그런데도 `same-token` 비교가 유효한 이유는 `token_pooling=True`가 시간·모달리티 축을
patch 단위로 pooling해 출력이 릴리스와 무관하게 공간 patch당 768-d 하나가 되기 때문이다.
→ **구조 차이가 지표를 무효화하는지는 pooling 지점이 어디인지로 결정된다.**

동시에 이 차이가 다른 곳에서 비대칭을 만든다. 10밴드 S2 입력은 v1에서는 `band_set 2` 부재로
표현할 수 있지만 v1.2에서는 단일 그룹이라 같은 방식이 없다. 즉 **같은 입력을 두 릴리스에
대칭적으로 줄 수 없는 조건이 존재한다.**
**확인 질문**: pooling 이전(토큰 수준)에서 두 릴리스를 비교하려면 무엇을 먼저 정의해야 하는가?
그리고 10밴드 입력을 대칭적으로 처리할 수 있는 방법이 있는가, 없다면 실험 설계를 어떻게 바꿔야 하는가?

### #21 계약 위반이 조용할 때 (2026-08-24, C2-A/M8)

rslearn은 Sentinel-2 mask를 **정적** 정의로 3 slice 만든다. 소비 측은 **로드된 모델의**
bandset 수만큼만 순회한다. v1.2는 1이므로 slice 1·2는 접근조차 안 된다 — 실측으로
byte-identical 확인.

더 나쁜 건 `fast_pass`가 입력 mask 전체를 보고 꺼진다는 점이다. slice 2를 MISSING으로 두면
pooling이 masked-average 경로로 바뀌지만 출력 mask에 MISSING이 없어 결과는 baseline과 같다.
**"선언했는데 아무 일도 안 일어나고 경고도 없다"** 가 가장 위험한 형태의 계약 위반이다.

**확인 질문**: 생산 측과 소비 측이 같은 이름의 축(band set)을 서로 다른 출처에서 가져올 때,
불일치를 조용히 통과시키지 않으려면 어디에 검사를 넣어야 하는가?

### #22 공개 split을 믿지 않는다 (2026-08-25, M9)

AI-Hub 71363의 공식 train/valid는 **valid 타일 110개 전부가 train 타일과 공간 중첩**이다
(642쌍). 타일이 10.24 km이고 sliding window로 잘렸을 가능성이 있는데, 그렇다면 결함은
타일링이 아니라 분할 방식이다.

같이 배운 것: 메타데이터의 `coordinates`가 중심인지 좌상단인지 **추측하지 않고**
라벨 폴리곤 범위와 대조해 판정했다 — upper_left 가설이 중위거리 4.2e-05 m로 정확일치,
center는 7,240 m. 기하 해석을 추측으로 넘기면 이후 모든 bbox가 조용히 틀린다.

**확인 질문**: 데이터셋이 제공한 split을 쓸 수 없다고 판정하려면 최소 몇 가지를 재야 하는가?
그리고 "타일당 날짜 수 1~8"이라는 사실이 split 설계에 주는 제약은 무엇인가?

### #23 반복 관측 수는 공간 anchor 수가 아니다 (2026-08-25, GK2A M15 재감사)

행정동 4곳을 24시각 반복 측정해 96개 비교가 생겨도 projection/offset을 식별하는 공간 정보는
4점뿐이다. 더구나 offset을 그 4점에서 고르고 같은 4점에서 평가한 0.8958은 held-out 정확도가
아니다. 시간 반복과 범주 class 수로 effective spatial sample size를 부풀리면 안 된다.

이번에는 더 싼 정답도 있었다. KMA API Hub가 KO/2 km의 lon·lat을 **grid 저장순서대로** 직접
제공한다. 공식 deterministic mapping이 있으면 범주값 일치로 CRS를 역추정하지 말고, 원본
checksum·shape·순서를 contract로 고정한다. fitted transform은 공식 파일의 sanity check나
역사적 실패 감사에만 남긴다.

**확인 질문**: 같은 행정동의 100시각 반복이 새로운 projection anchor가 아닌 이유는 무엇이며,
공식 lat/lon grid가 갱신됐을 때 어떤 hash·shape gate로 downstream join을 중단해야 하는가?

### #46 seed 고정과 결정론적 실행은 같은 계약이 아니다 (2026-08-26, M25)

Python/NumPy/PyTorch/DataLoader seed를 arm마다 reset해도 CUDA kernel이 nondeterministic하면 같은
seed 결과가 달라진다. 실제 P4 replay는 test IoU가 0.122826→0.143442(+16.8%)로 갈렸다.
`torch.use_deterministic_algorithms(True)`를 켜자 `max_pool3d`와 `avg_pool3d` backward가 지원되지
않는다는 오류가 드러났다. 경고로 후퇴하지 않고 pilot P2를 시간 pair 평균 + 2D spatial pool로
분해했고, 공식 3D U-Net이 아니라 P2-tiny라고 명시했다.

strict cuBLAS/cuDNN/TF32 계약 뒤 final P4 full-run과 P4-only는 checkpoint·모든 tensor·per-sample·
metric이 bitwise 동일(max-abs diff 0)이었다. 그러나 wall time은 950.5초 vs 520.0초였다. 즉
**수치 결정성은 cost 결정성이 아니다.** 비용은 isolated repeat·randomized order·cold/warm cache를
별도로 통제해야 한다.

**확인 질문**: 같은 seed의 두 CUDA 학습이 재현됐다고 말하려면 무엇을 비교해야 하며, 모델 수치는
bitwise 같지만 wall time이 1.8배 다를 때 accuracy-cost Pareto에는 어떤 측정 설계를 써야 하는가?

### #47 라벨 oracle 참고값은 learned decoder의 성능 상한이 아니다 (2026-08-26, M32 정정)

128×128 라벨을 4×4 블록으로 내린 뒤 블록 전체를 한 값으로 되올려 얻은 IoU는
`block-constant label oracle`이다. 라벨을 보고 규칙·threshold를 고르므로 배포할 수 없고,
decoder는 토큰 하나에서 블록 내부의 서로 다른 4×4 값을 출력할 수 있으므로 모든 decoder가
넘지 못하는 상한도 아니다. 이 값이 높다는 사실은 block-constant geometry로도 라벨 일부를
표현할 수 있다는 약한 참고일 뿐, 실제 embedding이 그 정보를 보존한다거나 40 m support가
병목이 아니라는 증거가 아니다.

**확인 질문**: label oracle, representational upper bound, deployable model 성능을 각각 정의하고,
32×32 token에서 128×128 mask를 복원할 때 어떤 추가 실험이 있어야 resolution bottleneck을
기각할 수 있는가?

### #48 factor 하나의 contrast와 전체 factorial main effect는 다르다 (2026-08-26, E1)

2×2 `context(tiled/full) × decoder(small/large)`에서 tiled-small→tiled-large가 좋아져도
`decoder main effect`가 확정된 것은 아니다. full-small→full-large contrast까지 같은 부호인지,
두 context contrast가 같은지, interaction이 있는지를 함께 봐야 한다. 더욱이 micro-IoU가 좋아지고
positive-patch macro IoU가 나빠지면 "더 좋은 segmentation"이라는 단일 결론도 불가능하다.
estimand와 metric hierarchy를 결과 전에 고정하고 paired spatial interval로 uncertainty를 붙인다.

**확인 질문**: `y00,y01,y10,y11`에서 context effect, decoder effect, interaction을 어떻게 계산하며,
한 셀의 test 결과만 읽은 상태에서 허용되는 가장 강한 주장은 무엇인가?

### #49 표현이 더 매끄럽다는 사실은 downstream sufficiency가 아니다 (2026-08-26, M37)

full-context cache는 crop 경계 이웃의 이질성을 줄였지만 E1 test IoU는 small decoder에서
0.1306→0.1166, large에서 0.1777→0.0814로 하락했다. seam/smoothness 진단은 표현의 한 성질을
측정할 뿐 task-relevant signal 보존을 측정하지 않는다. 넓은 context가 token을 과도하게
동질화하거나 positional/statistical contract를 바꾸고, decoder가 그 변화에 다르게 반응할 수도
있다. 따라서 proxy metric 개선을 causal mechanism이나 downstream 개선으로 승격하려면 paired
intervention과 task 결과가 함께 필요하다.

같은 결과에서 decoder 증가는 tiled에서는 +0.0471, full에서는 -0.0351이었다. main effect 평균이
작다고 "decoder가 무관"한 것도 아니고, 한 조건에서 좋다고 "decoder가 좋다"고도 할 수 없다.
부호 반전과 interaction을 먼저 보고 action을 joint recipe로 정의해야 한다.

**확인 질문**: representation smoothness가 높아져도 segmentation이 나빠질 수 있는 세 가지 기전을
말하고, spatial bootstrap이 optimization seed 불확실성을 해결하지 못하는 이유를 설명할 수 있는가?

### #50 지역 데이터셋은 자동으로 foundation-model modality가 되지 않는다 (2026-08-26)

AI-Hub label, ICIMOD 산사태 inventory, swissEO 영상, KMA 강우를 모두 “OLMoEarth에 넣을 데이터”로
부르면 연결이 막힌다. foundation model input에는 센서·밴드 순서·값 단위·정규화·GSD·시간축·
tokenization 계약이 있고, label은 target, 강우/DEM/경보는 local context, 새 센서는 adapter/PEFT
대상이다. OLMoEarth가 derived maps로 pretrain됐다는 사실도 임의의 지역 map channel을 append해도
된다는 뜻이 아니다.

따라서 지역 transfer는 local label과 **canonical supported EO**를 조인해 재고, missing-band product
shift는 별도 arm으로 잰다. Nepal Koshi처럼 U-Net-assisted label은 geography는 새로워도 label
mechanism이 독립 gold가 아니므로 silver target + manual adjudication으로 보고해야 한다.

**확인 질문**: swissEO 7-band 결과가 나빠졌을 때 geography shift, source-product shift,
missing-band shift 중 무엇이 원인인지 구별하려면 어떤 두 arm을 만들어야 하는가?

## 스터디 로그

### 2026-08-26 — 외부 데이터 onboarding·upstream PR 재감사

- 배운 것: 카드 #50. 한국·네팔·스위스 자산을 canonical input / local context / target으로
  분리해야 model update와 지역 transfer를 식별할 수 있다.
- label provenance 보정: Nepal Koshi 2024는 independent geography이지만 U-Net-assisted + manual-QC
  silver label이다. 수동 adjudication 없이는 untouched gold라고 부르지 않는다.
- upstream 보정: sample schema PR은 current main에도 유효하고 direct-materialize 후보는 current
  rslearn에서 해소됐다. SCL categorical scoring은 남지만 dependency API와 nearest fix를 분리한다.
- 다음 학습: AI-Hub v2 3-task action ranking을 먼저 닫고, public partial-band end-to-end repro를
  만든 뒤에만 release-aware mask를 upstream patch로 승격한다.

### 2026-08-26 — E1 원인진단·Earth embedding product prior-art 재정렬

- 배운 것: 카드 #47. M32의 0.607을 decoder 상한으로 부른 것은 오류였고, block-constant label
  oracle로 낮췄다. 해상도·가는 형태 가설은 열어 둔다.
- 배운 것: 카드 #48. tiled-large 한 셀의 큰 회복은 tiled context에서 capacity가 중요하다는
  신호지만 2×2 main effect가 아니며, micro/positive-patch metric도 교차한다.
- 배관 보정: alternate full128 embedding root와 base mask/raw/month/audit root를 분리하고 별도
  content seal을 추가했다. 기존 E1은 첫 새 셀 뒤 audit path 결합 오류로 중단됐다.
- 문헌 보정: OLMo/PANGAEA/PEFT는 strong decoder·adaptation, TESSERA v2는 shared product·seam,
  RALF는 downstream-regret refresh를 이미 다룬다. 남는 gap을 label-free EO task-risk와
  multi-action regret–cost policy로 좁혔다.
- 다음 학습: host identity를 확인한 뒤 동일 code SHA로 E1을 재실행하고, full factorial이 닫힌
  뒤 multi-level cached-token adapter와 encoder-changing PEFT를 서로 다른 action으로 설계한다.
- 실행 결과: 카드 #49. full-context의 평균효과는 -0.0552, decoder 효과는 tiled +0.0471/full
  -0.0351로 반전했다. seam 제거를 성능 proxy로 쓰지 않고, exact-time parity와 공통-seed 반복 뒤
  tiled-large recipe만 남길지 판정한다.

### 2026-08-26 — G-P pilot 재현성·지표·비용 계약 복구

- 배운 것: 카드 #46. RNG seed, deterministic kernel, artifact equality, wall-time repeatability는
  네 개의 다른 계약이다.
- 실행 보정: sampled AP를 all-pixel exact AP로, 300-mask pos_weight를 전체 5,542 mask로 바꾸고,
  checkpoint/per-sample SHA와 독립 aggregate verifier를 남겼다.
- 판정 보정: P4 IoU가 P2-tiny를 넘었어도 AP는 78.7%이며 official P2/P3와 timestamp parity가 없어
  G-P는 통과도 실패도 아닌 BLOCKED다. Chimanimani 추가 튜닝 대신 미열람 9지역 전 recipe를 닫는다.

### 2026-08-25 — GK2A 운영·좌표계·sensor contract 재감사

- 배운 것: 카드 #23. 96 repeated observations를 96 spatial anchors로 해석할 수 없고,
  deterministic official grid가 있으면 fitted categorical agreement보다 우선한다.
- 운영 보정: 57개는 파일 수가 아니라 data/NO_DATA를 포함한 예정 슬롯 수다. 폴더/파일 개수만
  세면 구 스케줄 extra가 누락을 가리므로 expected contract로 상태를 검사한다.
- sensor 보정: 같은 SAR라는 말로 frequency/polarization을 지울 수 없다. KOMPSAT-5 X-band
  single-pol과 Sentinel-1 C-band VV+VH는 platform-only 대조군이 아니다.

### 2026-08-24 — EarthKV 층위와 최신 공개 benchmark 보정

- 배운 것: 카드 #45. version provenance가 존재하는 것과 downstream compatibility가 검증된 것은
  다르다. novelty를 metadata 부재가 아니라 task-risk validation과 action cost에 둬야 한다.
- 데이터 보정: AvalCD와 Sen12Landslides가 MountainShift의 더 싼 Phase 0을 제공한다. AI-Hub 4해상도는
  ontology가 달라 동일 label ladder로 가정할 수 없다.
- 실행 확인: parent venv에서 128 tests가 통과했고 optional geospatial 1건만 skip됐다. system Python의
  PyYAML 부재 실패는 interpreter/runtime도 재현계약의 일부라는 기존 카드 #36을 다시 확인했다.
- 다음 학습: frozen-head task error와 calibration을 측정해 R@1 proxy를 실제 decision risk로 승격하거나
  실패하면 headline을 representation compatibility audit로 낮춘다.

### 2026-08-23 — audit-only pilot·immutable release smoke

- 배운 것: 카드 #32. checkpoint 파일의 blob 경로를 resolve하면 repo commit provenance가 사라져,
  파일 내용 hash와 release revision을 서로 다른 축으로 보존해야 한다.
- 배운 것: 카드 #33. 기존 14후보와 사후 공공근거는 실패 감사 자산이지 prevalence·정확도 test가
  아니며, t1 이후 관측과 assistant pre-annotation을 명시적으로 차단해야 한다.
- 배운 것: 카드 #34. 8 smoke는 7 spatial cluster이며 raw cross-version cosine이나 같은 표본에
  맞춘 Procrustes는 task utility가 아니다. CKA도 spatial-shift null과 cluster LOO 없이 픽셀을
  독립 반복으로 세면 과신하게 된다.
- 배운 것: 카드 #35. 전역 GPU process gate가 비어 있는 GPU0까지 막았고, selected UUID process
  gate로 고친 뒤 GPU1 학습과 충돌 없이 16개 출력을 완주했다.
- 배운 것: 카드 #36. stale preflight 한 파일 때문에 결과 체인이 완전히 닫히지 않았고, raw
  228파일·7.85GB 재해시와 분석 byte-identical 재실행을 한 뒤에야 실행 무결성을 확정했다.
- 다음 학습: BestClear는 대표 8개가 아니라 label-free stratified stress 8개로 부르고, 12기간
  선택 trace·SCL/reflectance hash·`changed/valid no-op`·2건 replay 계약을 코드로 만든 뒤에만
  materialize한다. cache 호환성은 별도의 held-out gallery가 생길 때까지 주장하지 않는다.
- full audit에서 추가로 배운 것: 카드 #37. sealed pooled CKA 0.9786과 거리 Spearman 0.9525가
  높아도 raw cross-version R@1은 0이고 선형 ridge도 0.697/0.609에 그쳤다. 패널 관계 구조와
  cache identity를 분리해 측정해야 하며, 이번 sealed 결과를 본 뒤의 새 bridge는 새 untouched
  split 없이는 검증이 아니다.
- public-context 문헌 재설계에서 추가로 배운 것: 카드 #38. inference-time fusion의 이득과
  privileged supervision으로 EO-only student가 좋아지는 표현 이득은 다른 estimand다.
- 자연 누락 설계에서 추가로 배운 것: 카드 #39. 한국 행정 record의 no-match·지연·오류는
  random modality dropout으로 대체할 수 없고 provenance와 coverage strata를 따로 보존해야 한다.
- embedding transfer 문헌 감사에서 추가로 배운 것: 카드 #40. multi-teacher feature 회귀와
  cross-model query/gallery compatibility는 별도 목표이며 task utility·비용과 함께 측정해야 한다.
- robotics/simulation 확장에서 추가로 배운 것: 카드 #41–#42. paired cross-view image는
  localization 자산이지 trajectory가 아니고, simulation은 pixel realism보다 real/held-out policy
  fidelity로 승격해야 한다.
- K-ALIGN 수렴에서 추가로 배운 것: 카드 #43. model release·새 EO 관측·public-record publication은
  갱신 시계가 달라 stable EO cache와 provenance-aware context residual을 분리해야 한다.
- 시간계약 재감사에서 추가로 배운 것: 카드 #44. 같은 768채널 출력도 시간창이 다르면 비교 자격이
  없고, 융합된 embedding에서 월별 축을 복구할 수 없으므로 overlap·month coverage·actual
  acquisition을 후보 생성 전에 fail-closed로 검사해야 한다.

### 2026-08-22 — 연구 프로그램·MARC 적용 재설계

- 배운 것: 카드 #17. 파트너 적용은 모델 능력에서 시작하지 않고, 파트너 target과 위성이
  실제 관측하는 proxy를 분리한 뒤 의사결정·표본설계·금지 주장을 먼저 고정해야 한다.
- 방향 수정: 제주를 돌고래 탐지 데모가 아니라 WorldShift × ModelShift 방법론의 생태 검증장과
  현장조사 우선순위 보조로 제한. PPI·릴리스 감사·선택적 갱신을 하나의 박사 연구축으로 연결.
- 다음 학습: MARC와 접촉 전 서식지 이용/인간 영향 연구의 관측 단위와 sampling effort를
  확인하고, PPI에서 nonuniform sampling을 다루는 가정을 정리한다.
- 실행에서 추가로 배운 것: 카드 #18. v5가 materialize 216/216을 통과한 뒤에도 폐기 예정
  설정·시간순서 기본값 경고가 나타났다. 다음 paired audit부터 item-order hash와 정규화 설정을
  남기지 않으면 “같은 입력”이라는 통제가 성립하지 않는다.
- 품질 감사에서 추가로 배운 것: 카드 #19. 12기간을 저장했지만 현재 모델 설정은 앞 4기간만
  소비한다. 2026 rolling 윈도우의 시작월도 달라 연도 변화와 계절 변화가 섞일 수 있으므로,
  품질 지표를 실제 모델 입력 범위와 전체 저장 범위로 분리하고 시간축부터 다시 검증한다.
- 등가성 진단에서 추가로 배운 것: 카드 #20. 설정 이름을 바꾼 것과 실행 의미를 바꾼 것은
  다르다. v5의 152 GiB 재계산은 실제 개입이 없는 중복 실행이었으며, 다음 실험부터 대표
  window에서 handler·item hash·pixel 차이를 확인하지 못하면 전체 materialize를 금지한다.
- v7 실행에서 추가로 배운 것: 카드 #21. compositor가 참조하는 보조 품질 밴드는 item URL의
  존재만으로 사용 가능하지 않고 data-source/tile-store 의존성에 등록돼야 한다. 범주형 SCL의
  nearest 점수와 연속 반사도의 bilinear 출력을 분리한 뒤에야 1-window 사전 게이트를 통과했다.
- 14후보 육안 감사에서 추가로 배운 것: 카드 #22. 같은 달·고정 stretch로 다시 보니 기존
  v3 대조는 구름/농경 계절성으로 닫혔고 동부 중산간 후보 일부는 다년 지속 토지전환으로
  분리됐다. 알고리즘 record 2개가 같은 site를 가리키는 중복도 발견해 사건 수를 따로 셌다.
- 한국 공공데이터 결합에서 추가로 배운 것: 카드 #23. 현재 지도 객체의 근접은 원인·시점
  증거가 아니고, 누락된 행정 데이터의 0건은 음성 증거가 아니다. 정밀좌표는 외부 API로
  보내지 않고 대한민국 전체 스냅샷을 받은 뒤 로컬 join했으며, parcel/boundary/nearest의
  증거 수준을 UI와 JSON에서 분리했다.
- 오름 368 전수 레지스트리에서 추가로 배운 것: 카드 #24. 부분집합 내부 순번을 공식 연번으로
  잘못 결합하면 대규모 가짜 충돌이 생긴다. 고정 분모 안에서 목록·위치·모델·원인근거 coverage를
  분리하고, A/B급 원인 근거가 10% 미만일 때 선택적 보류로 전환하도록 만들었다.
- 오름 RGB 감사에서 추가로 배운 것: 카드 #25. 4기간·12기간 합의 후보 8건이 공유 입력의
  구름 때문에 8/8 거짓 양성이었다. 모델 합의는 독립 증거가 아니며, 공통 입력 품질 게이트와
  사람 검수가 통과되기 전에는 조사 우선순위 이상의 의미를 주지 않는다.
- 공식 데이터 전수 탐색에서 추가로 배운 것: 카드 #26. PNU는 여러 행정표를 잇는 핵심 spine이지만
  대표 지번을 오름 경계로, 같은 필지를 동일 원인으로 바꾸지는 않는다. 공간 footprint·사건시점·
  출처 모집단이 모두 맞아야 하며, 이 조건을 `KOREA_PUBLIC_DATA_CATALOG.md`의 join contract와
  no-match 규칙으로 고정했다.
- FarmMap 실제 ingest에서 추가로 배운 것: 카드 #27. 제품 기준일·행정 갱신일·실제 항공 관측일은
  다르며, 관측일을 변화구간과 직접 비교해야 한다. 정확한 polygon hit도 상태근거일 뿐 사건·인과가
  아니고, upstream 위치가 OSM C급이면 downstream FarmMap 결합도 C급을 넘지 않는다.
- 공공 API 실수집에서 추가로 배운 것: 카드 #28. HTTP 200과 업무 성공을 분리하니 207요청 중
  의미상 성공은 200이었다. 서버가 강제한 100행 page를 111페이지 끝까지 소진했고, 로컬/VM
  VWorld key 실패와 GK2A 과거 조회 제한을 데이터 0건이 아니라 coverage 실패로 보존했다.
- K-EvidenceShift 논문 설계에서 추가로 배운 것: 카드 #29. GeoFM 전이 이득은 pretrained 점수
  자체가 아니라 input/decoder/label/compute가 맞은 scratch와의 그룹별 paired 차이며, 평균 양수와
  subgroup negative transfer가 동시에 존재할 수 있다.
- 제한 예산 라벨 획득 설계에서 추가로 배운 것: 카드 #30. disagreement 표본은 학습 개선용으로
  편향되어 있고 모집단 추론용 확률표본을 대신하지 못한다. PDE의 물리 parameter·모호성선·D-opt
  해석은 Earth estimand로 새로 정의하지 않으면 가져오지 않는다.
- VWorld 재승인 결합에서 추가로 배운 것: 카드 #31. 동일 request hash도 인증·시점에 따라 응답이
  달라지므로 raw SHA와 retrieved_at을 별도 lineage로 남겼다. FarmMap/VWorld PNU 충돌도 최신값으로
  덮어쓰지 않고 두 source anchor와 보류 사유를 candidate record에 보존했다.

### 2026-08-21 — 제주 v5 실행 상태 점검

- 배운 것: 카드 #16. `PER_PERIOD_MOSAIC`이 선택됐다는 사실과 실제 픽셀 유효성은 별개이며,
  FIRST_VALID의 nodata sentinel까지 입력 manifest·품질 검증 범위에 포함해야 한다.
- 다음 학습: materialize 완료 후 원본/합성 래스터의 0값·마스크 분포와 RGB 칩을 대조해
  이 경고가 무해한 기본값인지 실제 편향원인지 판정한다.

### 2026-08-21 — 릴리스 인지형 Earth Embedding 포지셔닝

- 배운 것: 카드 #11~#12. 최신 공개 현황상 Major TOM에 OlmoEarth 249k 임베딩이 이미 존재.
- 방향 수정: 데이터셋 최초성 대신 모델 릴리스 이동과 세계 변화의 분해, 의사결정 연속성,
  부분 재계산 비용-오차 곡선을 핵심 연구 질문으로 승격.
- 다음 학습: Earth Embeddings(2608.03410) 전문 정독, 기존 249k의 생성 recipe/manifest 공백,
  Procrustes·CKA·이웃 보존 지표의 가정과 bootstrap 설계.

### 2026-08-13 ~ 14 (루프 1)
- 배운 것: 카드 #1~#5. 설정 3종의 필드 → 라이브러리 흐름 추적 완료.
- 다음 학습: v1 논문 정독 (벤치마크 A 측정 항목을 논문 지표와 정렬하기 위해 필수).

### 2026-08-14 (2차) — v1 논문 정독 완료
- 배운 것: 카드 #6~#8. ViT enc-dec + Latent MIM Lite(동결 랜덤 프로젝션) + 밴드셋 마스킹.
  파인튜닝 레시피(20% 동결→해동, plateau)가 우리 설정과 동일함 확인.
- A 설계 시사점: 논문의 MACs 파레토(Fig 1)에 우리가 얹을 것 = 실배포 GPU-초/km²,
  단계별 분해(다운로드/조립/추론), 릴리스 간 델타. "Large ≠ 항상 우위" 자인은
  온보딩 킷의 "어떤 크기를 쓸 것인가" 질문의 팀 공식 근거.
- 다음 학습: v1.1/v1.2 변경점 (HF 모델카드/릴리스 노트) — A 매트릭스 확정 직전에.
