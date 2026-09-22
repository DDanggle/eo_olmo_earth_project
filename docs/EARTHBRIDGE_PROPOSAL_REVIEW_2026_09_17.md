# EarthBridge(Continuous OlmoEarth) 제안 검토 — 외부 시선 — 2026-09-17

작성: 2026-09-17. 사용자 첨부 제안문(연속 시공간 EO FM, PDE latent dynamics, shift/semigroup consistency, 확률 bridge) 검토.
새 실험 없음. 검증은 저장소 내 OlmoEarth v1 번역본, MEASURED_FINDINGS, PAPER_READING_LIST, JEPA 준비안(9/9) 기준.

## 0. 사실 검증

| 제안문 주장 | 판정 | 근거 |
|---|---|---|
| OlmoEarth는 고정 무작위 투영 latent target + contrastive | 확인 | v1 §2.4 Latent MIM Lite, bandset 내 대조 |
| v1.2가 p=0.5 전체 timestep masking 도입 | **미확인** | 저장소 내 v1.2 기록은 mask slice 소비 경로(M8)뿐. 기술보고서 원문 확인 필요 |
| LIANet(2604.07092), ALISE, DINo 존재 | 이름만 확인 | RELATED_WORK_STREAMING 표에 ALISE·LIANet 한 줄. 정독 없음 |
| Sentinel-3 revisit·해상도, GOES 10분 | 미검증 | 상식 범위이나 이번에 원문 확인 안 함 |
| AnySat·Copernicus-FM이 임의 sensor/resolution 처리 | AnySat 확인(reading list) | "임의 센서·해상도 query"는 이미 점유된 축 |

## 1. 기존 증거가 제안문에 주는 제약 (가장 중요)

1. **교차 센서 observation operator는 우리 실측에서 실패함.** MS-118/121: S2 캐시를 S1 새 관측으로 갱신 시 회복 0.1~1.3%, 사영기로 cos .62→.81 맞춰도 downstream 회복 없음. 제안문의 "sensor-specific decoder H_m으로 임의 센서 질의"는 이 결과와 정면 충돌. 넣으려면 왜 이번엔 되는지 먼저 답해야 함.
2. **Δt 조건 갱신은 이미 시험했고 등록 기준 미달.** MS-116-arch: Δt-GRU가 4/4 폴드에서 +2~3%p, 기준 +5%p 미달. continuous-time query의 첫 데이터점이 이미 있으며 약함.
3. **토큰 해상도 지렛대 두 번 기각.** MS-115/120. "임의 해상도 질의" 주장의 근거가 우리 데이터에는 없음.
4. **관측 0 예측 ≈0.** MS-116/117: 새 관측 없이 상태만으로 forecasting 불가. 제안문의 interpolation(smoothing)은 가능하나 forecasting 축은 현재 증거상 죽어 있음.
5. **이미 같은 방향의 준비안이 있음.** POSTTRAINING_JEPA_UPDATE(9/9): frozen encoder/LoRA × forecast loss. 제안문은 이것의 확대판이지 새 방향이 아님.

## 2. 외부 시선별 반론

### LeCun(JEPA·world model 관점)
- 픽셀(reflectance) 복원·diffusion은 구름·대기 노이즈를 모델링하는 데 용량을 씀. latent 예측으로 충분하며, OlmoEarth 자체가 이미 latent target임. 제안문의 L_spectral·L_prob(픽셀 diffusion)은 방향 역행.
- "PDE advection–diffusion on latent"는 latent 공간에 물리 좌표계가 있다는 가정. 검증된 적 없음. 붙일 거면 학습된 Φ_Δt와 semigroup 제약만으로 시작하고 PDE 항은 ablation.
- 불확실성은 픽셀 분포가 아니라 latent 다중 가설/energy로. 현재 우리 평가 판독기가 frozen이므로 latent 불확실성 → 판독기 출력 분산으로 잴 수 있음.

### 원격탐사·자료동화 교수
- GOES 고빈도로 Sentinel 간격 시뮬레이션 → 보간은 STARFM(2006)·ESTARFM·FSDAF 계열 시공간 융합의 20년 된 문제. 이 계보 없이 내면 즉시 기각. 현대판(딥 fusion, cloud removal: SEN12MS-CR 계열)도 baseline에 필요.
- "learned data assimilation"은 Aurora·GenCast 이후 기상에서 포화. EO 지표 상태로 옮기는 것의 차별점을 forcing 없는 setting에서 보여야 함.
- 계절·일주기 때문에 time-shift invariance 없음(제안문도 인정). semigroup은 day-of-year 조건 하에서만 성립하며, 이는 검증 가능한 좋은 주장.

### CVPR 리뷰어(비전)
- AnySat(CVPR25)·Copernicus-FM·ALISE·LIANet·DINo가 각각 임의 해상도·센서·불규칙 시간·연속 신경장·연속 PDE를 점유. 제안문 novelty 문장은 "이들의 합"이라 단일 기여로 안 읽힘. 한 축만 골라야 함.
- shift consistency 20 m 평가는 싸고 새로움. 기존 캐시(Sen12·Solar)로 GPU 없이 시험 가능. 단독 기여는 아니나 벤치마크 열로 좋음.
- 확률 head 없는 t1,t3→t2 L2는 blur. 리뷰어가 첫 질문으로 던짐. 그러나 latent-only면 회피 가능.

### 프로그램 매니저(자원)
- 마감 약 10주(GOAL 장부), GPU 1장, 1인. 제안문 전체(GOES 수집, diffusion, 다센서 operator, PDE)는 6개월·수 GPU 규모. 사전학습 재시작은 불가.
- 현재 자산(R1~R10) 폐기 비용이 큼. 제안문 중 현재 자산 위에 얹을 수 있는 것만 채택해야 함.

## 3. 채택안 — EarthBridge-lite

한 문장:
> frozen OlmoEarth 상태에 대해 **연속 Δt 질의가 가능한 latent updater**를 post-training하고, semigroup·shift consistency를 제약이자 벤치마크 열로 도입해, 보지 못한 시간 간격·공간 offset에서 frozen 판독기 성능을 유지한다.

| 제안문 요소 | 처리 |
|---|---|
| continuous-time query Φ_Δt | 채택. MS-116 GRU를 Δt-conditioned/ODE-style로 교체, 학습 6/12/24h→평가 3/9/18h 대신 Sen12 실제 간격(평균 52일)에서 unseen Δt 평가 |
| semigroup consistency loss | 채택. 싸고 검증 가능 |
| shift consistency 20 m | 채택. 기존 캐시로 CPU 평가 먼저, 손실은 그 다음 |
| latent 불확실성 | 조건부. 판독기 출력 분산으로 calibration 1개 표 |
| forcing(ERA5) | 부록 ablation. 관측0≈0 결과가 있어 기대 낮음 |
| PDE advection–diffusion | 보류. Φ_Δt 성립 후 ablation |
| sensor-specific observation operator | 제외. MS-118/121 음성 |
| 픽셀 diffusion, L_spectral | 제외. latent-only |
| GOES/Sentinel-3 1시간 실험 | 제외. 다음 학회 |
| 임의 해상도 질의 | 제외. MS-115/120 음성 |

기존 자산 배치: R7/R8(갱신)=본문 개발·외부 근거, R1~R6=empirical base 부록, N2/N3/N5=경계 조건. 9/4 "벤치마크" 결정과의 접점: unseen Δt·shift offset·release 전환을 열로 갖는 **continuity benchmark**로 평가표를 짜면 두 정체성이 합쳐짐.

## 4. 시작 전 닫아야 할 것
1. v1.2 time masking 원문 확인(기여 겹침 판단의 전제).
2. ALISE·LIANet·DINo·TerraFlow 정독 후 1쪽 차별표.
3. R8 clean 확증(NaN 제외 전 decoder 선택, FAR 분모).
4. shift 20 m CPU 파일럿(기존 캐시)으로 현상 존재 확인 → 있으면 트랙 개통.

## 5. 2026-09-17 갱신 — 검증 결과와 자원 완화 반영

검증:
- **v1.2 time masking 확인.** arXiv 2605.20804 §: 인스턴스마다 p_t=0.5로 timestep 전체 masking, 나머지는 random masking. timestep은 **월 단위 최대 12개**. 즉 "1,3으로 2 복원"은 월 격자 위에서 이미 사전학습됨.
- **LIANet은 CVPR 2026 채택작(IBM Research·UniBw).** 좌표(x,y,t)만으로 영상을 복원하는 연속 시공간 신경장. "연속 좌표 질의로 영상 복원"은 점유됨. 인접: GeoNDC(2603.25037) queryable neural data cube.
- 자원: GPU 2장 허용(사용자 9/17). olmoearth_pretrain v0.1.2 공개 → encoder continued pretraining 가능.

남는 진짜 틈(OlmoEarth 고유):
1. v1.2 시간축은 **월 bin 이산 격자**이고 우리 wrapper도 월 양자화(M39). 실제 취득일·불규칙 간격·월 경계 밖 질의는 사전학습 목표에 없음.
2. v1.2 time masking은 **창 내부 보간**만 학습. 창 밖 새 관측으로의 상태 갱신(우리 MS-116/117)과 semigroup 일관성은 없음.
3. LIANet은 지역별 재적합 신경장이며 FM 재사용·frozen readout 유지 문제를 다루지 않음.

갱신된 한 문장:
> OlmoEarth v1.2의 월 격자 time masking을 **연속 취득시각 조건 + semigroup/shift consistency 목표**로 확장해 continued pretraining하면, 보지 못한 시간 간격·월 경계 밖 날짜·20 m offset에서 frozen readout 성능이 유지되는가. 평가는 continuity benchmark(unseen Δt·offset·release 전환 열).

두 GPU 배분: GPU1 = post-training 본체(encoder LoRA/부분 해동 × 연속시간 목표), GPU0 = 벤치마크 평가·baseline(v1.2 원본, GRU updater, 선형 보간).

즉시 착수(독립 3건): (a) v1.2 원문 §시간 마스킹·timestep 정의 정독 후 1쪽 차별표에 LIANet·ALISE·DINo·TerraFlow·GeoNDC 추가, (b) shift 20 m CPU 파일럿(기존 Sen12 캐시), (c) olmoearth_pretrain v0.1.2 학습 진입점·데이터 포맷 확인 후 continued-pretraining 실행 가능성 감사.

## 6. 착수 조사 결과 (2026-09-17 저녁)

### 6.1 OlmoEarth 시간 부호화 — 틈 확정
- 사전학습 H5 데이터는 timestep별 `timestamps[T,3]=[day,month,year]`를 보유(CSV의 start/end_time 유래). deepwiki 답변, 코드 직접 확인 전.
- 모델 `CompositeEncodings`(`olmoearth_pretrain/nn/flexi_vit.py`)는 **`timestamps[:,:,1]`(월)만 사용** + timestep slot index의 1D sincos(3D RoPE 설정 시 생략). raw 파일 직접 확인.
- v1.2 보고서(2605.20804): 월 encoding 유지, 학습형 RoPE θt·t는 slot index 기준, 불규칙 간격·unseen gap 평가 없음. time masking p_t=0.5 확인.
- **결론:** day 정보가 데이터에 있으나 모델에 들어가지 않음. "연속 취득시각 조건 + 불규칙 간격 일반화"는 OlmoEarth 고유의 미점유 틈. LIANet(CVPR26)은 지역별 신경장·픽셀 복원이라 FM post-training과 다름.

### 6.2 continued pretraining 실행 가능성
- 진입점 `olmoearth_pretrain/internal/experiment.py` main, 빌더 `scripts/official/{nano,base}.py`, `torchrun ... train <run> local --dataset.h5py_dir=...`.
- masking `RandomTimeWithDecodeMaskingStrategy`, `encode_ratio/decode_ratio/random_ratio` override 가능.
- HF checkpoint 로드 `model_loader.py`; 학습 초기화는 `CheckpointerConfig/LoadStrategy` 경로 확인 필요(미검증).
- 데이터: 2560 m 셀 × 360일 H5. 우리 Sen12(128×128@10 m, 15 timestep, 실제 취득일 보유)를 H5 포맷으로 변환하는 어댑터가 필요. 이 어댑터가 첫 공학 작업.

### 6.3 shift 20 m 파일럿 — CPU 단독 불가, 소규모 GPU 필요
- P4 캐시 40 m 토큰, 64 crop 4개 고정 원점. offset 인자 없음. 재임베딩 필요.
- `raw_u16`(10×12×128×128)이 캐시 옆에 있어 NetCDF 불필요. `extract_olmo_variants.py`에 `--offset dy,dx` 추가하면 됨.
- 평가는 `resolution_contract_v2/p4_native_control/holdout_*_seed1_best.pt`(봉인 frozen readout) + `cache_decoder_train.py`의 `EmbDecoder/metrics` 재사용, eval-only. train-split 정규화 통계는 무이동 캐시로 고정.
- 설계 결정 필요: 라벨 프레임 정합. 20 m(2픽셀) 이동 시 예측을 되돌려 정렬하거나 공통 겹침 영역만 IoU 계산. 안 하면 misregistration과 표현 불안정이 섞임.
- 규모: 8 fold test split만 재임베딩. GPU0에서 가능.

### 6.4 다음 실행 순서
1. `extract_olmo_variants.py --offset` 추가 + eval-only shift 스크립트 + 정합 규칙 prereg 초안(config/shift_consistency_prereg_v0_draft.json).
2. Sen12→olmoearth_pretrain H5 어댑터 + timestamps day 실전달 확인(코드 직접 읽기).
3. 연속시간 목표 설계(day-of-year 연속 encoding + unseen Δt masking + semigroup) prereg 초안. 실행 전 사용자 동결.

## 7. 축별 유의미성 판정 + 사전 4건 재검토 (2026-09-17, 외부 교수 시점)

확률은 판단이지 측정이 아님. "현상 존재"와 "고쳐서 논문 기여" 확률을 분리함.

| 축 | 현상 존재 | 고쳐서 기여 | 근거·리스크 |
|---|---|---|---|
| A. 연속 취득시각 조건 | 중(50%) | 저~중(30%) | 모델은 월만 씀(확인). 사전학습 timestep이 월 합성이면 "일 단위"는 사전학습 분포 밖이라 효과 방향 불확실. MS-115에서 합성 월격자 날짜(.272)가 실제 날짜 p2(.246)보다 좋았던 관찰은 시간 부호화 민감성의 약한 신호. 진단 없이 학습 들어가면 안 됨 |
| B. semigroup 일관성 | 해당 없음(제약) | 저(20%) 단독 / 중 벤치마크 열 | neural ODE·Koopman에서 기존 기법. unseen Δt 일반화 개선이 있어야만 의미. MS-116-arch Δt-GRU +2~3%p가 현재 상한 근사 |
| C. 20 m shift 일관성 | 고(85%) | 중(40%) | patch-4 ViT의 half-patch aliasing은 잘 알려짐(Zhang 2019 anti-aliasing, ViT shift 민감성 문헌). 기여가 되려면 "다른 격자에서 만든 캐시를 재사용"이라는 배포 문제로 묶어야 함. 단순 shift 증강은 novelty 없음 |
| D. latent 불확실성 | — | 저 | 이번 논문 밖 |
| E. encoder continued pretraining(A+B+C 목표) | — | 저~중(25%, 10주 기준) | 어댑터 공학 2주+, 사후학습이 기존 캐시·frozen readout을 깨뜨림(R6 v1→v1.1 R@1=0이 선례). "호환성을 유지하는 post-training"으로 뒤집으면 기여 후보이나 난이도 최고. 대안: frozen encoder + LoRA/adapter |

종합: **A·C 진단이 양성이어야 E가 정당화됨.** 진단 음성이면 남는 것은 "OlmoEarth의 시간·격자 섭동 강건성 characterization"이며 CVPR main엔 약함(workshop·부록 수준).

### 사전 4건 재검토
1. v1.2 원문: **완료.** 추가로 닫을 것 하나: 사전학습 timestep이 월 합성인지 단일 취득인지(H5 생성 코드). 이게 A의 프레이밍을 결정함.
2. 문헌 1쪽: 요약표가 아니라 **"각 선행이 우리 진단 결과를 어떻게 예측하는가" 표**로. GeoNDC·TerraFlow·Copernicus-FM·AnySat, shift 문헌 추가. 2일 상한.
3. R8 clean 확증: **critical path에서 뺌.** 논문 몸통이 A/C/E로 가면 KuroSiwo는 외부 평가 열이고, 인용 전 확증은 필요하나 지금 1주를 쓸 일 아님. 몸통 결정 후 병렬.
4. shift 파일럿: CPU 불가 확인됨. **시간 진단과 묶어 GPU0 1~2일 "frozen 민감성 진단" 한 묶음으로 재정의.** (a) 공간: 2 fold test split을 (0,0)/(2,0)/(0,2)/(2,2)픽셀 offset 재임베딩, 봉인 readout으로 macro IoU와 토큰 cos 변화. (b) 시간: 같은 tile을 월 라벨 재배정(실제/합성 균등/±1개월 jitter/순서 셔플)으로 재임베딩, 동일 readout 평가. 판정 규칙을 실행 전 등록: 효과가 봉인 3-seed 표준편차의 2배 미만이면 해당 축 제거.

### 10주 배분(제안)
- 1주차: 진단 묶음(4) + 문헌표(2) + H5 timestep 확인(1).
- 2주차: 판정. 양성 축만으로 E 또는 adapter 설계 prereg.
- 3~7주: 학습·벤치마크 열 구축(GPU 2장).
- 8~10주: 확증·집필. R8 확증은 3주차 이후 병렬.

## 8. 1주차 gate 결과 (2026-09-17 저녁) — A·C 축 Sen12에서 폐기

MS-122: 두 폴드 모두 격자·시간 축 개통 규칙 불통과. 상세 표는 MEASURED_FINDINGS MS-122.
- §7 예측 대비: C(현상 존재 85%) 예측이 틀렸다. patch-4 half-token offset이 결정을 거의 바꾸지 않는다(기준자 대비 1.3~1.5배).
- A: 월 셔플이 표현을 cos .93까지 움직이지만 readout 결정·IoU는 불변. "표현은 변하나 과업 무관"이 가장 단순한 설명.
- 결론: A·C를 목표로 한 encoder post-training(E)은 Sen12 계약에서 동기 부족. 진행하지 않는다.
- 남는 선택지: (1) 시간에 민감한 과업(작물·홍수 시계열)에서 같은 진단을 1일 재실행해 A만 재검 — 현재 캐시 없음, 새 데이터 필요. (2) 논문 몸통을 갱신형 캐시(R7/R8)+continuity benchmark로 되돌리고 R8 clean 확증을 critical path로 올림. (3) 확률 재평가 후 사용자 결정.

## 9. v1 진단 결과 (2026-09-17 밤) — MS-123

- 연도 무시 확정, 구름 주입 무해(두 폴드).
- 결측: 6장 제거는 두 폴드 손실. 그러나 hiroshima의 "연속>무작위" 비대칭(c3 −.073 vs r3 +.001)은 indonesia에서 반대(c3 +.008, r6 −.076). **재현 실패.**
- 대안 설명: 시드로 정해진 연속 구멍이 사건 후 관측을 덮었는지 여부(정보 제거). `analyze_gap_position.py`로 post_index 기준 분리 중.
- 판정 보류. 구멍 위치 분석에서 "post 제거 여부"가 손실을 설명하면 이는 모델 실패가 아니라 관측 부재이며, A축 동기는 다시 0으로 돌아감. 설명 못 하면 위치 통제 arm(시작/중간/끝 고정)을 등록해 재시험.

## 10. 최종 판정 (2026-09-17 밤, MS-122~124)

세 차례 진단(격자·월·연도·구름·결측·구멍 위치) 결과 frozen OlmoEarth의 결정은 모두에 강건했고, 유일한 손실은 사건 후 증거 제거였다(2/2 폴드, 사전등록 규칙). **A(연속 취득시각)·C(격자 일관성) post-training 축은 Sen12에서 폐기.** §7의 C 85% 예측은 틀렸다.
살아남은 것: 진단 3종은 continuity benchmark의 "강건성 열"로 재사용 가능(라벨 없는 flip rate + 봉인 readout eval-only 절차). 창 끝 3장 제거가 사건 후 영상이 충분하면 무해하다는 관찰은 갱신형 캐시 연구에서 "최신성"보다 "사건 직후 관측 확보"가 가치라는 가설을 준다(미검).
남은 미검: 시간 정보를 실제로 쓰는 과업(PASTIS 작물, 다운로드 실패 상태)에서의 월 셔플 감도. 이것만이 A축을 되살릴 수 있는 마지막 시험이며 2~3일.
추천: 논문 몸통은 갱신형 캐시(R7/R8) + continuity benchmark로 복귀. R8 clean 확증을 critical path로.
