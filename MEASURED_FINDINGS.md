# 측정 장부 — 실제로 잰 것만

최종 갱신: 2026-08-26

이 파일에는 **실행해서 나온 수치만** 넣는다. 계획·설계·문헌 판단은 넣지 않는다
(그건 `K_ALIGN_PROGRAM_NOTE.md`, `K_ALIGN_CVPR_READINESS_AUDIT.md`, `K_GAIN_AXES.md`가 맡는다).

각 항목은 **주장 / 근거파일 / 허용범위 / 아직 말할 수 없는 것**을 함께 적는다.

---

## 0. 한눈에

| # | 측정 | 판정 | 상태 |
|---|---|---|---|
| M1 | OlmoEarth v1↔v1.2 cross-release 검색 identity (제주 216) | **실패** — 사전등록 8 gate 전부 | 봉인 완료 |
| M2 | Major TOM 249k 두 제품의 paired 여부·계약 필드 | **PAIRED / 계약필드 8개 전부 부재** | 완료 |
| M3 | 밴드 순서 계약 불일치 dose–response (v1) | 단조 곡선 확보 | 완료 |
| M4 | R@1 취약성 반증 대조군 | **위협 기각** | 완료 |
| M5 | dose–response 릴리스 복제 (v1.2) | **복제 실패 → 일반 주장 철회** | 완료 |
| M6 | LFMC 인코더 교체(v1→v1.2) frozen-head 실험 | **설계 결함 발견 → 실험 재설계** | 진단 완료 |
| M7 | v1/v1.2 토큰화 계약과 로딩 환경 (C1) | **M1 자작 아님 확인 + 비대칭 발견** | 완료 |
| M8 | v1.2 mask 소비 경로 실측 (C2-A) | **게이트 6/6 통과 — M1 방어 완결** | 완료 |
| M9 | AI-Hub 71363 인벤토리·split 감사 | **공식 split 사용 불가 (valid 110/110 누수)** | 완료 |
| M10 | AOI 군집 spatial holdout 구축·동결 (C2-S) | **게이트 6/6 통과, 동결 완료** | 완료 |
| M11 | Sen12Landslides 접근 감사 (G-0) | **통과 — CC BY 4.0, 지역 단위 선택 수신 가능** | 완료 |
| M12 | annotation-process 감사 (G-A) | **16지역 중 13이 저자 교락, MMU 최대 1,916배** | 완료 |
| M13 | 한국 폴리곤을 Sen12 16지역과 같은 자로 측정 | **한국은 Höhn군과 호환. Italy는 최대 이질 짝** | 완료 |
| M14 | OlmoEarth modality 전수 조사 | **15개. 기상용 비공간 슬롯 `era5_10`이 이미 있음** | 완료 |
| M15 | GK2A 2km 격자 좌표계 역공학 | **4-point in-sample fit 0.896 — 좌표계 확인 주장 철회, 공식 KMA grid로 대체** | 종결/폐기 |
| M16 | KMA API Hub 공식 격자 접근·봉인 | **승인됨. Seal A 완료(격자 파일) / Seal B 미결(창 오프셋)** | 진행 |
| M17 | ASOS 일자료로 `era5_10` 6변수 소급 확보 | **가능. 2022-04-17 실측 성공** | 완료 |
| M18 | apihub 활용신청 5건 실측 | **필요한 격자 API는 아직 403. ASOS 시간자료 열림** | 진행 |
| M19 | apihub 활용신청의 실제 단위 | **서비스가 아니라 개별 API 단위. 필요 목록 확정** | 완료 |
| M20 | 공식 KO 격자 확보 + 봉인 게이트 작동 | **V1·V2 통과, V4 실패 — `x0`가 해석 불가** | 진행 |
| M21 | ASOS 지점 → AOI 군집 결합 | **게이트 4/4. 중위 17.7 km** | 완료 |
| M22 | `era5_10` 6변수 60일 추출 | **5/6 커버리지 ~100%. 강수만 99.1% 공백** | 완료 |
| M23 | G-P pilot 1차 (8 epoch, 1 fold) | **개발 관측. AUPRC 표본추출 결함·test 노출로 확정표에서 제외** | 보존/제외 |
| M24 | 공식 저장소 binary benchmark와의 비교 가능성 | **비교 불가 — task·split·입력·학습이 다름. 난이도 순서도 단정 금지** | 정정 완료 |
| M25 | G-P strict 개발 pilot 독립 복구 | **P4 IoU는 P2-tiny 초과, AP는 78.7%. G-P는 strong baseline 부재로 BLOCKED** | 개발 측정 완료 |

**아직 confirmatory하게 측정하지 않은 것**: frozen OLMo의 full region-macro downstream 자격,
한국 공공데이터의 **표현 기여**(접근·인벤토리·split 감사는 M9·M10에서 했으나 모델 성능 기여는 0),
스위스·네팔 산악 데이터, 압축(PQ/int8) 하에서의 거동, ADC baseline.

---

## M1. 릴리스 전환에서 검색 identity가 깨진다

**근거**: `artifacts/release_audit_full216_v1/analysis_strict1/`

- 모집단: 제주 54 windows × 2023–2026 = **216 site-years**, 입력 5,616파일 56.68 GB SHA-256 고정
- split: calibration 30위치 / embargo 6 / **sealed test 16(64건)** / disclosed-audit 2

| bridge | v1.2→v1 R@1 | v1→v1.2 R@1 |
|---|---:|---:|
| native same-release ceiling | 1.0000 | 1.0000 |
| identity | **0.0000** | **0.0000** |
| calibration mean shift | 0.00024 | 0.0000 |
| translated Procrustes | 0.49097 | 0.43604 |
| affine ridge | 0.69727 | 0.60889 |

사전등록 4방법 × 2방향 **8 gate 전부 실패**. 동시에 pooled CKA **0.97857**,
거리 Spearman **0.95251**, 그런데 동일 token raw cosine **−0.00860**.

실행비 비대칭도 기록됐다: v1 3,756.12초(55.26 crops/s, peak 4,291 MiB) vs
v1.2 2,250.12초(92.25 crops/s, peak 2,719 MiB) = **1.67×**.

- **말할 수 있는 것**: 같은 입력에서 릴리스만 바꾸면 좌표계 identity가 깨진다. 관계구조는 남는다.
- **말할 수 없는 것**: task 정확도, 구름 강건성, 공공데이터 효과, 한국 일반화.
  이 sealed 64는 이미 결과를 봤으므로 **새 방법의 test로 재사용 금지**.

---

## M2. 공개 임베딩 제품 두 개는 paired다. 그러나 계약이 기계에 안 보인다

**근거**: `artifacts/results/majortom_contract_audit.json`
**대상**: `Major-TOM/Core-S2L2A-249k-OlmoEarth-Base`, `Major-TOM/Core-S2L2A-249k-Clay-v1_5`

### 조인

| 조인 키 | 1:1 | 교집합 |
|---|---|---:|
| `unique_id` | **False** | **0** |
| `grid_cell + product_id` | True | **248,719** |
| `grid_cell` | True | 248,719 |

두 데이터셋 모두 248,719행, **동일 스키마 15컬럼**, 첫 행들의 `grid_cell`·`product_id`가 일치.
그런데 **`unique_id`는 교집합이 0**이다 — 데이터셋별 content hash이지 공유 chip 식별자가 아니다.
이름과 스키마 위치가 같아서 **조인 키로 쓰면 조용히 빈 결과가 나온다**(`unique_id_is_a_trap: true`).

### 계약 필드

기계가 읽을 수 있는 스키마에 **8개 전부 부재**:
`model_weights_hash`, `acquisition_dates`, `temporal_recipe`, `band_order`,
`normalization`, `pooling`, `input_content_hash`, `output_content_hash`.

실제로는 두 제품이 다음에서 다르다 — 차원 768 vs 1024, pooling **unmasked token 평균 vs CLS**,
밴드 **12 vs 10**(Clay는 B01·B09 없음), 정규화 OlmoEarth normalizer vs Clay mean/std.
**이 차이는 데이터셋 카드의 산문에만 있다.**

- **말할 수 있는 것**: paired cross-model 실험대가 공개 자산으로 존재한다(총 약 2 GB, CC-BY-SA-4.0).
  그리고 출시된 공개 제품에 계약 메타데이터가 기계 판독 형태로 없다.
- **말할 수 없는 것**: 이건 cross-**family**이지 release pair가 아니다(Major TOM의 OlmoEarth
  릴리스는 하나뿐). 밴드·pooling·정규화가 달라 **모델 우열 비교 불가**.
  chip당 벡터 하나뿐이라 **token/공간 수준 분석 불가**. 라벨 없음.
- **미확인**: SatCLIP·SigLIP·DINOv2·FarSLIP·MMEarth·AlphaEarth·UniverSat의 계약.
  `grid_cell` 단독 조인은 이 subset에서 우연히 1:1일 뿐 **일반 조인 키로 쓰면 안 된다.**

---

## M3. 계약 불일치의 용량–반응 (OlmoEarth v1)

**근거**: `artifacts/results/contract_dose_v1_analysis.json`
**설계**: `config/olmo_release_v1_legacy.yaml`이 밴드 순서를 두 곳에 선언한다는 점을 이용 —
`data.inputs.sentinel2_l2a.bands`만 k쌍 치환하고 정규화기 `band_names`는 그대로 둔다.
정규화 통계가 다른 밴드에 붙지만 파일·차원·실행은 정상이다.
**대상**: disclosed-audit smoke **8 site-years** (sealed 64는 건드리지 않음), 원본 raster 재사용.

**타당성**: dose 0이 frozen `embeddings_audit_v1_legacy`와 **byte-identical 8/8**.
→ dose ≥1의 차이는 전부 밴드 순서 불일치에 귀속된다.

| dose | 이동칸 | same-token cos | linear CKA | dist Spearman | **R@1** |
|---|---:|---:|---:|---:|---:|
| 1 | 2 | +0.9643 | 0.9923 | +0.9617 | 0.9818 |
| 2 | 4 | +0.9584 | 0.9873 | +0.9275 | 0.9246 |
| 3 | 6 | +0.9430 | 0.9797 | +0.8925 | 0.6019 |
| 6 | 12 | +0.9145 | 0.9720 | +0.8318 | 0.2456 |
| reverse | 12 | +0.8628 | 0.9595 | +0.7743 | 0.1613 |

- **말할 수 있는 것**: 용량–반응이 단조롭다. 계측기가 작동한다.
  **dose 6과 reverse는 이동 칸수가 똑같이 12인데 손상이 다르다** — 이동 개수가 아니라
  이동 거리가 중요하며, 단순 카운트로 손상을 예측할 수 없다.
- **말할 수 없는 것**: R@1은 자기검색(표현 프록시)이지 task 정확도가 아니다.

---

## M4. R@1은 취약한 지표가 아니다 (반증 대조군)

**근거**: `artifacts/results/dose_brittleness_control.json`
**동기**: 한 window의 토큰이 공간적으로 인접해 거의 같다면, 작은 섭동만으로 최근접이 넘어가
M3의 R@1 붕괴가 `표현이 깨졌다`가 아니라 `지표가 원래 잘 깨진다`가 된다.

dose 0 임베딩에 크기를 아는 Gaussian 잡음을 넣었다.

| 잡음 | same-token cos | linear CKA | **R@1** | 오검색 |
|---:|---:|---:|---:|---|
| 0.01 | +0.9999 | 1.0000 | 1.0000 | 0건 |
| 0.1 | +0.9950 | 0.9967 | 1.0000 | 0건 |
| **0.3** | **+0.9577** | **0.9505** | **1.0000** | **0건** |

`smallest_noise_matching_dose6 = None` — **시험한 어떤 잡음도 dose 6의 R@1(0.2456)을
재현하지 못했다.** 30% 잡음에서도 4,096 토큰 전부 자기 자신을 찾았다.

- **말할 수 있는 것**: 위협 기각. M3의 R@1 붕괴는 지표 취약성이 아니라 **구조적 이동**이다.
  덤으로 `토큰이 인접해 거의 같다`는 우려도 틀렸다 — 토큰은 임베딩 공간에서 충분히 분리돼 있다.
- **v1에 한정한 관측**: 잡음 0.3은 CKA 0.9505 / R@1 1.0000인데, band-order dose 6은
  CKA 0.9720 / R@1 0.2456이다. **CKA 기준으로는 무해한 쪽이 더 달라 보인다**(순서 역전).
  단 아래 M5에 의해 **이것을 일반 주장으로 쓰지 않는다.**

---

## M5. 복제 실패 — `진단 눈멂` 일반 주장을 철회한다

**근거**: `artifacts/results/contract_dose_v12_analysis.json`,
`artifacts/results/overnight_complete.json`
**설계**: M3과 동일한 축·동일한 8 site-years를 **OlmoEarth v1.2**에서 반복.
**타당성**: v1.2 dose 0도 frozen `embeddings_audit_v1_2_legacy`와 **byte-identical 8/8**.

| dose | v1 CKA / R@1 | **v1.2 CKA / R@1** |
|---|---:|---:|
| 1 | 0.9923 / 0.9818 | **0.7928** / 0.7556 |
| 2 | 0.9873 / 0.9246 | **0.5274** / 0.1487 |
| 3 | 0.9797 / 0.6019 | **0.5193** / 0.0898 |
| 6 | 0.9720 / 0.2456 | **0.4172** / 0.0216 |
| reverse | 0.9595 / 0.1613 | **0.2749** / 0.0020 |

`blind_doses_by_release = {"v1": ["6","reverse"], "v1_2": []}`,
`replicates_across_releases = **False**`.

**v1.2에서는 CKA가 손상을 정확히 따라간다.** 눈멀지 않았다.

분석기에 사전 등록한 규칙(`모든 dose에서 CKA와 R@1이 같이 무너지면 W2 주장을 철회한다`)이
v1.2에 적용된다. 따라서:

- **철회**: `CKA는 계약 불일치에 눈멀었다` / `CKA의 순서가 역전된다`를 **일반 주장으로 쓰지 않는다.**
  v1에서 관측된 **모델 의존적 현상**으로만 기술한다.
- **유지**: 용량–반응 단조성(양 릴리스), 취약성 기각(M4), 하네스 타당성(양 릴리스 byte-identical).

### 대신 나온 탐색적 관측 — 승격하지 않음

같은 불일치에 **v1.2가 훨씬 취약하다**: dose 2에서 **6.2×**, dose 6에서 11×, reverse에서 80×.
운영적으로는 *파이프라인에 잠복한 밴드 순서 버그가 있을 때 v1→v1.2 업그레이드가 그 피해를
6~80배 키운다*는 뜻이다.

**승격하지 않는 이유**: 릴리스 2개는 `비교`이지 `법칙`이 아니고, **사전 등록되지 않았다.**
원인 가설(v1.2 표현이 밴드에 더 특화돼 순열이 전역 구조까지 흔든다)은 가설이며
effective rank·밴드별 ablation 민감도로 시험할 수 있다.

---

## M6. 미세조정은 파트너를 릴리스에 비가역적으로 결합시킨다

**근거**: `frozen_head_swap/frozen_head_swap.json`, `frozen_head_swap/encoder_finetune_diagnosis.json`
**의도**: 배포된 LFMC 모델의 인코더만 v1→v1.2로 올렸을 때 task가 얼마나 나빠지는지 측정.
`R@1=0`(좌표 호환성)과 task 실패 사이의 간극을 메우려 했다.

**실행**: 우리 학습 체크포인트(`epoch=33-step=22270`, test MSE 558.787)의 디코더 12키를 유지하고
인코더 231키를 릴리스 원본으로 교체해 키가 정확히 일치하는 병합 체크포인트를 만든 뒤
평소와 같은 `rslearn model test`로 평가했다.

**타당성 게이트 실패**: v1으로 병합한 대조군이 원본을 재현하지 못했다.

| arm | test MSE |
|---|---:|
| 원본 ep33 (참조) | 558.787 |
| v1 병합 재구성 | **1006.913** (|Δ| 448.126) |

**진단**: ep33의 인코더 텐서 **231개 중 206개가 HF 릴리스 원본과 다르다**
(최대 상대 L2 차이 0.0169, `blocks.*.attn.v.bias` 계열이 가장 큼). lfmc 레시피가
`FreezeUnfreeze(unfreeze_at_epoch=20)`로 백본을 해동하므로 ep33은 인코더까지 미세조정된 상태다.
병합이 그 13에폭을 원본으로 덮어써 성능이 절반으로 떨어졌다.

- **말할 수 있는 것**: end-to-end 미세조정된 배포 모델은 **인코더를 교체할 수 없다.**
  head와 인코더가 co-adapt돼 있어 릴리스 백본을 끼우면 즉시 붕괴한다(MSE 1.80배).
  따라서 파트너의 실제 선택지는 ① v1 유지 ② 전면 재미세조정 ③ 캐시 임베딩 + bridge 뿐이다.
  미세조정은 파트너를 릴리스에 **비가역적으로 결합**시킨다.
- **말할 수 없는 것**: 이 수치는 "v1.2로 올리면 1.8배 나빠진다"가 **아니다.**
  v1 대조군에서 이미 발생한 손해이므로 릴리스 효과가 아니라 **인코더 교체 자체의 비용**이다.
  v1.2 arm은 이 대조군이 무효이므로 실행하지 않았다.
- **재설계**: 논문 주제(아카이브 임베딩 재사용)에 맞는 설정은 **인코더 동결 + head 학습**이다.
  v1 동결 특징으로 디코더를 학습하고, 같은 디코더에 v1.2 동결 특징을 투입한다.
  두 arm 모두 "동결 인코더 + 같은 head"가 되어 바뀐 변수가 릴리스 하나로 통제된다.

## M7. M1은 자작이 아니다. 그러나 두 릴리스는 밴드 구조가 비대칭이다

**근거**: `release_tokenization_probe/release_tokenization_probe.json`,
`release_audit_p0/checkpoints.json`의 environment 기록
**질문 (C1)**: M1의 `R@1=0`이 v1.2를 잘못된 입력 계약으로 돌린 결과인가?

### 로드된 모델에서 직접 확인한 구조

| | v1 | v1.2 |
|---|---|---|
| 파라미터 | 88.96 M | **113.99 M** |
| S2 band group | override 없음 → rslearn 기본 **3 band_set** | **12밴드 단일 그룹** |
| `token_pooling` 기본값 | True | True |

rslearn `_prepare_modality_inputs`는 `num_band_sets`를 `Modality.get(m).band_sets`=**3**으로
계산한다. 릴리스와 무관하다. 즉 v1.2에는 mask 3-set과 model 1-group의 불일치가 존재한다.

**그럼에도 M1은 유효하다.** `token_pooling=True`가 시간·모달리티 축을 patch 단위로 pooling해
(model.py L373–377) 출력이 릴리스와 무관하게 공간 patch당 768-d 하나가 된다. 따라서
`same-token` 비교의 단위는 band group이 아니라 공간 patch이고, 두 릴리스 간 정의가 같다.

### 로딩 환경 — 재현에 필수인 사실

감사가 기록한 환경은 `rslearn 0.1.13 + olmoearth_pretrain 0.0.6`이다.

| 환경 | 패키지 | v1.2 로드 |
|---|---|---|
| `.venv` (uv.lock 고정) | rslearn 0.0.27 + `olmoearth_pretrain` 0.0.2 | **불가** |
| `.venv-master` (감사 환경) | rslearn 0.1.13 + `olmoearth_pretrain_minimal` 0.0.6 | 가능 |

`.venv`의 `ModelID`에는 v1 변종만 있고, `model_path`로 v1.2를 로드하면 원인 안내 없이
`RuntimeError: Error(s) in loading state_dict` — `per_modality_channel_embeddings.sentinel2_l2a`가
checkpoint `[1,192]` vs model `[3,192]`, `rope_mixed_freqs`·`pixel_proj`가 unexpected로 뜬다.
`.venv-master`의 `ModelID`에는 v1.1·v1.2 엔트리가 있다.

- **말할 수 있는 것**: M1은 로딩·계약 오류의 산물이 아니다. 밴드 순서도 v1.2 선언 순서와 일치했고
  (2026-08-24 확인), 구조 차이는 pooling으로 흡수된다. **재현에는 rslearn ≥0.1.x가 필요하며
  레포의 lockfile 환경으로는 v1.2 arm을 돌릴 수 없다.**
- **말할 수 없는 것**: (M8에서 해소됨 — forward 내부 mask 경로를 실측했다.)
- **파생 제약 (PhilEO P0 설계에 직접 영향)**: PhilEO S2는 10밴드로 `band_set 0+1`과 정확히 일치하고
  없는 B01·B09는 `band_set 2` 전체다. v1에서는 band_set 하나의 부재로 표현 가능하지만,
  **v1.2는 12밴드가 단일 그룹이라 같은 방식으로 표현할 수 없다.** 즉 10밴드 입력을 두 릴리스에
  **대칭적으로** 줄 방법이 없고, 어떤 처리를 하든 릴리스 의존적 차이가 주입된다.
  이것을 통제하지 못하면 P0의 task-risk 비교가 오염된다.

## M8. v1.2는 mask slice 0만 읽는다. band_set 2를 MISSING으로 표시해도 조용히 무시된다

**근거**: `mask_path_c2a/mask_path_c2a.json`, `code/probe_mask_path_c2a.py`
**환경**: rslearn 0.1.13 + olmoearth_pretrain_minimal 0.0.6 (M1을 만든 그 환경)
**설계**: 결정적 합성 입력 1개(32×32, T=2, 12채널, seed 20260824). 공간 위쪽 절반만 MISSING으로
표시해 "모든 토큰 masking" assertion을 회피했다. 사전 등록 게이트 6개.

| 게이트 | 내용 | 결과 |
|---|---|---|
| G1 | encoder 출력 S축: v1=3, v1.2=1 | PASS |
| G2 | rslearn 입력 mask S축 = 3 (릴리스 무관) | PASS |
| G3 | v1.2에서 slice 1·2를 MISSING → 출력 byte-identical | PASS |
| G4 | v1.2에서 slice 0을 MISSING → 출력 변화 | PASS |
| G5 | v1에서 slice 2를 MISSING → 출력 변화 | PASS |
| G6 | v1.2에서 slice 2 MISSING이 fast_pass만 끄고 출력은 동일 | PASS |

측정값:

| 릴리스 | num_bandsets | token shape | slice 0 MISSING | slice 1·2 MISSING | slice 2 MISSING |
|---|---|---|---|---|---|
| v1 | 3 | `[1,8,8,2,3,768]` | max\|Δ\| 5.38228 | 5.49309 | 4.79137 |
| v1.2 | 1 | `[1,8,8,2,1,768]` | max\|Δ\| 5.34337 | **0.0 (byte-identical)** | **0.0 (byte-identical)** |

**메커니즘 (설치된 소스에서 확인)**:
`flexi_vit.py` per-modality embedding 루프가 `for idx in range(num_band_sets)`이고
`num_band_sets = self.tokenization_config.get_num_bandsets(modality)` — 즉 **모델 쪽** 값이다.
v1.2는 1이므로 `modality_mask[..., 0]`만 읽히고 slice 1·2는 접근되지 않는다.

- **말할 수 있는 것**:
  1. M1의 same-token 비교는 유효하다. 두 릴리스 모두 pooling 후 공간 patch당 768-d 하나를 낸다.
     `R@1=0`은 토큰 개수를 임의로 대응시킨 결과가 아니다.
  2. **v1.2에서는 partial-group missingness를 표현할 수단이 없다.** band_set 2(B01·B09)를
     MISSING으로 표시하는 것이 v1에서는 실제 효과가 있고 v1.2에서는 완전히 무시된다.
  3. **G6 — 이 무시가 조용하다.** rslearn의 `fast_pass`는 입력 mask 3 slice 전체를 보고 결정되므로
     slice 2를 MISSING으로 두면 `fast_pass=False`로 바뀌어 pooling이 masked-average 경로로 간다.
     그런데 출력 mask는 S=1이고 MISSING이 없으므로 결과는 baseline과 byte-identical이다.
     사용자는 "밴드 부재를 선언했다"고 믿지만 아무 일도 일어나지 않았고, 경고도 없다.
- **말할 수 없는 것**: 이 측정은 합성 입력 1개다. 실제 PhilEO 타일에서 downstream 지표가
  어떻게 갈리는지는 별개(C2-B)다. `use_register_bottleneck_output` 경로는 시험하지 않았다.
- **논문 문구**: "R@1=0은 서로 다른 token 개수를 임의로 대응시킨 결과가 아니다. 두 릴리스 모두
  동일한 공간 patch마다 768차원 출력을 생성한다." 이것은 representation compatibility failure이며
  downstream task failure를 직접 의미하지는 않는다(그것이 C2-B의 질문이다).

## M9. AI-Hub 71363의 공식 split은 쓸 수 없다. valid 타일 110개 전부가 train과 겹친다

**근거**: `inventory/inventory_audit.json`, `inventory/split_leakage_audit.json`,
`code/build_aihub_inventory.py`, `code/audit_aihub_split_leakage.py`
**환경**: 네트워크 미사용. 수신한 zip만으로 판정했다.

### 먼저 — 내가 틀렸던 것

"678 타일 × 63 날짜"라고 썼다. **관측된 조합이 아니다.** Cartesian product로 만들면
42,714개의 인공 조합이 되고 원 분포와 다른 데이터셋이 된다. 실제는 다음이다.

| | 값 |
|---|---|
| 실제 (타일, 날짜) 쌍 | **2,699** |
| 고유 타일 | 594 (train 485 / valid 110) |
| 고유 날짜 | 60 (train 56 / valid 13) |
| 타일당 날짜 수 | 1 ~ 8 |
| 플랫폼 | SENTINEL-2A 1,961 / 2B 738 |
| WGS84 범위 | 125.14–129.59 E, 34.01–38.31 N |

수집 단위는 `(sample geometry, img_time, platform, original grid)`이며 곱집합이 아니다.

### A1 ID 조인 — 통과

메타데이터 3,000 · 라벨 2,700 · **교집합 2,699** (label_only 1, metadata_only 301).
`SA`/`SB` 접두는 두 쪽에 모두 있다(메타 SA 1,962 / SB 1,038). Major TOM식 교집합 0 함정은 없다.

### A2 기하 해석 — 추측을 제거했다

메타데이터 `coordinates`가 중심인지 좌상단인지 몰랐다. 라벨 폴리곤 범위와 대조해 판정했다.

| 가설 | 중위거리 |
|---|---|
| **upper_left** | **4.2e-05 m** (400/400 투표) |
| center | 7,240.77 m |
| lower_left | 10,240.00 m |

**좌상단이다.** 사실상 정확일치이므로 bbox는 `[x, y-10240, x+10240, y]`로 확정된다.

### split 누수 — 공식 split 사용 불가

| 게이트 | 결과 | 판정 |
|---|---|---|
| L1 tile_id 누수 | 공유 타일 1개 | FAIL |
| L2 날짜 누수 | 공유 날짜 9개 | FAIL |
| **L3 공간 중첩** | **642쌍, valid 타일 110/110 영향** | **FAIL** |
| L4 근접(1타일 폭 이내) | 646쌍 | FAIL |
| L5 AOI 군집 | 13군집 중 5개가 양쪽에 걸침 (최대 군집 180타일) | FAIL |

**valid 타일의 100%가 train 타일과 실제로 겹친다.** A2가 정확일치였으므로 bbox 계산
오류가 아니다. 타일이 설계상 중첩(sliding window)일 수 있으나, 그렇다면 결함은 타일링이
아니라 **분할 방식**에 있다.

- **말할 수 있는 것**: 이 데이터셋의 공식 train/valid split으로 낸 수치는 공간 누수로
  부풀려진다. 우리는 제공된 split을 쓰지 않고 **13개 AOI 군집 단위로 spatial holdout을
  직접 만든다.** 군집이 13개뿐이므로 leave-one-cluster-out이 자연스럽다.
- **말할 수 없는 것**: 중첩이 데이터 구축 의도인지 실수인지는 모른다(문서 미확인).
  누수가 실제 성능을 얼마나 부풀리는지는 아직 측정하지 않았다 — 누수 split과 우리 holdout을
  같은 head로 비교해야 수치가 나온다.
- **다음 조치**: 군집 단위 holdout을 먼저 만들고, test 군집을 **동결**한다.
  이미 탐색에 쓴 valid 300은 test로 쓰지 않는다.

## M10. AOI 군집 단위 spatial holdout — 동결 완료

**근거**: `artifacts/aihub71363_spatial_holdout.json`, `..._tile_assignment.jsonl`,
`..._loco_folds.json`, `code/build_aihub_spatial_holdout.py`
**전제**: M9에서 공식 split이 사용 불가로 판정됨 (valid 110/110 공간 중첩).

### 동결된 분할 (SHA-256은 타일 목록의 해시다 — 바뀌면 split이 바뀐 것)

| split | 군집 | 타일 | 관측쌍 | 벌목지 | 산사태 | sha256(앞16) |
|---|---|---|---|---|---|---|
| train | C01·C11·C13 | 393 | 1,733 | 131 | 56 | `50fdcb4b6b404d41` |
| val | C04·C09·C10 | 84 | 408 | 10 | 12 | `8e133c51db9b2eb5` |
| test | C02·C03·C05·C06·C07·C08·C12 | 113 | 542 | 26 | 22 | `3f44498758600c3f` |
| excluded | (C12 일부) | 4 | 16 | — | — | `e67c09a5b013cf73` |

합계 594 타일 / 2,699 관측쌍. `excluded`는 이미 집계 감사에 쓴 공식 valid 타일이라
test에서 뺐고, train으로 옮기면 공간 분리가 깨지므로 **어느 split에도 넣지 않았다.**

### 게이트 (사전 등록, 6/6 통과)

| | 내용 | 결과 |
|---|---|---|
| S1 | 군집 간 최소 이격 ≥ 1 타일 폭(10.24 km) | **20,480.0 m** |
| S2 | test가 두 희소 클래스를 모두 포함 | 통과 |
| S3 | val이 두 희소 클래스를 모두 포함 | 통과 |
| S4 | test에 기탐색 공식 valid 타일 없음 | 0건 |
| S5 | 모든 타일이 배정됨 | 594/594 |
| S6 | val·test가 희소 클래스마다 ≥10 타일 | 최소 10 |

### 도중에 고친 결함 세 개 (전부 기록에 남긴다)

1. **군집 간 거리를 bounding box로 쟀다.** 군집들이 지리적으로 맞물려 있어 extent는 겹치고
   `min_gap = 0`이 나왔다. **타일 단위 거리**로 다시 재니 정확히 20,480 m — union-find의
   연결 기준과 일치한다. 즉 구성상 보장된 성질을 잘못된 척도로 재서 놓칠 뻔했다.
2. **희소 클래스를 '존재 여부'로 셌다.** 그래서 val의 산사태가 4타일뿐이었다. '타일 수'로 바꿨다.
3. **test가 희소 군집을 먼저 독식했다.** 산사태 90타일 중 56이 대형 군집 C11·C13에 몰려 있어
   val 예산으로는 받을 수 없다. **두 split의 쿼터를 동시에 채우는** 규칙으로 바꿨다.
   예산을 특정 군집에 맞춰 바꾸는 것은 cherry-picking이므로 하지 않았다.

- **말할 수 있는 것**: 분할은 결정적이다(같은 입력 → 같은 분할, 사람의 선택 없음).
  군집이 13개뿐이므로 leave-one-cluster-out 13폴드도 함께 생성했다.
- **말할 수 없는 것**: 공식 split의 누수가 성능을 얼마나 부풀리는지는 **아직 측정하지 않았다.**
  같은 head를 두 분할에서 학습·평가해야 수치가 나온다 (C2-B에서 함께 잰다).
  test 군집의 희소 클래스 절대량(벌목 26 / 산사태 22 타일)은 작으므로, 신뢰구간을
  spatial bootstrap으로 반드시 붙인다.

## M11. Sen12Landslides 접근 감사 — G-0 통과

**근거**: `sen12landslides/audit/access_audit.json`, `code/audit_sen12landslides_access.py`
**출처**: HuggingFace `paulhoehn/Sen12Landslides` (실물 확인. 논문·검색 요약을 근거로 쓰지 않았다)

| 항목 | 실측 |
|---|---|
| 라이선스 | **CC BY 4.0** (`doi:10.57967/hf/5883`) — 파생물 공개 가능 |
| 총 용량 | 170.5 GB |
| harmonized S2 | **39.42 GB / 28 파트** |
| 파트 구성 | **지역별로 묶여 있다** (part01 = chimanimani 500개) → **필요한 지역만 수신 가능** |
| 패치 | 128×128 px @10 m, **15 timestep** |
| S2 밴드 | **B02–B12 (10밴드)** + SCL + MASK + DEM |
| 메타 속성 | `event_date`, `date_confidence`, `pre_post_dates`, `annotated`, `crs`, `center_lat/lon` |
| inventory | `inventories.zip` 22 MB, shapefile 1개 (74,956 폴리곤) |

**두 가지가 설계에 직접 영향을 준다.**

1. `data_harmonized`가 **ESA Baseline 04.00의 +1000 DN offset을 이미 보정**했다
   (2022-01-25 이후). 앞서 걱정한 PB04 문제를 데이터셋 저자가 처리했다.
   단 `data_raw`는 보정하지 않았으므로 **둘을 섞으면 안 된다.**
2. S2가 **B02–B12 10밴드로 B01·B09가 없다.** PhilEO와 같은 구조다. 즉 M8의 비대칭이
   여기서도 그대로 적용된다 — v1에서는 band_set 2 부재로 표현 가능하고 v1.2에서는 무시된다.
   설계가 정한 "v1 + B01·B09 band-group missing mask" 경로가 맞다는 뜻이다.

## M12. annotation 교락 — Sen12Landslides의 지역은 라벨 저자와 거의 같다

**근거**: `sen12landslides/audit/annotation_audit.json`, `code/audit_annotation_process.py`
**질문 (R2/G-A)**: leave-one-region-out 성능 하락이 지형 차이인가 라벨 차이인가?

### 측정 (inventory 74,956 폴리곤, 16지역, 저자 5명)

| region | 폴리곤 | MMU(p1) m² | median m² | 최다 저자 점유 |
|---|---|---|---|---|
| Italy | 47,522 | **62.9** | 409.8 | Ferrario 1.00 |
| DominicaMaria | 10,172 | 219.7 | 3,191.9 | Emberson 1.00 |
| Newzealand | 3,242 | 1,038.2 | 7,905.8 | Höhn 1.00 |
| Chimanimani | 2,513 | 441.7 | 2,813.4 | Höhn 1.00 |
| Kyrgyzstan1 | 2,405 | 2,230.3 | 8,640.2 | Höhn 1.00 |
| Hokkaido | 2,340 | 562.0 | 7,391.3 | Höhn 1.00 |
| Indonesia | 2,097 | 8,757.4 | 26,769.9 | Höhn 0.91 |
| USA_Alaska | 103 | **120,569.4** | 466,259.8 | Belair 1.00 |
| **Nepal** | **8** | 4,640.2 | 72,720.5 | Höhn 1.00 |

- **13/16 지역이 단일 저자 90% 이상** → leave-one-region-out은 부분적으로
  leave-one-**annotator**-out이다.
- **MMU 비가 최대 1,916배** (Italy 62.9 vs USA_Alaska 120,569). 10배 이상 차이나는 지역쌍 **50개**.
- median 면적이 409.8 m²(Italy) ~ 466,259.8 m²(USA_Alaska)로 **1,100배** 퍼져 있다.

### 내가 제안한 harmonization은 실패한다

전 지역 공통 면적 하한을 `max(MMU) = 120,569 m²`로 잡으면 Italy는 p99가 6,965 m²이므로
**사실상 전멸한다.** 단순 면적 하한으로는 조화가 불가능하다. 이 실패를 기록으로 남긴다.

### 해결 — 저자를 고정한 LOCO

Höhn et al. (2025) 단독으로 **14지역 16,306 폴리곤**을 덮는다.

| | 전체 저자 | **Höhn 단독** |
|---|---|---|
| 지역 | 16 | 14 (≥100 폴리곤 **11**) |
| MMU 비 | **1,916×** | **20×** |
| 공통 하한 8,821.8 m² 적용 | Italy 전멸 | 11지역 **15.2~99.0%** 보존, 합계 7,921 폴리곤 |

11지역: Chimanimani · China · Hiroshima · Hokkaido · Indonesia · Itogon ·
Kyrgyzstan1 · Kyrgyzstan2 · LanaoDelNorte · Newzealand · Thrissur

- **말할 수 있는 것**: annotation 교락을 **설계로 제거**할 수 있다. 저자 고정 11지역 LOCO는
  독립 표본 11개이고, 3국 LOCO(n=3)보다 신뢰구간을 만들 수 있다.
- **말할 수 없는 것**: 저자를 고정해도 MMU가 20배 남는다. 면적 하한의 보존율이
  Chimanimani 15.2%로 낮으므로, 하한값에 대한 **민감도 분석**이 필요하다.
  라벨 정확도 자체(폴리곤이 옳은가)는 검증하지 않았다.
### 이 교락이 원 논문의 결과에 실제로 닿는가 — 닿는다

원 논문(Sci Data 2025)을 확인했다.

- **Experiment 3에서 leave-one-cluster-out을 수행한다.** 6개 지리 군집:
  Americas / Europe / Africa / Central Asia / Southeast Asia / Oceania.
- 라벨 잡음은 인정한다 — *"the ground truth data was noisy, particularly in inventories
  derived from deep learning models"*. slope < 7% 제외 등 정제도 했다.
- **그러나 원 inventory 간 도화 기준(최소 도화 면적·상세도) 이질성에 대한 명시적 논의는 없다.**

우리 측정과 겹쳐 보면: inventory에 유럽 지역은 Italy 하나뿐이고 Italy = Ferrario 47,522
(전체의 63%, MMU 62.9 m²)다. 따라서 **Europe 군집을 hold-out하는 것은 사실상 한 저자를
hold-out하는 것**이고, 그 저자의 MMU는 다른 지역보다 최대 1,916배 작다.
Americas 군집도 DominicaMaria(Emberson) + USA_*(Belair)로 저자가 갈린다.

- **말할 수 있는 것**: 공개 benchmark의 지역 일반화 평가가 annotation 이질성과 교락돼 있고,
  원 논문이 이를 논의하지 않는다. M9(공식 split 공간 누수)와 **같은 종류의 benchmark validity
  결과**이며, 재현 스크립트가 있다.
- **말할 수 없는 것**: 원 논문의 6개 군집 각각의 정확한 지역 구성을 논문에서 확인하지
  않았다(우리 inventory 지역 목록으로 추정했다). 교락이 그들의 보고 수치를 **얼마나**
  바꾸는지도 측정하지 않았다 — 저자 고정 LOCO와 raw LOCO를 같은 head로 비교해야 나온다.

- **설계 변경**: **네팔은 이 inventory에서 폴리곤이 8개뿐이다.** 문서의 네팔 arm은
  Sen12Landslides로 성립하지 않는다. BIPAD/ICIMOD에서 따로 와야 하고 headline 지역이 아니다.

## M13. 한국은 Höhn 군집과 annotation 호환됨. Italy→Korea는 최대 이질 짝임

**근거**: `aihub/audit/korea_annotation_audit.json`,
`code/audit_korea_vs_sen12_annotation.py`, `code/korea_sliver.py`
**질문**: `Italy 학습 → Korea 적용` arm이 성립하는가. 한국을 저자 고정 LOCO에 넣을 수 있는가.

### 측정 대상

AI-Hub 71363 라벨 GeoJSON의 `ANN_CD=80`(산사태·토석류) 폴리곤 **8,350개 / 90타일**.
좌표가 EPSG:32652 미터라 shoelace로 면적을 바로 구했음. 재투영 안 했음.
`ANN_CD=70`(벌목지)도 같이 냈음 — 2,643개 / 167타일.

### 먼저 발견한 함정 — MMU가 0.046 m²로 나왔음

원인은 도화 기준이 아니라 **타일 경계 절단물**이었음.

| 면적 구간 | 개수 | 비율 |
|---|---|---|
| < 1 m² | 134 | 1.60% |
| < 100 m² | 150 | 1.80% |
| < 400 m² | 229 | 2.74% |

100 m² 미만 150개는 **전부 꼭짓점 4~5개짜리 사각형**임. 1024×1024 타일로 자를 때 생긴
조각임. 즉 `p1`을 MMU 대리로 쓰면 **클리핑 상태가 다른 데이터셋끼리 비교가 무효**가 됨.
Sen12Landslides inventory는 전역 shapefile이라 잘리지 않았고, AI-Hub는 타일 단위로 잘렸음.

**400 m² 하한을 걸면 2.74%만 버려지고 결과가 뒤바뀜.**

| | 필터 전 | **필터 후 (≥400 m²)** |
|---|---|---|
| n | 8,350 | 8,121 |
| MMU(p1) | 0.046 m² | **549.1 m²** |
| median | 5,410.8 m² | 5,731.8 m² |
| p99 | 281,983 m² | 290,935 m² |

### 한국의 자리 — Höhn 군집 한가운데임

| | MMU(p1) m² | median m² |
|---|---|---|
| Höhn 11지역 범위 | 216.2 ~ 8,757.4 | 2,813.4 ~ 46,104.1 |
| **Korea (필터 후)** | **549.1** | **5,731.8** |
| Italy (Ferrario) | 62.9 | 409.8 |
| USA_Alaska (Belair) | 120,569.4 | 466,259.8 |

한국은 MMU·median 모두 **Höhn 범위 안에 들어감**. Hiroshima(4,162)와 Kyrgyzstan2(5,512) 사이임.

### Italy → Korea 판정

| 비교 | 값 | G-A 임계 |
|---|---|---|
| MMU 비 (필터 후) | **8.7×** | 10× 미만 → 통과 |
| **median 비** | **14.0×** | — |
| Italy 점유율 | 전체 폴리곤의 **63%** | — |

**MMU 비만 보면 통과하지만 median 비가 14배임.** 즉 `MMU 비 ≥ 10×` 단독 기준은 불충분함.
분포 수준 비교(median 비 또는 log-면적 분포 거리)를 같이 걸어야 함.

- **말할 수 있는 것**:
  1. 한국을 저자 고정 LOCO의 **12번째 지역**으로 넣을 수 있음. annotation 호환됨.
  2. `Italy → Korea`는 **의도적인 최대 annotation-shift 짝**으로 쓰는 것이 정확함.
     라벨이 풍부한 곳에서 학습해 라벨이 없는 곳에 적용하는 현실 배치 시나리오와도 맞음.
  3. 면적 하한 400 m²는 **클리핑 정규화**이지 도화 기준 조화가 아님. 둘을 구분해야 함.
- **말할 수 없는 것**: 면적 분포가 비슷하다고 라벨이 같은 것을 가리킨다는 보장은 없음.
  현상 정의(산사태·토석류 vs debris flow)와 시점 의미(촬영일 vs event date)는 아직 대조 안 했음.
  한국 폴리곤의 정확도 자체도 검증 안 했음.
- **G-A 수정 사항 2개**:
  ① MMU 대리를 `p1` → **클리핑 정규화 후 p1**로 바꿈
  ② `MMU 비 ≥ 10×` 단독 판정 → **MMU 비와 median 비를 같이** 봄

## M14. OlmoEarth에는 기상 전용 비공간 modality 슬롯이 이미 있음

**근거**: `code/probe_modalities.py`, `.venv-master` (rslearn 0.1.13 + olmoearth_pretrain_minimal 0.0.6)
**질문**: 한국 위성·기상 자료를 넣으려면 새 modality를 만들어야 하는가?

### 등록된 modality 15개

| modality | band_set | is_spatial | tile_factor | 밴드 |
|---|---|---|---|---|
| `sentinel2_l2a` | **3** | True | 1 | B02B03B04B08 / B05B06B07B8AB11B12 / B01B09 |
| `sentinel1` | 1 | True | 1 | **vv, vh** |
| `landsat` | 2 | True | 1 | B8 / B1–B7,B9–B11 |
| `naip` / `naip_10` | 1 | True | 1 / **4** | R,G,B,IR |
| **`era5_10`** | 1 | **False** | **−256** | **2m-temperature, 2m-dewpoint-temperature, surface-pressure, 10m-u-component-of-wind, 10m-v-component-of-wind, total-precipitation** |
| `srtm` | 1 | True | 1 | srtm |
| `latlon` | 1 | False | 1 | lat, lon |
| `worldcover` / `worldpop` / `cdl` / `worldcereal` / `gse` / `openstreetmap_raster` / `wri_canopy_height_map` | 1 | True | 1 | (보조 레이어) |

### 결론 — 기존 슬롯은 후보일 뿐, source contract를 새로 검증해야 함

이것이 M8과 직결됨. 새 센서를 위해 아키텍처를 바꾸는 것이 아니라,
**이미 사전학습된 슬롯에 다른 출처의 자료를 넣었을 때 계약이 지켜지는가**가 질문임.

| 넣으려는 것 | 후보 슬롯 | 근거 |
|---|---|---|
| **아리랑 5호 (KOMPSAT-5)** | 직접 재사용 불가. 새 sensor adapter 또는 공통 VV-only baseline 후보 | K5는 **X-band 9.66 GHz·single polarization**, S1은 **C-band 5.405 GHz·통상 VV+VH**. 주파수·편파 계약이 다름 |
| **KMA ASOS 지상관측** | `era5_10`은 구조적 후보 | 변수 이름은 대응하지만 관측높이·단위·누적 강수·점→격자·u/v 변환이 달라 exact match가 아님 |
| GK2A 구름·에어로졸 | 대응 슬롯 없음 | CLD/AOD/FOG/COT/CT는 어느 슬롯에도 없음. `r_t`로 head 쪽에서 결합해야 함 |
| 차세대중형위성 4호 (5 m) | 없음 | GSD·밴드가 어느 슬롯과도 안 맞음 |

- **말할 수 있는 것**:
  1. `era5_10`이 **비공간(is_spatial=False, tile_factor=−256)** 슬롯이므로, 기상 시계열을
     받는 구조적 경로는 존재함. 그러나 ASOS를 넣어도 사전학습 분포 안이라는 뜻은 아님.
  2. **KMA ASOS 일자료는 이미 승인됐고 과거 이력이 있음.** GK2A 경량화 endpoint의
     D-1/D-2 창과 달리 소급 실험이 가능함. 따라서 retrospective forcing arm의 1순위는 ASOS임.
  3. 아리랑 5호는 `sentinel1`이라는 이름으로 재포장하면 안 됨. 현재 OLMo task config는
     Sentinel-1 IW의 `vv`,`vh` 두 밴드와 dB 변환을 요구하므로 K5 single-pol에서 fake VH를
     만들지 않는다.
- **말할 수 없는 것**: ASOS 관측값을 `era5_10`이 기대하는 단위·정규화로 맞출 수 있는지는
  **미확인**임. ERA5 total precipitation과 ASOS 강수의 시간 누적·단위도 같다고 가정할 수 없다.
  ERA5는 재분석 격자, ASOS는 지점 관측이므로 공간 보간이 필요하고 그 보간이 또 하나의 계약
  변경이다. K5↔S1은 주파수·편파·입사각·보정수준이 함께 바뀌는 cross-sensor stress test다.
  이 슬롯들에 다른 출처를 넣었을 때 M8식 조용한 무시가 일어나는지도 아직 시험하지 않았음.

## M15. GK2A 4-anchor 좌표 역공학은 판정 불가 — 공식 KMA grid로 대체

**근거**: `gk2a/_crs/grid_crs_test.json`, `code/solve_gk2a_grid_crs.py`,
`code/gk2a_offset_search.py`, 앵커 캐시 `gk2a/_crs/area_anchors.jsonl`
**문제**: 한반도(All) 응답에 CRS가 없음 (`gridKm=2.0 xdim=320 ydim=397 x0=63 y0=333` 뿐).
좌표를 못 붙이면 격자를 AOI에 쓸 수 없음.

### 당시 검정 설계 (사전 등록)

`Area` 계열이 행정동코드에 대해 `(lon, lat, value)`를 줌. 같은 시각 격자에서 그 위치의
칸 값을 뽑아 **일치율**을 봄. 구름탐지는 3클래스이므로 우연 수준은 약 0.33임.

| 일치율 | 판정 |
|---|---|
| ≥ 0.90 | 채택 |
| 0.50–0.90 | 오프셋 보정 후 재검정 |
| < 0.50 | 기각. 문서 확보 전까지 격자를 쓰지 않음 |

### 실측 결과와 재판정

| 단계 | 가정 | 일치율 |
|---|---|---|
| H1 | 동네예보 5 km 오프셋을 2 km로 환산 (XO2=107.5, YO2=340), y축 그대로 | **격자 밖** (전 관측 이탈) |
| H1′ | 같은 오프셋, **y축 뒤집음** | 0.385 (우연 수준) |
| **H2** | LCC 유지, 오프셋을 탐색 (xo·yo를 −200~400 / −100~700에서 전수) | **0.8958** (86/96) |

H2의 최적해는 `xo ≈ 124~125, yo = 664`, row-major, y축 안 뒤집음이었다. 최초 코드에는
cache key에서 `resultType`이 빠져 FOG Area가 CLD Area를 덮어쓴 채 CLD All과 비교될 수 있는
결함도 있었다. CLD만 명시적으로 filter한 재실행에서도 수치는 우연히 같은 0.8958(86/96)이었다.
그러나 96개 비교는
**고유 공간점 4개 × 24시각**이고, 두 offset을 같은 4개 지점으로 선택하고 평가했다. 범주별
기저확률도 균등 0.33으로 검증하지 않았다. 따라서 0.8958은 held-out projection score가 아니며,
LCC·저장순서·y축을 “확인”했다는 주장을 철회한다.

- **말할 수 있는 것**: 4-point fitted candidate가 86/96을 맞췄고, Area parsed row는
  192건(4지점 × CLD·FOG × 12슬롯 × 2일)이다.
- **말할 수 없는 것**: projection family, offset, row/column order, y direction의 외부 타당성.
- **대체 경로 발견**: KMA API Hub는 KO/2 km의 공식 lon·lat을 grid 저장순서대로 ASCII로
  조회하거나 NetCDF로 받는 endpoint를 제공한다. 좌표는 추정하지 않고 이 파일을 SHA-256 고정한다.
- **종결 기준**: 공식 lon/lat과 경량화 값이 모두 `320 × 397 = 127,040`이고 순서가 exact match한
  뒤에만 AOI join을 연다. 기존 offset 코드는 audit-only 실패 기록으로 남긴다.

## M16. 공식 KO/2km 격자 — Seal A 완료, Seal B 미결

**근거**: `gk2a/_grid/grid_seal.json`, `code/seal_gk2a_grid.py`
**배경**: M15의 역공학은 철회됐음(4점 in-sample fit). 기상청이 KO/2km lat/lon을
**격자 저장 순서대로** 제공하므로 역공학이 불필요함.

### 공식 경로 (2026-08-25 확인)

```
ASCII   apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-gk2a_latlon_api
        ?area=KO&grid=2&latlon=lon|lat&disp=A&authKey=
NetCDF  apihub.kma.go.kr/api/typ01/url/gk2a_latlon_file_down.php?area=KO&grid=2&authKey=
```

### 실측 — 401과 403을 구분해야 함

| 호출 | 응답 |
|---|---|
| 키 없음 | `401 유효한 인증키가 아닙니다` |
| **키 있음** + GK2A latlon | `403 활용신청이 필요한 API 입니다` |
| 키 있음 + NetCDF 다운로드 | `403` 동일 |
| 키 있음 + **전혀 다른 API** (`kma_sfctm2` 지상관측) | **`403` 동일** |

마지막 줄이 진단을 확정함. **키 자체는 유효하고**(401이 아님), apihub는 data.go.kr과 같이
**API별 활용신청**을 요구하며 현재 신청된 API가 없음.

### 봉인 스크립트는 준비됨 — 사전 등록 게이트 4개

`code/seal_gk2a_grid.py`. 활용신청이 승인되면 그대로 실행함.

| | 검증 | 실패 시 |
|---|---|---|
| V1 | lon·lat 원소 수 = `xdim × ydim` = 320 × 397 = 127,040 | 봉인 안 함 |
| V2 | row-major 저장 순서 — 행 내 경도 단조증가, 열 내 위도 단조 | 저장 순서가 다른 것이므로 봉인 안 함 |
| V3 | 경도 120~135, 위도 30~45 (한반도) | 봉인 안 함 |
| V4 | **자유 매개변수 0개** 앵커 검증. 공식 격자에서 최근접 칸을 찾아 같은 시각 CLD 값과 대조. 일치율 ≥ 0.90 | 봉인 안 함 |

V4가 M15와 결정적으로 다른 점: **아무것도 적합하지 않음.** 대응이 공식 파일에서 직접 오므로
높은 일치율은 증거가 되고 낮은 일치율은 반증이 됨. M15의 캐시 결함(`resultType` 누락으로
FOG가 CLD를 덮어씀)도 필터로 막았음.

- **말할 수 있는 것**: 공식 경로가 존재하고 키는 유효함. 봉인 절차와 판정 기준이 사전 등록됨.
- **말할 수 없는 것**: 격자 대응은 **아직 확정되지 않았음.** 320×397 순서 가정도 미검증임
  (V2가 그것을 검증할 항목임). GK2A admission 실험은 봉인 전에 열지 않음.
- **필요한 사용자 조치**: apihub.kma.go.kr에서 **GK2A 기상산출물** API 활용신청.
  겸해서 **지상관측(ASOS)** 도 신청하면 M14의 `era5_10` 슬롯 경로가 열림.

### 2026-08-25 갱신 — 활용신청이 승인되어 파일을 받았음. 위 403 절은 그 이전 상태임

`nph-gk2a_latlon_api`가 **http 200**으로 열렸음. 위의 401/403 진단 이력은 보존하되,
아래가 현재 상태임. **그리고 사전 등록 게이트 표의 V1·V3이 틀렸음을 실측이 드러냈음.**

#### 내가 사전 등록을 잘못했던 두 가지

| 게이트 | 사전 등록값 | 실측 | 왜 틀렸나 |
|---|---|---|---|
| V1 | 원소 수 = `320 × 397` = 127,040 | **810,000 = 900 × 900** | 공식 파일은 **KO 도메인 전체**이고, 경량화 응답의 320×397은 그 안의 **부분 창**임. 둘을 같다고 가정했음 |
| V3 | 경도 120~135, 위도 30~45 | 실제 **113.996~138.004 / 29.312~46.358** | 도메인이 한반도보다 넓음. 정상 격자가 게이트 때문에 실패로 나왔음 |

게이트를 실측 도메인(`113~139 / 29~46.5`)과 전체 격자 크기에 맞춰 고쳤음.
**사전 등록을 했다는 것만으로 게이트가 옳은 것은 아님** — 등록값이 틀리면 정상 데이터가
실패로 찍힘. 이 사례를 남김.

#### Seal A — 격자 파일 (통과)

| | 값 |
|---|---|
| 전체 격자 | **900 × 900 = 810,000** (헤더 `900, 900,=`와 일치) |
| 경도 파일 | 8,991,016 B · `sha256 2b1f43a28a8002e1…` |
| 위도 파일 | 8,181,016 B · `sha256 997ebe8c066194d6…` |
| 저장 순서 | **row-major**, 행 내 경도 단조증가 18/18, 열 내 위도 **북→남** 18/18 |
| 범위 | 경도 113.996 ~ 138.004, 위도 29.312 ~ 46.358 |
| 간격(중앙) | Δlon 0.02337° · Δlat −0.01850° |
| **Seal A 해시** | `0476c9ad622bbedb1e825e9c0203280306e6bcb736d23faf3a47672d9d481e1a` |

LCC 특성이 보임 — 행 0은 경도 폭 24.0°, 행 899는 18.5°로 극쪽 수렴함.
**투영 역공학은 영구히 불필요해졌음.**

`area=EA&grid=2`는 3000×2600(85.7 MB), `area=FD&grid=2`는 `error(-11)`로 미제공임.

#### Seal B — 창 오프셋 (미결)

경량화 응답이 준 `x0=63, y0=333`을 공식 격자에서 확인했음.

| 해석 | 결과 |
|---|---|
| `y0=333`을 행 인덱스로 | 위도 40.08 → 32.81. 한반도를 덮어 **그럴듯함** |
| `x0=63`을 열 인덱스로 | 경도 116.69 → 124.54. **한국보다 서쪽** |
| EA 격자(76.81 E 시작)의 열 63 | 더 서쪽. 안 맞음 |
| 앵커 값 일치율 | **0.2396** (임계 0.90) |

**앵커로 메우려는 시도는 식별 불가임.** `y0=333`을 고정해 `x0`만 탐색해도(자유도 1),
최악 앵커 거리 1.17 km(반 칸)를 내는 `x0`이 **213개 동률**임. 320폭 창이 앵커를 포함하기만
하면 거리가 항상 반 칸이 되기 때문임. **`y0`도 같은 이유로 확인된 것이 아님.**

- **말할 수 있는 것**: 공식 격자는 봉인됐음. 투영을 다시 추정할 필요가 없음.
- **말할 수 없는 것**: `x0/y0`가 이 격자의 어디를 가리키는지 **모름.** 따라서 격자를
  AOI에 붙이지 않고 GK2A admission 실험도 열지 않음.
- **필요한 것**: data.go.kr **위성자료 경량화 활용가이드(참고문서)** 의 `x0/y0` 정의.
  `dateTime`·`resultType`은 경험적으로 풀 수 있었으나 이건 자유도 2에 앵커 4곳이라
  **적합으로 풀면 M15를 반복하는 것**임.

## M17. `era5_10` 슬롯 6변수를 ASOS 일자료로 소급 확보할 수 있음

**근거**: `apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList` 실호출
**배경**: M14에서 OlmoEarth에 기상 전용 **비공간** modality `era5_10`이 있음을 확인했음.
GK2A는 보존 2일이라 소급이 불가능했으나, ASOS는 과거 이력이 있음.

### 실측 (2022-04-17, 서울 stn 108 — 71363 촬영일 중 하나)

`resultCode 00 NORMAL_SERVICE`. 반환 필드가 `era5_10` 6변수와 대응함.

| `era5_10` 밴드 | ASOS 필드 | 실측값 |
|---|---|---|
| 2m-temperature | `avgTa` | 14.0 |
| 2m-dewpoint-temperature | `avgTd` | 3.5 |
| surface-pressure | `avgPa` / `avgPs` | 1007.7 / 1017.9 |
| 10m-u/v-component-of-wind | `avgWs` + `maxWd` | 2.3, 250° |
| total-precipitation | `sumRn` | 공백(무강수) |

- **말할 수 있는 것**: 6변수 모두 존재하고 **과거 날짜로 조회됨**. 이 서비스는 2026-05-18에
  이미 승인돼 있어 추가 신청이 필요 없음. 따라서 retrospective live-residual arm의
  1순위는 GK2A가 아니라 ASOS임(M14의 판단이 실측으로 확인됨).
- **말할 수 없는 것**: 세 가지가 미해결임.
  1. **단위·정규화**가 `era5_10`이 기대하는 것과 같은지 미확인. ERA5는 K·Pa·m/s 재분석이고
     ASOS는 °C·hPa·m/s 지점관측임
  2. `avgWs`+`maxWd`는 **평균 풍속과 최대 풍향**이라 u/v 성분으로 바로 바꿀 수 없음.
     같은 시각의 풍향이 아니므로 벡터 분해가 부정확함. 시간자료가 필요할 수 있음
  3. **지점 → 격자 보간**이 필요하고, 그 보간이 또 하나의 계약 변경임(M8 계열 위험)
- **다음**: 지점 좌표(지상관측 지점정보)를 확보해 AOI 최근접 지점을 정하고,
  단위 변환표를 사전 등록한 뒤 결합함. 보간 없이 **최근접 지점 값**부터 시작하는 것이 안전함.

## M18. apihub 활용신청 5건 실측 — 필요한 격자 API는 아직 아님

**근거**: 2026-08-25 승인 5건에 대한 직접 호출.

| # | 승인된 API | 실측 |
|---|---|---|
| 1 | 단기예보 `fct_shrt_reg.php` | **200**, 30,208 B |
| 2 | 레이더합성 `rdr_cmp_file.php` | `404 유효하지 않은 API` — 파라미터/경로 형태가 다름 |
| 3 | 레이더 파일목록 `rdr_stn_file_list.php` | 200이나 46 B (`#`만) — 파라미터 미확정 |
| 4 | **ASOS `kma_sfctm2.php`** | **200**, 26,441 B — **열림** |
| 5 | 천리안 2A호 **LE1B 기본관측자료** | (별도 경로) |
| — | **`nph-gk2a_latlon_api`** | **403 여전히 활용신청 필요** |
| — | **`gk2a_latlon_file_down.php`** | **403 여전히** |

**5번은 우리가 필요한 API가 아니었음.** LE1B는 16채널 기본관측자료이고, 봉인에 필요한 것은
**기상산출물 계열의 위경도 조회/파일**임.

**2026-08-25 갱신**: 이 표 이후 `nph-gk2a_latlon_api` 활용신청이 승인되어 **http 200**으로
파일을 받았음. M16의 **Seal A는 완료**됐고, 남은 것은 경량화 응답의 창 오프셋(Seal B)임.
위 403 행은 승인 이전 상태 기록으로 보존함.

### ASOS 시간자료가 M17의 결함 하나를 해결함

`help=1` 헤더로 컬럼 정의를 확인했음. **시간자료**이므로 같은 시각의 값이 함께 옴.

| 필요 | ASOS 시간자료 컬럼 |
|---|---|
| 기온 | `TA` (C) |
| 이슬점 | `TD` (C) |
| 기압 | `PA` 현지기압 / `PS` 해면기압 (hPa) |
| **풍향·풍속** | **`WD` (36방위) + `WS` (m/s)** — 같은 시각이므로 **u/v 분해가 정확함** |
| 강수 | `RN` (mm, 4~10월 1시간 / 11~3월 3시간), `RN_DAY`, `RN_INT` |
| (보너스) 적설 | `SD_HR3`, `SD_DAY`, `SD_TOT` |
| (보너스) **전운량** | **`CA_TOT` (1/10)**, `CA_MID`, `CH_MIN` 최저운고 |

M17에서 문제로 적었던 "`avgWs`+`maxWd`는 평균풍속과 최대풍향이라 u/v 분해가 부정확함"이
**시간자료로 해소됨.** 그리고 `CA_TOT` 전운량은 GK2A와 **독립적인 지상 관측 기반 관측조건
변수**이므로 admission control의 대조군이 될 수 있음.

- **말할 수 없는 것**: 응답에 **지점 좌표(경위도)가 없음.** `지상관측 지점정보`를 따로
  받아야 AOI 최근접 지점을 정할 수 있음. 강수 `RN`의 집계 구간이 계절에 따라 1시간/3시간으로
  달라지는 것도 결합 시 통제해야 함.

## M19. apihub 활용신청은 **개별 API 단위**임 — 필요 목록을 확정했음

**근거**: 명세서(datawiki 지상관측)에 나온 하위 API 4개를 직접 호출.

`지상관측 > 종관기상관측(ASOS)` 항목에서 **1.1만 승인**된 상태로 나머지를 호출했음.

| 명세 번호 | API | 결과 |
|---|---|---|
| 1.1 시간자료 | `kma_sfctm2.php` | **200** (승인됨) |
| 1.2 시간자료 기간조회 | `kma_sfctm3.php` | **403** |
| 1.5 요소별 조회 | `kma_sfctm5.php` | **403** |
| 2. 지상 평년값 | `sfc_norm1.php` | **403** |
| 4.2 지상관측지점일람표 | `getSfcStnLstTbl` | **403** |

**즉 활용신청 단위는 "종관기상관측(ASOS)" 같은 서비스가 아니라 그 아래 개별 API임.**
앞서 M18에서 "천리안 2A호"를 신청했는데 필요한 위경도 API가 403이었던 것도 같은 이유임.

### 명세에서 확인한 사실 — 소급 실험이 충분히 가능함

| 항목 | 값 |
|---|---|
| ASOS 보유기간 | **1904년 4월 ~ 현재** (지점별 상이) |
| 지점 수 | 96 (2020-04-01 기준) |
| 생산주기 | 분·시간·일·월·연 |
| `kma_sfctm3.php` 1회 조회 한도 | **최대 31일** |
| `kma_sfctm5.php` 출력 | **`LON`(deg), `LAT`(deg), `HT`(m)** 포함 |
| `getSfcStnLstTbl` 출력 | `lat`, `lon`, `ht` + 기압계·온도계·풍속계·우량계 높이 |
| `sfc_norm1.php` | 1991~2020 등 4개 평년기간의 일·순·월·연 평년값 |

- **말할 수 있는 것**:
  1. M17·M18에서 막혔던 **지점 좌표 문제는 `kma_sfctm5.php` 또는 `getSfcStnLstTbl`로 풀림.**
     둘 다 아직 403이라 신청이 필요함.
  2. `kma_sfctm3.php`(31일 일괄)가 있으면 2019~2022 소급 수집이 실용적임.
     없으면 시각마다 개별 호출이라 비용이 커짐.
  3. `sfc_norm1.php`의 **기후평년값**은 설계의 `z_region`(정적 지역 residual) 재료임.
     기온·강수·풍속·전운량·안개계속시간·지중온도까지 있음.
- **말할 수 없는 것**: 신청 후 승인 리드타임을 모름. 일일 호출 한도도 미확인임.
  `sfc_norm1.php`의 평년값이 `era5_10` 정규화에 쓸 수 있는 형태인지도 미검증임.

## M20. 공식 KO/2km 격자를 받았고 봉인 게이트가 제 역할을 했음 — `x0` 해석 불가

**근거**: `gk2a/_grid/grid_seal.json`, `gk2a/_grid/window_convention.json`,
`code/seal_gk2a_grid.py`, `code/gk2a_window_convention.py`
**전제**: apihub 활용신청 #4·#5 승인(`gk2a_latlon_file_down.php`, `nph-gk2a_latlon_api`).

### 받은 것

| | |
|---|---|
| lon 파일 | 8,991,016 B, sha256 `2b1f43a28a8002e1…` |
| lat 파일 | 8,181,016 B, sha256 `997ebe8c066194d6…` |
| 헤더 | **`900, 900,=`** → KO/2km는 **900×900 = 810,000 칸** |
| 전체 범위 | lon 113.9964~138.0036, lat 29.3122~46.3580 |
| 코너 | (0,0) 113.996/45.729 · (899,899) 135.247/29.312 |

`area`/`grid` 조합도 전수 확인했음: KO 2/1/0.5km = 900²/1800²/3600², EA 2km = 3000×2600,
**FD는 제공되지 않음**(`#latlon_data_read:error(-11)`).

### 게이트 결과

| | 결과 |
|---|---|
| V1 개수 | **통과** — 810,000 = 헤더가 말한 900×900 |
| V2 저장순서 | **통과** — lon 행 단조증가 18/18, lat 열 단조감소 18/18 → **row 0 = 북쪽, row-major** |
| V3 범위 | **실패 → 내 사전등록 오류.** 실제 범위가 113.99~138.00 E인데 게이트를 120~135로 좁게 잡았음. KO 영역은 한반도보다 넓음. 데이터가 아니라 임계값이 틀렸음 |
| V4 앵커 | **실패 (0.4583)** — 아래가 이유임 |

### V4 실패의 실체 — `x0`가 KO 격자 열 인덱스가 아님

앵커 4곳의 최근접 칸을 900×900에서 직접 찾았음.

| 앵커 | j | i | 격자 lon/lat | 거리 |
|---|---|---|---|---|
| 서울 종로 | 472 | 491 | 126.9637 / 37.5794 | 0.0084° |
| 서울 중구 | 473 | 492 | 126.9866 / 37.5607 | 0.0076° |
| 부산 | 604 | 585 | 129.0292 / 35.1045 | 0.0105° |
| 울산 | 579 | 598 | 129.3428 / 35.5541 | 0.0060° |

한국 본토(124.5~131.0 E, 33.0~38.7 N)의 인덱스 범위는 **`j 406~721, i 381~680`** 임.
응답이 준 창은 `320×397 @ (x0=63, y0=333)`.

| | 창 범위 | 한국 |
|---|---|---|
| 행 (`y0=333`, 높이 397) | 333~729 | 406~721 → **전부 포함** |
| 열 (`x0=63`, 폭 320) | 63~382 | 381~680 → **거의 전부 제외** |

**즉 `y0`는 KO-900 행 인덱스로 맞지만 `x0=63`은 열 인덱스가 아님.**
행/열 원점(북↔남, 서↔동)과 응답 배열 뒤집기까지 **16개 규약을 전수로 시험**했으나
최고 0.4583, 차선 0.4375로 **격차가 0.0208**뿐이었음 — 어떤 규약도 분리되지 않음.
그리고 어떤 창도 앵커를 제대로 포함하지 않았음(서울 126.97이 창 밖).

apihub 경량화(`WthrSatlitInfoService/getGk2aIrAll`)도 **동일한 `x0=63.0, y0=333.0`** 을 주며
추가 좌표정보가 없음. 즉 data.go.kr 쪽 문제가 아니라 산출물 자체의 메타데이터가 부족함.

- **말할 수 있는 것**:
  1. 공식 KO/2km 격자를 확보하고 해시로 봉인 가능한 상태임. 저장순서(row-major, 북→남)도 확정됐음.
  2. **봉인은 보류가 맞음.** 사전 등록한 V4가 정확히 이 불일치를 잡아냈음.
     M15 때처럼 적합해서 밀어붙이지 않았음.
  3. 경량화 응답의 `x0`는 이 격자로 해석되지 않음. 문서화 공백임.
- **말할 수 없는 것**: `x0=63`의 의미를 모름. 다른 area/grid로도 설명되지 않음
  (KO 900²/1800²/3600², EA 3000×2600 전부 확인).
- **다음**: 앵커를 **행정동코드 수십~수백 개**로 늘려 정수 오프셋 `(i0, j0)`를 탐색함.
  4개로는 16규약이 0.02 차이로 붙었지만, 앵커가 100개면 참 오프셋만 높은 일치율을 냄.
  이때도 **연속 매개변수 적합이 아니라 정수 오프셋 선택**이며, 최고와 차선의 격차를 함께 보고함.
  격차가 작으면 여전히 봉인하지 않음.
- **우회로**: `Area` 계열은 lon/lat을 직접 주므로 **격자 확정 없이도 AOI 작업이 가능함.**
  admission 실험을 행정동 단위로 먼저 하는 것이 막히지 않는 경로임.

## M21. ASOS 지점 → AOI 군집 결합 — 게이트 4/4

**근거**: `asos/asos_aoi_join.json`, `code/build_asos_aoi_join.py`
**전제**: apihub `stn_inf.php` 승인(M19에서 막혀 있던 지점 좌표).

96지점 좌표를 받아 M10에서 동결한 13군집 중심에 최근접 결합했음. **보간하지 않음.**

| 군집 | split | 타일 | 산사태 | 최근접 지점 | km |
|---|---|---|---|---|---|
| C11 | train | 180 | 42 | 217 정선군 | 20.7 |
| C13 | train | 129 | 14 | 295 남해 | 20.1 |
| C12 | test | 22 | 12 | 251 고창군 | 10.4 |
| C09 | val | 12 | 8 | 276 청송군 | 16.1 |
| C08 | test | 6 | 6 | 259 강진군 | 9.1 |
| C05 | test | 8 | 4 | 133 대전 | 11.0 |
| C10 | val | 38 | 4 | 289 산청 | 23.5 |
| C01 | train | 84 | 0 | 268 진도군 | 17.7 |
| C04 | val | 34 | 0 | 283 경주시 | 10.9 |
| C06 | test | 30 | 0 | 284 거창 | 20.1 |
| C03 | test | 28 | 0 | 129 서산 | 29.2 |
| C07 | test | 14 | 0 | 108 서울 | 11.8 |
| C02 | test | 9 | 0 | 262 고흥 | **61.2** |

거리 최소 9.1 · **중위 17.7** · 최대 61.2 km.

| 게이트 | 결과 |
|---|---|
| J1 모든 군집 결합 | 13/13 |
| J2 중위 ≤ 30 km | 17.7 |
| J3 60 km 초과 군집 표시 | C02 하나 (61.2 km). **산사태 0타일이라 primary task 영향 없음** |
| J4 한 지점이 4개 이상 군집 담당 안 함 | 통과. 담당 중복이 없으므로 residual이 복제되지 않음 |

**J4가 중요함**: 한 지점이 여러 군집을 담당하면 같은 값이 복제되어 지역 간 변동이 사라지고,
negative control(region-shuffle)에서 이득이 남는 위험이 생김. 통과했음.

지점명이 AOI 정체를 확인해줌 — **C10 산청(지리산), C09 청송군(주왕산), C11 정선군(오대산권),
C12 고창군(내장산권)**. 산사태가 있는 7개 군집은 전부 **23.5 km 이내**임.

## M22. `era5_10` 6변수를 60 촬영일 전부 추출했음 — 강수만 문제임

**근거**: `asos/era5_10_residual.jsonl`, `asos/era5_10_residual_summary.json`,
`code/build_era5_10_residual.py`

`kma_sfctm2.php`가 `stn=0`으로 96지점을 한 번에 주므로 **60일 × 5시각 = 300회 호출**로 끝났음.
결과 3,898행(주 시각 780행 = 60일 × 13군집).

### 사전 등록한 규약 (사후에 바꾸지 않음)

| | |
|---|---|
| 주 시각 | **11:00 KST** (Sentinel-2 통과가 현지 10:30 전후) |
| 민감도 | 09·10·12·13시도 함께 받음 |
| 바람 | `WD`(36방위)×10° 를 **불어오는 방향**으로 보고 `u = −WS·sin θ`, `v = −WS·cos θ` |
| 단위 | **변환하지 않음.** °C·hPa·m/s·mm 원값 보존. ERA5의 K·Pa 변환은 학습 직전에 명시적으로 |
| 보간 | **없음.** 최근접 지점 값만 |

### 커버리지 (주 시각 780행)

| 변수 | 커버리지 |
|---|---|
| temperature (°C) | **1.0000** |
| surface_pressure (hPa) | **1.0000** |
| cloud_total (1/10) | **1.0000** |
| dewpoint (°C) | 0.9987 |
| wind u/v (m/s) | 0.9987 |
| **precipitation (mm)** | **0.0090** |
| precipitation_day (mm) | 0.0795 |
| snow_depth (cm) | 0.0077 |

### 강수 공백의 정체 — 계절 규칙이 아님

명세는 "11월~3월은 3시간강수량, 4~10월은 1시간 강수량"이라고 하므로 겨울철 공백을 예상했음.
캐시 28,591행을 세어보니 **그게 아님**.

| 구간 | RN 값 | RN 공백 | 공백률 |
|---|---|---|---|
| 4~10월 | 279 | 15,482 | **98.2%** |
| 11~3월 | 159 | 12,671 | **98.8%** |

두 계절이 같음. 그리고 RN이 공백일 때 `RN_DAY`도 26,199건 함께 공백이고, 값이 있는 1,954건은
0.0·0.1·0.2… 로 소량 강수임. 96지점 모두 강수를 상시 측정하므로 "미측정"으로 98%는 설명되지 않음.
→ **ASOS 텍스트 피드는 무강수를 0으로 적지 않고 생략함.**

- **말할 수 있는 것**: 6변수 중 5개는 커버리지 ~100%로 곧바로 쓸 수 있음.
  강수는 원값(`precipitation_mm`, None 가능)과 **가정 적용값**(`precipitation_mm_zerofilled`,
  0.0 채움)을 분리 보존하고 `precip_was_blank`로 표시했음. 조용히 채우지 않았음.
- **말할 수 없는 것**: 공백=0 이라는 해석은 **정황 증거**임(98% 공백률, RN_DAY 동반 공백,
  상시 측정 사실). 기상청 문서로 확인하지 않았음. 그리고 **산사태 forcing에는 시각 강수가 아니라
  선행강우 누적이 필요함** — 촬영시각 11시의 1시간 강수는 산사태 유발과 거의 무관함.
- **다음**: 선행강우지수(antecedent precipitation index)를 설계함. 촬영일 이전 1·3·7·15일
  누적을 `kma_sfcdd`/`kma_sfctm3`로 받아 계산하고, 누적 창 길이를 **사전 등록**함.
  창을 성능 보고 고르면 자기기만임.

## M23. G-P pilot 1차 — “frozen OLMo 전 지표 우위”는 개발 신호였고 확정표에서 제외한다

**근거**: `sen12_gp_pilot/holdout_chimanimani_pilot_8ep.json`, `logs/gp_pilot.log`,
`code/extract_sen12_fold_cache.py`, `code/pilot_sen12_gp_heads.py`
**환경**: GPU1 전용(`CUDA_VISIBLE_DEVICES=1`), GPU0은 다른 프로젝트 62.6 GB로 미사용.

> **2026-08-25 독립 감사 정정**: 아래 표는 8-epoch 개발 관측으로 보존하지만 확정 성능표가 아니다.
> AUPRC 구현이 매 batch의 동일한 pixel offset을 반복 표본추출해 공간 편향됐고, `positive_pixel_frac`도
> 전체가 아닌 그 표본에서 계산됐다. 8-epoch test를 열람한 뒤 40-epoch로 수정했으므로 Chimanimani는
> 이후 confirmatory test가 아니라 development holdout이다. v2는 exact AUPRC·전체 train pos_weight·
> arm별 RNG reset·checkpoint/per-sample seal로 재실행한다.

### 준비 — fold 캐시

`holdout_chimanimani` (test chimanimani 1,133 / val china 159 / train 5,542).
**봉인 해시 3/3 일치**를 코드가 검증하고 불일치면 중단한다.

| 캐시 | 용량 |
|---|---|
| `emb_fp16` frozen OLMo 768×32×32 | 10.75 GB |
| `raw_u16` S12q 실관측 10밴드 10×12×128×128 | 26.87 GB |
| `mask_u8` 128×128 | 0.11 GB |

추출 6,834개 · 6.2~6.8 sample/s · peak CUDA 0.74 GB · 약 18분.

### 1차 결과 (8 epoch, seed 1, 동일 예산)

| arm | 학습 파라미터 | 학습 s | peak GB | **test IoU** | test F1 | test AUPRC | **test ECE** | val IoU | val F1 |
|---|---|---|---|---|---|---|---|---|---|
| P1 raw 시간평균 U-Net | 474,849 | 67 | 0.47 | 0.0727 | 0.1356 | 0.0534 | 0.0949 | 0.0294 | 0.0571 |
| P2 raw 12시점 3D U-Net | 265,649 | 138 | 0.96 | 0.1030 | 0.1868 | 0.4757 | 0.1010 | 0.0464 | 0.0888 |
| **P4 frozen OLMo + decoder** | **237,537** | **62** | **0.32** | **0.1153** | **0.2067** | **0.5501** | **0.0738** | **0.0927** | **0.1698** |

`pos_weight` 35.002 (세 arm 동일), test 양성 픽셀 비율 0.90%.

이 8-epoch 실행 안에서는 P4가 **IoU·F1·근사 AUPRC·ECE 전부에서 1위**이고 동시에
**학습 파라미터 최소·head 학습시간 최단·head 학습 메모리 최소**였다.
P4/P2 비는 IoU 112% · F1 111% · AUPRC 116%. val에서는 IoU가 2.0배(0.0927 vs 0.0464)다.

### 그런데 이 비교는 아직 공정하지 않다

epoch별 손실을 보니 **세 arm 모두 8 epoch에서 단조 하강 중**이었다.

```
P1  0.9431 → 0.5738   (8 epoch까지 계속 하강)
P2  0.9606 → 0.3894
P4  0.4064 → 0.1857
```

전부 미수렴이다. 그리고 P4는 시작 손실이 0.406으로 P1/P2의 ~0.95보다 훨씬 낮다 —
frozen feature가 이미 정보를 담고 있어서다. 즉 **짧은 예산은 P4에 유리하다.**
사전 등록한 `8 epoch 고정`이 이 경우 편향을 만든다.

- **말할 수 있는 것**: 이 실행에서 세 arm은 같은 S12q 표본 계약을 썼고, 이미 만든 cache 위의
  **head fit 범위**에서는 P4가 더 적은 학습 파라미터·시간·메모리를 썼다.
- **말할 수 없는 것**: frozen OLMo encoder 88.96M 파라미터와 10.75 GB cache extraction을 제외한
  62초·0.32 GB만으로 end-to-end 비용 우위를 주장할 수 없다. multi-task 수에 따른 amortization 표가
  나오기 전에는 `cached-head cost`로만 부른다.
- **말할 수 없는 것**:
  1. **정확도 우위는 아직 확정할 수 없다.** 미수렴 상태의 비교다
  2. 절대값이 낮다. IoU 0.115 · F1 0.207. 양성 픽셀이 0.9%인 극단 불균형 task이고
     Sen12 논문 baseline과 비교하지 않았다
  3. **fold 1개 · seed 1개**다. G-P 게이트의 `catastrophic fold 없음`을 판정할 수 없다
  4. **P3(U-TAE)을 돌리지 않았다.** 게이트는 `max(P2,P3)의 95%`이므로 U-TAE가 훨씬 강하면
     결과가 바뀔 수 있다
  5. AUPRC는 픽셀 표본추출 + 101 threshold sweep 근사다
- **프로토콜 수정 (사유 공개)**: 미수렴을 근거로 예산을 **모든 arm에 동일하게 40 epoch**로
  늘리고 **val IoU로 best epoch을 고른다**(test는 선택에 쓰지 않는다). 1차 결과는 폐기하지 않고
  이 항목에 그대로 남긴다. 2차 결과는 별도 M-항목으로 기록한다.

## M24. 공식 저장소의 후속 binary benchmark와 우리 수치는 비교 대상이 아니다

**근거**: `PaulH97/Sen12Landslides` 현재 README·configs와 Scientific Data 본문
**질문**: M23의 IoU 0.115가 낮은 것인가?

### 공식 저장소가 추가 보고한 S12LS-LD binary benchmark (S2+DEM, seed 42/123/777 평균)

| Model | AP | F1 | **IoU** | Precision | Recall |
|---|---|---|---|---|---|
| U-TAE | 67.75 | 61.80 | **44.74** | 53.19 | 74.90 |
| U-ConvLSTM | 65.13 | 61.95 | 44.88 | 60.59 | 63.92 |
| Unet3d | 62.08 | 58.82 | **41.66** | 55.75 | 62.56 |
| ConvGRU | 60.00 | 59.06 | 41.91 | 56.72 | 61.77 |

### 왜 비교할 수 없는가 — 네 가지가 다르다

| | 공식 저장소 S12LS-LD benchmark | 우리 (M23) |
|---|---|---|
| **task** | README의 **마스크 >50픽셀 표본만** (S2 4,988개, 100% annotated) | headline 6,834개, **음성 포함**, 양성 픽셀 0.90% |
| **split** | random 80/20 (`seed 42`) | **leave-one-region-out** (M9·M12의 요지) |
| **입력** | 11채널 (밴드 + SCL + **DEM**), **15 timestep** | 10밴드, **12 timestep**, DEM 없음 |
| **학습** | **75 epoch**, `BCEDiceLoss(pos_weight 5, dice_w 0.5)` | 8 epoch, `BCE(pos_weight 35)` |

두 표는 네 축이 동시에 달라 **난이도 순서까지 단정할 수 없다.** LOCO는 random split보다 공간 전이에
엄격하지만, 음성 표본 증가는 지표에 따라 난이도를 올리거나 background-dominated metric을 쉽게 만들 수
있다. 또한 위 표는 논문 원표가 아니라 현재 공식 저장소가 별도로 제공하는 binary benchmark다.
Scientific Data 논문 자체는 50 epoch·cross-entropy 설정과 geographic cluster leave-one-out 실험도
보고하므로, 이를 “원 논문은 같은 지역 random split만 썼다”라고 요약하면 틀린다.

- **말할 수 있는 것**: M23의 IoU 0.115와 저장소 benchmark 0.447을 직접 나눠 성능 격차로 읽으면 안 된다.
  또한 이 표는 **P3(U-TAE)이 3D U-Net보다 IoU 44.74 vs 41.66으로 우수함**을 보여주므로,
  G-P 게이트의 `max(P2,P3)`에서 P3가 기준선이 될 가능성이 높다 → **P3를 반드시 돌려야 한다.**
- **말할 수 없는 것**: task만 맞춘 부분집합(마스크 >50) 수치로도 **split·입력·epoch이 여전히
  다르므로** 직접 비교가 아니다. 그들의 LOCO(Experiment 3) 수치는 확인하지 못했다.
- **조치**: v2 pilot에 `ld_iou`/`ld_f1`(마스크 >50 부분집합)을 같은 pass에서 함께 내도록 추가했고,
  산출물에 원 논문 표와 **`not_comparable_because` 4항목**을 함께 박았다. 숫자만 옮겨 비교하는
  일을 코드 수준에서 막았다.

## M25. strict G-P 개발 pilot — P4는 유망하지만 “전 지표 1위”가 아니다

**근거**: `artifacts/sen12_gp_pilot_audit/determinism/final/`,
`docs/GP_PILOT_VALIDATION_AUDIT.md`
**환경**: Python 3.12.3, NumPy 1.26.4, PyTorch `2.7.0a0+7c8ec84dab.nv25.03`, CUDA 12.8,
cuDNN 9.8, NVIDIA H200 physical GPU1. 코드 SHA-256 `478c6af5…`.

strict CUDA 결정성, arm별 RNG/DataLoader reset, 전체 train pos_weight 46.53, val-best checkpoint,
모든 pixel exact AP, checkpoint/per-sample SHA를 적용했다. cache 6,834/6,834 content audit도 통과했다.

| arm | val IoU | test IoU | exact AP | F1 | positive-patch macro IoU | LD IoU | head fit+val 초 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 raw mean | 0.03685 | 0.054055 | 0.077737 | 0.102565 | 0.115962 | 0.14488 | **387.3** |
| P2-tiny factorized | 0.06197 | 0.134989 | **0.286074** | 0.237869 | **0.195021** | 0.21327 | 1,455.7 |
| P4 frozen OLMo | **0.11181** | **0.141643** | 0.225115 | **0.248139** | 0.183479 | **0.23041** | 950.5 |

- **말할 수 있는 것**: 이 개발 fold에서 P4는 P1보다 강하며 P2-tiny 대비 IoU 104.9%·F1 104.3%,
  LD IoU 108.0%다. OLMo spatial cache에 산사태 segmentation 신호가 있다는 viability evidence다.
- **동시에 말해야 하는 것**: P4의 AP는 P2-tiny의 **78.7%**, positive-patch macro IoU는 94.1%다.
  따라서 사전 G-P의 IoU+AUPRC 95% 조건을 충족하지 않는다. calibration 지표는 P4가 낮지만
  background-dominated pixel-micro라 router calibration 근거가 아니다.
- **말할 수 없는 것**: P2는 공식 Sen12 3D U-Net이 아니라 deterministic factorized-pool tiny
  stand-in이고 P3 U-TAE가 없다. P4만 acquisition timestamp를 받았다. Chimanimani test는 이미
  M23에서 열람했다. 따라서 strong baseline 우위·region 일반화·confirmatory test·CVPR gate를
  주장할 수 없다.
- **재현성 복구**: RNG reset만 한 pre-fix P4 replay는 test IoU `0.122826→0.143442`(+16.8%)로
  갈렸다. strict algorithms/cuBLAS/cuDNN/TF32 계약 뒤 P1/P4 smoke와 P2-tiny smoke는 각각
  checkpoint tensor max-abs diff 0을 보였다. final P4-only 40-epoch도 full run과 history(시간 제외),
  모든 지표, per-sample/checkpoint SHA, tensor가 bitwise 일치했다(max-abs diff 0).
- **비용 범위**: 표의 시간은 cached head fit+val이다. P4 cold extraction은 1,130.05초,
  embedding 10.75 GB, frozen encoder 88.96M 파라미터다. 같은 P4의 wall time도 full sequence
  950.5초 vs 단독 replay 520.0초로 갈려, isolated 반복 전에는 P4 end-to-end/cost 우위가 미측정이다.

## 이 장부에 없는 것 (혼동 방지)

M23 이후 개발 pilot에는 Sen12Landslides S2가 실제로 들어갔다. 그러나 아래 지역·공공데이터의
**표현 기여**는 여전히 어떤 성능표에도 들어가지 않았다.

| | 사용량 |
|---|---|
| GLAMOS·swissALTI3D (스위스) | **0** |
| ICIMOD·HKH (네팔) | **0** |
| 한국 공공데이터 — 표현 기여(모델 성능) | **0** (2026-08-25 현재도 미측정) |
| 한국 공공데이터 — 접근·계약 감사 | M9·M10에서 수행 (AI-Hub 71363, GK2A 10/10, VWorld) |
| 사람 판독 라벨 | **0** |

Sen12Landslides frozen OLMo 수치는 이미 열람한 Chimanimani 1개 개발 fold에서만 나왔다.
미열람 9지역·공식 P2/P3·timestamp parity를 갖춘 full G-P는 0이다. 네팔 BIPAD/ICIMOD와
스위스 event join도 0이다.

---

## 다음 측정 후보 (우선순위는 `K_ALIGN_WIDE_ANGLE.md`)

1. **downstream task** — M3/M5의 R@1 붕괴가 실제 태스크를 망가뜨리는지. 남은 최대 위협.
2. **ADC baseline** — 새 query → 선형 map → old float 공간 → old PQ codebook.
   통과하면 quantizer-aware 방법 논문이 없어진다.
3. **두 번째 계약 축** — 정규화 scaling. 밴드 순서 하나로는 일반화 못 한다.
4. **세 번째 릴리스** — Clay v1.0/v1.5는 둘 다 공개돼 있고 Major TOM에 v1.5 임베딩도 있다.
5. **embeddings-stac gap 이슈/PR** — M2가 이미 1차 증거다.
