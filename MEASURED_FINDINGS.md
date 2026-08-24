# 측정 장부 — 실제로 잰 것만

최종 갱신: 2026-08-24

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

**아직 한 번도 측정하지 않은 것**: downstream task 정확도, 한국 공공데이터의 표현 기여,
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

## 이 장부에 없는 것 (혼동 방지)

이 세션의 실험에 들어간 입력은 **제주 Sentinel-2 영상과 공개 Major TOM parquet뿐**이다.

| | 사용량 |
|---|---|
| GLAMOS·swissALTI3D (스위스) | **0** |
| ICIMOD·HKH (네팔) | **0** |
| 한국 공공데이터 (BuildingHUB·EIA·VWorld PNU·GK2A·FarMap) | **0** |
| 사람 판독 라벨 | **0** |

`MOUNTAIN_EVIDENCE_TRANSFER.md`의 Phase 0(Glacial-Lake-Bench·Landslide4Sense)은
다운로드·라이선스 확인조차 시작하지 않았다.

---

## 다음 측정 후보 (우선순위는 `K_ALIGN_WIDE_ANGLE.md`)

1. **downstream task** — M3/M5의 R@1 붕괴가 실제 태스크를 망가뜨리는지. 남은 최대 위협.
2. **ADC baseline** — 새 query → 선형 map → old float 공간 → old PQ codebook.
   통과하면 quantizer-aware 방법 논문이 없어진다.
3. **두 번째 계약 축** — 정규화 scaling. 밴드 순서 하나로는 일반화 못 한다.
4. **세 번째 릴리스** — Clay v1.0/v1.5는 둘 다 공개돼 있고 Major TOM에 v1.5 임베딩도 있다.
5. **embeddings-stac gap 이슈/PR** — M2가 이미 1차 증거다.
