# 한 발 물러나서 — K-ALIGN 프로그램의 광각 보정

작성 2026-08-23  
역할: `K_ALIGN_BIG_PICTURE.md`(무엇을 푸는가)와 `K_ALIGN_CVPR_READINESS_AUDIT.md`(어떻게 푸는가)는
이미 충분히 깊다. 이 문서는 **그 두 문서 바깥에 있는 것**만 다룬다. 중복되는 내용은 쓰지 않는다.

전제: 아래 W1–W9는 전부 **제안**이다. 실행 결과가 아니다.

**2026-08-23 검증 갱신.** 처음 작성 시 "확인하지 않았다"고 표시한 네 항목(W4·W5·W6·W9)을
실제로 확인했다. 그중 **W9는 결론이 바뀌었고 W4는 규모가 크게 커졌다.** 검증된 항목은
✅로 표시한다. 나머지는 여전히 제안이다.

---

## W1. 6개월 중첩은 버그가 아니라 계측기다 — 이번 조사에서 가장 큰 것

지금 이 발견은 "우리가 실수를 찾았다"로 쓰이고 있다. 그런데 실제로 손에 쥔 것은 다른 것이다.

> **알려진 양의 계약 불일치를 주입하면 알려진 양의 거짓 확신이 나온다는 것을, 원본 재다운로드
> 없이 임의 횟수로 재현할 수 있는 장치.**

184일 중첩은 우연히 만들어진 **50% 용량의 1회 투여**다. 용량을 조절하면 곡선이 된다.

| 조작 축 | 용량 | 재실행 비용 |
|---|---|---|
| 시간창 중첩 | 0 / 25 / 50 / 75 / 100% | 창 재정의 + encoder 재실행 (raster 재사용) |
| 계절 offset | 0 / 3 / 6 / 9개월 | 위와 동일 |
| 밴드 순서 | identity / 인접 swap / 역순 | **입력 텐서 치환만.** encoder만 재실행 |
| 반사율 scaling·정규화 | 원본 / ×1.1 / 다른 정규화 | 위와 동일 |
| pooling·stride | 계약별 | 위와 동일 |
| 모델 릴리스 | v1 / v1.2 | 이미 있음 |

각 셀에서 **Top-k 후보의 z-score와 실제 아티팩트 비율**을 잰다. 결과는 dose–response 곡선이다.

**이것이 바꾸는 것 세 가지.**

1. **Figure 1이 일화에서 곡선이 된다.** "z=10.6인데 100% 인공물" 한 점보다, "불일치 용량이
   늘수록 z는 올라가고 진짜 비율은 떨어진다"는 곡선이 압도적으로 강하다.
2. **release pair 부족 문제가 풀린다.** W5 참조 — characterization의 최대 약점은 공개
   release pair가 몇 개 없다는 것인데, **합성 계약 변경은 ground truth가 알려진 release pair를
   무제한 생산한다.** 예측기를 여기서 fit하고 실제 release pair(Olmo v1→v1.2, Prithvi 1.0→2.0)에서
   held-out 평가하면 "예측 가능한가"가 비로소 답할 수 있는 질문이 된다.
3. **가장 싼 실험이다.** 재다운로드·재materialize 없이 이미 있는 raster에서 encoder만 돌린다.
   밴드 순서·정규화 축은 창 재정의조차 필요 없다.

**주의.** 합성 불일치와 실제 release drift가 같은 종류라고 가정하면 안 된다. 합성으로 fit한
예측기가 실제 pair에서 실패하는 것 자체가 보고할 결과다. 두 축을 한 표에 섞지 않는다.

---

## W2. 기여는 계약 명세가 아니라 "기존 진단이 전부 눈멀었다"는 것이다

계약 명세(release + 시간창 + 밴드 + GSD + pooling + hash)는 옳지만, **명세는 누구나 쓸 수 있고
리뷰어는 그것을 기여로 세지 않는다.** 실제로 새로운 것은 이미 손에 있다.

- CKA 0.97857 — "거의 같다"고 말했다. 실제 cross-release R@1은 0.0000이었다.
- 거리 Spearman 0.95251 — "구조가 보존됐다"고 말했다. 같은 token cosine은 −0.00860이었다.
- z = 10.6 — "매우 유의하다"고 말했다. 입력의 절반이 같은 두 합성본의 차이였다.

**세 경우 모두, 우리가 쓰는 값싼 진단이 전부 통과시켰다.** 이것이 논문의 문장이다.

> 계약 불일치는 기존 표현 유사도 지표·신뢰도 점수에 **보이지 않는다**. 따라서 재사용 자격은
> 유사도가 아니라 계약으로 판정해야 한다.

이건 반증 가능한 주장이고 실험이 명확하다.

```text
진단 K개 × 불일치 유형 M개 → 탐지 여부 행렬
진단: CKA(linear/RBF/row-norm), 평균 cosine, kNN overlap, PCA subspace overlap,
      effective rank, MMD, Procrustes residual, 예측 신뢰도/z
불일치: 릴리스, 시간창 중첩, 계절 offset, 밴드 순서, 정규화, pooling, GSD
```

**대부분의 칸이 "탐지 못함"인 표 자체가 결과다.** 그리고 한두 칸이 탐지에 성공한다면 그것이
곧 값싼 screen이 된다 — 라벨 없이, encoder 재실행 없이.

### ✅ 2026-08-24 1차 측정 결과 — 예상보다 강하다

제주 8 site-years, OlmoEarth v1, 밴드 순서 축에서 실제로 쟀다
(`artifacts/results/contract_dose_v1_analysis.json`, `dose_brittleness_control.json`).
하네스 타당성은 dose 0이 기존 frozen 출력과 **byte-identical 8/8**로 확인했다.

| | same-token cos | **linear CKA** | **R@1** |
|---|---:|---:|---:|
| band-order dose 1 (2칸) | +0.9643 | 0.9923 | 0.9818 |
| band-order dose 3 (6칸) | +0.9430 | 0.9797 | 0.6019 |
| band-order dose 6 (12칸) | +0.9145 | **0.9720** | **0.2456** |
| band-order reverse (12칸) | +0.8628 | 0.9595 | 0.1613 |
| **무작위 잡음 30%** | +0.9577 | **0.9505** | **1.0000** |

**세 가지가 확인됐다.**

1. **용량–반응이 단조롭다.** R@1 0.98 → 0.92 → 0.60 → 0.25 → 0.16. 계측기가 작동한다.
2. **취약성 위협이 기각됐다.** 시험한 어떤 잡음(최대 30%)도 dose 6의 R@1을 재현하지 못했다
   (`smallest_noise_matching_dose6 = None`). 30% 잡음에서 4,096 토큰 **오검색 0건**이다.
   따라서 R@1 붕괴는 지표의 취약성이 아니라 **구조적 이동**이다.
3. **핵심 — CKA의 순서가 역전돼 있다.** 검색이 완벽한 잡음(R@1 1.000)을 CKA 0.9505로,
   검색의 75%가 깨진 계약 불일치(R@1 0.2456)를 CKA 0.9720으로 평가한다.
   **CKA 기준으로는 무해한 쪽이 더 달라 보인다.**

이로써 주장이 바뀐다.

> ~~CKA는 계약 불일치에 덜 민감하다~~ (알려진 성질이라 약함)
> → **CKA의 순서는 운영 결과에 대해 역전될 수 있다** (알려진 성질이 아님)

### ⛔ 2026-08-24 2차 측정 — 위 주장의 **일반성은 기각됐다**

같은 축을 OlmoEarth **v1.2**에서 반복했다. `replicates_across_releases = **False**`.

| dose | v1 CKA / R@1 | **v1.2 CKA / R@1** |
|---|---:|---:|
| 1 | 0.9923 / 0.9818 | **0.7928** / 0.7556 |
| 2 | 0.9873 / 0.9246 | **0.5274** / 0.1487 |
| 3 | 0.9797 / 0.6019 | **0.5193** / 0.0898 |
| 6 | 0.9720 / 0.2456 | **0.4172** / 0.0216 |
| reverse | 0.9595 / 0.1613 | **0.2749** / 0.0020 |

**v1.2에서는 CKA가 손상을 정확히 따라간다.** 눈멀지 않았다.
v1.2 dose 0도 frozen 출력과 byte-identical 8/8이므로 데이터 문제가 아니다.

분석기에 미리 적어둔 규칙(`모든 dose에서 CKA와 R@1이 같이 무너지면 W2 주장을 철회한다`)에 따라:

- **철회** — `CKA는 계약 불일치에 눈멀었다`와 `순서 역전`을 **일반 주장으로 쓰지 않는다.**
  v1에서 관측된 **모델 의존적 현상**으로만 기술한다.
- **유지** — 용량–반응 단조성(두 릴리스 모두), 취약성 기각(잡음 30%에서 오검색 0건),
  하네스 타당성(dose 0 byte-identical, 양 릴리스).

### 대신 나온 탐색적 관측 (사전 등록 안 됨)

**같은 불일치에 v1.2가 훨씬 취약하다** — dose 2에서 6.2×, dose 6에서 11×, reverse에서 80×.
운영적으로는 *파이프라인에 잠복한 밴드 순서 버그가 있을 때 v1→v1.2 업그레이드가 그 피해를
6~80배 키운다*는 뜻이고, 이는 계약 축의 원래 이야기와 더 잘 맞는다.

**그러나 승격하지 않는다** — 릴리스 2개는 `비교`이지 `법칙`이 아니고, 사전 등록되지 않았다.
별도 축·별도 릴리스에서 **사전 등록 후** 확인해야 한다.

부수 발견: dose 6과 reverse는 이동 칸수가 **똑같이 12**인데 손상이 다르다(R@1 0.2456 vs 0.1613).
**이동 개수가 아니라 이동 거리가 중요하다** — 단순 카운트로 손상을 예측할 수 없다.

**남은 위협 하나**: downstream task를 아직 측정하지 않았다. 자기검색 R@1은 표현 프록시이며,
이것이 깨질 때 실제 태스크가 망가지는지는 별도로 보여야 한다. 이 표만으로 `조용한 오류`의
운영 비용을 주장하지 않는다.

이 프레이밍의 부수 효과: 이 논문은 remote sensing 논문이 아니라 **표현 평가 논문**이 된다.
CVPR 심사자가 읽을 이유가 생긴다.

---

## W3. EO 밖으로 나갈 것인가 — 지금 결정한다

"아카이브된 임베딩은 계약 안에서만 의미가 있다"는 EO만의 명제가 아니다. RAG 벡터 DB, 추천
아이템 임베딩, 생체인식 gallery가 전부 같은 구조다. **EO가 특별한 이유는 계약 축이 더 많다는
것**이다 — 텍스트는 (모델, tokenizer, pooling, 정규화) 4축인데 EO는 여기에 시간창·밴드·GSD·
temporal recipe가 더 붙는다.

두 갈래다.

| 선택 | 얻는 것 | 잃는 것 |
|---|---|---|
| EO 한정 | 안전. 주장 범위가 증거와 일치 | remote sensing 트랙으로 읽힘 |
| 비-EO 데모 1개 추가 | "일반 표현 문제"로 읽힘. CVPR main 적합도 상승 | 범위 확대 위험, 작업량 |

**권고: 아주 작게 (b).** 텍스트 임베딩 모델 하나의 버전 교체 × 고정 벡터 DB에서 같은 실패가
나오는지만 본다(반나절). 성공하면 intro 한 문단과 부록 표 하나, 실패하면 "EO 고유"라는 근거가
생긴다. **어느 쪽이든 이득이고, 범위를 넓히는 것이 아니라 경계를 긋는 실험이다.**

---

## W4. ✅ "왜 지금인가" + Major TOM은 이미 쓸 수 있는 실험대다

### 왜 지금인가

최근 약 18개월 동안 공개된 것들이다.

- AlphaEarth (64d 전지구 embedding field)
- TESSERA
- Major TOM 임베딩 계열
- Ai2 Studio embedding export
- ESSD 초경량 Earth embedding DB (전지구 육지 1년 ≈ 2.4 TB)

전부 **아카이브**이고 version transition이 실제 문제가 된다. 그러나 2026-08-24 재조사 결과
**`재사용 프로토콜을 가진 곳은 하나도 없다`는 문장은 철회한다.** AlphaEarth는 model/process/data
version을, TESSERA convention은 dataset/model/build version과 cross-version 혼합 금지를 명시한다.
남은 빈칸은 version 필드의 존재가 아니라 **동일 task에서의 호환성 검증, 불일치 시 action,
위험–재계산 비용 곡선**이다.

### ✅ 검증 결과 — 예상보다 훨씬 좋다

[Major TOM 조직 페이지](https://huggingface.co/Major-TOM)를 직접 확인했다. 총 25개 데이터셋이다.

- **`Core-S2L2A-249k-OlmoEarth-Base`는 실재한다.** GOAL.md의 2026-08-21 기록이 맞았다.
- 그런데 그게 전부가 아니다. **동일한 249k chip 위에 여러 모델의 임베딩이 이미 계산돼 있다.**

| 249k subset (동일 chip) | 그 밖의 공식 embedding 데이터셋 |
|---|---|
| `Core-S2L2A-249k-OlmoEarth-Base` | `Core-S2L2A-MMEarth` |
| `Core-S2L2A-249k-Clay-v1_5` | `Core-AlphaEarth-Embeddings` |
| `Core-S2L2A-249k-SatCLIP` | `Core-S2L2A-UniverSat` |
| `Core-S2RGB-249k-SigLIP` | `Core-S2L1C-SSL4EO`, `Core-S1RTC-SSL4EO` |
| `Core-S2RGB-249k-DINOv2` | `Core-S2L1C-DeCUR`, `Core-S1RTC-DeCUR` |
| `Core-S2RGB-249k-FarSLIP` | `Core-S2RGB-DINOv2`, `Core-S2RGB-SigLIP` |

### ✅ 두 데이터셋 카드를 직접 확인한 정밀 사양

| | `249k-OlmoEarth-Base` | `249k-Clay-v1_5` |
|---|---|---|
| chip 수 | **248,719** | **248,719** |
| chip 크기 | 384×384 | 384×384 |
| 차원 | **768** | **1024** |
| pooling | **unmasked spatial token 평균** | **CLS token** |
| 밴드 | **12개 전부** (재정렬됨) | **10개** (B02·03·04·05·06·07·08·8A·11·12) |
| 정규화 | OlmoEarth 사전학습 normalizer | Clay S2 mean/std |
| L2 정규화 | 미적용(명시) | 미기재 |
| 라이선스 | CC-BY-SA-4.0 | CC-BY-SA-4.0 |
| 용량 | **824 MB** | **1.08 GB** |

공통 스키마: `unique_id, embedding, timestamp, product_id, grid_cell, grid_row_u, grid_col_r,
geometry, centre_lat, centre_lon, utm_footprint, utm_crs, pixel_bbox, parquet_row, parquet_url`.

**Clay 카드가 "동일한 249k grid cell과 동일 원본 영상을 다른 Major TOM 249k 임베딩 데이터셋과
공유한다"고 명시한다.** 실제 parquet 감사 결과 진짜 paired set이지만, 조인 키는
`grid_cell + product_id`다. **`unique_id` 교집합은 0**이므로 공유 chip ID로 쓰면 조용히 빈 조인이 난다.

### 이 표 자체가 우리 논문의 증거다

같은 chip 위의 두 공개 제품이 **pooling(평균 vs CLS)·밴드 수(12 vs 10)·정규화**에서 다르다.
그리고 이 차이는 데이터셋 카드의 **산문**에만 있고 기계가 읽을 수 있는 필드에는 없다.
모델 가중치 hash는 아예 없다. → W9의 gap 분석에 그대로 들어간다.

**이것이 바꾸는 것.**

1. **paired-input cross-model 실험대가 이미 존재한다.** 계약이 요구하는 "동일 acquisition·동일
   AOI에서 여러 모델" 조건이 공개 자산으로 충족된다. 우리가 계산할 필요가 없다.
2. **규모가 216 site-years에서 248,719 chip으로 올라간다.** `E_compat`의 gallery-size 곡선
   (`10³–10⁶`)을 실제 임베딩으로 그릴 수 있게 된다 — 지금까지는 외삽해야 했던 구간이다.
3. **단, 이들은 cross-family이지 release pair가 아니다.** `S1→S0` 호환성 질문에 직접 답하지
   않는다. release 축은 여전히 W5에서 따로 확보해야 한다.

### 가능한 것과 불가능한 것 — 경계를 명확히

**오늘 당장 가능하다** (GPU 불필요, 총 약 2 GB 내려받기, 노트북 규모):

1. 248,719 chip을 `grid_cell + product_id`로 조인한 **cross-model paired 임베딩 집합**.
2. **gallery-size 곡선을 실측으로** — `10³–10⁵` 구간을 외삽이 아니라 실제 임베딩으로 그린다.
   지금까지 216 site-years에서 외삽하던 구간이다.
3. W2의 **진단 눈멂 행렬을 실제 공개 제품에서** 실행 (CKA·kNN overlap·subspace overlap 등).
4. CC-BY-SA-4.0이므로 인용·재배포 조건이 명확하다.

**불가능하다 — 이 네 가지는 이걸로 못 한다:**

1. **release pair가 아니다.** Major TOM에 OlmoEarth 릴리스는 **하나뿐**이다. 우리 핵심 질문인
   `S1 query → S0 gallery`(같은 family, 다른 릴리스)에 직접 답하지 않는다. 이건 cross-family다.
2. **모델 우열 비교 불가.** 밴드 12 vs 10, pooling 평균 vs CLS, 정규화가 전부 다르다.
   여기서 "Olmo가 Clay보다 낫다"를 읽으면 **계약 교란**이다.
3. **token/공간 수준 분석 불가.** chip당 벡터 하나뿐이다. 우리 실패(R@1 0.0000, 동일 token
   cosine −0.00860, window 내부 spatial CKA 0.427)는 **token 수준**에서 일어났다.
   그 분석은 여전히 우리 216 site-years 로컬 raster에서만 가능하다.
4. **라벨이 없다.** downstream 평가에는 별도 조인이 필요하다.

**요약: 규모·paired·공개성은 얻고, release 축과 token 축은 못 얻는다.** 두 자산은 대체재가
아니라 보완재다 — Major TOM은 넓고 얕게, 우리 제주 216은 좁고 깊게.

### 재계산 대조에 대한 사전 예측 (실행 전에 적는다)

`249k-OlmoEarth-Base`를 로컬에서 재계산해 대조할 때, **우리 216 파이프라인은 768×256×256
dense token raster를 만들고 Major TOM은 unmasked token 평균 1개를 만든다.** 따라서:

> **예측: 동일하게 mean-pool하지 않으면 일치하지 않는다. 동일하게 pool해도 밴드 재정렬과
> normalizer가 정확히 같아야 일치한다.**

이 예측이 맞으면 "같은 모델·같은 영상인데 계약이 달라 값이 다르다"의 **공개 제품 사례**가 되고,
틀리면(그냥 일치하면) 계약 재현 절차 자체가 산출물이다. 어느 쪽이든 손해가 없다.

### 여전히 유효한 공짜 외부 증거

`Core-S2L2A-249k-OlmoEarth-Base`의 표본을 **명시된 모델로 로컬 재계산**해 대조한다.

- 비용: 하루 이하. 임베딩과 모델 둘 다 공개돼 있다.
- 일치하면 계약 재현 절차 자체가 산출물이다.
- 불일치하면 **출시된 공개 제품에서 계약 불일치를 찾은 것**이고, 이슈/PR 하나로 L7이 닫히며
  Figure 1이 가정이 아니라 오늘 쓰이는 제품의 문제가 된다.

## W5. characterization은 release pair가 모자라면 답할 수 없는 질문이다

"예측 가능한가"는 예측기를 fit하고 **보지 않은 pair에서 평가**해야 성립한다. 현재 축은
Olmo(v1/v1.2)와 Prithvi(1.0/2.0) 둘뿐이다. **pair 2개로는 예측기를 fit할 수 없다.**

두 가지를 동시에 해야 한다.

1. **실제 release pair 재고조사.** ✅ 일부 확인했다.

   | family | 확인된 릴리스 | 상태 |
   |---|---|---|
   | OlmoEarth | v1 / v1.2 | ✅ 보유·실행 완료 |
   | Prithvi-EO | 1.0 / 2.0 (+300M/600M) | 공개 확인, 입력계약 미검증 |
   | **Clay** | **v1.0 (2024-06-06) / v1.5 (가중치 2024-11-19)** | ✅ 둘 다 공개. **Major TOM에 v1.5 임베딩도 있음** |
   | SatlasPretrain / DOFA / CROMA / TerraMind | 미조사 | A0 잔여 항목 |

   Clay는 v1.0→v1.5가 실재하는 **세 번째 release pair**이고, 게다가 Major TOM이 v1.5 임베딩을
   이미 공개했으므로 new-side를 계산할 필요가 없다. 크기 변형(300M/600M, Base/Large)은 릴리스
   교체와 다른 종류이므로 **별도 축**으로 표시한다.

   현재 확실한 release pair는 **3개**다. 예측기를 fit하고 held-out 평가하기에는 여전히 부족하다.
2. **W1의 합성 계약 변경으로 fit용 pair를 만든다.** ground truth 불일치 용량이 알려져 있으므로
   예측기 학습에 이상적이다. 실제 pair는 held-out 평가에만 쓴다.

**이 두 개가 없으면 "predictability"는 endpoint가 아니라 희망이다.** 재고조사는 GPU가 필요 없고
반나절이면 된다. A0에 넣어야 한다.

---

## W6. fixed quantizer의 진짜 killer baseline은 ADC다

UniBCT·BiCT·DMU는 이미 경고에 들어가 있다. 그런데 리뷰어 중 ANN 검색을 아는 사람이 던질
질문은 문헌이 아니라 **표준 기법**이다.

> Product quantization에서는 원래 query를 양자화하지 않는다. **Asymmetric Distance
> Computation(ADC)** — query는 float으로 두고 gallery의 PQ code와 직접 거리 계산한다.
> 그러면 "old codebook 고정"은 제약이 아니라 **원래 설계 그대로**다.

✅ 출처 확인: Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*,
IEEE TPAMI 33(1):117–128, 2011. 원문이 "an asymmetric version increases precision, as it
computes the approximate distance between a vector and a code"라고 명시한다. 즉 **query를
양자화하지 않는 것이 PQ의 원래 권장 사용법**이다. 15년 된 표준 기법이므로 리뷰어가 모를 리 없다.

따라서 quantizer-aware 방법을 제안하기 전에 반드시 먼저 돌려야 하는 baseline은 이것이다.

```text
새 query → (calibration에서 학습한) 선형/affine map → old float 공간
         → old PQ codebook에 대해 ADC 검색
```

**이 baseline이 gate를 통과하면 quantizer-aware 방법 논문은 없다.** 반대로 이것이 실패하는
정확한 조건(코드북이 old 분포에 과적합된 정도, 잔차의 비선형성, subspace 회전)을 특정하면
그것이 곧 방법의 존재 이유가 된다.

현재 baseline 표의 `decode-then-align / align-then-quantize`와 가깝지만 **같지 않다.**
ADC는 decode를 하지 않는다. 별도 행으로 명시해야 한다.

---

## W7. 최악의 결과가 사실 좋은 논문이다 — 바닥을 다시 매긴다

현재 중단 기준은 실패 시 "제주 파이프라인 재현성·데이터 품질 보고서로 남긴다"이다.
**이건 바닥을 너무 낮게 잡은 것이다.**

전처리 불일치가 R@1=0을 전부 설명한다고 하자. 그러면 남는 것은 이것이다.

> 공개 EO 임베딩 아카이브의 재사용은 계약 불일치에 취약하며, 불일치는 기존 진단에 보이지 않고,
> 용량–반응 관계가 있으며, 그 결과 고확신 거짓 탐지가 생긴다. 여기 재현 가능한 계측기와
> 진단 행렬이 있다.

이건 위로상이 아니라 **감사 논문**이고, 방법 논문보다 인용될 가능성이 높다. 실제로 이 분야
수상작 다수가 방법이 아니라 감사였다(DivShift, GRAM의 문제선택·자산·coverage).

**실무적 함의:** 바닥을 이렇게 매기면 자원 배분이 달라진다. W1(계측기)과 W2(진단 행렬)는
방법이 성공하든 실패하든 쓰이므로 **먼저** 해야 한다. quantizer-aware 방법은 그 다음이다.
지금 계획의 Day 0(비용곡선)과 충돌하지 않는다 — 비용곡선은 *방법이 필요한가*를 묻고,
W1·W2는 *문제가 실재하는가*를 묻는다. 둘 다 방법보다 앞이다.

---

## W8. (2026-08-24 갱신) AAAI 마감이 지났으므로 당분간 CVPR 하나다

처음에는 한국 트랙의 집으로 AAAI AISI를 제안했다. **사용자 확인 결과 AAAI 마감은 이미 지났다.**
따라서 이 주기의 목표는 **CVPR 단일**이다.

- **내부 완료 목표: 2026-10-31.** CVPR 2027 공식 paper deadline은 아직 확인되지 않았으므로
  이것을 공식 마감이라고 쓰지 않는다.
- 한국 트랙의 venue 결정은 **보류**한다. NeurIPS D&B·ISPRS JPRS·TGRS·다음 회차 AAAI AISI가
  모두 후보로 남아 있으나, 지금 고르는 것은 이득이 없다. CVPR 결과와 라벨 진척(B1)을 보고 정한다.
- 실무적 함의 하나: **마감이 하나뿐이므로 "두 트랙 동시 제출"이라는 압력이 사라졌다.**
  한국 트랙을 CVPR main에서 분리한 결정이 일정 면에서도 옳았다는 뜻이다. 한국 데이터는 이번
  주기에 **CVPR 논문의 failure atlas·stress case**로만 기여하면 된다.
- FoldRefresh의 AAAI-27 AISI 제출 건은 그대로 심사 중이며, K-ALIGN에서는 인용하는 선행
  자산으로만 다룬다(중복 제출 아님).

## W9. ✅ spec의 자리는 이미 있었다 — 그리고 빈칸이 정확히 우리 실패다

모델은 교체된다. adapter는 다음 릴리스에서 다시 짜야 한다. 남는 것은 **계약 자체**다.
처음에는 "STAC 확장을 새로 쓰자"고 제안했다. **확인해 보니 자리는 이미 있었다.**

| 저장소 | 상태 |
|---|---|
| [`stac-extensions/mlm`](https://github.com/stac-extensions/mlm) | 활성. ML **모델**의 메타데이터(가중치 위치, 밴드, 하이퍼파라미터, 추론 런타임) |
| [`stac-extensions/ml-model`](https://github.com/stac-extensions/ml-model) | mlm으로 대체됨(deprecated) |
| [`geo-embeddings/embeddings-stac-specification`](https://github.com/geo-embeddings/embeddings-stac-specification) | **"Proposal", v0.0.1.** 지리공간 벡터 **임베딩 컬렉션**을 기술하는 확장 |

### 빈칸이 정확히 우리가 증명한 두 실패다

세 번째 저장소를 직접 열어 필드를 확인했다.

| 계약 필드 | embeddings-stac v0.0.1 | 우리 근거 |
|---|---|---|
| 시간 합성 창 | `emb:temporal_resolution` (있음) | |
| GSD | `gsd` (있음) | |
| 정규화·후처리 | `emb:preprocessing` / `emb:postprocessing` (있음) | |
| pooling | `emb:runtime_parameters`로 가능 (불명확) | |
| 소스 처리 버전 | `processing:version` (있음) | |
| **모델 가중치 hash** | **없음** | v1→v1.2에서 R@1 0.0000. 같은 이름·같은 차원인데 좌표계가 다르다 |
| **실제 acquisition 날짜** | **없음** | jeju25↔jeju26r **184일 중첩** |
| **temporal recipe 상세** | **없음** | 4기간 계절편향 경로 |
| **밴드 순서** | **없음** | 계약 불일치 축 |
| **input/output content hash** | **없음** | 재현 검증 불가 |

즉 **우리가 문서화한 두 개의 고확신 실패는, 떠오르는 표준이 아직 기록하지 않는 바로 그
필드에서 발생했다.** 이건 우연이라기보다 같은 원인이다 — 사람들이 중요하다고 생각하지 않은
필드가 곧 조용히 깨지는 필드다.

### 그래서 할 일이 바뀐다 (더 싸지고 더 강해진다)

새 확장을 쓰지 않는다. **기존 Proposal에 gap 분석과 근거를 붙여 기여한다.**

1. 필드 대조표를 만든다 (위 표의 완성본).
2. 각 빈칸에 **재현 가능한 실패 증거**를 붙인다 — 우리는 이미 두 개를 갖고 있다.
3. 이슈 또는 PR로 제안한다. v0.0.1 "Proposal" 단계이므로 **지금이 정확히 열려 있는 시점**이다.

- 새 표준을 만드는 것보다 **채택 가능성이 훨씬 높다**. 이미 있는 것에 근거를 대는 일이다.
- L7(공개 증거)을 즉시 닫는다.
- 논문의 마지막 절이 "우리는 이 필드들을 표준에 넣자고 제안했고 그 근거가 이 실험이다"로 닫힌다.
- **비용이 거의 0이다.** GPU가 필요 없다.

**미확인 잔여**: `mlm`과 `embeddings-stac`의 역할 경계, 그리고 두 확장이 이 필드들을 각각
어디에 두는 것이 맞는지. 제안 전에 두 저장소의 이슈를 읽어야 한다.

## W10. (2026-08-24) 한국 트랙은 포기된 게 아니라 **날짜가 없다**

사용자 질문: *"한국식으로 이거 하는 건 깔끔하게 벌써 포기한 거야?"*

정직한 답은 **아니다. 그러나 지금 상태로 두면 포기된다**이다.

### 실제로 빠진 것과 남은 것

| 결정 | 빠졌나 | 실제 내용 |
|---|---|---|
| 필지 경계를 기여로 | **빠짐** | 전지구 10 m 필지지도(241개국 31.7억 polygon)가 이미 있다. anchor로는 계속 쓴다 |
| 한국판 FLAIR-HUB | **빠짐** | 장르 선점 + dense 주석이 환경부 토지피복 재포장이 된다 |
| "한국 데이터로 정확도 향상"을 headline으로 | **빠짐** | PANGAEA: full-label에서 supervised 우세. 주장 형태가 라벨 절감으로 바뀐 것이지 축이 사라진 게 아니다 |
| 한국 데이터가 CVPR 논문에 들어가는가 | **남음** | Figure 1 failure atlas(184일 중첩 → z=10.6 인공물), stress case, motivation |
| `E_repr`/`E_fusion` | **남음, 연기됨** | B1 라벨이 10주 안에 안 끝나서 이번 주기 headline에서 빠졌다 |
| event-first 재설계 + `R/V/T` | **남음, 가장 새로움** | 사용자 본인 판단대로 잠재 novelty가 가장 높다 |

즉 **빠진 셋은 전부 "이미 남이 차지했거나 순환논증"이라 빠진 것**이고, 한국 고유 기여는 하나도
반증되지 않았다. 다만 전부 **연기**됐다.

### 진짜 위험은 연기의 누적이다

3일 동안 한국 트랙은 네 번 밀렸다.

1. derivability screen 도입 → `E_repr`의 재료가 먼저 검증돼야 함
2. 문헌 감사 → 정확도 headline이 라벨 절감으로 강등
3. B1 비용 실측 → 라벨이 임계경로에서 제외
4. CVPR 단일 마감 → 한국 트랙이 다음 주기로

**네 번 다 근거가 옳았다.** 그런데 네 번 다 **날짜를 붙이지 않았다.** 근거 있는 연기가
날짜 없이 쌓이면 그게 조용한 포기다. 이 프로젝트의 L4(판정 기준을 먼저 정한다)를
일정에도 적용해야 한다.

### 고칠 방법 — 14일 P0 안에 한국 데이터 P0를 같이 넣는다

`K_ALIGN_CVPR_READINESS_AUDIT.md` §3.3에 이미 5단계 데이터 P0가 있다. 그 문서 스스로
**"이 단계는 GPU보다 데이터 정의가 병목"**이라고 적었다. 그렇다면 14일 P0와 **경쟁하지 않는다.**

| | 14일 P0 (CVPR 방법) | 병행 (한국 데이터) |
|---|---|---|
| D1–5 | split·adapter 구축 (GPU) | BuildingHUB 8,794행 → event universe 재구성 |
| D6–8 | bridge 비교 (GPU) | unique event/중복·정정 분리, `published_time` 필드 실재 확인 |
| D9–11 | 공개 task 4-cell (GPU) | 200 event 층화추출 → S2 전후 가용성·구름·footprint |
| D12–14 | 압축·비용 (GPU) | **NGII lead time 측정 결과 회수** |

D1–14는 GPU-bound이고 위 오른쪽 열은 전부 CPU·API·문서 작업이다. **한 사람이 겹쳐 할 수 있는
분량인지가 유일한 제약**이며, 그렇지 않다면 오른쪽 열을 절반으로 줄이되 **NGII 신청만은
D1에 넣는다**(승인 대기가 길어 지금 넣지 않으면 다음 주기도 못 연다).

**판정**: 이 병행 P0가 14일 안에 착수되지 않으면, 그때는 "연기"가 아니라 실제로 포기한 것이므로
문서에 그렇게 적는다.

---

## W11. event-first 재설계에 빠진 칸 하나 — cell D

사용자의 순서 뒤집기(사건 universe → 전후 EO → matched control → 독립 label)는 옳다.
old 방식의 치명적 문제(positive coverage 0/14)를 정면으로 푼다. 다만 한 칸이 빠졌다.

BuildingHUB에서 표집을 시작하면 **행정기록이 있는 것이 표본틀의 정의**가 된다. 그러면 행정기록을
보는 모델은 그 틀 위에서 자동으로 이긴다. 필요한 것은 2×2다.

| | 행정기록 있음 | 행정기록 없음 |
|---|---|---|
| **실제 변화 O** | **A** 허가 후 실제 착공 | **D** 무허가·미기록 변화 |
| **실제 변화 X** | **B** 허가 후 미착공·취소 | **C** 배경 |

- **A·B는 BuildingHUB에서 공짜로 나온다** — event universe에서 뽑아 항공사진으로 확인만 하면 된다.
- **C도 공짜다** — 무작위 표집.
- **D가 비싸고, D가 과학적 요점 전부다.** D는 행정근거가 **실패하는** 칸이다.
  A·B·C만 있으면 만든 것은 "허가기록 검증기"이지 연구 기여가 아니다.

**D가 비어 있지 않다는 증거는 이미 이 저장소에 있다**: 개발행위허가 제주 **2023·2024가 0행**이다.
그 두 해의 모든 실제 변화는 정의상 cell D다. 즉 D는 잔여 범주가 아니라 **구조적으로 큰 모집단**이다.

**표집 설계 권고**: 모델로 D 후보를 고르면 표본이 모델에 의존해 편향된다. 대신
**행정 coverage 상태(있음/없음) × 연도로 층화한 뒤 층 내부에서 무작위 추출 → 블라인드 판독**한다.
무변화 사이트를 더 읽어야 하므로 비싸지만, 두 칸 모두 편향 없이 추정된다.
그리고 이 층화 자체가 Paper B(*행정근거 coverage가 누가 감시받는지를 결정한다*)의 주 결과다.

**부수 효과 — B1이 싸진다.** old 설계에서 사람은 변화를 **발견**해야 했다(368 분모, 희소 positive).
새 설계에서 A·B·C의 사람 작업은 **확인**이다. 발견보다 훨씬 빠르고 신뢰도도 높다.
비용은 D에 집중되며, 그건 줄일 게 아니라 **의도적으로 쓸 예산**이다.

---

## W12. `R/V/T` 분해의 경고등 하나 — `R≈0`인데 `T>0`이면 축하할 일이 아니다

단일 derivability 기준을 `R_source`(EO에서의 회복 가능성) / `V_source`(추론 시 독립 task 가치) /
`T_source`(train-only context가 EO-only student로 전달되는가)로 나눈 것은 명백한 개선이다.
기존의 hard exclusion보다 정확하다.

여기에 정보이론적 제약 하나를 명시해야 한다.

> 공공정보가 EO 관측에서 **정말로** 회복 불가능하면(`R ≈ 0`), 추론 때 EO만 보는 student는
> 그 정보를 운반할 수 없다. 따라서 **`R ≈ 0`이면 `T ≈ 0`이어야 한다.**

그러므로 `R ≈ 0`인데 `T > 0`이 관측되면 셋 중 하나이며, **전부 조사 대상이지 성공이 아니다.**

| 원인 | 어떻게 구분하나 |
|---|---|
| ① 누수 — student가 보면 안 될 것을 봤다 | leak sentinel, split 재검사, feature/label role 중복 감사 |
| ② `R` 측정 실패 — probe가 너무 약했다 | probe 용량·학습량을 올려 `R` 재측정 |
| ③ context가 정보원이 아니라 **정규화·커리큘럼**으로 작동했다 | context를 무작위 셔플해도 같은 이득이 나오는지 확인 |

③이면 그것도 실제 발견이지만 **다른 주장**이다 — "공공데이터가 EO 표현에 정보를 넣었다"가 아니라
"공공 context가 학습 정규화로 작동했다"이다. 두 문장을 같은 표에 섞으면 안 된다.

역으로 `R`이 매우 높은 source는 `T > 0`이 나와도 기여가 약하다. EO에서 이미 회복 가능한 것을
다시 넣은 것이기 때문이다. **따라서 사용자가 정한 "R 중간 + V·T 양수"라는 창(window)이 옳고,
그 창이 좁은 이유가 바로 이것**이다. 문서에 이유를 남겨야 나중에 스스로 창을 넓히지 않는다.

---

## W13. (2026-08-24) VLM 방향 — 구조는 맞고, **제목 질문이 틀렸다**

제안된 논문 질문:

> "Earth VLM은 두 EO 임베딩을 비교하면 안 되는 순간을 알 수 있는가?"

### 이 질문은 거의 확실히 "아니오, 그리고 알 필요도 없다"로 답해진다

**계약 불일치는 시각적 사실이 아니라 메타데이터 사실이다.**

- mean pooling인지 CLS인지, 밴드가 12개인지 10개인지, 가중치가 v1인지 v1.2인지 —
  **어느 것도 픽셀에 없다.** VLM은 원리적으로 볼 수 없다.
- 반대로 메타데이터가 있으면 **10줄짜리 결정론적 검사가 100%로 답한다.** VLM이 더할 게 없다.

따라서 제안된 5-way ablation에서 `계약 gate만`이 `계약 gate + VLM`을 이길 가능성이 높고,
그러면 논문은 자기 제목에 "아니오"라고 답한다. 그건 정직한 음성 결과이고 workshop 한 편은
되지만, **VLM 장치 전체를 켜 놓고 그 결론에 도달하는 것은 자원 낭비다.**

한 가지 예외: 계약 필드가 **없는** 아카이브에서 VLM이 *결과*를 보고 역추정할 수는 있다.
"이 두 임베딩의 top-k가 전부 구름이다 → 무언가 어긋났다." 그러나 이건 계약 탐지가 아니라
**결과 이상 탐지**이고, 그 역할은 통계 검정이 더 잘한다.

### 그런데 제안 안에 강한 논문이 이미 들어 있다 — 역할 2번

> "영상 변화는 있지만 공식 근거는 없음"처럼 구조화해서 답하게 하고, **근거가 없으면 원인을
> 만들어내지 않고 보류한다.**

이게 진짜 기여다. 이름을 붙이면 **행정근거 결손 아래의 원인 환각(fabricated cause attribution)**이다.

**선행 조사 결과 — 인접 영역은 붐비지만 이 칸은 비어 있다.**

| 이미 점유됨 | 내용 |
|---|---|
| [GeoChat (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Kuckreja_GeoChat_Grounded_Large_Vision-Language_Model_for_Remote_Sensing_CVPR_2024_paper.html), [EarthDial (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Soni_EarthDial_Turning_Multi-sensory_Earth_Observations_to_Interactive_Dialogues_CVPR_2025_paper.html) | EO 대화·grounding·다중분광 |
| [GEOBench-VLM (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Danish_GEOBench-VLM_Benchmarking_Vision-Language_Models_for_Geospatial_Tasks_ICCV_2025_paper.html) | 지리공간 VLM 벤치마크 (최고 MCQ 41.7%) |
| [RSHallu](https://arxiv.org/pdf/2602.10799), [DDFAV/RSPOPE](https://doi.org/10.3390/rs17040719), [UHR-Micro](https://arxiv.org/pdf/2605.12237), [VLRS-Bench](https://arxiv.org/pdf/2602.07045), [CHOICE](https://arxiv.org/pdf/2411.18145) | RS VLM **환각 벤치마크** |
| [ChangeVLM](https://doi.org/10.3390/rs18101671), [VLM-BCD](https://dl.acm.org/doi/10.1145/3595916.3626357), [ViLaCD-R1](https://arxiv.org/pdf/2512.23244), [Decoding the Delta](https://arxiv.org/pdf/2604.14044) | VLM 변화탐지 |
| [Knowing When Not to Answer](https://arxiv.org/html/2604.14799) | 다중모달 abstention 평가 |

**핵심 구분점**: 기존 RS 환각 벤치마크는 전부 **객체 존재**를 검사한다(POPE 계열 — "비행기가
있는가?"). **변화의 원인 주장**을 외부 기록으로 검증하는 것은 검색에 나오지 않았다.
이유는 명확하다 — **시점이 찍힌 행정기록을 가진 팀이 거의 없기 때문이다.** 한국은 있다.

### 그래서 바꿔야 할 것은 구조가 아니라 과녁이다

제안된 아키텍처(규칙 먼저 → VLM → 공공근거)는 **옳다.** 역할 분담만 정확히 쓴다.

| 층 | 답하는 질문 | 성격 | 기대 성능 |
|---|---|---|---|
| 계약 gate | 이 비교가 **유효한가** | 메타데이터·결정론적 | 100% (또는 필드 부재로 ABSTAIN) |
| VLM | 유효한 비교에서 이게 **진짜 변화인가** | 지각 | 불안정 (GEOBench 41.7% 참고) |
| **한국 공공근거** | VLM이 만든 **원인 설명이 사실인가** | 기록 대조 | **여기가 기여** |

바뀐 제목 질문:

> **시점이 찍힌 행정근거는 Earth VLM의 허위 원인 설명을 줄이는가? 그리고 그 근거의 coverage가
> 비어 있는 곳에서 환각은 누구에게 집중되는가?**

이 형태의 장점:

1. 계약 gate가 **버려지지 않고 상류 필터**가 된다 — 인공물 후보를 VLM이 보기 전에 제거한다.
   184일 중첩 5건·4기간 5건이 정확히 그 역할의 실증이다.
2. **cell D와 직접 연결된다.** cell D(변화는 있는데 기록이 없음)가 VLM이 원인을 지어낼 바로
   그 칸이고, 정답은 "보류"다.
3. **coverage 편향이 결과가 된다.** 개발행위허가 제주 2023·2024가 0행이므로, "행정근거가 없는
   해·지역에서 환각이 증가하는가"를 측정할 수 있다. 이것이 Paper B의 주 결과와 같은 축이다.
4. 41.7%짜리 teacher로 증류하지 않는다 — 역할 3(VLM→embedding 증류)은 계속 보류가 맞다.

### 하루짜리 사전 검사 — 이걸 먼저 한다

새 데이터 없이 지금 있는 것으로 5-way를 돌린다.

```text
표본: 14 candidates + v5 blind pair 5쌍 + 184일 중첩 노출 5건 + 4기간 source 5건 (약 20~29건)
비교: EO만 / VLM만 / 계약 gate만 / gate+VLM / gate+VLM+공공근거
측정: 잘못된 REUSE 비율, 원인 분류 정확도, 근거 일치율, 보류 정확도
```

**판정 규칙(먼저 적는다)**: `계약 gate만`이 `gate+VLM`과 잘못된 REUSE 비율에서 같으면,
"VLM이 계약 불일치를 안다"는 주장을 **버리고** 원인 환각 축으로만 간다.
이 20건은 이미 사람 판독·구조 결함 분류가 끝나 있어 정답이 있다. **GPU도 새 라벨도 필요 없다.**

### 마지막으로 정직하게 — 이건 4일 만의 네 번째 방향이다

compat/method → 한국 event-first → wide-angle 계측기 → VLM. **아이디어는 매번 좋아지고 있다.**
그러나 마감은 10주, 인력은 1인, 14일 P0는 이미 방법 트랙에 배정돼 있다.

- 위 하루짜리 사전 검사는 **P0와 경쟁하지 않는다**(GPU 불필요). 그것만 먼저 한다.
- 결과가 "gate만으로 충분"이면 VLM은 원인 환각 축으로 **범위가 좁아진 채** 살아남는다.
- 결과가 "VLM이 뭔가 더한다"면 그때 트랙으로 승격한다.
- **어느 쪽이든 14일 P0의 D1–14는 건드리지 않는다.**

---

## W14. (2026-08-24) MountainShift 검토 + 다섯 방향 통합 우선순위

### 좋은 점 — 그리고 아무도 안 세고 있는 진짜 이유

1. **ETH 정합성.** GLAMOS는 ETH가 공동운영한다. 다섯 방향 중 **지원 목표와 자산이 겹치는 유일한
   방향**이다. 이건 과학 논거가 아니라 경력 논거이고, 그렇게 부르고 쓰면 정당하다. 숨기면 안 된다.
2. **Phase 0이 라벨 병목을 통과한다.** AvalCD(4지역 bi-temporal SAR)와
   Sen12Landslides(15지역, refined 74,956 landslides)는 공개 benchmark로 시작할 수 있다.
   한국 트랙은 라벨 1,200건 + NGII 승인이 필요했다. **지금까지 나온 것 중 실제 region-holdout
   transfer 수치에 가장 빨리 닿는 경로다.** 단 annotation license와 OlmoEarth 입력 변환은
   20 sample gate를 먼저 통과해야 한다.
3. dual-speed 구조(`z_global / z_region / r_t`)가 K-ALIGN의 stable/residual 분리와 **같은 구조**다.
   새로 만드는 게 아니라 같은 설계를 다른 도메인에 놓은 것이다.

### 비용 1 — MountainShift는 우리의 가장 희소한 자산을 버린다

한국의 희소 자산은 산악이 아니라 **필지에 결속된 시점 있는 행정기록**이다.
BuildingHUB 8,794 event행 × PNU × 허가·착공·사용승인 일자.

**산사태와 산불에는 건축 인허가가 없다.** MountainShift에서 한국은 "또 하나의 산사태 지역"이 되고,
cell D·coverage 편향·`published_time` 같은 행정 provenance 계측기는 **대부분 적용되지 않는다.**

즉 MountainShift는 **희소 자산을 흔한 도메인 전이 연구와 맞바꾼다.** 그 교환이 나쁘다는 게 아니라,
**교환이라는 사실이 문서 어디에도 없다.** 결정하려면 이걸 알고 해야 한다.

완화책: Track B/C에서 한국의 역할을 "산사태 지역"이 아니라 **"행정근거가 있는 유일한 지역"**으로
둔다. 알프스·HKH에는 GLAMOS/ARPA/ICIMOD inventory가 있지만 **필지 단위 인허가 시각은 없다.**
그러면 한국은 대체 가능한 4번째 지역이 아니라 **evidence-aware 축의 유일한 근거지**가 된다.

### 비용 2 — 지금까지 중 가장 큰 범위 확장

전체 프로그램으로 승격하면 GLAMOS, swissALTI3D, ARPA SIFraP, ARPA 눈사태 portal, ICIMOD RDS,
산림청 위험지도·발령이력이 새로 붙고 각각 라이선스·다운로드·시간정렬 확인이 필요하다.
그러나 **Phase 0에서는 이 기관별 결합을 열지 않는다.** AvalCD와 Sen12Landslides만으로 먼저
transfer 신호를 죽이거나 살린 뒤, 통과했을 때만 5-cell × 다지역 × label 1/5/10/50/100% 격자를 연다.

**오늘 GPU는 0장이고 마감은 10주다.** Phase 0(4 method × 2 benchmark × region holdout)만 해도
이미 상당한 GPU 프로그램이다.

### 비용 3 — "여러 데이터 합친 게 낫다"의 답은 부분적으로 이미 알려져 있다

5-cell 설계 자체는 옳다. 다만 PANGAEA(저라벨에서 GFM 우세, full-label에서 supervised 우세)와
AnySat(다센서 공동학습 전이)이 이미 있으므로 **"다지역이 저라벨에서 낫다"는 재현될 가능성이 높다.**

따라서 headline을 거기에 두면 약하다. 새로운 것은 **worst-region 비악화**, **API 누락·시점이동에서의
보류**, **location shortcut 배제**다. 문서의 사전 성공 기준은 이미 그렇게 적혀 있으니, **논문 제목만
그쪽으로 옮기면 된다.**

### 그런데 하나는 오히려 MountainShift에서 더 강해진다

**계절 눈이 계약 불일치를 훨씬 극적으로 만든다.** 제주에서 6개월 창 offset은 구름·식생 차이였지만,
**알프스에서 6개월 offset은 적설 유무를 통째로 뒤집는다.** W1 dose–response 계측기를 산악에서
돌리면 인공물 크기가 제주보다 훨씬 크고 육안으로 명백하다.

즉 계약 축은 MountainShift에서 **버려지는 게 아니라 증폭된다.** 이게 두 방향의 합류점이다.

---

### 다섯 방향 통합 우선순위 — 1·2는 완료, 현재 queue는 다시 좁혔다

원칙: **(결정력 × 저렴함) ÷ (새 의존성)**. 그리고 하드룰 하나 —
**측정 하나가 착지하기 전에는 새 방향을 열지 않는다.**

먼저 각 작업이 **몇 개 방향에 동시에 쓰이는지** 센다. 이게 진짜 우선순위 신호다.

| 작업 | compat | 계약/감사 | 한국 | VLM | Mountain | GPU |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **W1 dose–response 계측기** | ○ | ○ | ○ | ○ | ○ | 필요 |
| Major TOM 계약 감사 | ○ | ○ | — | — | — | 불필요 |
| frozen probe (water/snow/debris/veg-loss) | ○ | — | — | — | ○ | 소량 |
| Phase 0 benchmark 다운로드·라이선스 확인 | — | — | — | — | ○ | 불필요 |
| 한국 데이터 P0 + NGII 신청 | — | — | ○ | ○ | △ | 불필요 |
| ADC baseline | ○ | — | — | — | — | 소량 |
| 5-cell 다지역 격자 | — | — | — | — | ○ | **대량** |

**W1이 유일하게 다섯 칸 전부에 들어간다.** 이게 1순위다.

| 순위 | 작업 | 왜 | 조건 |
|---:|---|---|---|
| **1** | **Major TOM 계약 감사** (`code/audit_majortom_contract.py`) | **GPU 0장인 오늘 할 수 있는 유일한 결정적 측정.** paired 전제를 살리거나 죽인다. 결과표가 embeddings-stac gap에 바로 들어감 | 지금 |
| **2** | **W1 dose–response 최소판** (밴드 순서·정규화) | 다섯 방향 전부에 쓰임. raster 재사용. **산악으로 확장하면 눈 때문에 더 강해짐** | GPU 나면 즉시 |
| **3** | **Phase 0 다운로드·라이선스 확인** (AvalCD, Sen12Landslides) | MountainShift 전체의 gate. 하루. GPU 불필요. **여기서 막히면 MountainShift는 없다** | 1과 병행 |
| **4** | **NGII 신청 + 한국 event universe** | 승인 대기가 길어 지금 안 넣으면 다음 주기도 못 엶 | 1과 병행 |
| **5** | frozen probe (제주 임베딩에서 water/snow/debris) | MountainShift 1단계이자 compat 기계와 동일. 소량 GPU | 2 이후 |
| **6** | ADC baseline | quantizer 방법을 **싸게 죽일 수 있음** | 2 이후 |
| **7** | 14일 방법 P0 | 이미 설계됨 | 2·6 결과 반영 후 |
| **8** | MountainShift Phase 1 (지역별 20건) | 3이 통과했을 때만 | 3 통과 후 |
| **보류** | VLM 트랙 | 하루짜리 5-way 사전검사만. 그 외 동결 | 사전검사 결과 |
| **보류** | 5-cell 다지역 격자 | GPU 대량. Phase 0 신호 없이 열지 않음 | 8 이후 |

### 이 순서가 옳은지 검사하는 방법

**질문: 1·2·3이 전부 음성으로 나오면 무엇이 남는가?**

- Major TOM이 paired가 아니다 → 규모 축을 잃지만 제주 216 token 축은 남는다.
- dose–response에 반응이 없다 → **계약 축 전체가 죽는다.** 그러면 MountainShift가 주 방향이 된다.
- Phase 0 다운로드가 막힌다 → MountainShift가 죽는다. 계약 축과 한국 축이 남는다.

**세 개가 동시에 죽을 확률은 낮고, 어느 하나가 죽어도 나머지가 산다.** 이게 이 순서를 고른 이유다.
반대로 5-cell 격자부터 시작하면 GPU를 다 쓰고도 세 축 중 어느 것도 판정하지 못한다.

**후속 실행 상태(2026-08-24).** 1은 M2, 2는 M3–M5로 완료됐고 M5가 일반 주장을 철회시켰다.
따라서 위 표를 새 작업 순서로 반복하지 않는다. 현재 임계경로는
① public downstream frozen-head task table, ② 외부 sample-schema PR 또는 재현 report,
③ AvalCD/Sen12Landslides 20-sample transform·license gate다. 상세 상태는 `GOAL.md`가 SSOT다.

---

## 우선순위 — 무엇을 먼저 하는가

기존 계획의 Day 0(compact 재임베딩 비용곡선)은 유지한다. 그 옆에 병렬로 놓는다.

| 순위 | 항목 | 비용 | 방법 성패와 무관하게 쓰이는가 |
|---:|---|---|---|
| 1 | Day 0 비용곡선 (기존) | 중 | 예 — 필요성 판정 |
| 1 | **W1 계측기: 합성 계약 불일치 dose–response** | **낮음** (raster 재사용) | **예** |
| 1 | **W4 Major TOM 계약 대조 + 재계산 검증** ✅사양확인·약 2 GB·GPU 불필요 | **매우 낮음** (하루 이하) | **예** |
| 2 | **W2 진단 × 불일치 행렬** | 낮음 | **예** |
| 2 | **W5 잔여 release pair 조사** ✅3개 확보 | 매우 낮음 (GPU 불필요) | 예 — endpoint 성립 여부 |
| 3 | **W6 ADC baseline** | 낮음 | 예 — 방법 존재 이유 |
| 3 | **W9 embeddings-stac gap 분석 + 이슈/PR** ✅자리확인 | 거의 0 (GPU 불필요) | 예 |
| 4 | W3 비-EO 데모 | 낮음 (반나절) | 예 — 경계 확정 |
| 5 | quantizer-aware 방법 | 높음 | 아니오 |
| **병행** | **한국 데이터 P0 (§3.3) + NGII 신청 D1** | 낮음 (CPU·API) | **예 — 다음 주기의 전제** |

**한 줄 요약:** 지금 가장 싼 항목들이 가장 방법-독립적이고, 방법이 실패해도 전부 살아남는다.
그것들을 먼저 한다.

---

## 동의하고 넘어가는 것

- `R@1=0 → affine 61–70%`를 비선형 잔차의 증거로 쓰지 않는다는 정정에 동의한다.
  밴드 순서·정규화·timestamp·pooling 배제가 먼저다. W2의 진단 행렬이 이 배제 작업과 같은 실험이다.
- 37.5분/216건을 재임베딩 비용으로 외삽하지 않는다는 지적에 동의한다. 49.1 GiB GeoTIFF 쓰기가
  포함돼 있다.
- "오름은 보전지역이라 행정사건 0"을 아직 증명되지 않은 것으로 둔 판단에 동의한다.
- 한국 트랙을 CVPR main에서 분리한 결정에 동의한다.
