# MountainShift 실험 설계 (2026-08-25)

`MOUNTAIN_EVIDENCE_TRANSFER.md`의 연구 질문을 **실행 가능한 arm·지표·게이트**로 옮긴 문서다.
연구 질문과 자산 목록은 그쪽이 authoritative하고, 여기는 "무엇을 어떤 순서로 돌리고
무엇이 나오면 죽이는가"만 다룬다.

## 0. 먼저 — 설계에 아직 없던 위험 세 개

### R1. headline이 n=3이고, 그중 둘은 접근조차 미측정이다

3-way leave-one-country-out은 **독립 표본이 3개**다. 신뢰구간을 만들 수 없고,
한국을 뺀 두 나라(네팔·스위스)는 snapshot join을 **한 번도 하지 않았다.**
어느 한 나라가 막히면 headline이 2-way로 무너지는데, 그 경우의 대비가 문서에 없다.

**그런데 이미 기록된 자산에 답이 있다.** Sen12Landslides는 S1/S2+DEM에
refined 74,956 landslides와 event date/confidence를 담고 있다. 2026-08-25 실측 결과
inventory의 지역은 **16개**이고(문서의 "15지역"은 부정확했다), R2의 저자 교락을 통제하면
**11개**가 남는다 (M11·M12).

→ **headline spine을 공개 데이터의 저자 고정 11지역 leave-one-region-out으로 옮긴다.**
3국은 headline이 아니라 **annotation-shift + live-residual 확장**으로 붙인다.

| | 저자 고정 11지역 LOCO (공개) | 3국 LOCO |
|---|---|---|
| 독립 표본 | **11** (저자 고정 후) | 3 |
| 재배포 제약 | 없음 | AI-Hub 있음 |
| 접근 위험 | 이미 공개 | 네팔·스위스 미측정 |
| 신뢰구간 | 만들 수 있다 | 사실상 불가 |
| 죽을 확률 | 낮음 | 두 나라 중 하나만 막혀도 |

이렇게 하면 **네팔·스위스가 전부 막혀도 논문이 산다.** 3국은 기여를 더하는 층이지
논문의 생사가 걸린 층이 아니게 된다.

### R2. annotation 생성 과정이 지역마다 다르다 — 2026-08-25 측정 완료 (M12)

우려가 아니라 **측정된 사실**이다. Sen12Landslides inventory 74,956 폴리곤을 감사했다.

| | 실측 |
|---|---|
| 지역 | 16 |
| 라벨 저자 | 5 |
| **단일 저자 90% 이상인 지역** | **13 / 16** |
| **MMU(p1 면적) 최대 비** | **1,916×** (Italy 62.9 m² vs USA_Alaska 120,569 m²) |
| MMU 10배 이상 차이나는 지역쌍 | **50** |
| median 면적 범위 | 409.8 ~ 466,259.8 m² (**1,100배**) |

**지역과 저자가 거의 같은 변수다.** Italy 47,522 = Ferrario 47,522.
따라서 순진한 leave-one-region-out은 부분적으로 leave-one-**annotator**-out이고,
성능 하락이 지형 차이인지 라벨 차이인지 분리되지 않는다.

#### 실패한 대안 (기록으로 남긴다)

전 지역 공통 면적 하한으로 조화하려 했다. `max(MMU) = 120,569 m²`가 되어
**Italy가 전멸한다**(p99가 6,965 m²). 단순 면적 하한은 불가능하다.

#### 채택 — 저자를 고정해 교락을 **설계로** 제거한다

Höhn et al. (2025) 단독이 14지역 16,306 폴리곤을 덮는다.

| | 전체 저자 | **저자 고정** |
|---|---|---|
| 지역 | 16 | 14 (≥100 폴리곤 **11**) |
| MMU 비 | 1,916× | **20×** |
| 공통 하한 8,821.8 m² | Italy 전멸 | 11지역 15.2~99.0% 보존, **7,921 폴리곤** |

**headline은 저자 고정 11지역 LOCO다.**
Chimanimani · China · Hiroshima · Hokkaido · Indonesia · Itogon ·
Kyrgyzstan1 · Kyrgyzstan2 · LanaoDelNorte · Newzealand · Thrissur

Italy·DominicaMaria·USA_* 는 headline에서 빼고, **annotation-shift 전용 arm**으로 따로 쓴다
— "같은 지형인데 다른 저자가 그리면 성능이 얼마나 달라지는가"는 그 자체로 측정 가치가 있다.

→ 게이트 **G-A**는 이제 아래 §4의 구체 절차로 확정한다.

### R3. `region residual → F1 +2%p`는 그 자체로 novelty가 아니다

산사태 모델에 DEM/slope를 넣으면 좋아지는 것은 **예상된 결과**다. 리뷰어는
"geomorphometric feature를 더한 것 아니냐"고 한다. 기여는 개선폭이 아니라 아래 셋에 있다.

1. **transfer** — 봉인한 지역/국가에서 저라벨로도 이득이 유지되는가
2. **cutoff 유효성** — live residual이 미래정보 없이 이득을 내는가
3. **분해** — `E_static / E_live / E_transfer / E_refresh`를 따로 보고

따라서 성공 기준의 무게를 `+2%p`가 아니라 **`worst-region ≤ 1%p 저하`와
`negative control에서 이득 소멸`**에 둔다.

---

## 1. 표기와 arm

```
z_global  frozen OLMoEarth 임베딩 (느린 cache)
z_region  지역 정적 residual  — DEM·slope·aspect·기후평년·장기 토지피복
r_t       live residual       — 강우·적설·경보·freshness (cutoff 유효)
ŷ = h_task( z_global ⊕ z_region ⊕ r_t )
```

| arm | 입력 | 무엇을 가른다 |
|---|---|---|
| **A0** | z_global (지역 내부 학습만) | local-only 하한 |
| **A1** | z_global (전 지역 pooled) | naive pooling |
| **A2** | z_global + local head | 공유 표현 + 지역 head |
| **A3** | A2 + z_region | **E_static** |
| **A4** | A3 + r_t | **E_live** |
| **T-m** | A4를 Höhn 11지역에서 학습 → Korea 적용 | matched-annotation 전이 |
| **T-x** | A4를 Italy에서 학습 → Korea 적용 | mismatched-annotation 전이. `T-m − T-x` = **E_annotation** |
| **B1** | region-ID one-hot | residual이 그냥 "지역 식별자"인지 반증 |
| **B2** | 위경도 (lat/lon) | 같음 |
| **B3** | raw-spectral retrieval | 임베딩이 필요하긴 한지 |
| **C1** | U-TAE 또는 3D-UNet (task-specific) | GeoFM이 이기긴 하는지 |
| **C2** | Prithvi-EO-2.0 **또는** TerraMind 하나 | 방향이 backbone 무관한지 |

`E_transfer` = A4의 LOCO 성능 − A0(hold-out 지역 자체 학습).
`E_annotation` = T-m − T-x. 도화 기준 차이가 전이에 주는 단독 손실임.
`E_refresh` = FoldRefresh arm에서만. **local/live 정확도 기여로 세지 않는다.**

## 2. 평가 단위와 지표

- **단위는 지역(region/country)이다.** 타일이 아니다.
- 지표: F1, AUPRC (segmentation) / Recall@20, nDCG@20 (event retrieval)
- 보고는 **항상 넷을 같이** 둔다:
  ① 지역별 원자료 ② region-macro 평균 ③ LOCO 폴드 전체 ④ **지역 단위 bootstrap/jackknife CI**
- **금지 표현**: "N개의 독립 사례" (N이 타일 수일 때). 독립 표본은 지역 수다.
- 라벨 regime: zero-shot / target label **1% / 5% / 10%** 곡선

## 3. 실행 순서 — 공개 spine 먼저, 3국은 확장

```
P1  Sen12Landslides **저자 고정 11지역** leave-one-region-out  ← headline spine (공개)
P2  G-A annotation-process 감사   **[완료 M12]**        ← 모든 cross-region 주장의 전제
P3  한국(봉인됨 M10)을 12번째 지역으로 + T-m/T-x arm     ← annotation shift 측정 (M13)
P4  네팔·스위스 access audit 통과분만 추가                ← live residual 확장
P5  FoldRefresh continuity/cost 표                       ← E_refresh
```

**P1이 통과하지 못하면 P3~P5를 열지 않는다.** 공개 데이터에서 안 되는 방법을
제한 데이터로 살리려는 시도는 하지 않는다.

C2-C(한국 12밴드 복구)는 **P3의 지원 gate이며 최대 1일**이다. 실패하면 한국은
v1/10밴드 + B01·B09 band-group missing mask로 간다. P1을 막지 않는다.

## 4. 사전 등록 게이트

| ID | 시점 | 통과 조건 | 실패 시 |
|---|---|---|---|
| **G-0** | P1 착수 전 | Sen12Landslides 다운로드·라이선스·지역 분리 확인 **[통과 M11]**. split 파일 단위 정의는 미확인 | MountainShift 중단. 여기서 막히면 없다 |
| **G-P** | P1 probe | frozen OLMo probe가 scratch/U-TAE의 **95% 이상** 또는 raw-spectral retrieval보다 우수 | GeoFM을 backbone으로 쓰지 않는다. task model로 전환 |
| **G-A** | 모든 cross-region 주장 전 | 아래 4단계를 통과해야 한다 (M12에서 1·2 실행 완료) | 위반 지역을 headline에서 빼고 annotation-shift arm으로 |
| **G-S** | E_static 주장 | region-macro F1 또는 Recall@20 **+2%p**, **worst-region 저하 ≤1%p**, 지역 bootstrap CI 하한 > 0, **B1·B2를 유의하게 상회** | E_static 주장 철회 |
| **G-N** | E_static·E_live 주장 전 | **negative control**: region-shuffle / time-shift 시 이득이 **소멸**해야 한다 | 이득이 남으면 그것은 지역 정보가 아니라 leakage다. 주장 전부 철회 |
| **G-L** | E_live 주장 | cutoff replay에서 observed/published/retrieved 시각 95% 이상 확보, **미래정보 0건**, AUPRC 또는 detection lead-time 개선 | E_live를 inference-fusion으로만 보고 |
| **G-B** | 일반화 주장 | OLMo 외 backbone(C1 또는 C2) **하나**에서 방향 재현 | "OLMo 한정" 명시 |
| **G-C** | 착수 전 | 총 run 수 × 1 run 시간이 GPU 예산 안에 들어감 (§5) | arm 또는 seed 축소를 **미리** 결정 |

G-N이 가장 중요하다. **이득이 negative control에서도 남으면 그것은 leakage다.**
이 게이트를 사후에 만들면 자기기만이 된다.

### G-A 상세 절차 (사전 등록)

annotation 차이를 "확인하고 제외"하는 게 아니라 **측정하고 통제한다.**

**1단계 — descriptor 측정** (지역마다, inventory 속성에서 직접)

| | 무엇 | 왜 |
|---|---|---|
| A1 | 폴리곤 개수 | 표본량 |
| A2 | 면적 분포 min/p1(MMU)/median/p99/max + log10 히스토그램 | 도화 상세도 |
| A3 | 저자 구성과 최다 저자 점유율 | **교락 여부** |
| A4 | event_type(유발요인) 구성 | 현상 차이 |
| A5 | 날짜 3종 존재율 + `event_conf` 분포 | cutoff replay 가능성 |
| A6 | type(현상) 구성 | debris flow vs ice avalanche |

**2단계 — 교락 판정 (임계값 사전 고정. M13에서 2회 수정했음)**

- 최다 저자 점유율 ≥ **0.90** → 그 지역은 `author-confounded`
- **클리핑 정규화 후** MMU(p1) 비 ≥ **10×** → `직접 비교 불가`
- **median 면적 비 ≥ 10×** → `분포 이질` (MMU만으로는 부족했음)

**클리핑 정규화가 왜 필요한가**: AI-Hub 한국 폴리곤은 1024×1024 타일 경계에서 잘려
100 m² 미만 사각형 조각이 150개(1.80%) 섞여 있었음. 그대로 재면 MMU가 0.046 m²로 나와
Italy와 1,367배 차이가 남. 400 m² 하한으로 2.74%만 버리면 MMU가 549.1 m²가 되고
Italy와 8.7배가 됨. **잘린 상태가 다른 데이터셋끼리 MMU를 비교하면 무효임.**

**3단계 — 통제 (제외가 아니라 통제가 기본)**

우선순위대로 시도하고, 통과한 첫 방법을 쓴다.

| 순위 | 방법 | 조건 |
|---|---|---|
| 1 | **저자 고정** — 단일 저자가 덮는 지역만으로 LOCO | 지역 ≥ 8, 지역당 폴리곤 ≥ 100 |
| 2 | **면적 하한 조화** — 공통 하한 적용 후 재평가 | 모든 지역 보존율 ≥ 10% |
| 3 | **면적 밴드 층화** — 모든 지역이 공유하는 크기 구간에서만 평가 | 밴드 내 지역당 ≥ 50 |
| 4 | 해당 지역을 headline에서 제외 | 위 셋 모두 실패 |

M12 기준 **1번이 통과했다** (11지역, MMU 비 20×).

**3.5단계 — annotation-shift를 제거만 하지 말고 **측정**함 (M13)**

한국을 잰 결과 Höhn 11지역 범위 안에 들어갔음 (MMU 549.1, median 5,731.8).
따라서 한국은 저자 고정 LOCO의 **12번째 지역**이 됨.

동시에 Italy는 한국과 median 14배 차이나는 **최대 이질 짝**임. 이걸 버리지 않고 2×2로 씀.

| | source → Korea | 도화 기준 |
|---|---|---|
| **matched** | Höhn 11지역 → Korea | 같음 (M13 확인) |
| **mismatched** | Italy → Korea | median 14×, Italy가 전체의 63% |

**두 값의 차이 = annotation shift 단독 효과**임. 지형 차이와 분리됨.
그리고 `Italy → Korea`는 "라벨이 많은 곳에서 배워 라벨 없는 곳에 쓴다"는 현실 배치
시나리오와 정확히 같음. 따라서 headline이 아니라 **배치 스트레스 테스트**로 씀.

주장 문구는 이렇게 제한함.

> matched-annotation 전이에서 얻은 이득이 mismatched-annotation 전이에서 얼마나 남는가.

`Italy 학습 모델이 한국에 잘 적용된다`고 단독으로 쓰지 않음. matched 짝과 함께 보고함.

**4단계 — 민감도와 반증**
- 면적 하한값을 ±50% 흔들어 결론이 뒤집히는지 본다. 뒤집히면 결론은 하한 선택의 산물이다
- **annotation-shift arm**: 같은/유사 지형에서 저자만 다른 쌍(예: Höhn Hiroshima 1,937 vs
  전체 Hiroshima 2,211)으로, 저자 차이 단독의 성능 영향을 따로 잰다
- 보고 시 항상 `raw` 결과와 `저자 고정` 결과를 **둘 다** 싣는다

## 5. 계산 예산 — 착수 전에 고정한다 (G-C)

```
P1 headline:  arm 5(A0..A4) × LOCO 11폴드 × seed 3          = 165 run
반증 baseline: arm 3(B1..B3) × 11폴드 × seed 1              =  33 run
backbone:      arm 2(C1,C2) × 11폴드 × seed 1               =  22 run
라벨 regime:   A4만 × 4regime(0/1/5/10%) × 11폴드 × seed 1  =  44 run
                                                       합계  = 264 run
```

H200 1장, frozen backbone + 가벼운 head 기준. **1 run이 10분을 넘으면 44시간을 넘는다.**
따라서 착수 전에 1 run 시간을 재고, 초과하면 **seed를 3→1로 줄이거나 LOCO를 11→8폴드로**
줄이는 것을 미리 정한다. 돌리다가 줄이면 선택 편향이 들어간다.

## 6. 무엇이 CVPR이고 무엇이 워크숍인가

| 결과 | 판정 |
|---|---|
| G-S·G-N·G-L·G-B 전부 통과 + 저라벨 transfer 곡선 | **CVPR method/transfer 후보** |
| G-S 통과, G-L 실패 | 지역 적응 논문. EarthVision 등 워크숍 급 |
| G-S 실패, G-N에서 이득 소멸 | **negative result.** "전지구 EO 임베딩에 지역 residual을 더해도 봉인 지역에서 이득이 없다"는 보고 가치가 있다 |
| G-P 실패 | GeoFM이 이 task에서 task-specific 모델에 못 미친다는 결과. 이것도 발표 가치가 있다 |

## 7. 이미 측정된 자산의 자리 — 고아로 만들지 않는다

우선순위가 MountainShift로 옮겨가면서 M1·M8·M9·M10이 어디에도 속하지 않을 위험이 있다.
명시적으로 배치한다.

| 자산 | 자리 |
|---|---|
| **M1·M8** (릴리스 간 cache 붕괴, mask 계약 silent failure) | `EarthEmbedContract`. FoldRefresh arm(P5)의 **전제**이자 별도 제출 자산 |
| **M9** (공식 split valid 110/110 누수) | **독립 발표 가능.** 어느 논문에도 종속되지 않는다. benchmark validity 결과이며 취업·사업 양쪽에 바로 쓰인다 |
| **M10** (군집 holdout 동결, 계약 4층 봉인) | P3 한국 arm의 평가 기반. 동시에 M9의 해결책으로 같이 보고 |

**M9는 지금 상태로도 닫을 수 있는 유일한 결과다.** MountainShift가 전부 실패해도 남는다.

## 8. 지금 열려 있는 미측정 항목 (정직하게)

- ~~P1 Sen12Landslides 접근~~ → **G-0 통과 (M11)**. CC BY 4.0, harmonized S2 39.42 GB /
  28 파트, 파트가 지역별로 묶여 있어 필요한 지역만 수신 가능. `data_harmonized`는 PB04
  +1000 DN offset을 이미 보정했다(`data_raw`와 섞지 말 것). S2는 B02–B12 10밴드로
  **B01·B09가 없다** — M8의 비대칭이 그대로 적용되므로 v1 + band-group missing mask 경로가 맞다
- ~~G-A annotation 감사~~ → **측정 완료 (M12)**. 저자 고정 11지역 LOCO로 확정
- **네팔은 이 inventory에 폴리곤이 8개뿐이다.** 네팔 arm은 Sen12Landslides로 성립하지 않고
  BIPAD/ICIMOD에서 따로 와야 하며 headline 지역이 아니다
- S12LS-LD / S12LS-AD split의 **파일 단위 정의**: 미확인 (README에 개수만 있음)
- 네팔 BIPAD/ICIMOD snapshot join: 0
- 스위스 Bern/SLF event join: 0
- frozen OLMo probe 성능: 0
- 1 run 소요시간: 미측정 → G-C를 아직 통과시킬 수 없다

현재 말할 수 있는 것은 **실행 가능한 설계**까지다. transfer 성능·live 개선·CVPR 적격성은
전부 0% 측정 상태다.
