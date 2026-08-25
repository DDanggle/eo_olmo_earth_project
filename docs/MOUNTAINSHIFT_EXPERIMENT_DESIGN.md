# MountainShift 실험 설계 (2026-08-25)

`MOUNTAIN_EVIDENCE_TRANSFER.md`의 연구 질문을 **실행 가능한 arm·지표·게이트**로 옮긴 문서다.
연구 질문과 자산 목록은 그쪽이 authoritative하고, 여기는 "무엇을 어떤 순서로 돌리고
무엇이 나오면 죽이는가"만 다룬다.

## 0. 먼저 — 설계에 아직 없던 위험 세 개

### R1. headline이 n=3이고, 그중 둘은 접근조차 미측정이다

3-way leave-one-country-out은 **독립 표본이 3개**다. 신뢰구간을 만들 수 없고,
한국을 뺀 두 나라(네팔·스위스)는 snapshot join을 **한 번도 하지 않았다.**
어느 한 나라가 막히면 headline이 2-way로 무너지는데, 그 경우의 대비가 문서에 없다.

**그런데 이미 기록된 자산에 답이 있다.** Sen12Landslides는 **15지역** S1/S2+DEM,
refined 74,956 landslides, event date/confidence 포함이다
(`MOUNTAIN_EVIDENCE_TRANSFER.md` 2026-08-24 보정표).

→ **headline spine을 공개 15지역 leave-one-region-out으로 옮긴다.**
3국은 headline이 아니라 **annotation-shift + live-residual 확장**으로 붙인다.

| | 15지역 LOCO (공개) | 3국 LOCO |
|---|---|---|
| 독립 표본 | **15** | 3 |
| 재배포 제약 | 없음 | AI-Hub 있음 |
| 접근 위험 | 이미 공개 | 네팔·스위스 미측정 |
| 신뢰구간 | 만들 수 있다 | 사실상 불가 |
| 죽을 확률 | 낮음 | 두 나라 중 하나만 막혀도 |

이렇게 하면 **네팔·스위스가 전부 막혀도 논문이 산다.** 3국은 기여를 더하는 층이지
논문의 생사가 걸린 층이 아니게 된다.

### R2. annotation 생성 과정이 나라마다 다르다 — domain shift와 구분되지 않는다

세 나라의 산사태 라벨은 **만들어진 방식이 다르다.**

| 국가 | 라벨 출처 | 날짜의 의미 |
|---|---|---|
| 한국 | AI-Hub, S2 10 m 위에서 사람이 그린 폴리곤 | 영상 촬영일 |
| 네팔 | Sen12Landslides refined inventory | event date + confidence |
| 스위스 | Bern natural-event cadastre | 행정 신고/기록일 (EO 파생 아닐 수 있음) |

**최소도화면적(min mapping unit)·날짜 의미·폴리곤 정밀도가 다르면, 국가 간 성능 차이는
"지역이 달라서"인지 "라벨을 다르게 그려서"인지 분리되지 않는다.**
이건 앞서 "4단 해상도 ladder"로 틀렸던 것과 **같은 종류의 오류**다.

→ 게이트 **G-A**를 headline 주장 앞에 둔다 (아래 §4).

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
| **B1** | region-ID one-hot | residual이 그냥 "지역 식별자"인지 반증 |
| **B2** | 위경도 (lat/lon) | 같음 |
| **B3** | raw-spectral retrieval | 임베딩이 필요하긴 한지 |
| **C1** | U-TAE 또는 3D-UNet (task-specific) | GeoFM이 이기긴 하는지 |
| **C2** | Prithvi-EO-2.0 **또는** TerraMind 하나 | 방향이 backbone 무관한지 |

`E_transfer` = A4의 LOCO 성능 − A0(hold-out 지역 자체 학습).
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
P1  Sen12Landslides 15지역 leave-one-region-out         ← headline spine (공개)
P2  G-A annotation-process 감사                          ← 3국 주장의 전제
P3  한국(AI-Hub, 봉인됨) 을 16번째 지역으로 추가          ← annotation shift 측정
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
| **G-0** | P1 착수 전 | Sen12Landslides 다운로드·라이선스·split 정의 확인, 지역 경계가 실제로 분리됨 | MountainShift 중단. 여기서 막히면 없다 |
| **G-P** | P1 probe | frozen OLMo probe가 scratch/U-TAE의 **95% 이상** 또는 raw-spectral retrieval보다 우수 | GeoFM을 backbone으로 쓰지 않는다. task model로 전환 |
| **G-A** | 3국 주장 전 | 국가별로 ① 라벨 출처 센서 ② min mapping unit ③ 날짜 의미(event/mapping/신고) ④ 폴리곤 면적 분포를 표로 확정. **min mapping unit이 2배 이상 차이나거나 날짜 의미가 다르면** headline에서 제외 | 3국 결과를 `annotation-confounded`로 명시하고 auxiliary로 강등 |
| **G-S** | E_static 주장 | region-macro F1 또는 Recall@20 **+2%p**, **worst-region 저하 ≤1%p**, 지역 bootstrap CI 하한 > 0, **B1·B2를 유의하게 상회** | E_static 주장 철회 |
| **G-N** | E_static·E_live 주장 전 | **negative control**: region-shuffle / time-shift 시 이득이 **소멸**해야 한다 | 이득이 남으면 그것은 지역 정보가 아니라 leakage다. 주장 전부 철회 |
| **G-L** | E_live 주장 | cutoff replay에서 observed/published/retrieved 시각 95% 이상 확보, **미래정보 0건**, AUPRC 또는 detection lead-time 개선 | E_live를 inference-fusion으로만 보고 |
| **G-B** | 일반화 주장 | OLMo 외 backbone(C1 또는 C2) **하나**에서 방향 재현 | "OLMo 한정" 명시 |
| **G-C** | 착수 전 | 총 run 수 × 1 run 시간이 GPU 예산 안에 들어감 (§5) | arm 또는 seed 축소를 **미리** 결정 |

G-N이 가장 중요하다. **이득이 negative control에서도 남으면 그것은 leakage다.**
이 게이트를 사후에 만들면 자기기만이 된다.

## 5. 계산 예산 — 착수 전에 고정한다 (G-C)

```
P1 headline:  arm 5(A0..A4) × LOCO 15폴드 × seed 3          = 225 run
반증 baseline: arm 3(B1..B3) × 15폴드 × seed 1              =  45 run
backbone:      arm 2(C1,C2) × 15폴드 × seed 1               =  30 run
라벨 regime:   A4만 × 4regime(0/1/5/10%) × 15폴드 × seed 1  =  60 run
                                                       합계  = 360 run
```

H200 1장, frozen backbone + 가벼운 head 기준. **1 run이 10분을 넘으면 60시간을 넘는다.**
따라서 착수 전에 1 run 시간을 재고, 초과하면 **seed를 3→1로 줄이거나 LOCO를 15→8폴드로**
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

- P1 Sen12Landslides: 다운로드 0, split 정의 확인 0
- G-A annotation 감사: 0
- 네팔 BIPAD/ICIMOD snapshot join: 0
- 스위스 Bern/SLF event join: 0
- frozen OLMo probe 성능: 0
- 1 run 소요시간: 미측정 → G-C를 아직 통과시킬 수 없다

현재 말할 수 있는 것은 **실행 가능한 설계**까지다. transfer 성능·live 개선·CVPR 적격성은
전부 0% 측정 상태다.
