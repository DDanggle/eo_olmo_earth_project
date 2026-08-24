# Mountain Evidence Transfer

최종 갱신: 2026-08-24  
상태: 아이디어 검토 완료, 데이터 다운로드·학습 미실행

## 약점부터

- 한국에는 현재 빙하가 없으므로 `히말라야 빙하호 → 한국`을 같은 task의 직접 전이라고 부를 수 없다.
- 기관별 inventory는 공간해상도·관측시점·발행시점이 다르다. polygon이 있다고 같은 정답은 아니다.
- 기존 OlmoEarth embedding이 산악 변화를 실제로 담고 있는지는 아직 probe하지 않았다.
- 따라서 현재 말할 수 있는 것은 **실행 가능한 연구 설계**이지, 전이 성능이나 개선 결과가 아니다.

## 한 문장 질문

> **알프스·히말라야·한국처럼 기후와 관측체계가 다른 산악지역에서, 글로벌 Earth embedding의
> 어떤 변화 신호가 전이되고 어디서 실패하며, 지역 공공근거를 얼마나 넣어야 그 실패를 안전하게
> 보정할 수 있는가?**

가칭은 `MountainShift`다. 핵심은 산악 데이터를 많이 모으는 것이 아니라
`TRANSFER / LOCAL-ADAPT / RE-EMBED / ABSTAIN`을 구분하는 것이다.

## 지역별로 실제 연결할 수 있는 공식 자산

| 지역 | 공식 자산 | 맡길 역할 |
|---|---|---|
| ETH권 스위스 알프스 | [GLAMOS](https://swiss-glaciers.glaciology.ethz.ch/en/downloads)의 glacier inventory·길이·질량·부피 변화, [swissALTI3D](https://www.swisstopo.admin.ch/en/height-model-swissalti3d) 0.5/2 m DEM, [ETH landslide monitoring](https://engineeringgeology.ethz.ch/research/landslide-monitoring.html) | 정밀 관측이 있는 source·검증 지역 |
| Ostana·Monviso/Piemonte | [ARPA SIFraP](https://www.arpa.piemonte.it/dato/sistema-informativo-frane-piemonte-sifrap) 산사태 inventory, [빙하 현장조사](https://www.arpa.piemonte.it/sites/default/files/media/2026-03/Relazione_glaciologica_2025__0.pdf), [눈사태 portal](https://webgis.arpa.piemonte.it/portale_valanghe/), Monviso 암벽붕괴 사진측량 | 소규모 빙하·암벽붕괴·눈사태의 local transfer |
| Hindu Kush Himalaya | [ICIMOD RDS](https://rds.icimod.org/)의 1990–2020 빙하 변화, 1533–2025 GLOF 766건, 2000–2022 연도별 토지피복 | 넓은 cryosphere source와 장기 변화 inventory |
| 한국 산악 | 산림청 [산사태 위험지도](https://www.data.go.kr/tcs/dss/selectFileDataDetailView.do?publicDataPk=15074817)·[발령 이력](https://www.data.go.kr/data/15074798/openapi.do), [산불 이력](https://www.data.go.kr/data/15121205/fileData.do), 기상·토지피복·항공사진·환경영향평가·필지/건축 자료 | 빙하가 아닌 산사태·산불·식생훼손·인간개입 target과 행정 근거 |

`open`은 포털 전체가 아니라 **개별 자료의 이용조건과 실제 다운로드 가능성**을 다시 확인해야 한다.
GLOF 사건 목록도 곧바로 pixel mask가 되는 것은 아니다.

## 2026-08-24 최신 공개자산 보정

기관별 원자료를 처음부터 꿰매기 전에 아래 공개 benchmark로 반증한다.

| 자산 | 확인된 내용 | 이 프로젝트에서의 정확한 역할 |
|---|---|---|
| **AvalCD (2026)** | Sentinel-1 pre/post, LIA·DEM·slope·aspect·mask·polygon, Livigno·Nuuk·Pish·Tromsø 4지역, 1.1 GB | 눈사태 cross-region bi-temporal Phase 0. annotation license 표기는 별도 확인 |
| **Sen12Landslides (2025)** | 15지역, refined 74,956 landslides, S2 13,628 patches와 S1 asc/desc·DEM·event date/confidence | 산사태 unseen-region·future-time Phase 0의 1순위 |
| **GlaViTU benchmark (Nature Communications 2024)** | 약 400 GB, 전지구 빙하의 9%, 11지역, optical/SAR/DEM, 독립 시공간 test | cryosphere global/local/region-encoding strong baseline |
| **HKH Glacier Mapping (LILA BC)** | Landsat-7+SRTM, 35 tiles, 14,190개 512×512 patches, clean/debris glacier mask | 작고 쉬운 HKH smoke. 2002–2008 inventory라 최신 변화 task로 쓰지 않음 |
| **ICIMOD GLOF** | 1533–2025 766 events, 2025-12 갱신, CC BY 4.0 | event/date/impact inventory. 좌표·날짜 정밀도와 pre/post 영상 가능성 감사 뒤에만 pixel task 승격 |

WSL/SLF의 공개 설명에서 스위스 2018+2019 눈사태 polygon은 **24,778개**다. Norway·Greenland를
합쳐 이 숫자를 부풀리지 않는다. 2023 정리 논문은 Norway 약 6,300, Greenland 약 800을 보고하지만
센서·지역·라벨 계약이 달라 단순 합산 표본수는 과학적 장점이 아니다.

한국 AI-Hub 국립공원 데이터는 50,000장(0.1 m 20k / 0.5 m 25k / 10 m 3k / 30 m 2k)이 맞다.
그러나 **네 해상도가 동일 장면·동일 polygon·동일 ontology라는 증거는 없다.** 실제 class는
sensor마다 다르고 Landsat에는 산사태/토석류 class가 없다. 따라서 `같은 라벨의 4단 해상도 ladder`는
다운로드 후 co-registration·scene ID·label mapping을 감사하기 전까지 금지한다. 원본·재가공 데이터는
재배포하지 않고, 공개 논문은 재현 가능한 public benchmark를 반드시 함께 둔다.

## 하나로 합치되, 같은 현상인 척하지 않는다

### Track A — Cryosphere

- HKH ↔ Swiss Alps ↔ Monviso만 사용한다.
- task: glacier/lake boundary, lake expansion, snow/ice confusion, debris-covered ice.
- 한국은 이 task의 target도 negative도 아니다.

### Track B — Mountain disturbance

- 모든 지역을 사용한다.
- 공통 시각 primitive: `water`, `snow/ice`, `bare rock/debris`, `vegetation loss`,
  `slope failure scar`.
- 원인은 별도 local evidence head가 맡는다. 같은 bare patch라도 산사태·공사·산불을 영상만으로
  같은 원인이라 단정하지 않는다.

### Track C — Evidence-aware decision

- 영상 변화와 공식 근거를 분리해 `visual change`, `evidence available`, `event supported`, `cause`를
  각각 저장한다.
- 한국 API, GLAMOS/ARPA/ICIMOD inventory는 teacher이자 감사 근거다. 근거가 없으면 보류한다.

## 기존 embedding을 보정하는 네 단계

1. **Probe** — frozen embedding에서 선형/작은 MLP로 water·snow·debris·vegetation loss를 읽을 수
   있는지 먼저 본다. 읽히면 정보는 이미 있고 head만 부족한 것이다.
2. **Residual adapter** — embedding에 DEM/slope/aspect와 시점이 맞는 공공자료를 작은 adapter로
   결합한다. `EO-only`와 반드시 비교한다.
3. **Privileged distillation** — 학습 때만 공공근거를 본 teacher에서 EO-only student로 증류한다.
   공공정보가 영상에서 회복 불가능하면 student 개선을 주장하지 않는다.
4. **Re-embed** — single pooled vector나 잘못된 시간창 때문에 정보가 없으면 S1/S2/DEM과 정렬된
   dense·multi-temporal 입력으로 다시 추출한다. adapter가 잃어버린 월·공간정보를 복원할 수는 없다.

## 지역성은 embedding에, 실시간성은 residual에 둔다

모든 정보를 한 벡터에 영구히 섞지 않는다. 산악 자연보존 시스템은 두 속도로 나눈다.

```text
z_global = Earth encoder(S1/S2, frozen input contract)
z_region = local adapter(z_global, DEM, slope, geology, climate normal)
r_t      = API encoder(weather/fire/landslide/snow/evidence, observed_at, freshness, missingness)
h_t      = gated fusion(z_global, z_region, r_t)
```

| 속도 | 데이터 예시 | 역할 |
|---|---|---|
| 느림 | DEM, slope/aspect, 지질, 장기 기후평년, 보호구역, baseline land cover | 지역 embedding/adapter |
| 중간 | 계절 식생, glacier/lake inventory, 연도별 토지피복 | timestamped snapshot |
| 빠름 | 강우·적설·구름, 산불·산사태 경보, 센서 상태 | 매 요청 갱신하는 `r_t` |
| 사후 | 확정 피해조사, 복구보고, 사후 인허가 | label/검증 근거; 과거 예측 input 금지 |

공공 API가 바뀌면 `r_t`만 갱신한다. 전체 EO gallery를 재임베딩하지 않는다. API가 없거나 오래됐거나
서로 충돌하면 missingness/freshness token이 residual을 0 또는 `ABSTAIN` 방향으로 gate한다.

## 단일지역 모델보다 다지역 모델이 좋은지 묻는 정확한 비교

`여러 데이터를 합친 모델`을 하나만 만들면 원인을 알 수 없다. 같은 label·compute에서 다섯 칸을 둔다.

1. **Local-only** — 지역마다 따로 학습. 지역 특화 ceiling.
2. **Naive pooled** — 모든 지역을 그대로 합친 단일 모델. negative-transfer 기준점.
3. **Shared backbone + local head** — 공통 시각 primitive만 공유.
4. **Shared backbone + local adapter** — 지형·기후 차이를 작은 지역 adapter로 보존.
5. **4 + timestamped API residual** — 실시간 관측을 gated fusion하고 근거 부족 시 보류.

평가는 `지역별 1/5/10/50/100% label`, `unseen-region`, `future-year`, `API missing/stale`로 나눈다.
다지역 모델은 특히 저라벨·새 지역에서 local-only보다 빨리 올라가야 의미가 있다. full-label local model을
모든 지역에서 이길 필요는 없지만, macro 평균만 오르고 worst-region이 내려가면 성공으로 세지 않는다.

### 사전 성공 기준

- local-only/naive pooled 대비 unseen-region 또는 저라벨 macro F1·AUPRC `+2%p`, spatial bootstrap CI>0.
- worst-region 저하 `≤1%p`, high-cloud/snow false positive 감소.
- API time-shift·region-shuffle에서 이득이 사라져야 실제 시공간 context 이득으로 인정.
- `region_id`/위경도만 넣은 baseline과 같으면 물리적 지역 embedding 주장을 중단.
- `E_repr`(EO-only student 개선)와 `E_fusion`(API를 볼 때만 개선)을 반드시 별도 표로 보고한다.

## 가장 작은 반증 가능한 실험

### Phase 0 — 공개 benchmark로 먼저 확인

- avalanche: [AvalCD](https://zenodo.org/records/15863589) 4지역 bi-temporal SAR.
- landslide: [Sen12Landslides](https://www.nature.com/articles/s41597-025-06167-2)
  15지역 S1/S2+DEM 시계열. Landslide4Sense는 보조 구형 baseline으로만 둔다.
- cryosphere: [GlaViTU benchmark](https://www.nature.com/articles/s41467-024-54956-x)를
  strong baseline으로, HKH LILA BC를 작은 smoke로 둔다. Glacial-Lake-Bench는 license·download를
  확인한 뒤 추가한다.
- 비교: task-specific U-Net/scratch, frozen OlmoEarth probe, Prithvi-EO-2.0, Olmo residual adapter.
- 먼저 `region holdout`에서 frozen Olmo가 scratch보다 나은지 확인한다. 평균만 보지 않는다.

**24시간 promotion gate**:

1. AvalCD annotation license와 Sen12Landslides split/license를 확인한다.
2. OlmoEarth 입력으로 변환 가능한 band/time/GSD mapping을 20 sample에서 검증한다.
3. frozen Olmo linear probe가 scratch baseline의 95%에도 못 미치면 MountainShift를 Paper 2에서
   응용 보고서로 내린다.
4. ICIMOD 766건 중 `좌표 + 월 이하 날짜정밀도 + pre/post usable observation`이 100건 미만이면
   GLOF pixel task를 열지 않는다.

### Phase 1 — 네 지역 evidence pilot

- 지역별 event 20건만 먼저: 변화 전·후 acquisition, 좌표, DEM, 공식 evidence snapshot,
  published/observed time을 한 record로 고정한다.
- 입력 비교:
  1. optical only
  2. optical + SAR/DEM
  3. EO embedding + local public evidence
  4. 3 + 선택적 보류
- 지표: event/region별 F1·mIoU, source→target transfer delta, cloud/snow false positive,
  worst-region, calibration, AURC(risk–coverage 면적).

### Phase 2 — 성공했을 때만 방법 논문

- 지역과 현상에 따라 `transfer/local-adapt/re-embed/abstain`을 고르는 mountain router를 학습한다.
- 새 지역을 봉인하고 label 1/5/10/50% 곡선을 보고한다.
- 여러 backbone에서 반복될 때만 foundation-model 일반화를 주장한다.

## 중단 기준

- public-evidence adapter가 EO-only 대비 독립 지역에서 +2%p 미만이고 worst-region 개선이 없으면 중단.
- 개선이 지역명·위경도만으로 재현되면 location shortcut으로 판정한다.
- frozen embedding probe가 실패하지만 raw S1/S2/DEM baseline은 성공하면 `embedding 보정`을 버리고
  재임베딩 문제로 전환한다.
- HKH→한국 개선이 빙하 class나 눈 유무 같은 비공통 현상에만 의존하면 cross-region transfer 주장을
  삭제한다.
- official evidence의 관측/발행시점을 확인할 수 없으면 prospective early-warning 주장을 하지 않는다.

## 논문이 되는 지점

빙하호 segmentation 자체와 산사태 segmentation 자체는 이미 경쟁 연구가 있다. 강한 질문은 다음이다.

> **산악 Earth representation은 어떤 물리적 변화 primitive를 대륙 간에 옮길 수 있으며,
> 지역 공공근거는 언제 표현을 강화하고 언제 보류 신호로만 사용되어야 하는가?**

이 질문을 여러 지역·backbone·label budget에서 답하면 CVPR형 representation/domain-shift 논문이 될
수 있다. 한 지역의 예쁜 변화지도만 만들면 좋은 응용 연구지만 main-track 방법 기여로는 부족하다.
