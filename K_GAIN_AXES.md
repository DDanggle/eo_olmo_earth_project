# 한국 공공데이터 × OlmoEarth — 무엇이 좋아지는가: 네 축의 기전·선행연구·빈칸

작성 2026-08-23  
역할: "한국 데이터를 결합하면 좋아진다"를 **네 개의 서로 다른 주장**으로 분해하고, 각 축마다
① 물리적/정보적 기전 ② 이미 점유된 선행연구 ③ 남은 빈칸 ④ 현재 자산으로의 실현가능성을
판정한다. K-ALIGN 계약(`KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md`)에 넣을 것과 뺄 것을 가른다.

근거 등급은 `PAPER_READING_LIST.md` 규약을 따른다. 2026-08-23 3차 감사에서 PANGAEA,
GeoLink, MMEarth, OmniSat, SatMIP, Galileo, WildSAT, Auxiliary Modality Learning,
BCT/FCT/LCE/AdvBCT/BT²와 E-07은 공식 proceedings·PMLR·OpenReview·arXiv 본문으로 다시
확인했다. 다만 `M`은 저자 주장을 확인했다는 뜻이지 우리 환경에서 재현했다는 뜻이 아니다.
나머지 최신 preprint와 검색 후보는 계속 `W`로 두며, 원문 확인 전 수치·우월 주장의 근거로 쓰지 않는다.

---

## 0. 결론부터 — 네 축의 한 줄 판정

| 축 | 주장 | 판정 | 이유 |
|---|---|---|---|
| **A. 정확도** | 한국 공공데이터로 EO task 정확도가 오른다 | **조건부 가능. 저라벨 구간에서만** | PANGAEA는 full-label에서 UNet이 GFM을 이기고 10% label에서만 GFM이 이긴다고 보고한다. 따라서 주장 형태는 "+2%p 정확도"가 아니라 **"라벨 N% 절감"**이어야 한다 |
| **B. 임베딩** | 한국 공공데이터가 표현 자체를 강화한다 | **가장 좁지만 가장 우리 것** | GeoLink·CLIP4Geo·WildSAT이 "비-EO 기록으로 EO 표현 강화"를 이미 점유했다. 남은 빈칸은 **시각(時刻)이 명시된 행정기록**뿐이다 |
| **C. 속도·비용** | 한국 데이터로 더 싸진다 | **직접 기전 없음. 재구성 필요** | 압축은 이미 binary 32×·PCA64+int8이 sweet spot이다. 한국 데이터는 비용을 줄이지 않는다. 줄이는 것은 **갱신 범위**다 |
| **D. 위성 유도** | 어디를 다시 볼지 결정한다 | **가장 큰 그림. 가장 준비 안 됨** | tip-and-cue·EO 스케줄링·온보드 배포가 각각 점유됐다. 우리 빈칸은 **행정근거 결손이 재관측 가치를 결정하는 경우** |

**가장 중요한 한 줄.** 네 축을 하나의 논문에 넣으면 전부 약해진다. 축 B가 K-ALIGN 본편이고,
축 A는 그 평가 방식이며, 축 C는 gate 수치의 단위이고, 축 D는 다음 프로그램(EarthRoute)이다.

---

## 축 A — 정확도

### A.1 기전 다섯 개

한국 공공데이터가 정확도를 올릴 수 있는 경로는 다섯 개이고, 서로 난이도가 다르다.

| # | 기전 | 한국 자산 | 난이도 |
|---|---|---|---|
| A-i | **집계 통계 제약** — 지역별 공식 집계에 지도를 맞춘다 | 제주시 산지이용 2008–2026 19행, 통계청 농업면적조사, 시군구 토지이용 통계 | 낮음 |
| A-ii | **약지도 라벨** — 공식 polygon을 잡음 있는 라벨로 쓴다 | FarmMap 289,379 polygon, 연도별 토지피복 | 낮음 |
| A-iii | **공간 사전분포** — 필지 경계를 segmentation prior로 쓴다 | VWorld 연속지적 PNU | 중간 |
| A-iv | **시간 사전분포** — 인허가 일자로 변화 시점을 좁힌다 | BuildingHUB 8,794 event행, EIA | 중간 |
| A-v | **품질 층화** — 독립 구름/관측품질로 표본을 나눈다 | GK2A 2 km·10분, SCL | 낮음 |

### A.2 선행연구가 이미 차지한 것

- **A-i은 Wang이 이미 했다.** `W-23` [Two shifts for crop mapping: Leveraging aggregate crop
  statistics to improve satellite-based maps in new regions](https://www.sciencedirect.com/science/article/abs/pii/S0034425721002066)
  (RSE 2021)는 집계 통계로 새 지역의 위성 지도를 보정한다. 우리 코퍼스 안에 있다.
  **따라서 "한국 집계 통계로 보정했다"는 방법 기여가 아니라 적용이다.**
- **A-ii/A-iii은 전지구 자산이 생겼다.** [The first global agricultural field boundary map at
  10m resolution](https://arxiv.org/pdf/2605.11055)이 PRUE 모델로 241개국 **31.7억 필지
  polygon**을 만들었다. [PRUE](https://arxiv.org/pdf/2603.27101),
  [CadastreVision](https://www.sciencedirect.com/science/article/pii/S0924271624003150),
  [APBD 리뷰](https://arxiv.org/pdf/2508.14558)도 있다. **FarmMap이 한국 필지 경계를 준다는
  사실 자체는 이제 기여가 아니다.**
- **국가 지도기관 다중모달 데이터셋은 프랑스가 했다.** [FLAIR-HUB](https://arxiv.org/abs/2506.07080)
  (IGN)는 항공·S1/S2·SPOT·지형·과거 항공 6모달이고 최고 성능은 거의 전 모달 사용 시
  OA 78.2 / mIoU 65.8이다.
  **2026-08-23 정정**: 초록을 확인하니 면적은 프랑스 전토가 아니라 **2,528 km²**이고,
  "630억 픽셀"은 그 면적을 20 cm로 나눈 dense raster 화소 수다(2,528 km² ÷ 0.04 m² ≈ 632억).
  **사람이 630억 번 판정한 것이 아니다.** 제주(약 1,850 km²)의 1.4배 면적이므로
  "주석 예산에서 이길 수 없다"는 앞선 서술은 틀렸다.
  하지 않을 진짜 이유는 셋이다: ① **장르를 FLAIR-HUB가 이미 정의했다**(국가기관 다중모달
  토지피복 데이터셋) ② 한국판의 dense 주석은 사실상 **환경부 토지피복지도를 재포장**하는 것이라
  독립 정답이 아니다 ③ 우리 차별점(비동기 provenance)과 아무 관계가 없다.
  단, **FLAIR-HUB는 우리가 만들 asset의 형식 템플릿으로는 유용하다** — 모달 정렬, 라이선스
  (CC BY-SA 4.0), 벤치마크 동봉 방식.
- **약지도 계열도 포화다.** [Global high categorical resolution land cover mapping via weak
  supervision](https://www.sciencedirect.com/science/article/pii/S0924271624004878),
  [저해상 라벨 정제](https://www.tandfonline.com/doi/full/10.1080/01431161.2024.2443612),
  `W-09`/`W-18`/`W-27`(Wang 계열 pseudo-label·weak supervision)이 있다.

### A.3 남은 빈칸과 주장 형태

- [PANGAEA](https://arxiv.org/abs/2412.04204)는 **full label에서 UNet 등 supervised baseline이
  대부분의 GFM을 이기고, 10% label에서만 일부 GFM이 이긴다**고 보고한다.
  [농업 GFM 벤치마크](https://arxiv.org/html/2606.29664v1)도 같은 방향이다.
- 따라서 축 A의 정직한 주장은 **"정확도가 올랐다"가 아니라 "같은 성능을 더 적은 라벨로 냈다"**다.
- 빈칸은 **A-iv(행정 시각)**다. 집계·경계·약지도는 전부 점유됐지만, *인허가 일자가 변화
  시점의 사전분포가 되는가*는 남아 있다. 다만 단일 derivability 점수로 source를 제외하지 않는다.
  EO 회복가능성 `R`, 독립 task의 추가가치 `V`, EO-only student 전이효과 `T`를 함께 측정한다.

> **계약 반영**: `E_repr`의 primary metric을 **labels-to-target(라벨 절감)**으로 승격하고
> 정확도 +2%p는 secondary로 내린다. FLAIR-HUB·전지구 필지지도·W-23을 baseline 문단에 명시한다.

---

## 축 B — 임베딩 (K-ALIGN 본편)

### B.1 기전

EO-only student가 학습 시에만 비-EO 기록을 보고, 추론 시에는 영상만으로 더 나은 표현을 갖는다
(privileged information / auxiliary-modality distillation).

### B.2 선행연구가 이미 차지한 것

- [GeoLink](https://arxiv.org/abs/2509.26016)는 OSM으로 RS foundation model의 자기지도
  사전학습을 다중입도 신호로 강화한다. **우리 계약의 baseline에 이미 있다. 유지.**
- [Towards A New Era of Geo-Foundation Models (CLIP4Geo)](https://dl.acm.org/doi/10.1145/3748636.3762756)
  (SIGSPATIAL 2025)는 위성영상 + LiDAR + POI + 텍스트를 통합한다.
- [WildSAT](https://arxiv.org/pdf/2412.14428)은 **야생동물 관찰 기록**으로 위성 표현을 학습한다.
  비-EO 기록이 EO 표현을 강화한다는 기전의 직접 선례다.
- [Spatial Representation Learning Beyond Pixels: Unifying Raster Data and Vector Semantics](https://arxiv.org/pdf/2606.02374)는
  래스터와 **벡터 시맨틱**의 통합을 다룬다. 한국 공공 벡터 데이터 아이디어와 가장 가깝다. **정독 필요.**
- [GeoMeld](https://arxiv.org/html/2604.10591), [Emerging Flexible Designs for Geospatial
  Multimodal FMs](https://arxiv.org/pdf/2606.12595), [DFR-Gemma](https://arxiv.org/pdf/2604.07490)도
  같은 방향의 최근 preprint다.
- 위치 인코더 계열은 우리 코퍼스에 있다: `M-02` Space2Vec, `M-03` Sphere2Vec, `M-04` CSP,
  `M-09` GAIR, `M-14` polygon representation learning, `M-11` KnowWhereGraph, `M-12` Earth Embeddings.
- privileged-modality distillation 자체도 점유됐다:
  [JDCNet](https://arxiv.org/pdf/2603.29167)(confidence-gated privileged-modality distillation),
  [InfraNet](https://arxiv.org/pdf/2607.03795)(학습 때만 RGB, 추론 때 제거),
  [Auxiliary Modality Learning with Generalized Curriculum Distillation](https://proceedings.mlr.press/v202/shen23f/shen23f.pdf).

### B.3 남은 빈칸

위 전부에 없는 것은 **시각(時刻)**이다.

- OSM·POI·야생동물 관찰은 "언제 공개됐는가"가 관리되지 않는다. cutoff 이전에 알 수 있었는지가
  정의되지 않으므로 prospective 주장을 할 수 없다.
- 한국 행정기록은 `event_time / observed_time / published_time / retrieved_time`을 분리할 수
  있다. 이것이 우리가 가진 유일한 구조적 차별점이다.
- 그리고 그 위에 **모델 릴리스가 바뀌어도 cache를 재사용하는 문제**가 얹힌다 — 이건 위 어느
  논문도 다루지 않는다.

> **계약 반영**: baseline 목록에 CLIP4Geo·WildSAT·Beyond-Pixels·JDCNet·InfraNet을 추가한다.
> "비-EO 기록으로 EO 표현을 강화한다"를 novelty로 쓰지 않고, **"공개 시각이 관리된 기록으로,
> 릴리스가 바뀌어도 재사용되는 좌표계에서"**로만 쓴다.

---

## 축 C — 속도·비용

### C.1 먼저 잘못된 기대를 지운다

**한국 공공데이터는 추론을 빠르게 하지 않는다.** 오히려 입력이 늘어 느려진다. 이 축에서
한국 데이터의 역할은 연산량이 아니라 **무엇을 다시 계산하지 않아도 되는지**를 결정하는 것이다.

### C.2 선행연구가 정한 현재 기준선 — 우리 gate를 재보정한다

- [Neural Embedding Compression for Efficient Multi-Task EO Modelling](https://arxiv.org/pdf/2403.17886)
- [Democratizing planetary-scale analysis: an ultra-lightweight Earth embedding database](https://essd.copernicus.org/articles/18/5375/2026/)
  (ESSD 2026) — 전지구 육지 1년치를 약 **2.4 TB**로.
- 실무 보고([TerraBit/DeltaBit](https://geospatialml.com/posts/compressing-earth-embeddings/))는
  binary quantization·PCA+int8의 경험적 결과를 제시한다. 이는 peer-reviewed 공통 기준이 아니므로
  수치를 논문 사실로 인용하지 않고, PCA64+int8·PQ/OPQ·binary를 우리 데이터에서 재실행할
  **engineering baseline 후보**로만 쓴다.
- [Inferring Height from Earth Embeddings (AlphaEarth)](https://arxiv.org/pdf/2602.17250),
  NeuCo-Bench(EO 손실 신경압축 벤치마크)도 같은 구간을 점유한다.

**이것이 바꾸는 것**: 현재 계약의 `E_refresh` gate에 있는 "embedding bytes 8× 절감"은
**float32 대비라면 이미 알려진 압축을 성과로 세는 것**이다. 기준선을 바꾼다.

> **계약 반영**: embedding bytes 비교를 float32 하나에 두지 않고 **PCA(64)+int8, PQ/OPQ,
> binary quantization**을 모두 포함한다. 이 강한 baseline의 utility–bytes–latency Pareto를
> 못 이기면 효율 주장을 하지 않는다. NeuCo-Bench protocol도 함께 명시한다.

### C.3 남은 빈칸

압축은 **한 시점의 표현을 작게 만든다**. 아무도 다루지 않는 것은 **모델이 바뀌었을 때 이미
압축해 둔 것을 어떻게 하는가**다. binary quantization된 gallery는 재계산 비용이 원본보다 크다
(원본 픽셀이 없으므로 전량 재수집·재추론). 여기가 K-ALIGN의 `E_refresh`가 서는 자리다.

---

## 축 D — 위성 유도 (관측 획득)

### D.1 기전

모델과 행정근거가 "말할 수 없다"고 판정한 지점이 곧 **다음에 관측할 가치가 있는 지점**이다.

### D.2 선행연구가 이미 차지한 것

- [An Automated Tip-and-Cue Framework for Optimized Satellite Tasking](https://arxiv.org/html/2512.09670) —
  tip 추출부터 연속시간 스케줄링까지 end-to-end.
- [Optimizing EO Satellite Schedules under Unknown Operational Constraints](https://arxiv.org/pdf/2604.13283) —
  능동 제약 획득.
- [Onboard Deployment of Remote Sensing Foundation Models: Architecture, Optimization, Hardware](https://www.mdpi.com/2072-4292/18/2/298)
  (Remote Sensing 2026) — 온보드 배포 종합 리뷰.
- 기존 EarthRoute 노트의 THOR·RingMoE·ZoomEarth·EO-Gym·OpenEarth-Agent도 여기 속한다.
- 우리 코퍼스: `O-02` 저비용 위성 온보드 홍수 매핑, `W-P03` 발전소 NO2 플룸의 탐지가능성,
  `W-07` 상업위성 아카이브 개방.

### D.3 운영 동기를 주는 보조 사례 — `E-07`

[Embedding-Only Uplink for Onboard Retrieval Under Shift](https://openreview.net/forum?id=IbzEpGdblY)
(ICLR 2026 공식 OpenReview 본문 확인): 지상국이 임베딩과 메타데이터를 보내 궤도상 검색을 하는
시뮬레이션을 다룬다. 본문의 약 598–690 B/query는 주로 내려오는 JSON telemetry 크기이며,
hint gallery 업링크는 `N_hints × D × bytes_per_value`로 별도 계산해야 한다.

**이것이 K-ALIGN에 주는 것**: 저대역폭 환경에서 표현·검색 효율이 중요하다는 **운영 동기**다.
이 논문이 전체 gallery backfill의 물리적 불가능이나 1 KB gallery 예산을 증명한 것은 아니다.

> **계약 반영**: `E_refresh`의 동기 문단에 온보드 링크 사례를 보조적으로 넣는다. 단, 우리는
> 위성을 운용하지 않으므로 대역폭 가정은 simulation으로만 다루고 온보드 실측·물리적 불가능을
> 주장하지 않는다.

### D.4 한국 고유 자산과 그 한계

- **GK2A**: 2018-12 발사, AMI, 2 km·고빈도. 한국이 가진 진짜 물리적 자산이다.
  그러나 [정지궤도 기반 NDVI gap filling](https://doi.org/10.3390/s26051731) 등 정지궤도×극궤도
  융합은 이미 연구되고 있고, 우리는 **과거 GK2A를 현재 endpoint로 소급 조회할 수 없다**
  (최근 2일 제한, 6 관측일 실패 확인). → **용도를 "구름 상태의 센서 독립 감사"로 한정한다.
  super-resolution·gap filling으로 확장하지 않는다.**
- **국내 위성 tasking**: KOMPSAT 계열 재관측 요청은 기관 협력 없이는 불가능하다.
  현실적 위치는 "우리가 위성을 유도한다"가 아니라 **"재관측이 필요한 지점의 근거를 만든다"**다.

---

## 교차 판정표 — 무엇을 K-ALIGN에 넣고 무엇을 뺄 것인가

| 기전 | novelty 여유 | 필요 자산 | 현재 준비도 | 결정 |
|---|---|---|---|---|
| A-iv 행정 시각을 시간 사전분포로 | **높음** | BuildingHUB·EIA + 독립 라벨 | 낮음(시간정렬 0/14) | **K-ALIGN `E_repr`의 핵심 재료. A3에서 먼저 측정** |
| B 시각 관리된 기록의 privileged distillation | **높음** | 위와 동일 | 낮음 | **K-ALIGN 본편** |
| C 릴리스 전환 시 압축 gallery의 갱신 | **높음** | 이미 있음(216×2) | 중간 | **`E_refresh`. 기준선을 PCA64+int8로 교체** |
| D 근거 결손 기반 재관측 우선순위 | 중간 | 없음 | 매우 낮음 | **EarthRoute로 이월** |
| A-i 집계 통계 보정 | 낮음(W-23) | 산지이용 19행 등 | 중간 | 보조 실험. 논문 주장 아님 |
| A-ii/A-iii 필지 경계·약지도 | **없음**(전지구 지도 존재) | FarmMap | 높음 | **기여로 세지 않는다.** anchor로만 사용 |
| A-v 독립 구름 감사 | 낮음 | GK2A | 중간 | 평가 stratum으로만 |
| 한국판 FLAIR-HUB | **없음** | 대규모 주석 예산 | 없음 | **하지 않는다** |
| GK2A×S2 융합/초해상 | 낮음 | 과거 GK2A(불가) | 없음 | **하지 않는다** |

---

## 이번 조사가 바꾸는 계약 수정 다섯 개

1. **`E_repr` primary를 라벨 절감으로.** PANGAEA가 full-label에서 supervised baseline 우세를
   보고하므로, 정확도 우위를 primary로 두면 강한 baseline에 진다.
2. **`E_refresh`의 bytes 기준선을 강한 압축군으로.** PCA64+int8·PQ/OPQ·binary 대비
   utility–bytes–latency Pareto를 보고, float32 대비 절감만으로는 성과를 세지 않는다.
3. **필지 경계를 기여에서 제외.** 31.7억 polygon 전지구 지도가 공개됐다. 한국 필지는 anchor다.
4. **baseline 목록 확장.** CLIP4Geo, WildSAT, Beyond-Pixels(raster+vector), JDCNet, InfraNet,
   FLAIR-HUB, 전지구 필지지도, NeuCo-Bench 계열.
5. **온보드 임베딩 링크를 `E_refresh`의 보조 동기로.** 약 598–690 B의 downlink JSON과
   별도 gallery/hint upload를 혼동하지 않고 simulation으로만 다룬다.

---

## 새로 확인한 문헌과 남은 watchlist

| 등급 | 제목 | 축 | 왜 중요한가 |
|---|---|---|---|
| **M** | [Embedding-Only Uplink for Onboard Retrieval Under Shift](https://openreview.net/forum?id=IbzEpGdblY) | C·D | ICLR 2026 공식 본문 확인. 저대역폭 운영 동기; gallery 비용 증거는 아님 |
| `M` | [PANGAEA benchmark](https://arxiv.org/abs/2412.04204) | A | 본문 확인. GFM이 supervised baseline을 일관되게 이기지 않음 |
| `M` | [FLAIR-HUB](https://arxiv.org/abs/2506.07080) | A | 초록 확인. 국가기관 다중모달 asset 형식의 비교 기준 |
| `W` | [First global 10m field boundary map](https://arxiv.org/pdf/2605.11055) | A | 241개국 31.7억 polygon. 필지 경계 기여 소멸 |
| `W` | [PRUE](https://arxiv.org/pdf/2603.27101) | A | 위 지도의 모델 |
| `W` | [CadastreVision](https://www.sciencedirect.com/science/article/pii/S0924271624003150) | A | 지적 경계 벤치마크 |
| `W` | [APBD 리뷰](https://arxiv.org/pdf/2508.14558) | A | 필지·경계 추출 종합 리뷰 |
| `W` | [Benchmarking GFMs for Agriculture](https://arxiv.org/html/2606.29664v1) | A | PANGAEA와 같은 방향의 재확인 |
| `W` | [CLIP4Geo / Expert-Guided Multimodal Alignment](https://dl.acm.org/doi/10.1145/3748636.3762756) | B | 위성+LiDAR+POI+텍스트 통합 GeoFM |
| `M` | [WildSAT](https://openaccess.thecvf.com/content/ICCV2025/html/Daroya_WildSAT_Learning_Satellite_Image_Representations_from_Wildlife_Observations_ICCV_2025_paper.html) | B | ICCV 2025 본문 확인. 비-EO 기록으로 위성 표현 학습 |
| `W` | [Spatial Representation Learning Beyond Pixels](https://arxiv.org/pdf/2606.02374) | B | raster×vector 통합. **우선 정독** |
| `W` | [GeoMeld](https://arxiv.org/html/2604.10591) | B | 의미 기반 RS foundation model |
| `W` | [Emerging Flexible Designs for Geospatial Multimodal FMs](https://arxiv.org/pdf/2606.12595) | B | 설계 공간 정리 |
| `W` | [DFR-Gemma](https://arxiv.org/pdf/2604.07490) | B | dense geospatial embedding 위 추론 |
| `W` | [JDCNet](https://arxiv.org/pdf/2603.29167) | B | confidence-gated privileged-modality distillation |
| `W` | [InfraNet](https://arxiv.org/pdf/2607.03795) | B | 학습 때만 보조 모달, 추론 때 제거 |
| `M` | [Auxiliary Modality Learning w/ Curriculum Distillation](https://proceedings.mlr.press/v202/shen23f.html) | B | ICML 2023 본문 확인. train-only 보조모달 baseline |
| `M` | [Neural Embedding Compression for EO](https://arxiv.org/abs/2403.17886) | C | 공식 원문 확인. 압축 baseline |
| `W` | [Ultra-lightweight Earth embedding database (ESSD)](https://essd.copernicus.org/articles/18/5375/2026/) | C | 전지구 1년 2.4 TB |
| `W` | [Inferring Height from Earth Embeddings](https://arxiv.org/pdf/2602.17250) | C | 임베딩 소비 태스크 |
| `W` | [Automated Tip-and-Cue Framework](https://arxiv.org/html/2512.09670) | D | 위성 tasking end-to-end |
| `W` | [EO Schedules under Unknown Constraints](https://arxiv.org/pdf/2604.13283) | D | 능동 제약 획득 |
| `W` | [Onboard Deployment of RS Foundation Models](https://www.mdpi.com/2072-4292/18/2/298) | D | 온보드 배포 리뷰 |
| `W` | [Geostationary NDVI gap filling](https://doi.org/10.3390/s26051731) | D | GK2A 융합이 이미 연구 중임 |
| `W` | [Scalable and Trustworthy EO Foundation Models](https://arxiv.org/pdf/2607.07758) | 전체 | 신뢰성 축 정리 |
| `W` | [PEFT of GFMs via Embedding Deflection](https://arxiv.org/pdf/2503.09493) | A·C | 적응 비용 baseline |

기존 코퍼스에서 승격할 것: `W-23`(집계 통계), `W-09`/`W-18`/`W-27`(약지도), `M-04`/`M-09`/`M-14`
(위치·폴리곤 인코더), `M-11`(KnowWhereGraph), `M-12`(Earth Embeddings), `E-03`(pooling),
`E-06`(AlphaEarth), `E-07`(업링크), `O-02`(온보드).

---

## 다음 독서 순서 (원문 정독)

1. compatibility 원문 묶음(BCT/FCT/LCE/AdvBCT/BT²/ICML 2025 hyperbolic BCRL) — novelty 경계 확정.
2. [Beyond Pixels: raster+vector](https://arxiv.org/pdf/2606.02374) — perspective인지 실행 가능한 baseline인지 구분.
3. 압축 원문(Neural Embedding Compression / NeuCo-Bench / PQ·OPQ) — `E_refresh` gate 수치 확정.
4. [JDCNet](https://arxiv.org/pdf/2603.29167)·InfraNet — 출판 상태와 privileged distillation 구현 확인.
5. BuildingHUB·EIA source schema — 논문보다 먼저 실제 `published_time` 존재 여부를 확인.
