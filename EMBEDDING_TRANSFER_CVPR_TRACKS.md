# Earth embedding transfer·robotics·simulation CVPR 트랙 감사

최종 갱신: 2026-08-23  
역할: OlmoEarth 임베딩을 다른 backbone·view·policy로 전이하는 연구를 **실행 가능한 논문 질문**으로
나누고, 현재 제주 자산으로 할 수 있는 것과 새 데이터가 필요한 것을 구분한다.

## 0. 먼저 약점과 금지 주장

- 현재 full audit은 제주 한 개 9×6 legacy grid, 216 site-years, task label 0개다. 두 OlmoEarth
  릴리스의 표현 이동은 검증했지만, 다른 모델로 전이했을 때 정확도·로봇 성능이 좋아진 증거는 없다.
- sealed 64 site-years 결과를 이미 봤다. 따라서 지금 고른 nonlinear bridge·distillation의 test로
  그 split을 다시 쓰지 않는다. 기존 결과는 **문제 동기**로만 쓰고 새 지역·새 acquisition의
  untouched test를 방법 선택 전에 hash-freeze한다.
- pooled CKA 0.9786은 같은 token 좌표나 cache identity를 뜻하지 않는다. 실제 raw cross-release
  R@1은 양방향 0이고, calibration-only affine ridge도 0.6973/0.6089로 0.95 gate를 실패했다.
- `Olmo 768d → 다른 모델 입력`을 연결했다는 사실만으로 transfer가 아니다. 새 모델의 독립 태스크,
  cross-model query/gallery, 비용·지연 중 최소 하나에서 강한 baseline을 이겨야 한다.
- 로봇 주장은 실제 action-conditioned trajectory와 성공률/SPL·충돌 평가가 있을 때만 쓴다.
  위성–드론/지상 paired image만 있으면 cross-view localization 연구이지 embodied navigation이 아니다.
- simulation의 핵심 평가는 FID나 보기 좋은 영상이 아니라 geometry/semantic consistency와 그
  simulator에서 학습한 policy의 real 또는 held-out-world 성능이다.

## 1. `earth_paper` 연구 지도 감사

사용자가 지목한 실제 파일은 `../earth_paper/dashboard.html`이다(`dashboard.hmtl`은 오타).
2026-08-10 생성본의 현 상태는 논문 185편, PDF 110편(59%), 완독 0, 읽는 중 1, 확장 아이디어
3개다. 인용 그래프는 유용한 inventory지만 아직 종합된 evidence map은 아니다.

### 현재 corpus에서 이어지는 씨앗

| ID | 현재 연결 | 새 트랙에서의 위치 | 한계 |
|---|---|---|---|
| `E-03` | Earth embedding pooling | token/pooling·압축 baseline | note가 stub이고 cross-model 호환성은 다루지 않음 |
| `E-06` | AlphaEarth 64d embedding field | compact embedding product baseline | 로봇·release migration 실험 아님 |
| `G-08` | Decision Transformer | trajectory를 sequence로 보는 고전 | EO map conditioning이나 perception transfer 아님 |
| `W-13` | street-view+satellite crop mapping | aerial–ground 관측 결합의 응용 계보 | localization·policy·모델 갱신과는 다른 target |
| `K-05` | dashboard-camera flood | ground sensor와 EO 결합 후보 | note/PDF가 비어 있고 paired cross-view protocol 미정 |
| `O-02` | onboard satellite flood ML | edge inference·지연/전력 제약 | embedding bus나 robot policy 연구 아님 |
| `O-09` | hybrid physics–ML climate simulation | simulator fidelity에 대한 먼 계보 | embodied world model과 직접 연결하면 안 됨 |
| `M-07` | Spatial-Agent | geospatial agent/tool reasoning | continuous control·cross-view perception과 다름 |

따라서 기존 corpus의 강점인 `EO 표현 → 통계적 추론 → 환경 의사결정`은 유지하되, robotics를
위해 별도 문헌축 `heterogeneous distillation → compatible embedding → cross-view localization →
latent world model → task-faithful simulation`을 추가해야 한다. 로봇을 기존 paper ID 몇 개로
정당화하는 것은 불충분하다.

### 2026-08-23 초점 수렴과 5차 보정

사용자 지시에 따라 넓은 후보 중 Track 1을 **K-ALIGN Bus**로 승격했다. stable EO cache는
Olmo/TerraMind 등 heterogeneous teacher에서 배우고, 한국 공공데이터는 cutoff-valid privileged
signal과 별도 dynamic residual로 사용한다. 두 표현을 분리해야 모델 릴리스와 public-record
publication을 서로 다른 비용으로 갱신할 수 있다. authoritative 계약은
`KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md`다.

5차 보정에서는 이를 더 좁혔다. current Track 1은 multi-teacher/context bundle이 아니라
**re-embedding necessity → alignment predictability → fixed-quantizer intervention**이다. 한국
context residual은 event-first 데이터 gate 뒤의 후속축이다. 최신 순위와 중단 기준은
`K_ALIGN_BIG_PICTURE.md`와 `K_ALIGN_CVPR_READINESS_AUDIT.md`가 대체한다.

## 2. “다른 모델로 전이”의 정확한 네 형태

| 형태 | 학습하는 것 | 가능한 경우 | 증명할 것 |
|---|---|---|---|
| 좌표 bridge | `g(z_old) ≈ z_new` projector | 동일 관측을 두 모델로 계산 가능 | cross query/gallery retrieval·task head 유지 |
| teacher→student distillation | 작은 student가 teacher token/관계/태스크를 모사 | teacher는 frozen black box여도 됨 | student task utility와 compute/storage 절감 |
| multi-teacher representation bus | student 하나가 Olmo·TerraMind·Galileo 등 여러 teacher와 호환 | 공통 canonical input/view 계약이 있음 | family/release 교차 호환성과 worst-task 성능 |
| embodied state transfer | EO/map embedding이 local video·action dynamics의 조건이 됨 | paired location/trajectory/action이 있음 | localization 또는 planning 성공, OOD·sim-to-real |

단순 feature MSE는 차원·stride·시간축이 다른 모델에서 가장 약한 baseline이다. 주 방법은
teacher별 projector와 함께 spatial/temporal **관계**를 보존하고, task head가 실제로 이용하는 정보와
cross-model retrieval을 함께 평가해야 한다.

## 3. 선행연구가 이미 차지한 경계

- [AM-RADIO, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ranzinger_AM-RADIO_Agglomerative_Vision_Foundation_Model_Reduce_All_Domains_Into_One_CVPR_2024_paper.html)는
  서로 다른 vision foundation model의 지식을 한 학생 인코더에 증류했다. 따라서 “여러 teacher를
  한 모델로 합친다” 자체는 새롭지 않다.
- [Theia, CoRL 2024](https://proceedings.mlr.press/v270/shang25a.html)는 여러 vision foundation
  model을 로봇 시각 인코더로 증류해 teacher와 기존 robot representation을 비교했다. EO teacher를
  robot encoder로 증류한다는 한 문장만으로도 novelty가 아니다.
- [BCT, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html),
  [FCT, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ramanujan_Forward_Compatible_Training_for_Large-Scale_Embedding_Retrieval_Systems_CVPR_2022_paper.html),
  [LCE, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Meng_Learning_Compatible_Embeddings_ICCV_2021_paper.html)는
  모델·데이터·차원 변화 아래 embedding compatibility를 이미 연구했다. 우리는 old Olmo를 재학습할
  수 없는 공개 black-box release와 multi-temporal EO task를 빈칸으로 삼아야 한다.
- [Matryoshka Representation Learning, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html)은
  한 embedding의 prefix를 여러 dimension/compute budget에 쓰는 방법을 점유한다. dimension 축소만
  새 기여로 내세우지 않고 cross-family/release compatibility와 결합해야 한다.
- [GeoBridge, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Song_GeoBridge_A_Semantic-Anchored_Multi-View_Foundation_Model_Bridging_Images_and_Text_CVPR_2026_paper.html),
  [UniGeoRS, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_UniGeoRS_A_Unified_Benchmark_for_Tri-view_Geo-Localization_CVPR_2026_paper.html),
  [PAUL, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_PAUL_Uncertainty-Guided_Partition_and_Augmentation_for_Robust_Cross-View_Geo-Localization_under_CVPR_2026_paper.html)는
  satellite–drone–ground alignment, unified benchmark, GPS-noisy partial correspondence를 각각
  점유한다. 일반 tri-view alignment는 이미 혼잡하다.
- [DINO-WM, ICML 2025](https://proceedings.mlr.press/v267/zhou25t.html)는 pixel 대신 pretrained
  patch latent의 action-conditioned 미래를 예측해 계획한다. [Navigation World Models, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Bar_Navigation_World_Models_CVPR_2025_paper.html)와
  [Vid2Sim, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_Vid2Sim_Realistic_and_Interactive_Simulation_from_Video_for_Urban_Navigation_CVPR_2025_paper.html)은
  navigation video generation과 real-video-derived interactive simulation을 다룬다. latent world model
  또는 simulator를 만들었다는 사실보다 **EO 조건이 어떤 OOD planning 실패를 줄이는지**가 빈칸이다.

## 4. CVPR main 후보 순위

| 순위 | 트랙 | CVPR 잠재력 | 현재 준비도 | 추천 |
|---:|---|---|---|---|
| 1 | EO Upgrade Characterization + Fixed-Quantizer Compatibility | 높음 | 중간 | **cost-first P0** |
| 2 | Release-Stable Earth-to-Embodied Cross-View Transfer | 높음 | 낮음 | paired drone/ground 확보 후 |
| 3 | Budgeted Matryoshka Edge–Cloud Earth Embedding | 중–높음 | 중간 | 1번의 efficiency 축으로 결합 |
| 4 | EO-Conditioned Latent World Model for Active Sensing | 중–높음/고위험 | 낮음 | trajectory partner 또는 표준 sim 먼저 |
| 5 | Task-Faithful Satellite-to-Ground Simulation | 중간/고위험 | 매우 낮음 | 별도 장기 트랙 |

### Track 1 — EO Upgrade Characterization + Fixed-Quantizer Compatibility

**질문**  
frozen EO release가 바뀔 때 compact re-embedding보다 compatibility가 필요한 운영 구간이 실제로
존재하는가? release/input 변화의 어떤 지표가 사후 정렬 가능성을 예측하며, old PQ/int8 codebook을
고정한 상태에서 새 모델 task utility와 기존 gallery 검색을 함께 보존할 수 있는가?

**방법 후보**

아래 multi-teacher/nested-bus 항목은 cost·predictability·fixed-quantizer P0를 통과한 뒤의 확장
백로그다. 현재 P0에서는 구현하지 않는다.

1. canonical S2 input과 model-native input을 분리해 동일 acquisition ID를 고정한다.
2. 학생 `s(x)`에서 teacher별 projector `g_m(s)`가 각 teacher의 pooled/spatial token을 예측한다.
3. token MSE 하나가 아니라 spatial affinity, temporal difference, cross-site neighbor, task logit을
   보존하는 relational distillation을 함께 쓴다.
4. 안정 bus head에는 old/new/family 교차 query–gallery loss를 적용한다.
5. 64/128/256/768d nested prefix를 학습해 edge query와 cloud gallery가 같은 bus를 쓴다.
6. missing modality·teacher dropout으로 특정 teacher shortcut을 진단한다.
7. EO-only `z_stable`과 cutoff-valid `r_context`를 분리하고 public record 갱신에는 residual만
   refresh한다. `E_repr`, `E_compat`, `E_fusion`, `E_refresh`를 별도 표로 보고한다.

**필수 비교군**

- identity, train-only mean shift, Procrustes, affine ridge
- PCA/PQ와 독립 저차원 모델, Matryoshka Representation Learning
- single-teacher feature/logit distillation
- BCT, FCT, LCE, AdvBCT
- AM-RADIO식 multi-teacher distillation
- 각 teacher frozen probe, compact student scratch, task-specific U-Net/ViT

**데이터·split**

- 최소 3지역×2연도, 공통 Sentinel-2 canonical view, 독립 task label 2종 이상
- calibration/train 지역, validation 지역, **새 untouched geographic-future test**
- 현재 full-216 sealed 64는 동기 Figure 1의 failure evidence만 허용하고 방법 test에서 제외
- model-family native ceiling은 별도 표; input advantage를 pretraining advantage로 해석 금지

**지표와 사전 gate**

- task: best matched-input teacher 대비 평균 −1%p 이내, 어느 task/region도 −2%p 초과 저하 금지
- compatibility: old/new, new/old, family/family R@1·mAP가 bus/bus native의 95% 이상
- 효율: query latency/FLOPs 5× 또는 embedding bytes 8× 절감, gallery backfill bytes 10× 절감
- public alignment: no-context student 대비 EO-only probe +2%p 또는 label 20% 절감, shuffle/time-shift에서
  관측 이득 80% 이상 소멸, future/duplicate-role sentinel 100% 차단
- 통계: site/location cluster bootstrap, worst-region/year, seed 5개
- **kill**: per-teacher affine projector가 같은 task·compatibility를 내거나, 절감이 없거나, 한 teacher
  성능만 좋아지고 worst-task가 무너지면 CVPR 방법 기여를 중단한다.

이 트랙은 현재 Olmo v1/v1.2의 강한 failure motivation, exact-input harness, H200 실행 체인을 직접
재사용할 수 있어 가장 현실적이다. 다만 다른 family checkpoint adapter와 task label이 새 병목이다.

### Track 2 — Release-Stable Earth-to-Embodied Cross-View Transfer

**질문**  
satellite EO teacher와 timestamped public context를 drone/ground student에 증류해 GPS 불안정·구름·
지도 노후화에서 localization/navigation을 높이면서, satellite model release 교체가 field robot의
gallery/policy를 깨뜨리지 않게 할 수 있는가?

**방법 후보**

- satellite map teacher + drone/ground query encoder + stable bus
- semantic/geometry anchor와 partial-correspondence uncertainty
- public context는 timestamp-valid map token으로만 사용하고 좌표/text shortcut shuffle을 둔다.
- release swap 때 robot student를 재학습하지 않는 frozen-policy/frozen-gallery cell을 핵심으로 둔다.

**비교군·데이터**

- GeoBridge, UniGeoRS/CAME, PAUL, MMGeo, FG², DINOv2/CLIP/Olmo retrieval
- VIGOR·GTA-UAV·UniGeoRS·GeoLoc 등 표준 data + 한국 실제 satellite–drone/ground pair
- navigation을 주장하려면 image pair와 별도로 실제 trajectory/action을 확보한다.

**지표/gate**

- localization Recall@K, median meter error, 3DoF pose; GPS-noise/weather/region OOD
- navigation은 Success, SPL, collision, edge latency
- 두 표준 benchmark에서 Recall@1 +2%p 또는 median error 20% 감소, 한국 real set도 같은 방향
- release swap 저하 ≤1%p, edge budget 충족
- **kill**: synthetic-only, 표준 benchmark 부재, 좌표 metadata shuffle에서 성능 유지 실패, 또는 일반
  cross-view baseline과 같은 결과면 main-track 주장을 중단한다.

시각적 매력과 CVPR fit은 높지만 generic tri-view 공간은 이미 혼잡하다. 이 프로젝트의 고유성은
`release-stable satellite teacher + timestamped public context + field-side no-retrain` 조합에 있다.

### Track 3 — Budgeted Matryoshka Edge–Cloud Earth Embedding

**질문**  
하나의 student가 64/128/256/768차원 prefix를 내고, 저전력 위성·드론이 작은 query를 보내도
cloud의 큰 gallery 및 여러 EO release와 호환되게 할 수 있는가?

- baseline: PCA, product quantization, 독립 dimension model, MRL, switchable/self-compatible model
- 지표: task/retrieval 대 bytes·FLOPs·latency·energy, device별 calibration, cross-dimension/family R@K
- gate: 8× bytes 절감에서 task −1%p 이내 또는 retrieval relative −5% 이내, 5× query speed
- kill: compression-only가 동일하거나 compatibility를 빼면 novelty가 사라지는 경우

단독 논문보다 Track 1의 deployment/efficiency 축으로 넣는 것이 강하다. 모든 dimension과 teacher를
한 번에 최적화해 표가 산만해지면 64/256/768 세 점만 사전 고정한다.

### Track 4 — EO-Conditioned Latent World Model for Active Sensing

**질문**  
static EO/public-map latent와 local camera latent를 분리한 action-conditioned world model이, 지도
없는 DINO-WM/Navigation-WM보다 unseen terrain·weather에서 다음 관측 위치와 경로를 더 잘 고르는가?

```text
z_earth(location, cutoff) + z_local(video_t) + action_t
                      → predicted z_local(t+1:t+h), uncertainty
```

- baseline: DINO-WM, Navigation World Models, map-free policy, raster map, DINOv2-only/Olmo-only,
  oracle map
- 데이터: action·pose가 있는 AirSim/Isaac/Urban simulation + 최소 하나의 real trajectory set
- 지표: future-latent error는 보조; Success/SPL, collision, planning horizon, unseen-world/weather가 주지표
- gate: 최소 3 unseen environments에서 relative Success/SPL +10%, in-domain −2%p 이내, 작은 real
  deployment에서도 같은 방향
- kill: action-conditioned trajectory가 없거나 synthetic image quality만 보고하면 중단

이것은 제주 EO pipeline의 자연스러운 다음 논문이 아니다. trajectory partner나 표준 simulator를
먼저 얻고, Track 1에서 compact stable map state가 검증됐을 때 여는 고위험 트랙이다.

### Track 5 — Task-Faithful Satellite-to-Ground Simulation

**질문**  
EO embedding과 public context로 만든 ground/aerial simulation이 단지 그럴듯한 영상이 아니라
terrain·road·flood·traversability를 보존해 real navigation/inspection policy를 실제로 개선하는가?

- baseline: Sat2GroundScape, Sky2Ground, Vid2Sim, URBAN-SIM, geometry-only simulator
- 필요 데이터: posed ground/drone video, depth/geometry 또는 strong multi-view correspondence,
  실제 policy evaluation
- 지표: FID/LPIPS는 보조; geometry/semantic consistency와 real zero-shot Success/SPL이 primary
- gate: generated simulation으로 학습한 policy가 real success를 relative 10% 이상 개선
- kill: 예쁜 제주 fly-through만 만들거나, 실측 policy/geometry가 없으면 논문 트랙이 아니다.

## 5. 추천 포트폴리오: 한 논문에 다 넣지 않는다

| 시간 | 논문 | 중심 기여 | 로봇/시뮬레이션 관계 |
|---|---|---|---|
| 현재 | Paper A/C 통합후보 `K-ALIGN` | public-context EO-only distillation + multi-teacher compatible bus | 로봇 없음 |
| gate 한 축 실패 | `Context Under Coverage` 또는 `Compatible Earth Bus` | 통과한 한 질문만 보존 | 로봇 없음 |
| 자산 확보 후 | Paper D `Earth-to-Embodied` | satellite→drone/ground stable cross-view transfer | localization, trajectory 있으면 navigation |
| 장기 | World-model/simulation | EO-conditioned action dynamics 또는 task-faithful sim | 별도 CVPR/CoRL/ICRA 트랙 |

`K-Context + representation bus + robot + simulator + federated learning`을 한 제출에 넣지 않는다.
주 논문 하나는 질문 하나, 새 방법 하나, 독립 test 하나로 닫는다.

## 6. 지금 가능한 4주 P0 — GPU를 쓰기 전의 검증 순서

### Week 0: 데이터·평가 계약

1. Olmo v1/v1.2 외 첫 teacher는 TerraMind Base 하나로 제한한다.
2. 공통 S2 acquisition, band order/scaling, AOI support, timestamp를 manifest에 고정한다.
3. 기존 sealed 위치와 겹치지 않는 calibration/validation/untouched test 지역을 먼저 freeze한다.
4. label-free compatibility proxy와 supervised task utility 표를 분리한다.

### Week 1: 표현 adapter smoke

- 32 canonical windows에서 Olmo/TerraMind spatial token shape·stride·mask·runtime 계약을 검사한다.
- linear projector, two-layer MLP, relational projector 세 개만 train split에서 비교한다.
- output hash, train seed, teacher checkpoint/revision, trainable parameter와 GPU-hour를 기록한다.
- validation에서 affine가 MLP와 같으면 nonlinear bridge 확장을 멈춘다.

### Week 2: compact student feasibility

- student 1개, single-teacher vs two-teacher distillation, 256/768d 두 dimension만 사용한다.
- frozen linear probe task 1개와 cross-model retrieval을 동시에 본다.
- label 300 gate가 없으면 task claim 없이 representation engineering feasibility로만 남긴다.

### Week 3–4: 승격 결정

- untouched test는 방법·hyperparameter 동결 뒤 한 번만 연다.
- `public-context E_repr + compatibility ≥95% + efficiency ≥5×/8×` 중 둘만 통과해도 부족하다.
- 세 gate를 모두 통과하면 Track 1 full matrix로, 실패하면 current release audit의 negative result와
  K-Context Paper A에 집중한다.

## 7. 결과 해석 계약

허용되는 문장:

- “공통 관측에서 여러 frozen EO teacher의 지식을 compact student로 증류할 수 있는지 평가했다.”
- “모델 family/release 교차 retrieval과 독립 EO task를 함께 측정했다.”
- “edge/cloud dimension–비용 곡선에서 사전 기준을 통과/실패했다.”

추가 증거 없이 금지되는 문장:

- “Olmo embedding을 TerraMind로 옮겨 정확도를 높였다.”
- “한국 공공데이터가 로봇의 세계 이해를 향상시켰다.”
- “시뮬레이션으로 real-world navigation을 해결했다.”
- “제주에서 검증된 범용 Earth robot foundation model이다.”
