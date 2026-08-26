# OlmoEarth × K-Earth 논문 검색·독서 장부

최종 갱신: 2026-08-24  
역할: 논문 제목을 모으는 목록이 아니라 **어떤 선행연구가 어떤 주장을 이미 차지했고,
그 결과 우리 실험을 어떻게 바꿔야 하는지** 기록하는 살아 있는 장부다.

OlmoEarth v1 한 편의 상세 정독 내용은 `PAPER_NOTES_v1.md`에 둔다. 이 파일은 검색 범위,
비교군, 반증 문헌과 다음 독서 순서를 관리한다.

## 상태와 근거 규칙

| 상태 | 뜻 |
|---|---|
| `M` | 논문명·저자·연도·초록을 arXiv, 학회 proceedings, 저자 공식 저장소 중 하나에서 확인 |
| `R` | 본문과 부록까지 정독하고 별도 노트 또는 재현 메모를 남김 |
| `X` | 이 저장소의 실험·중단 기준·표 한 칸에 직접 연결 |
| `W` | watchlist. 매우 최근 preprint 또는 아직 독립 재현이 없어 결론으로 인용 금지 |

- 마지막 메타데이터 확인일은 별도 표시가 없으면 **2026-08-23**다.
- `M`은 결과를 재현했다는 뜻이 아니다. 초록의 저자 주장은 우리 환경에서 재현될 때까지 가설이다.
- `최초`, `아무도`, `최고` 같은 부재·우월 주장은 체계적 검색 로그 없이 쓰지 않는다.
- AAAI/NeurIPS 수상 여부는 방법의 타당성을 대신하지 않는다. 이 장부는 수상 경력보다 연구 설계에
  미치는 결정을 우선한다.

## 2026-08-23 (3차) 조사 — 첨부 26편 재감사와 CVPR 경계

2차 조사 당시 대부분을 검색 snippet 수준으로 둔 뒤, 이번에는 논문 설계를 실제로 바꾸는 핵심
문헌을 공식 proceedings·PMLR·OpenReview·arXiv 본문으로 다시 확인했다. 상세 판정과 실험표는
`K_ALIGN_CVPR_READINESS_AUDIT.md`에 있다. `M`은 저자 주장 확인이지 재현 완료가 아니다.

| 결정 | 확인한 1차 문헌 | 실험에 미치는 영향 |
|---|---|---|
| 단순 `EO+지도`는 novelty가 아님 | GeoLink, MMEarth, OmniSat, SatMIP, Galileo, WildSAT | 동적 provenance·자연 누락·EO-only transfer를 함께 검증 |
| 단일 derivability 제외 규칙 폐기 | Auxiliary Modality Learning | `R_source`(EO 회복성), `V_source`(독립 task 가치), `T_source`(student 전이)를 분리 |
| 호환성 자체도 혼잡함 | BCT, FCT, LCE, AdvBCT, BT², hyperbolic BCRL, XBT | frozen third-party EO release·band/GSD/time-grid·compressed gallery의 결합으로 경계를 좁힘 |
| raw image 부재·gallery evolution도 선점됨 | UniBCT(IJCAI 2022), BiCT, Darwinian Model Upgrades(AAAI 2023), online backfill | raw raster 부재만 novelty로 쓰지 않고 fixed old PQ code 직접검색과 비용 경계를 비교 |
| GFM 우월성을 가정하지 않음 | PANGAEA | supervised U-Net/ViT, frozen probe, PEFT, full FT를 같은 split·budget으로 비교 |
| 압축은 보조축 | Neural Embedding Compression, NeuCo-Bench | FP32 절감이 아니라 PCA64+int8·PQ 대비 utility–bytes–latency Pareto를 요구 |
| E-07은 운영 동기일 뿐 | ICLR 2026 OpenReview | 약 598–690 B downlink JSON과 hint/gallery upload를 분리; 1 KB backfill 증거 주장 금지 |

현재 가장 방어력 있는 질문은 다음이다.

> 원 학습 파이프라인을 통제할 수 없는 frozen EO model release가 바뀔 때, 공개 anchor로 학습한
> 작은 adapter/student가 새 모델의 downstream utility와 기존 compressed gallery 검색을 함께
> 보존할 수 있는가?

한국 공공데이터는 이 논문의 유일한 정답셋이 아니라 release drift의 실제 motivation,
시간·geometry·coverage가 있는 stress test로 둔다. 공공데이터 자체의 표현 이득은 event-first 표본,
독립 label, `R/V/T` gate가 생긴 뒤 별도 track으로 승격한다.

## 2026-08-23 (2차) 조사 — 네 축(정확도·임베딩·속도·위성유도) 감사

전체 표와 링크는 `K_GAIN_AXES.md`에 있다. 이 절은 **이 장부의 결정만** 기록한다.
아래 표는 당시의 2차 조사 기록이다. 근거 등급과 E-07 해석은 위 3차 감사가 대체한다.

| # | 결정 | 근거 문헌 |
|---|---|---|
| 1 | `E_repr`의 primary를 **라벨 절감**으로. 정확도 우위는 secondary | PANGAEA(full-label에서 supervised 우세, 10%에서 GFM 우세), 농업 GFM 벤치마크 |
| 2 | `E_refresh`는 **PCA64+int8·PQ/OPQ·binary** 압축군과 utility–bytes–latency Pareto로 비교. 실무 블로그 수치는 재실행 전 사실로 인용하지 않음 | Neural Embedding Compression, NeuCo-Bench, ESSD DB, 실무 보고는 baseline 후보만 |
| 3 | **필지 경계를 기여에서 제외.** anchor로만 사용 | 전지구 10 m 필지지도 241개국 31.7억 polygon, PRUE, CadastreVision, APBD 리뷰 |
| 4 | **한국판 FLAIR-HUB를 만들지 않는다. 이유는 예산이 아니라 장르 선점** — 면적은 2,528 km²(제주의 약 1.4배)이고 630억은 20 cm 화소 수이지 사람 판정 횟수가 아니다. 한국판 dense 주석은 환경부 토지피복지도 재포장이 된다. 단 **asset 형식은 템플릿으로 차용** | FLAIR-HUB(IGN, 6모달, 2,528 km², OA 78.2/mIoU 65.8, CC BY-SA 4.0) — 초록 확인 |
| 5 | "비-EO 기록으로 EO 표현 강화"를 novelty로 쓰지 않는다. **공개 시각(published_time)이 관리된 기록**만 우리 빈칸 | GeoLink, CLIP4Geo, WildSAT, Beyond-Pixels(raster+vector) |
| 6 | privileged distillation baseline 확장 | JDCNet(confidence-gated), InfraNet(학습전용 보조모달), Auxiliary Modality Learning |
| 7 | `E_refresh`에 온보드 embedding link를 **보조 동기**로 추가. downlink JSON과 gallery upload를 분리 | `E-07` Embedding-Only Uplink (ICLR 2026 공식 본문, `M`) |
| 8 | GK2A 용도를 **구름 상태의 센서 독립 감사**로 한정. 융합·초해상으로 확장 금지 | 정지궤도 NDVI gap filling 선행연구 + 과거 GK2A 소급 조회 불가(실측) |
| 9 | 위성 tasking은 EarthRoute로 이월 | Tip-and-Cue 프레임워크, EO 스케줄링, 온보드 RSFM 배포 리뷰 |

갱신된 집중 독서 순서: BCT/FCT/LCE/AdvBCT/BT² → PANGAEA → GeoLink/WildSAT →
Auxiliary Modality Learning → 압축·NeuCo-Bench → 최신 privileged-distillation watchlist.

## 이번 검색이 바꾼 결론

1. **전이학습은 제품이 아니라 비용 절감 가설이다.** 현지 라벨이 매우 적을 때 이득이 보고되지만,
   PANGAEA와 EarthShift는 GeoFM이 강한 supervised baseline을 항상 이기지 않고 실제 분포 이동에서
   평균 성능이 크게 하락할 수 있음을 보인다.
2. **confidence가 높다는 이유만으로 말하게 해서는 안 된다.** 선택적 분류·conformal risk control과
   행정자료 coverage/missingness는 다른 문제다. `모델 risk`, `모집단 추론`, `근거 누락`을 각각
   측정해야 한다.
3. **범용 Earth routing은 이미 혼잡하다.** THOR, AnySat, RingMoE, ZoomEarth, EO-Gym,
   OpenEarth-Agent가 patch·sensor/modality·crop·tool routing의 여러 부분을 차지한다.
4. 우리 빈칸은 모델 하나가 아니라 **한국 공식 근거의 누락을 포함한 결정 연속성**이다. 즉 새
   관측·모델·행정근거·사람검증 중 무엇을 추가해야 현재 결론을 안전하게 갱신할 수 있는지 묻는다.
5. **전이 효과를 먼저 측정한 뒤 라벨 획득 방법을 만든다.** scratch·일반 vision/EO·GeoFM의
   지역/연도/센서/구름별 paired 차이를 모르면 active learning의 headroom도 정의할 수 없다.
   disagreement는 유망하지만 제주 8/8 cloud 공통오류처럼 오염만 반복 선택할 수 있으므로 품질,
   다양성, target 대표성, 비용을 함께 통제한다.
6. **단순 EO+지도 결합은 이미 주기여가 아니다.** NeurIPS 2025 GeoLink는 127만 EO–OSM pair에서
   region/object alignment와 object-patch fusion을 제안했고, ECCV 2024 MMEarth·SatMIP와 ICML
   2025 Galileo도 multimodal·time/location supervision의 표현 이득을 점유한다.
7. 새 빈칸은 **동적 공공 context의 provenance와 자연 누락 아래 EO-only 표현까지 강화되는가**다.
   train-only privileged supervision과 inference-time fusion을 분리하고, `event/observed/published/
   retrieved time`, coverage, 지연·충돌, 미래정보 누출을 benchmark 축으로 만들어야 한다.
8. workshop의 simple fusion 결과를 약한 비교군으로 취급하면 안 된다. Rao–Rolf의 STACK·TOKEN-FUSE가
   저라벨/OOD에서 learned fusion보다 강할 수 있으므로, 이를 못 이기면 새 adapter의 방법 기여는
   중단하고 benchmark/negative finding으로 범위를 낮춘다.
9. **이종 모델 임베딩 전이는 가능하지만 그 자체는 새롭지 않다.** AM-RADIO는 여러 vision
   foundation model을 한 student로 합쳤고 Theia는 multi-teacher distillation을 robot encoder까지
   연결했다. 우리 빈칸은 multi-temporal EO family/release의 cross-query/gallery 호환성과 task·비용을
   동시에 유지하는 stable representation bus다.
10. **일반 satellite–drone–ground alignment도 이미 혼잡하다.** GeoBridge·UniGeoRS·PAUL이
    multi-view semantic anchor, tri-view benchmark, noisy GPS correspondence를 점유한다. 로봇 트랙은
    release-stable satellite teacher, timestamped public context, field-side no-retrain 중 실제로 새로
    검증할 축이 있어야 한다.
11. **world model에는 action과 task가 필요하다.** DINO-WM·Navigation World Models·Vid2Sim 이후에는
    latent 또는 simulator를 만들었다는 사실이 아니라 EO 조건이 unseen-world planning·sim-to-real을
    개선하는지를 Success/SPL·collision으로 보여야 한다.
12. **embedding 효율은 compatibility와 함께 봐야 한다.** Matryoshka Representation Learning이
    nested dimension을 이미 점유하므로 64/128/256/768d 압축만으로는 부족하다. model family/release와
    edge/cloud 사이의 호환성, backfill byte, task utility가 함께 유지돼야 한다.
13. **한국 공공 context와 cache compatibility는 stable/residual로 분해한다.** cutoff-valid public
    data는 EO-only stable bus를 가르치는 privileged signal이자 추론 시 별도 residual이다. model
    release는 bus 좌표를, public-record publication은 residual을 갱신하므로 `E_repr / E_compat /
    E_fusion / E_refresh`를 따로 측정해야 한다.

## 먼저 읽을 26편

| 순서 | 문헌 | 읽고 답할 질문 | 바꿀 산출물 |
|---:|---|---|---|
| 1 | [GeoLink](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f681209306654a0c1f690f65810e8e45-Abstract-Conference.html) | 정적 OSM 결합이 이미 차지한 범위와 동적 공공기록의 남은 빈칸은 무엇인가? | context adapter·ablation |
| 2 | [MMEarth](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8085_ECCV_2024_paper.php) | train-time multimodal pretext가 EO-only 표현을 어떻게 강화하는가? | privileged-train/EO-only-test track |
| 3 | [Galileo](https://proceedings.mlr.press/v267/tseng25a.html) | 대규모 weather/SAR/DEM fusion과 경량 local adapter를 어떻게 공정 비교하는가? | native-ceiling table |
| 4 | [SatMIP](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3849_ECCV_2024_paper.php) | 위치·시간 metadata 이득과 좌표 shortcut을 어떻게 분리하는가? | location/year·shuffle control |
| 5 | [EarthShift](https://arxiv.org/abs/2605.29330) | 다섯 shift 중 한국 실험이 실제로 만드는 shift는 무엇인가? | spatial/temporal/sensor holdout 표 |
| 6 | [MMEarth-Bench](https://arxiv.org/abs/2602.06285) `W` | multimodal pretraining 이득이 low-shot·Africa OOD에서 언제 사라지는가? | 저자 표기 ECCV 2026; 공식 proceedings 확인 전 외부 재현 후보 |
| 7 | [PANGAEA](https://arxiv.org/abs/2412.04204) | 비-GFM supervised baseline을 어떤 조건으로 공정하게 비교하는가? | transfer matrix |
| 8 | [Parameter-Efficient Self-Supervised Geospatial Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2024/html/Scheibenreif_Parameter_Efficient_Self-Supervised_Geospatial_Domain_Adaptation_CVPR_2024_paper.html) | 1–2% adapter로 local context signal을 흡수할 수 있는가? | SLR adapter baseline |
| 9 | [Using Multiple Input Modalities to Improve Satellite-Based Land Use Classification](https://proceedings.mlr.press/v292/rao25a.html) `workshop` | hard-coded STACK/TOKEN-FUSE가 learned fusion보다 강한가? | simple-fusion kill baseline |
| 10 | [Beyond Accuracy: Calibration of GeoFMs under Distribution Shifts](https://arxiv.org/abs/2608.16614) `W` | risk–coverage 곡선이 구름·corruption에서 어떻게 무너지는가? | SCL 품질층별 abstention baseline |
| 11 | [Backward-Compatible Prediction Updates](https://arxiv.org/abs/2107.01057) | selective recompute와 negative flip을 FoldRefresh와 어떻게 분리하는가? | 릴리스 갱신 baseline |
| 12 | [Regression Coefficient Estimation from Remote Sensing Maps](https://arxiv.org/abs/2407.13659) | 확률표본 ground truth가 없을 때 어떤 주장을 포기해야 하는가? | PPI 표본설계 |
| 13 | [Conformal Risk Control](https://arxiv.org/abs/2208.02814) | map-level risk로 허용할 monotone loss는 정확히 무엇인가? | risk certificate 정의 |
| 14 | [Mapping on a Budget](https://ojs.aaai.org/index.php/AAAI/article/view/41162) | 검수 이동비용과 정보이득을 어떻게 action cost로 넣는가? | 후속 active-label baseline |
| 15 | [Detecting Environmental Violations with Satellite Imagery in Near Real Time](https://arxiv.org/abs/2208.08919) | 영상·허가구역·전문가 검수를 어떤 순서로 결합했는가? | K-Earth 파일럿 산출물 |
| 16 | [On the Generalizability of Foundation Models for Crop Type Mapping](https://arxiv.org/abs/2409.09451) | 현지 라벨 10–100개 이득이 한국 두 번째 지역에서도 남는가? | label-efficiency curve |
| 17 | [CLUE: Active Domain Adaptation](https://openaccess.thecvf.com/content/ICCV2021/html/Prabhu_Active_Domain_Adaptation_via_Clustering_Uncertainty-Weighted_Embeddings_ICCV_2021_paper.html) | uncertainty와 target diversity를 왜 함께 골라야 하는가? | 후속 active-label baseline |
| 18 | [AllClear](https://arxiv.org/abs/2410.23891) | cloud 복원 화질이 아니라 downstream 변화 판정까지 어떻게 층화할 것인가? | cloud stress protocol |
| 19 | [AM-RADIO](https://openaccess.thecvf.com/content/CVPR2024/html/Ranzinger_AM-RADIO_Agglomerative_Vision_Foundation_Model_Reduce_All_Domains_Into_One_CVPR_2024_paper.html) | 서로 다른 teacher의 token·task를 한 student에 합칠 때 무엇을 보존하는가? | multi-teacher distillation baseline |
| 20 | [Theia](https://proceedings.mlr.press/v270/shang25a.html) | VFM distillation이 robot policy에 실제로 전달됐음을 어떻게 증명했는가? | embodied transfer claim boundary |
| 21 | [Matryoshka Representation Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html) | dimension별 독립모델 없이 coarse-to-fine 정보를 어떻게 학습하는가? | edge/cloud 64–768d baseline |
| 22 | [DINO-WM](https://proceedings.mlr.press/v267/zhou25t.html) | pixel이 아닌 patch latent 미래예측이 planning으로 이어지는 최소 계약은 무엇인가? | EO-conditioned world-model baseline |
| 23 | [GeoBridge](https://openaccess.thecvf.com/content/CVPR2026/html/Song_GeoBridge_A_Semantic-Anchored_Multi-View_Foundation_Model_Bridging_Images_and_Text_CVPR_2026_paper.html) | satellite·drone·street·text 결합에서 이미 점유된 novelty는 무엇인가? | cross-view 경쟁 경계 |
| 24 | [UniGeoRS](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_UniGeoRS_A_Unified_Benchmark_for_Tri-view_Geo-Localization_CVPR_2026_paper.html) | real+synthetic tri-view 평가를 한국 P0보다 먼저 재사용할 수 있는가? | external localization benchmark |
| 25 | [Navigation World Models](https://openaccess.thecvf.com/content/CVPR2025/html/Bar_Navigation_World_Models_CVPR_2025_paper.html) | controllable video 예측과 실제 navigation 성공을 어떻게 연결하는가? | action-conditioned baseline |
| 26 | [Vid2Sim](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_Vid2Sim_Realistic_and_Interactive_Simulation_from_Video_for_Urban_Navigation_CVPR_2025_paper.html) | photorealism이 아니라 task-faithful simulation을 무엇으로 측정하는가? | sim-to-real kill gate |

K-ALIGN P0의 집중 독서 순서는 `GeoLink → MMEarth → Galileo → SatMIP → AM-RADIO →
Theia → BCT/FCT/LCE → MRL`이다. robotics/world-model 22–26은 stable bus가 task·compatibility gate를
통과하거나 paired trajectory가 생길 때까지 구현 백로그다.

## 1. OlmoEarth·Earth embedding·평가 기준

| 상태 | 문헌 | 선점한 것 / 중요한 반례 | 우리 결정 |
|---|---|---|---|
| `R+X` | [OlmoEarth v1: Stable Latent Image Modeling](https://arxiv.org/abs/2511.13655) | 멀티모달·시공간 latent MIM과 광범위 downstream 평가 | v1 수치는 `PAPER_NOTES_v1.md`에서만 사용; 공개 checkpoint 재현은 별도 감사 |
| `M+X` | [OlmoEarth v1.1](https://arxiv.org/abs/2605.20804) | 더 작은 모델군과 효율 변경 | 모델 크기 표는 릴리스별로 분리 |
| `M+X` | [OlmoEarth v1.2 공식 보고서](https://allenai.org/papers/olmoearth-v1-2) | 같은 학습 데이터에서 RoPE 등 알고리즘 변경; 평균 성능이 같아도 태스크별 회귀 존재 | v1↔v1.2 paired release audit의 직접 축 |
| `M` | [Earth Embeddings](https://arxiv.org/abs/2608.03410) | embedding product의 유형·저장·재현·불확실성 문제를 정리 | 평범한 embedding API를 사업으로 만들지 않음 |
| `M` | [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291) | 전지구 연간 10 m, 64차원 embedding field와 sparse-label mapping | product baseline이자 LUCAS 외부 재현 축 |
| `M` | [Better Together: Complementarity of Earth Embedding Models](https://arxiv.org/abs/2605.18667) | 단일 embedding 승자보다 모델 결합 가능성 | 여러 모델 결합은 single-model baseline 이후에만 |
| `M+X` | [GeoLink](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f681209306654a0c1f690f65810e8e45-Abstract-Conference.html) | 127만 EO–OSM pair, region/object alignment와 object-patch cross-attention | 단순 지도 fusion을 novelty에서 제거; dynamic provenance/natural missingness로 차별화 |
| `M+X` | [MMEarth](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8085_ECCV_2024_paper.php) | 120만 위치·12 modality의 multimodal masked pretraining으로 optical representation 강화 | public context를 train-only privileged supervision으로 쓰는 직접 baseline |
| `M+X` | [Galileo](https://proceedings.mlr.press/v267/tseng25a.html) | optical·SAR·elevation·weather·pseudo-label을 한 foundation model에서 학습 | weather/SAR 결합 자체를 novelty로 주장하지 않고 native multimodal ceiling으로 비교 |
| `M+X` | [SatMIP](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3849_ECCV_2024_paper.php) | time/location metadata를 text-like supervision으로 image와 정렬 | location/year-only와 shuffled metadata control을 필수화 |
| `M+W` | [MMEarth-Bench](https://arxiv.org/abs/2602.06285) | 12 modality·5 task·geographic OOD와 multimodal reconstruction TTT; arXiv에 ECCV 2026 게재 표기 | 공식 proceedings·독립재현 전 우월 결론 근거 금지, 외부 protocol 후보 |
| `M+W` | [Rao & Rolf: Multiple Input Modalities](https://proceedings.mlr.press/v292/rao25a.html) | STACK·TOKEN-FUSE 같은 단순 결합이 저라벨/OOD에서 learned fusion보다 강할 수 있음 | workshop 문헌이지만 반드시 이겨야 할 hard-coded baseline으로 사용 |
| `M` | [GEO-Bench](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a0644215d9cff6646fa334dfa5d29c5a-Abstract-Datasets_and_Benchmarks.html) | 6 classification + 6 segmentation의 표준 평가 | 임의 random split 대신 표준 protocol 참조 |
| `M` | [GEO-Bench-2](https://arxiv.org/abs/2511.15658) | 태스크·모달리티·제약별로 단일 최고 모델이 없다는 확장 평가 | router의 model action에 실제 headroom이 있는지 먼저 확인 |
| `M` | [PANGAEA](https://arxiv.org/abs/2412.04204) | 다양한 센서·해상도·시간·지역; GeoFM이 supervised baseline을 일관되게 이기지 않음 | U-Net/temporal baseline을 반드시 유지 |
| `M` | [Prithvi-EO-2.0](https://arxiv.org/abs/2412.02732) | HLS 기반 멀티시계열 GeoFM, SME 공동설계 | second-family release/transfer baseline 후보 |
| `M` | [SatMAE](https://arxiv.org/abs/2207.08051) | 시간·다중분광 masked autoencoding | OlmoEarth만 비교하는 자기참조 평가 방지 |
| `M` | [Scale-MAE](https://openaccess.thecvf.com/content/ICCV2023/html/Reed_Scale-MAE_A_Scale-Aware_Masked_Autoencoder_for_Multiscale_Geospatial_Representation_Learning_ICCV_2023_paper.html) | 알려진 ground sampling distance를 학습에 반영 | resolution action의 강한 representation baseline |
| `M` | [CROMA](https://papers.neurips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html) | SAR–optical contrastive + masked autoencoding | S1/S2/both sensor ablation baseline |
| `M` | [AnySat](https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html) | 다양한 sensor·resolution·scale을 한 모델에서 처리 | sensor routing 필요성이 모델 유연성으로 사라지는지 검사 |
| `M` | [TerraMind](https://openaccess.thecvf.com/content/ICCV2025/html/Jakubik_TerraMind_Large-Scale_Generative_Multimodality_for_Earth_Observation_ICCV_2025_paper.html) | any-to-any 생성과 missing modality 보완 | 실제 센서 취득 vs 생성 modality의 비용·위험 비교 |
| `M` | [Copernicus-FM / Copernicus-Bench](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Towards_a_Unified_Copernicus_Foundation_Model_for_Earth_Vision_ICCV_2025_paper.html) | 18.7M aligned pretraining image와 15개 Sentinel 계층 task; cloud도 별도 task로 포함 | cloud segmentation과 cloud 아래 change/abstention을 구분 |
| `M+W` | [REOBench](https://arxiv.org/abs/2505.16793) | 6 task × 12 synthetic/realistic corruption에서 모델별 degradation 비교 | 구름·기하 오염 stress의 보조 protocol; 실제 한국 shift 대체 금지 |
| `M` | [AllClear](https://arxiv.org/abs/2410.23891) | S2/Landsat/S1 약 4M 영상과 cloud-cover 층화 복원 평가 | PSNR 향상을 change accuracy 향상으로 오인하지 않음 |
| `M` | [CloudSEN12+](https://doi.org/10.1038/s41597-022-01878-2) | expert cloud/thin-cloud/shadow labels와 S1/보조자료 | input-quality benchmark로만 사용; 변화·원인 정답 아님 |

## 2. 전이·분포 이동·미세조정

| 상태 | 문헌 | 선점한 것 / 중요한 반례 | 우리 결정 |
|---|---|---|---|
| `M+X` | [EarthShift](https://arxiv.org/abs/2605.29330) | 8개 GeoFM·11태스크·5 shift에서 OOD 저하를 정면 측정 | 두 지역·두 태스크군·두 릴리스의 무튜닝 holdout 고정 |
| `M+W` | [Beyond Accuracy: Calibration of GeoFMs under Distribution Shifts](https://arxiv.org/abs/2608.16614) | 매우 최근 calibration/OOD 연구; confidence abstention 실패 가능성 | 출판·재현 전 단정 금지, 품질층별 비교 baseline으로만 사용 |
| `M` | [On the Generalizability of Foundation Models for Crop Type Mapping](https://arxiv.org/abs/2409.09451) | 5대륙 crop transfer; 제한된 현지 라벨에서 EO pretraining 이득 | 10/25/50/100 label budget과 spatial holdout |
| `M+W` | [Benchmarking GeoFMs for Agriculture Applications](https://arxiv.org/abs/2606.29664) | 지역 분리 crop segmentation/change detection와 rare-class 문제 | macro 평균 외 rare-class recall·worst-region 보고 |
| `M` | [Parameter Efficient Self-Supervised Geospatial Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2024/html/Scheibenreif_Parameter_Efficient_Self-Supervised_Geospatial_Domain_Adaptation_CVPR_2024_paper.html) | 1–2% 파라미터 적응과 제한 라벨 이득 | linear probe/adapter/full FT의 실측 비용까지 비교 |
| `M` | [Shaping Fine-Tuning of GeoFMs](https://proceedings.mlr.press/v292/castiglioni25a.html) | 낮은 라벨에서 full FT가 partial FT보다 나을 수 있는 반직관 | PEFT가 항상 효율적이라는 전제 폐기 |
| `M` | [Deep Learning Model Transfer in Forest Mapping](https://arxiv.org/abs/2308.05005) | 서로 다른 산림지역·다중센서 transfer | LFMC/산림 태스크의 second-region 설계 참고 |
| `M` | [Smallholder Field Delineation with Transfer Learning and Weak Supervision](https://arxiv.org/abs/2201.04771) | 프랑스 경계→인도 소농 필지 적응 사례 | 한국 FarmMap을 라벨 정답으로 쓰기 전 시점·정의 차이 검사 |
| `M+W` | [Deploying GeoFMs in the Real World: Lessons from WorldCereal](https://arxiv.org/abs/2508.00858) | operational crop mapping에서 adaptation·배포 마찰 | 논문 성능과 반복 운영비를 별도 열로 기록 |

### 전이 주장의 사전 게이트

`transfer learning이 유의미하다`고 쓰려면 최소한 다음을 모두 보고한다.

1. `frozen encoder + linear probe`, parameter-efficient adaptation, full fine-tuning,
   task-specific supervised baseline.
2. 두 태스크군 × 두 지역 × 두 모델 릴리스. 한 셀이라도 target-region tuning을 했다면
   zero-shot/무튜닝 전이라고 부르지 않는다.
3. label-efficiency, calibration, worst-region/rare-class 성능, risk–coverage, 실제 label·I/O·GPU 비용.
4. 사업 promotion 기준은 우선 `같은 품질에서 현지 라벨 50% 절감` 또는 `같은 라벨에서 강한
   baseline보다 5%p 개선` 중 하나와, 두 번째 태스크 반복이다. 이 수치는 시장 사실이 아니라
   실험 전 잠금용 가설이다.

### 2.1 M37 이후 action-utility claim을 위한 최우선 문헌 (2026-08-26)

아래는 단순 관련 문헌이 아니라 EarthRoute가 반드시 이겨야 하거나 주장을 좁히게 만든 직접
경쟁선이다. 상세 claim contract는 `docs/PAPER_CLAIM_EXPANSION_2026_08_26.md`를 따른다.

| 우선 | 문헌 | 이미 점유한 것 | 우리 실험에 강제하는 것 |
|---:|---|---|---|
| 1 | [EarthShift](https://earthshift.github.io/) | 8 GeoFM·11 task·5 real shift의 robustness 측정 | 공개 paired shift를 재사용하고 label-free action selection만 추가 |
| 2 | [CrossEarth-Gate](https://openaccess.thecvf.com/content/CVPR2026/html/Cao_CrossEarth-Gate_Fisher-Guided_Adaptive_Tuning_Engine_for_Efficient_Adaptation_of_Cross-Domain_CVPR_2026_paper.html) | Fisher-guided spatial/semantic/frequency PEFT 선택 | PEFT routing을 novelty에서 제거; cache lifecycle action으로 차별화 |
| 3 | [DARN](https://openaccess.thecvf.com/content/CVPR2026F/html/Yadav_DARN_Dynamic_Adaptive_Regularization_Networks_for_Efficient_and_Robust_Foundation_CVPRF_2026_paper.html) | sample difficulty 기반 decoder regularization/capacity gate | decoder-size router를 headline에서 제거; difficulty-only baseline 추가 |
| 4 | [GdScore](https://openreview.net/forum?id=FIWHRSuoos) | one-step gradient norm 기반 unsupervised accuracy estimation | dense segmentation용 gradient baseline과 cache-only 한계 보고 |
| 5 | [ODD](https://proceedings.mlr.press/v286/mishra25a.html) | source–target overlap-aware target error estimation | domain-overlap feature/baseline, support 밖 abstention |
| 6 | [RALF / Feature Store Freshness](https://escholarship.org/uc/item/5xk0f4z9) | downstream error feedback 기반 feature regret scheduling | label 도착 전 cold-start와 delayed-feedback baseline을 분리 |
| 7 | [IUPM](https://proceedings.mlr.press/v258/koebler25a.html) | gradual shift monitoring + uncertainty-aware active labels | temporal shift와 소량 audit-label action 비교 |
| 8 | [Model Assessment under Temporal Shift](https://proceedings.mlr.press/v235/han24b.html) | rolling-window loss 추정과 pairwise model selection | feedback delay·target-unlabeled 조건을 명시 |
| 9 | [Agreement-on-the-Line TTA](https://openreview.net/forum?id=iEFMwP5wng) | label-free accuracy·TTA selection·calibration | model agreement baseline; shift family 밖 실패를 따로 보고 |
| 10 | [Adapting Prediction Sets without Labels](https://proceedings.mlr.press/v286/kasa25a.html) | unlabeled shifted target의 conformal set 조정 | 예측 구간·risk–coverage·abstain action |
| 11 | [Testable Learning with Distribution Shift](https://proceedings.mlr.press/v247/klivans24a.html) | 통과 가능한 shift test 아래의 성능 certification | 보편 label-free 보장 금지; predefined shift family와 support test |
| 12 | [How to Embed Matters](https://openaccess.thecvf.com/content/CVPR2026W/EarthVision/html/Gilch_How_to_Embed_Matters_Evaluation_of_EO_Embedding_Design_Choices_CVPRW_2026_paper.html) | embedding design/aggregation recipe 의존성 | M37을 novelty가 아닌 proxy–utility 출발 evidence로 한정 |
| 13 | [ChronoEarth-492K](https://uiuctml.github.io/ChronoEarth492K/) | static/short/long horizon, spatial-temporal OOD, cross-satellite transfer | staleness/history action의 public stress 후보 |
| 14 | [MMEarth-Bench](https://mmearth-bench.com/) | 5 environmental task·geographic OOD·multimodal TTT | test-time adaptation ceiling과 task별 rank heterogeneity 비교 |

읽기 결과로 고정한 경계:

1. `target-unlabeled`는 source/development label을 쓴다는 사실을 숨기지 않는다.
2. 현재 risk가 아니라 **reuse 대비 각 action의 gain과 구간**을 예측한다.
3. `reuse / repair / re-embed / task-raw`를 test 전에 봉인한다.
4. representation extraction 비용은 여러 task가 공유하고, head/raw 비용은 task별로 계산한다.
5. leave-one-region뿐 아니라 leave-one-shift-family-out에서 baseline을 이겨야 한다.

## 3. 제한 예산 라벨 수집·능동 전이

| 상태 | 문헌 | 선점한 것 / 중요한 반례 | 우리 결정 |
|---|---|---|---|
| `M+X` | [Mapping on a Budget](https://ojs.aaai.org/index.php/AAAI/article/view/41162) | 공간적으로 군집된 SatML 라벨과 이질적 수집비용 아래 dataset augmentation을 최적화 | 단순 uncertainty보다 이동·검수·공식자료 취득비용을 포함 |
| `M` | [CLUE](https://openaccess.thecvf.com/content/ICCV2021/html/Prabhu_Active_Domain_Adaptation_via_Clustering_Uncertainty-Weighted_Embeddings_ICCV_2021_paper.html) | domain shift에서 uncertainty-only와 diversity-only의 한계를 지적하고 둘을 결합 | 한국 target pool의 강한 기본 baseline |
| `M` | [RIPU](https://openaccess.thecvf.com/content/CVPR2022/html/Xie_Towards_Fewer_Annotations_Active_Learning_via_Region_Impurity_and_Prediction_CVPR_2022_paper.html) | domain-shift semantic segmentation의 region impurity + uncertainty query | pixel query가 아니라 site/event annotation cost로 재정의 |
| `M` | [Active Learning under Label Shift](https://proceedings.mlr.press/v130/zhao21b.html) | class proportion이 이동할 때 naive class-balanced/uncertainty sampling의 bias–variance 문제 | 연도·지역별 class prevalence 이동을 무시하지 않음 |
| `M` | [Active Transfer Learning under Model Shift](https://proceedings.mlr.press/v32/wangi14.html) | source/target model shift 아래 라벨 query와 transfer를 함께 다룸 | `active transfer` 자체를 최초라고 주장하지 않음 |
| `M` | [Active-DDC](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Dual_Domain_Control_via_Active_Learning_for_Remote_Sensing_Domain_ICCV_2025_paper.html) | 센서·고도·지역이 순차로 바뀌는 RS object detection의 active replay/adaptation | 변화탐지·행정근거 문제와 차이를 명시 |
| `M` | [Disagreement-on-the-Line](https://proceedings.mlr.press/v202/lee23o.html) | 분포 이동에서 모델 disagreement와 오류의 경험적 관계 및 조건 | disagreement를 진실이나 물리적 `모호성선`으로 부르지 않음 |

### active-label 주장의 사전 게이트

1. 먼저 scratch/generic vision/OlmoEarth/다른 GeoFM의 지역·연도·센서·구름별 transfer effect를
   compute/data-matched 조건에서 측정한다.
2. random, 층화 random, uncertainty, disagreement, k-center/log-det diversity, CLUE, cost-aware
   spatial sampling을 같은 pool·oracle·budget·seed로 비교한다.
3. 주 지표는 AULC, labels-to-target, worst-group risk/AURC다. 선택된 라벨 수만 줄었다고 성공이 아니다.
4. disagreement query는 cloud/nodata quality gate와 spatial dedup을 먼저 통과한다. 같은 입력을
   공유한 모델들의 합의·불일치는 독립 증거가 아닐 수 있다.
5. adaptive label은 probability sample이 아니다. 봉인 test와 별도 층화 확률표본 없이 모집단
   prevalence나 PPI 신뢰구간을 보고하지 않는다.
6. PDE의 `beta`, advection–diffusion, 모호성선, D-opt의 물리·식별 해석을 전이하지 않는다.
   그룹별 transfer effect, 경험적 disagreement region, feature-diversity baseline으로 새로 정의한다.

## 4. 유효한 지도 추론·보류·누락 근거

| 상태 | 문헌 | 선점한 것 / 중요한 반례 | 우리 결정 |
|---|---|---|---|
| `M+X` | [Prediction-Powered Inference](https://arxiv.org/abs/2301.09633) | 많은 ML 예측 + 작은 무작위 ground truth로 모집단 parameter의 유효한 CI | Top-k 검수를 모집단 prevalence 추정으로 오용하지 않음 |
| `M+X` | [Regression Coefficient Estimation from Remote Sensing Maps](https://arxiv.org/abs/2407.13659) | remote-sensing map-only 회귀 편향을 PPI로 보정 | Sherrie Wang 계보의 직접 baseline; probability sample 필수 |
| `M` | [PPI++](https://arxiv.org/abs/2311.01453) | prediction quality에 따른 PPI 효율 개선 | 기본 PPI 재현 뒤에만 추가 |
| `M` | [Stratified Prediction-Powered Inference](https://arxiv.org/abs/2406.04291) | 이질적 집단에서 층화로 효율 개선 | 토지피복·지역·evidence coverage 층과 연결 |
| `M+X` | [Selective Classification for Deep Neural Networks](https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) | coverage를 포기해 선택적 risk를 낮추는 reject option | `367 abstain / 1 investigate`를 risk–coverage 곡선으로 평가 |
| `M+X` | [Conformal Risk Control](https://arxiv.org/abs/2208.02814) | monotone loss의 기대 risk 제어 | confidence threshold와 보증을 혼동하지 않음 |
| `M` | [Uncertainty Quantification in EO Using Conformal Prediction](https://arxiv.org/abs/2401.06421) | EO 지도에서 conformal 구현 사례 | pixel set coverage와 map statistic CI를 분리 |
| `M` | [Remote Control: Debiasing Remote Sensing Predictions for Causal Inference](https://iclr.cc/virtual/2023/14903) | 예측오차가 정책변수와 상관되면 downstream 효과추정이 편향 | 인과효과 주장은 별도 설계 없이는 금지 |
| `M` | [Mapping on a Budget](https://ojs.aaai.org/index.php/AAAI/article/view/41162) | 공간 라벨 수집을 이동·예산 제약 아래 최적화 | 모델 불확실성만이 아닌 이동비용 baseline 추가 |
| `M` | [Detecting Environmental Violations with Satellite Imagery in Near Real Time](https://arxiv.org/abs/2208.08919) | 고해상도 영상·허가구역·전문가 검수·층화평가를 실제 조사 우선순위와 연결 | K-Earth business wedge의 가장 가까운 공개 선례 |
| `M+W` | [Blind Spots in Environmental Permitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6690595) | 행정 허가목록 누락을 영상과 표적 검수로 연구 | working paper로 표시; no-match≠음성 원칙의 인접 증거 |

### 서로 섞으면 안 되는 세 보증

| 층 | 질문 | 후보 방법 | 현재 K-Earth 상태 |
|---|---|---|---|
| 예측 | 말한 사례 중 오답 위험은 얼마인가? | selective classification / conformal risk | 사람 라벨 표본 부족, 미검증 |
| 모집단 | 제주 전체 면적·비율·계수의 CI가 유효한가? | 확률표본 + PPI/설계기반 추정 | Top-k만 있어 아직 불가 |
| 증거 | 공식 원인자료가 실제로 조회 가능한가? | coverage/missingness audit | A/B급 원인 근거 0/368 |

EarthRoute의 `risk certificate`는 위 세 수치를 각각 내야 한다. 하나의 confidence score로
합치면 안 된다.

## 5. 적응형 관측·계산·EO agent

| 상태 | 문헌/시스템 | 이미 차지한 것 | EarthRoute에 남긴 빈칸 |
|---|---|---|---|
| `M` | [THOR](https://arxiv.org/abs/2601.16011) | 한 가중치에서 다양한 sensor native resolution과 patch-size compute trade-off | fetch 전 관측 선택, 행정근거, map statistic 보증은 별도 |
| `M` | [RingMoE](https://arxiv.org/abs/2504.03166) | modality expert와 dynamic expert pruning | observation acquisition과 evidence escalation은 별도 |
| `M` | [ZoomEarth](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_ZoomEarth_Active_Perception_for_Ultra-High-Resolution_Geospatial_Vision-Language_Tasks_CVPR_2026_paper.html) | 메모리에 있는 UHR 이미지의 active crop/zoom | 지도 생산 전 sensor/time 구매와는 다름 |
| `M` | [EO-Gym](https://arxiv.org/abs/2605.01250) | 35 tools와 상호작용 EO agent 환경 | benchmark/agent와 deterministic production policy 구분 |
| `M` | [OpenEarth-Agent](https://arxiv.org/abs/2603.22148) | 열린 환경 workflow planning과 tool creation | 생성된 도구의 통계적 보증·compiled runtime은 별도 |
| `M` | [OpenEarthAgent](https://arxiv.org/abs/2602.17665) | tool-augmented geospatial reasoning trace | 하이픈 있는 동명 논문과 혼동 금지 |
| `M+X` | [OlmoEarth Platform infrastructure](https://allenai.org/blog/olmoearth-infrastructure) | 대륙 규모 분산 실행, provider index, least-cloudy scene, 향후 alerts·agents·global embeddings | 범용 scheduler가 아니라 결정·근거 감사로 차별화 |

현재 조사 범위에서 `fetch 전 data + compute + administrative evidence + human review`를
map-level decision risk 아래 공동 선택하는 공개 benchmark를 직접 찾지 못했다. 이것은 **novelty
확정이 아니라 검색 가설**이다. 논문 투고 전 DBLP/Semantic Scholar/OpenAlex 검색식과 제외 로그를
별도 보존한다.

## 6. 모델 갱신·embedding compatibility·부분 refresh

| 상태 | 문헌/자산 | 선점한 것 | 우리 결정 |
|---|---|---|---|
| `M` | [Towards Backward-Compatible Representation Learning](https://arxiv.org/abs/2003.11942) | gallery backfill 없이 old/new embedding을 직접 비교하도록 새 모델을 학습 | 모델 학습을 바꿀 수 없는 공개 릴리스 상황과 구분 |
| `M` | [Learning Backward Compatible Embeddings](https://arxiv.org/abs/2206.03040) | 여러 historical embedding consumer의 호환성 | downstream consumer 유지 baseline |
| `M` | [Backward-Compatible Prediction Updates](https://arxiv.org/abs/2107.01057) | 제한 예산 prediction update와 negative flip | FoldRefresh의 모집단 statistic 목표와 instance-level flip을 함께 보고 |
| `M` | [Boundary-Aware Backward-Compatible Representation](https://openaccess.thecvf.com/content/CVPR2023/html/Pan_Boundary-Aware_Backward-Compatible_Representation_via_Adversarial_Learning_in_Image_Retrieval_CVPR_2023_paper.html) | retrieval 호환성과 새 표현 판별력 trade-off | Procrustes 하나만으로 호환성 문헌을 대표하지 않음 |
| `M` | [Forward Compatible Training for Large-Scale Embedding Retrieval](https://openaccess.thecvf.com/content/CVPR2022/html/Ramanujan_Forward_Compatible_Training_for_Large-Scale_Embedding_Retrieval_Systems_CVPR_2022_paper.html) | 미래 모델을 위해 기존 표현에 side information을 준비 | old release를 다시 학습할 수 없는 경우의 경계 명시 |
| `X` | FoldRefresh 로컬 재현 체인 (`../decision-ready-earth-ai/REPRODUCTION.md`) | v1→v1.2 모집단 통계의 design-based partial refresh | **동료심사 통과 논문으로 취급 금지**; 이 저장소에는 rslearn port만 미완료 |

FoldRefresh의 실험·claim verifier·preregistration은 별도 로컬 저장소에서 확인되었다. 다만 실제
OpenReview 최종 제출 상태는 로컬 체크리스트의 수동 항목과 첨부 노트만으로 독립 확인하지 못했다.
따라서 이 프로젝트에서는 `local artifact verified / venue status externally unverified`로 표기한다.

## 7. Federated learning — 실제 silo가 있을 때만

| 상태 | 문헌 | 이미 차지한 것 / 경고 | 우리 결정 |
|---|---|---|---|
| `M+W` | [FedRS-Bench](https://arxiv.org/abs/2505.08325) | 8 RS classification dataset, 135 geographic/source clients, 10 FL algorithms | 시간변화·dense task·공식근거는 빈칸이지만 synthetic province split만으로 novelty 주장 금지 |
| `M` | [FedSense](https://openaccess.thecvf.com/content/ICCV2025/html/Tan_Towards_Privacy-preserved_Pre-training_of_Remote_Sensing_Foundation_Models_with_Federated_ICCV_2025_paper.html) | remote-sensing FM의 federated pretraining과 communication reduction | adapter-level downstream FL과 구분; FL 자체가 privacy 보증은 아님 |
| `M` | [GeoFed](https://arxiv.org/abs/2404.09292) | remote-sensing geographic heterogeneity를 FL 문제로 다룸 | 자연 지역 split의 strong baseline 후보 |
| `M` | [FedAG](https://proceedings.mlr.press/v267/wang25bk.html) | cross-silo foundation-model adapter aggregation | 실제 기관 silo 확보 시 adapter FL 비교군 |
| `M` | [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html) / [FedProx](https://proceedings.mlsys.org/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html) / [SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html) | 평균집계, heterogeneity regularization, client-drift correction | local-only·pooled upper bound와 함께 최소 baseline |
| `M` | [Gradient inversion](https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html) | gradient만으로도 training data가 노출될 수 있음 | secure aggregation/DP 없이 `privacy-preserving` 금지 |

FL은 독립 기관 3곳 이상, 반출 불가 라벨/영상, 실제 분산 실행, 중앙화 불가 근거가 모두 있을 때만
본문으로 승격한다. 공개 data.go.kr 자료를 시도별로 나눈 실험은 `synthetic geographic stress test`다.
최소 비교는 local-only, pooled centralized upper bound, FedAvg full/adapters, FedProx/SCAFFOLD adapters,
personalized head이며 macro/worst-client와 upload/download bytes를 함께 보고한다.

## 8. 사회적 영향·데이터 자산 설계 참고

| 상태 | 문헌 | 배울 설계 | 이 프로젝트에 그대로 가져오지 않을 것 |
|---|---|---|---|
| `M` | [SustainBench](https://arxiv.org/abs/2111.04724) | 위성·현장조사·SDG target의 표준 task와 데이터 문서화 | 공동 연구를 특정 연구실 단독 자산처럼 표현하지 않음 |
| `M` | [DivShift](https://ojs.aaai.org/index.php/AAAI/article/view/35060) | 시민과학의 공간·시간·관측자·사회경제 편향을 shift로 분해 | 데이터가 많다는 이유로 표본대표성을 가정하지 않음 |
| `M` | [PlantTraitNet](https://ojs.aaai.org/index.php/AAAI/article/view/41272) | 약한 시민과학 관측 + 불확실성 + 독립 생태자료 검증 | 자동지도를 field truth로 대체하지 않음 |
| `M` | [GRAM](https://ojs.aaai.org/index.php/AAAI/article/view/41227) | 여러 도시·기관 경계를 연결한 고해상도 자산과 unseen-region 평가 | award/규모를 방법 novelty의 증거로 사용하지 않음 |
| `M` | [Street-view + satellite crop mapping](https://arxiv.org/abs/2309.05930) | 다른 관측체계의 라벨을 전국 위성지도와 결합 | 한국 도로영상 접근성을 가정하지 않음 |
| `M` | [ClimSim](https://papers.neurips.cc/paper_files/paper/2023/hash/45fbcc01349292f5e059a0b8b02c8c3f-Abstract-Datasets_and_Benchmarks.html) | 큰 자산 + 명확한 coupling target + 공개 baseline | 실행 trace만 쌓은 data dump를 자산 논문이라 부르지 않음 |
| `M` | [PRISM Alignment Dataset](https://proceedings.neurips.cc/paper_files/paper/2024/hash/be2e1b68b44f2419e19f6c35a1b8cf35-Abstract-Datasets_and_Benchmarks_Track.html) | 누구의 피드백인지 인구통계·맥락과 함께 보존 | LLM alignment 결과를 EO 성능 근거로 사용하지 않음 |

## 9. 이종 임베딩 전이·cross-view robotics·simulation

| 상태 | 문헌 | 이미 차지한 것 / 중요한 경계 | 우리 결정 |
|---|---|---|---|
| `M+X` | [AM-RADIO](https://openaccess.thecvf.com/content/CVPR2024/html/Ranzinger_AM-RADIO_Agglomerative_Vision_Foundation_Model_Reduce_All_Domains_Into_One_CVPR_2024_paper.html) | 여러 상이한 VFM teacher를 한 효율적 student에 증류 | multi-teacher 자체를 novelty에서 제거; EO temporal/spatial relation과 compatibility가 필요 |
| `M+X` | [Theia](https://proceedings.mlr.press/v270/shang25a.html) | multi-teacher VFM distillation을 robot learning encoder로 연결 | EO→robot feature 연결만으로 기여 금지; policy/task 성능 필수 |
| `M+X` | [Matryoshka Representation Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html) | 한 표현의 nested prefix로 dimension·compute를 조절 | 64/128/256/768d efficiency baseline; family/release 호환성과 결합 |
| `M+X` | [Learning Compatible Embeddings](https://openaccess.thecvf.com/content/ICCV2021/html/Meng_Learning_Compatible_Embeddings_ICCV_2021_paper.html) | architecture·data·loss·dimension 변화의 embedding compatibility | BCT/FCT/AdvBCT와 함께 stable EO bus의 필수 비교군 |
| `M+X` | [DINO-WM](https://proceedings.mlr.press/v267/zhou25t.html) | pretrained patch feature의 action-conditioned future prediction과 planning | EO token을 latent state로 쓸 때의 직접 baseline; future error보다 planning primary |
| `M` | [Navigation World Models](https://openaccess.thecvf.com/content/CVPR2025/html/Bar_Navigation_World_Models_CVPR_2025_paper.html) | 관측과 navigation action으로 controllable future video 생성 | trajectory/action 없는 EO 시계열을 world model이라 부르지 않음 |
| `M+X` | [Vid2Sim](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_Vid2Sim_Realistic_and_Interactive_Simulation_from_Video_for_Urban_Navigation_CVPR_2025_paper.html) | real video에서 photorealistic·interactive urban simulator와 sim-to-real 평가 | FID보다 real policy Success/SPL을 simulation promotion gate로 사용 |
| `M` | [URBAN-SIM](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Towards_Autonomous_Micromobility_through_Scalable_Urban_Simulation_CVPR_2025_paper.html) | 여러 embodiment·task를 가진 scalable urban simulation | 제주 fly-through를 embodied benchmark로 포장하지 않음 |
| `M+X` | [GeoBridge](https://openaccess.thecvf.com/content/CVPR2026/html/Song_GeoBridge_A_Semantic-Anchored_Multi-View_Foundation_Model_Bridging_Images_and_Text_CVPR_2026_paper.html) | drone·street·satellite·text pair와 semantic-anchor multi-view FM | generic tri-view alignment를 novelty에서 제거 |
| `M+X` | [UniGeoRS](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_UniGeoRS_A_Unified_Benchmark_for_Tri-view_Geo-Localization_CVPR_2026_paper.html) | real+synthetic satellite·drone·ground benchmark와 unified matching | 새 한국 dataset 전에 external benchmark 재사용 |
| `M+X` | [PAUL](https://openaccess.thecvf.com/content/CVPR2026/html/Li_PAUL_Uncertainty-Guided_Partition_and_Augmentation_for_Robust_Cross-View_Geo-Localization_under_CVPR_2026_paper.html) | GPS drift에 따른 noisy/partial correspondence | 공공 좌표를 완전 alignment로 가정하지 않는 baseline |
| `M` | [MMGeo](https://openaccess.thecvf.com/content/ICCV2025/html/Ji_MMGeo_Multimodal_Compositional_Geo-Localization_for_UAVs_ICCV_2025_paper.html) | image·point cloud·depth·text query의 satellite geolocation | modality 수 증가 자체를 기여로 세지 않음 |
| `M` | [FG²](https://openaccess.thecvf.com/content/CVPR2025/html/Xia_FG2_Fine-Grained_Cross-View_Localization_by_Fine-Grained_Feature_Matching_CVPR_2025_paper.html) | fine-grained ground-to-aerial feature matching과 3DoF localization | cross-view pose의 strong task baseline |
| `M` | [Sky2Ground](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Sky2Ground_A_Benchmark_for_Site_Modeling_under_Varying_Altitude_CVPR_2026_paper.html) | ground/aerial/satellite site modeling에서 satellite input이 오히려 noise가 될 수 있음 | EO context가 항상 유익하다는 가정 금지; input ablation 필수 |
| `M` | [Satellite to GroundScape](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Satellite_to_GroundScape_-_Large-scale_Consistent_Ground_View_Generation_from_CVPR_2025_paper.html) | satellite-conditioned consistent ground-view generation | 생성 화질만으로 navigation 기여 주장 금지 |

현재 프로젝트에서 가장 방어력 있는 중심은 stable multi-teacher bus와 한국 공공 context residual을
합친 `K-ALIGN`이다. public-context privileged distillation과 cross-model compatibility가 모두 gate를
통과할 때만 한 논문으로 묶고, 하나만 통과하면 다시 분리한다. authoritative 실험 계약은
`KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md`, 이후 순서는 `release-stable satellite→drone/ground
transfer` → trajectory 기반 `EO-conditioned world model`이며 넓은 트랙 경계는
`EMBEDDING_TRANSFER_CVPR_TRACKS.md`에 둔다.

## 10. 산악·cryosphere 지역 전이

| 상태 | 문헌 | 이미 차지한 것 / 중요한 경계 | 우리 결정 |
|---|---|---|---|
| `M+X` | [Landslide4Sense](https://arxiv.org/abs/2206.00515) | S2+DEM+slope, 4지역 3,799 patch의 산사태 segmentation benchmark | 새 benchmark를 먼저 만들지 않고 region-holdout frozen/adapter 검증에 사용 |
| `M` | [Globally scalable glacier mapping](https://www.nature.com/articles/s41467-024-54956-x) | optical+SAR+DEM과 여러 regional inventory로 glacier mapping을 전문가 수준과 비교 | glacier mapping 자체를 novelty로 주장하지 않음 |
| `M` | [Glacial lake mapping using Geo-Foundation Model](https://elib.dlr.de/220257/) | Prithvi 기반 glacial-lake segmentation과 small lake·shadow 문제 | Olmo/Prithvi/scratch 비교와 shadow stratum의 직접 baseline |
| `W+X` | [Glacial-Lake-Bench](https://essd.copernicus.org/preprints/essd-2026-474/) | 19,115 S1+S2+DEM pair, challenge subset, leave-one-region-out; 2026 ESSD 심사 중 preprint | 출판 결론으로 쓰지 않고 Phase 0 cryosphere transfer protocol 후보로 사용 |

산악 통합 연구는 빙하호를 한국에 억지로 전이하지 않는다. cryosphere는 HKH↔알프스·Monviso,
공통 산악 disturbance는 전 지역에서 평가한다. 공식 data/evidence와 kill gate는
`MOUNTAIN_EVIDENCE_TRANSFER.md`에 고정한다.

## 검색식과 갱신 절차

### 핵심 검색식

```text
(earth observation OR remote sensing OR geospatial) AND
  (foundation model) AND (geographic transfer OR temporal shift OR calibration)

(satellite OR geospatial) AND
  (adaptive inference OR active perception OR sensor selection OR compute routing)

(embedding OR prediction) AND
  (model update OR backward compatible OR stale OR partial refresh OR backfill)

(remote sensing map) AND
  (selective prediction OR conformal OR prediction-powered inference OR abstention)

(earth observation OR satellite) AND
  (active learning OR label acquisition OR dataset optimization) AND
  (domain shift OR transfer OR spatial cost)

(remote sensing OR geospatial foundation model) AND
  (federated learning OR cross-silo OR personalized adapter)

(environmental permitting OR land-use permit) AND
  (satellite imagery OR change detection) AND (missing records OR audit)

(embedding OR representation) AND
  (cross-model OR multi-teacher distillation OR backward compatible OR matryoshka) AND
  (earth observation OR remote sensing)

(satellite OR aerial OR drone OR street view) AND
  (cross-view localization OR robot navigation OR visual place recognition) AND
  (domain shift OR model update OR noisy correspondence)

(world model OR interactive simulation OR sim-to-real) AND
  (remote sensing OR satellite map OR geospatial context) AND
  (planning OR active sensing OR navigation)
```

### 한 논문을 승격하는 조건

1. `candidate → M`: primary URL, 연도/venue/preprint 상태, 저자 주장 범위를 확인한다.
2. `M → R`: 표본 단위, split, baseline, metric, 실패 사례, license를 본문/부록에서 기록한다.
3. `R → X`: 이 프로젝트의 config·표·kill rule 중 하나를 실제로 바꾸고 경로를 남긴다.
4. 2026 preprint는 출판 상태가 바뀔 수 있으므로 실험 착수와 투고 전에 다시 확인한다.

## 다음 독서 산출물

- `EarthShift + Beyond Accuracy`: 제주 SCL/토지피복별 OOD·calibration protocol 1쪽.
- `PANGAEA + CLUE + Mapping on a Budget`: matched-scratch transfer table과 active-label offline replay.
- `AllClear + CloudSEN12 + CROMA`: cloud 화질지표와 downstream change/AURC를 분리한 ablation.
- `Backward-Compatible Prediction Updates + FoldRefresh`: instance flip과 population statistic을
  함께 비교하는 baseline 표.
- `RSE PPI + StratPPI + Mapping on a Budget`: 오름 368의 확률표본·이동비용 설계.
- `Environmental Violations + Blind Spots`: 사후환경영향평가 업체 인터뷰용 현재 workflow 도식.
- `THOR + EO-Gym + OlmoEarth infrastructure`: EarthRoute oracle의 최소 action space와 경쟁표.
- `FedRS-Bench + FedSense + FedAG`: 실제 기관 silo가 생겼을 때만 여는 adapter-FL protocol.
- `AM-RADIO + Theia + BCT/FCT/LCE + MRL`: multi-teacher stable Earth bus의 loss·효율·호환성 표.
- `GeoBridge + UniGeoRS + PAUL + Sky2Ground`: satellite–drone–ground 공간에서 이미 선점된 질문과
  release-stable/no-retrain 빈칸 정리.
- `DINO-WM + Navigation World Models + Vid2Sim`: EO-conditioned latent planning과 task-faithful
  simulation의 trajectory·metric·sim-to-real 최소 계약.
