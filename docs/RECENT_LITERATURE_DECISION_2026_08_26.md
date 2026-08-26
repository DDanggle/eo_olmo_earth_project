# 최근 연구 대비 판정 — E1 이후 무엇이 논문 기여인가

조사일: 2026-08-26. 목적은 결과를 뒷받침할 논문을 모으는 것이 아니라, 이미 선점된 주장과
아직 남은 질문을 분리하는 것이다.

## 냉정한 판정

E1의 큰 decoder가 frozen OLMoEarth 성능을 회복하는 것은 **중요한 실험 수정**이지만 새 방법은
아니다. OLMoEarth 원 논문은 frozen linear probe와 full fine-tuning을 분리하고, full fine-tuning은
초기 20% epoch만 encoder를 동결한 뒤 전체 모델을 푼다. partner segmentation에는 transposed-conv
또는 multi-scale encoder용 U-Net decoder를 쓴다. PANGAEA도 dense task에서 encoder의 네 intermediate
level과 UPerNet을 사용한다. 따라서 마지막 layer + 237K decoder로 GeoFM을 판정한 M30은
**embedding-product viability 시험**으로는 유효하지만, model potential의 최종 판정으로는 약했다.

M37 완결 결과는 이를 더 좁혔다. large decoder의 회복은 tiled cache에서만 나타났고
full-context에서는 small/large가 모두 악화됐다. 따라서 `더 넓은 context`나 `더 큰 decoder`를
일반 처방으로 제안하지 않고, **serving context와 head를 결합한 recipe contract**로 다룬다.
이 interaction 역시 한 개발 지역·seed의 분석 결과이지 방법 novelty가 아니다.

- OLMoEarth paper: <https://arxiv.org/pdf/2511.13655>
- PANGAEA: <https://arxiv.org/abs/2412.04204>
- PEFT systematic study: <https://arxiv.org/abs/2504.17397>
- DEFLECT (ICCV 2025): <https://openaccess.thecvf.com/content/ICCV2025/html/Thoreau_Parameter-Efficient_Adaptation_of_Geospatial_Foundation_Models_through_Embedding_Deflection_ICCV_2025_paper.html>

frozen GFM이 scratch U-Net에 지는 현상도 실패가 아니라 알려진 평가 양상이다. PANGAEA는 task에
따라 supervised U-Net이 모든 GFM을 이기는 사례를 보고했고, 2026 biomass benchmark도 frozen
encoder와 embedding product가 서로 다른 순위를 보인다고 보고한다.

- 2026 biomass benchmark: <https://arxiv.org/abs/2608.04792>

## shared Earth embedding cache는 이미 하나의 분야다

AlphaEarth, TESSERA, OlmoEarth Studio는 모두 embedding을 여러 downstream 분석에 재사용하는
`embeddings-as-data`를 전면에 둔다. TESSERA v2는 29-task suite, 16–128차원 storage frontier,
v1의 along-track/tile-seam artifact까지 이미 다룬다. 따라서 아래는 novelty가 아니다.

- embedding을 한 번 계산해 여러 task에 쓴다.
- tile seam이 있다.
- embedding dimension·storage와 성능을 비교한다.
- 서로 다른 embedding product를 공통 loader로 읽는다.

근거:

- OlmoEarth embeddings: <https://allenai.org/blog/olmoearth-embeddings>
- AlphaEarth: <https://arxiv.org/abs/2507.22291>
- TESSERA: <https://arxiv.org/abs/2506.20380>
- TESSERA v2: <https://arxiv.org/abs/2607.03949>
- Earth Embeddings as Products: <https://arxiv.org/abs/2601.13134>

M34의 seam 측정은 우리 실행 결함을 고친다는 점과 Ai2-facing engineering evidence로는 가치가
있지만, seam 자체를 CVPR headline으로 쓰지 않는다.

## refresh router도 일반 시스템 주장만으로는 부족하다

Berkeley RALF는 downstream prediction feedback으로 stale feature의 `feature store regret`을 정의하고
중요한 feature update를 우선하는 시스템을 이미 제안했다. 따라서 “downstream 영향을 보고 cache를
갱신한다”만으로는 새롭지 않다.

- RALF/feature freshness thesis: <https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/31523.html>

남는 EO-specific gap은 더 좁다.

> 라벨을 즉시 볼 수 없는 EO 운영에서 지역·관측창·센서·모델 release가 변할 때, 서로 다른
> downstream task의 손실을 **관측 가능한 contract/quality/representation 진단으로 미리 예측**하고,
> `reuse / cached-token adapter / re-embed / PEFT / task-specific raw model` 중 budget-aware action을
> 고를 수 있는가?

이 질문은 단순 freshness가 아니다. 동일 지리 셀의 입력 증거, 공간·시간 support, 센서 modality,
embedding generator/version과 task semantics가 함께 변한다. novelty는 router라는 단어가 아니라
**EO shift별 task loss의 예측 가능성, action regret, 외부 지역 전이**에서 나온다.

## 실험 사슬

### Stage A — representation viability를 공정하게 닫기

1. E1: tiled/full context × small/large convolutional decoder — M37 완료. full-context는 중단하고
   tiled-large만 exact-time·공통 seed에서 재검증한다.
2. multi-level/UNet decoder: tiled-large가 positive-macro·비용을 회복하지 못하고 OLMo intermediate
   feature를 실제로 노출할 수 있을 때만 별도 arm.
3. adapter 위치를 분리한다.
   - cached-token adapter: encoder 재실행 없이 task별 head/adapter만 학습.
   - LoRA/partial/full fine-tuning: encoder가 바뀌므로 **shared cache 재생성이 필요한 action**.
4. raw UNet3D/U-TAE와 accuracy·training·re-embedding·storage를 함께 보고한다.

큰 decoder로 P2를 이기더라도 이것은 Stage A 통과이지 CVPR method 결과가 아니다.

### Stage B — router가 필요한지 먼저 반증

같은 spatial unit에 최소 세 task를 둔다.

- land-cover/forest state segmentation
- deforestation/disturbance segmentation 또는 retrieval
- landslide segmentation 또는 event retrieval

각 task에 동일한 representation states를 제공한다.

- context shift: tiled vs full-context
- observation shift: stale vs current temporal window
- input-contract shift: missing bands/sensor or resampling
- model shift: OLMo release 또는 두 번째 embedding family

task별 action gain의 순위가 거의 같으면 router를 중단한다. 평균 metric 교차나 서로 다른 지표의
승자 교차는 근거가 아니다. 같은 목적함수에서 sample/region별 action-value heterogeneity와
oracle routing gain이 있어야 Stage C를 연다.

### Stage C — label-free risk predictor와 action policy

입력 feature 후보는 label을 사용하지 않는다.

- contract: model/data/build version, band/modality, GSD, temporal support
- quality: valid coverage, cloud/SCL, missing observation, mosaic count
- representation: norm/anisotropy, tiled-full discrepancy, neighbor stability, release drift
- task descriptor: output support, object-size prior, temporal horizon, label budget

평가는 degradation prediction 자체와 최종 policy를 분리한다.

- prediction: rank correlation, calibration, worst-group error
- decision: oracle regret, accuracy–GPU/storage/latency Pareto, fixed-schedule·uncertainty-only·RALF-style
  downstream-feedback baseline과 비교

### Stage D — 지역과 모델 전이

- method development: 공개 multi-task benchmark/PANGAEA-compatible tasks
- operational demonstration: coverage-valid AI-Hub v2 + KMA/산림·지질 evidence
- untouched transfer: Nepal 또는 Switzerland 한 곳을 최종 외부 region으로 사용
- representation family: OLMoEarth custom sub-annual embedding + TESSERA v2 또는 AlphaEarth annual
  product. annual product는 event timing을 놓칠 수 있으므로 강한 대조군이지 하위 모델이 아니다.

제주·지리산·네팔·스위스를 모두 training geography로 섞지 않는다. 방법 개발, 한국 운영 데모,
최종 외부 transfer의 역할을 분리한다.

## CVPR 판정

현재는 **CVPR method paper가 아니다.** 지금 확보한 것은 강한 재현성/계약 감사, 한 개발 지역의
frozen-small 실패, decoder capacity 회복 신호, AI-Hub materialization 결함이다.

CVPR 후보로 승격하려면 최소한 다음이 모두 필요하다.

1. 2개 representation family × 3개 task × 3개 이상의 사전 정의 shift.
2. task별 action heterogeneity와 양의 oracle routing gain.
3. label-free risk predictor가 단순 contract rule·uncertainty·fixed refresh를 이김.
4. 동일 accuracy에서 compute/storage/latency를 줄이거나 동일 budget에서 accuracy를 높이는 Pareto.
5. 미열람 외부 지역에서 policy ordering이 유지됨.
6. spatial/event 단위 interval, 여러 seed, test 노출 없는 protocol.

여기서 2가 실패하면 router를 접고, 결과는 `when frozen Earth embeddings fail` benchmark/analysis로
전환한다. 3–5가 실패하면 system/data/workshop paper로 낮춘다. 이 kill path를 유지해야 박사 연구로
신뢰할 수 있다.
