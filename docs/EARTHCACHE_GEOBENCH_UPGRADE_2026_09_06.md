# EarthCache × GEO-Bench-2 업그레이드 판정

작성: 2026-09-06  
상태: **설계 준비 완료, selector 학습 전**  
기계 판독 계약: `config/geobench_cache_action_prereg_v0.json`

## 결론

MS-109는 분명한 진전이다. Solar에서 OlmoEarth cache는 Galileo cache보다 macro IoU `+.149`,
AP `+.131`이고 8/8 fold에서 모두 우위였다. Sen12와 다른 과업에서 순서가 반복됐으므로
“OlmoEarth cache의 가치가 산사태 한 과업의 우연”이라는 반론은 약해졌다.

그러나 이 결과는 **selector의 필요성까지 증명하지 않는다.** 지금 관측한 두 과업에서는
OlmoEarth가 모두 1등이다. 항상 OlmoEarth를 고르는 고정 정책이 이미 최적이면, 행동 선택기는
정확도를 올릴 여지가 없다. 따라서 다음 단계는 selector 코드를 만드는 것이 아니라 아래 질문을
먼저 닫는 것이다.

> **과업·지역·계절·모델 규모·비용 예산이 바뀔 때 실제 최적 행동이 바뀌는가?**

최적 행동의 순위 역전 또는 정확도–비용 Pareto 교차가 없다면, 이 프로젝트는 선택 방법 논문이
아니라 **어떤 Earth cache가 언제 가치 있는지 보이는 특성화 벤치마크**로 남긴다.

## 1. MS-109에서 확실히 된 것과 안 된 것

서버의 16개 raw report를 다시 집계한 값은 다음과 같다.

| Solar, seed 1 | macro IoU | AP | fold 승수 |
|---|---:|---:|---:|
| OlmoEarth base cache | 0.608072 | 0.926177 | 8/8 |
| Galileo base cache | 0.458675 | 0.795246 | 0/8 |
| paired 차이 | +0.149397 | +0.130930 | one-sided Wilcoxon p=.00390625 |

근거 상태를 구분한다.

- **강한 관측**: OlmoEarth cache가 서로 다른 두 downstream 과업에서 raw 및 Galileo보다 강했다.
- **반복이지 독립 표본 8개가 아님**: Solar 8개는 하나의 데이터셋에서 만든 UTM group fold다.
  독립 과업 수는 현재 2개다.
- **한 seed**: 방향은 8/8로 선명하지만 seed 불확실성은 남는다.
- **selector 증거 0**: 두 과업 모두 OlmoEarth가 1등이라 top action이 아직 바뀌지 않았다.
- **프로브 강등 유지**: effective rank는 model family를 가르는 탐색 신호일 뿐, family 내부에서
  어느 cache가 좋은지 예측하지 못했다. 절대 성능 예측기로 쓰지 않는다.

현재 보이는 작은 교차는 Galileo와 raw 사이에 있다. Sen12에서는 raw `.197`이 Galileo `.153`보다
높고, Solar에서는 Galileo `.459`가 raw `.333`보다 높다. 하지만 OlmoEarth가 두 경우 모두 위에
있으므로, **OlmoEarth cache를 항상 쓸 수 있는 action set에서는 이 교차만으로 routing 이득이 없다.**
OlmoEarth nano가 Sen12에서 raw보다 아주 조금 낮은 `.194 < .197`인 결과도 한 seed·절대 `.003`
차이라 아직 순위 역전 근거로 승급하지 않는다.

## 2. 행동 선택기의 가장 큰 장점

가장 큰 장점은 “좋은 모델을 추천한다”가 아니다.

> **배포 전에 모든 파이프라인을 다시 돌리지 않고, 정확도 손실과 재계산 비용을 함께 통제한다.**

Earth observation에서는 원본 영상 읽기, 여러 시점 정렬, encoder 추론, cache 쓰기가 반복된다.
모델 release·밴드·GSD·시간창이 달라지면 기존 embedding이 조용히 무효가 될 수도 있다. 선택기가
유효하면 다음 네 가지를 한꺼번에 얻는다.

1. **성능**: 고정 모델 하나가 아닌 과업별 최적 또는 근접 action을 골라 task-normalized score를
   올린다.
2. **비용**: cache가 충분한 경우 불필요한 raw I/O와 re-embedding을 피한다.
3. **안전**: action 차이가 불확실하거나 계약이 깨졌을 때 억지로 고르지 않고 label 추가 또는
   re-embed를 요청한다.
4. **운영 가능성**: 모델 release가 바뀔 때 “전부 갱신”과 “그냥 재사용” 사이를 task와 예산에 맞게
   결정한다.

반대로 benchmark score가 오르지 않고 비용만 줄면 CVPR vision method 주장은 약하다. 그래서
이번 계약은 **고정 compute budget에서 GEO-Bench-style normalized IQM 상승**과 **동일 성능에서
실측 비용 절감**을 별도 gate로 둔다.

## 3. 비슷한 논문과 정확한 차이

generic model selection은 이미 오래된 문제다. 따라서 “모델 성능을 미리 예측한다”만으로는
novelty가 없다.

| 연구 | 이미 해결한 것 | EarthCache가 추가로 보여야 하는 것 |
|---|---|---|
| [Task2Vec, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Achille_Task2Vec_Task_Embedding_for_Meta-Learning_ICCV_2019_paper.html) | label과 Fisher 기반 task embedding으로 feature extractor 선택 | dense·multispectral·temporal EO, cache 상태와 실측 비용, label-free/K-shot 분리 |
| [LEEP, ICML 2020](https://proceedings.mlr.press/v119/nguyen20b.html) | target label 한 번의 forward로 transferability 추정 | cache/re-embed/adapt action과 공간·시간 shift |
| [LogME, ICML 2021](https://proceedings.mlr.press/v139/you21b.html) | label evidence로 pretrained model을 fine-tuning 없이 평가 | task별 action regret, dense task, cache lifecycle 및 비용 |
| [Capabilities Encoding, CVPRW 2025](https://cvpr.thecvf.com/virtual/2025/35652) | RS foundation model 성능 예측과 선택 | 실제 action outcome, held-out task/family, release·contract·cache cost |
| [REMSA, 2025 preprint](https://arxiv.org/abs/2511.17442) | 150개 이상 RSFM metadata와 자연어 요구로 모델 추천 | metadata 추천이 아니라 실제 downstream counterfactual 및 regret |
| [GEO-Bench-2](https://arxiv.org/abs/2511.15658) | 19개 dataset, 5개 task type, capability별 static ranking | 저장 cache의 재사용·적응·재계산 결정과 budget별 결과 |
| [How do SSL RS models transfer?, 2026 preprint](https://arxiv.org/abs/2606.13896) | task·adaptation·layer에 따라 GeoFM ranking이 바뀜 | 순위 변화의 재관찰이 아니라 **변화를 사전에 예측해 행동** |
| [Beyond Accuracy, 2026 preprint](https://arxiv.org/abs/2608.16614) | corruption/budget에서 순위 변화, confidence abstention 실패 | confidence 하나가 아닌 contract·cache·cost 기반 fail-closed 정책 |
| [RALF, Berkeley 2024](https://escholarship.org/uc/item/5xk0f4z9) | downstream feedback으로 stale feature refresh 우선순위화 | 즉시 feedback 없는 EO cold-start, third-party release와 센서/시간 계약 |
| [BCT, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html) | 새 encoder를 호환되게 학습해 gallery backfill을 줄임 | 학습을 통제할 수 없는 frozen GeoFM release에서 reuse/repair/re-embed 선택 |

가장 가까운 위협은 Task2Vec/LogME/Capabilities Encoding이다. 이들과 구분되는 최소 novelty는
**모델 하나를 고르는 것**이 아니라 다음 counterfactual 표를 공개하고 예측하는 데 있다.

```text
동일 deployment episode
  × model family / release / scale
  × cached-head / head-adapt / re-embed / PEFT-re-embed / request
  × label budget
  × 실제 GPU·raw-I/O·storage budget
  → downstream utility와 action regret
```

## 4. GEO-Bench-2를 다시 써도 되는가

**된다. 오히려 적절하다.** GEO-Bench-2 저자들도 하나의 모델이 모든 task에서 우세하지 않다고
보고했고, adaptation 연구를 허용하는 유연한 protocol을 명시한다. 다만 사용법을 두 층으로 나눈다.

### 층 A — 논문의 주 실험: EarthCache decision benchmark

- GEO-Bench-2의 공개 dataset과 official split을 재사용한다.
- 모든 action에 같은 label·decoder·seed·budget 계약을 적용한다.
- task마다 metric 방향과 범위가 다르므로 절대 IoU/accuracy/RMSE를 섞지 않는다.
- 각 task 안에서 oracle 대비 regret를 계산하고, GEO-Bench 방식의 task-normalized IQM으로 모은다.
- 항상 Olmo, 항상 cache, 항상 re-embed, 가장 싼 action, Task2Vec/LogME류, oracle과 비교한다.

이 결과는 “공식 GEO-Bench leaderboard를 몇 점 올렸다”가 아니라 **같은 GEO-Bench 데이터에서
정해진 예산으로 고정 정책보다 normalized score가 올랐다**는 새 protocol의 결과다.

### 층 B — leaderboard-compatible 보조 실험

리뷰어가 standard benchmark 성능을 요구할 수 있으므로, 최종 방법 한 개와 강한 static baseline
한 개만 공식 seed/HPO/metric 계약으로 별도 실행한다. 공식 protocol을 완전히 맞추지 못하면
leaderboard 숫자와 직접 비교하지 않는다.

현재 PASTIS·Fields of the World·DynamicEarthNet 세 개만으로는 모두 dense segmentation 쪽이라
“GEO-Bench 전반”을 말할 수 없다. 최소한 classification 1개와 regression 1개를 추가한다.

- segmentation/time: PASTIS, DynamicEarthNet
- segmentation/contract shift: Fields of the World
- classification: TreeSatAI 또는 BigEarthNet-v2 중 하나
- regression: BioMassters
- cloud/sensor stress: CloudSEN12

Sen12와 Solar는 이미 본 개발 과업으로만 사용한다. 새 공개 과업을 final first-look으로 둔다.

## 5. 방법을 만들기 전 0번 실험

복잡한 selector보다 아래 **oracle headroom 표**를 먼저 만든다.

1. 모든 개발 episode에 같은 action set을 실행한다.
2. seed 불확실성보다 큰 best-action 순위 역전 수를 센다.
3. budget을 고정했을 때 best static action과 per-episode oracle의 normalized score 차이를 계산한다.
4. 실측 GPU·I/O·storage를 넣어 Pareto 교차를 확인한다.

승급 기준은 사전에 고정한다.

- 독립 task/deployment group 최소 2개에서 top action 순위 역전, **또는** 실제 budget에 따른 Pareto
  최적 action 교차.
- oracle routing이 best static보다 normalized score `+.02` 이상인 budget track 최소 1개.
- 이 조건을 못 넘으면 learned selector를 만들지 않는다.

이 gate가 중요한 이유는 단순하다. 선택지가 달라 보이게 많은 model variant를 넣으면 oracle 이득은
자동으로 커진다. family·fold를 독립 표본처럼 세지 않고, 정말 새로운 task와 family에 일반화하는지
leave-one-task-out와 leave-one-family-out으로 검증해야 한다.

## 6. selector가 통과하면 만들 방법

절대 macro IoU를 회귀하지 않는다. 과업 난이도를 맞히는 지름길이 되기 때문이다. 예측 대상은
각 episode 안의 **pairwise action gain** 또는 **oracle-normalized regret**다.

입력은 행동 전에 알 수 있는 것만 쓴다.

- contract: 센서, 밴드, GSD, 시간 길이, cloud/missingness
- cache: model family/release/scale, token support, dtype/bytes, unlabeled summary
- shift: source-target mean/covariance distance, effective rank는 후보 하나로만
- cost: warm read, cold encode, raw I/O, cache write/invalidation
- K-shot track만: LogME/Task2Vec, support loss·gradient·label composition

출력은 하나의 모델 이름이 아니라 `CACHED_HEAD / HEAD_ADAPT / REEMBED / PEFT_REEMBED /
REQUEST_MORE_LABELS`다. 차이가 불확실하면 보류한다. 이 보류가 confidence threshold 하나보다
나은지는 Beyond Accuracy가 보여준 confidently-wrong 실패를 baseline으로 직접 시험한다.

## 7. CVPR 가능성 판정

| 제출 형태 | 현재 | 필요한 증거 |
|---|---|---|
| OlmoEarth가 좋다는 application paper | 약함 | 이미 benchmark 논문들과 겹침 |
| EarthCacheBench characterization | **성립 중** | 공개 Task-3, 실측 비용, scale/family/release 표 완결 |
| generic model selector | 약함 | Task2Vec/LogME/Capabilities Encoding 대비 novelty 부족 |
| cost-aware Earth cache action benchmark + selector | **main 후보, 아직 미실증** | G0 headroom + held-out task/family regret 개선 + normalized score 상승 + 비용 절감 |

한 문장 paper pitch는 다음이 가장 정확하다.

> **Static GeoFM rankings say which encoder wins after evaluation; EarthCacheBench asks which already
> materialized representation should be reused, adapted, or recomputed before paying that evaluation cost,
> and measures the decision with held-out utility regret and real data-system cost.**

한국 공공데이터는 여기서 논문을 장식하는 모달리티가 아니다. 마지막 외부 배포 사례에서 cloud,
필지, 행정 coverage를 contract/evidence state로 주고, 근거가 부족할 때 `REQUEST`가 실제 오탐을
줄이는지 보여주는 역할이다. 공개 benchmark의 빈칸을 한국 사례만으로 대신하지 않는다.

## 8. 바로 실행할 순서

1. PASTIS 다운로드 무결성을 완료한다. FOTW와 DynamicEarthNet은 검증된 상태를 DONE marker와
   구분해 표시한다.
2. 모든 GEO-Bench 결과를 보기 전에 chipping·time·band·metric·action set을 이 문서의 JSON에서
   final freeze한다.
3. `code/geobench_action_headroom.py`로 Solar/Sen12 기존 표의 oracle headroom을 먼저 계산한다.
   현재 top-1 headroom이 0일 가능성을 그대로 보고한다. 이 코드는 seed를 먼저 접고 task를
   일반화 단위로 세며, 비용 때문에 불가능한 action을 0점으로 채우지 않는다.
4. Core-6 중 classification/regression 각 1개를 확보한다.
5. 비용 harness를 먼저 실측하고, 그 다음 action matrix를 GPU1에서 실행한다.
6. G0가 통과한 경우에만 simple pairwise ranker를 만든다. deep router·MoE·VLM은 금지한다.
7. leave-one-task-out + leave-one-family-out first-look 뒤에만 Korea external track을 연다.

## 9. 코드·산출물 감사에서 발견한 작은 결함

- Solar 결과 자체는 raw report와 실제 `galileo_audit.json`으로 확인됐다.
- `extract_galileo_cache.py`는 source의 `cache_audit.json`을 target에 symlink해 downstream report가
  OlmoEarth source audit을 Galileo audit처럼 담을 수 있었다. 실제 Galileo audit은 별도 파일이라
  성능 수치는 무효가 아니지만 provenance가 혼동된다.
- 같은 파일의 contract 문자열은 Solar가 실제 `T=4`인데 `T=12`로 고정돼 있었다. future run은
  target `cache_audit.json` 자체를 Galileo audit으로 쓰고 실제 timestep 집합을 기록하도록 고친다.
- `status.sh`는 DONE marker가 없는 검증 완료 데이터도 “받는중”으로 표시했다. `검증OK`가 있으면
  `검증완료`, 실패 marker가 있으면 `실패`가 우선하도록 바꾼다.

이 결함들은 MS-109의 16개 metric을 바꾸지는 않지만, 다음 benchmark의 report hash와 cache
provenance를 믿기 위해 지금 고쳐야 한다.
