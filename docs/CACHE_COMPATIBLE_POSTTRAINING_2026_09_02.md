# CacheTune 설계 — Earth embedding cache를 버리지 않는 post-training

작성: 2026-09-02  
상태: **METHOD PREREGISTRATION DRAFT · NO GPU RUN · NO KOREA TEST ACCESS**  
범위: Sen12Landslides S12q → Korea 3-task external transfer. Nepal은 K=0 운영 사례,
Switzerland는 후속 missing-band contract stress다.

## 결정

기존 `Reuse or Retrain?` 측정 축은 보존하되, 논문의 method 후보를 다음 질문으로 확장한다.

> **새 지역에 task label K개와 계산·저장 예산 B가 주어졌을 때, 기존 OLMoEarth embedding
> cache를 그대로 재사용할지, cache 위에서만 저랭크 적응할지, encoder를 PEFT해 재임베딩할지,
> raw task model을 다시 학습할지를 어떻게 결정하는가?**

제안 method의 작업명은 **CacheTune**이다.

> **Post-train the Earth embedding product, not the encoder.**

`LoRA`, `MoE`, `label curve`, `embedding reuse`는 각각 이미 선행연구가 많다. novelty는 이 부품들의
이름이 아니라 **cache validity를 보존하는 적응 action과, 정확도–라벨–I/O–재임베딩 비용을 같은
frontier에서 비교하는 문제 설정**에 둔다.

## 왜 지금 증거에서 자연스럽게 이어지는가

| 기존 증거 | 새 설계에서의 역할 |
|---|---|
| P4 frozen OLMoEarth `.2722` > P2 `.1966`, P3 `.1834` | cache 안에 새 지역으로 전이되는 task 정보가 있다는 현상 증명 |
| Presto pooled/native `.1092/.1261` | 아무 frozen embedding이면 자동으로 되는 현상은 아니라는 한정적 control |
| MS-90B/91/92 fusion·gate 실패 | prediction을 섞는 router가 아니라 representation의 적응 위치를 비교해야 한다는 근거 |
| M8 release/input-contract 감사 | encoder 또는 release가 바뀌면 cache 의미와 유효성이 바뀐다는 engineering 문제 |
| AI-Hub 동일 cube 3-task | cache 한 번 + task adapter 여러 개의 amortization을 실측할 자산 |
| Nepal NP-88/89 | K=0에서 embedding 변화가 검토 후보를 만드는 운영 사례. 본선 method 증거는 아님 |

현재 432-run source-label curve는 필요한 baseline이지만 method novelty는 아니다. 이것은
`source supervision을 줄여도 K=0 target transfer가 유지되는가`를 답한다. CacheTune은 다른 질문,
즉 `target에 K개의 support label이 생겼을 때 어디를 post-train할 것인가`를 답한다.

## action ladder

```text
raw Sentinel cube
        │
        ├─────────────────────────────── A4 raw task model retrain
        │
        ▼
frozen OLMoEarth encoder
        │                         encoder weight 변경
        │                         └── A3 APLA/LoRA/full FT
        │                                  │
        ▼                                  └── target 전체 재임베딩
sealed spatial cache z: 768×32×32
        │
        ├── A0 reuse: source decoder 그대로
        ├── A1 head adapt: cache 유지, decoder만 target 적응
        └── A2 CacheTune: cache 유지, low-rank spatial residual 적응
```

| action | target에서 갱신 | raw cube 필요 | 기존 cache 유효 | 배포 artifact |
|---|---|---:|---:|---|
| **A0 Reuse** | 없음 | 아니요 | 예 | source decoder |
| **A1 Head** | decoder 일부/전체 | 아니요 | 예 | target decoder |
| **A2 CacheTune** | cache residual + calibration | 아니요 | 예 | 작은 adapter |
| **A3 Encoder PEFT** | OLMo attention/MLP APLA 또는 LoRA | 예 | **아니요** | PEFT weights + 새 cache |
| **A4 Raw** | U-TAE/UNet3D | 예 | 사용 안 함 | raw task model |

FoldRefresh/VersionAdapterCache는 A2와 경쟁하는 새 이름이 아니다. release v1→v1.2처럼 **encoder
version이 바뀌어 cache migration이 필요한 경우**의 기존 repair action으로 보존한다. CacheTune은
같은 frozen encoder·같은 cache contract에서 지역/task를 적응한다.

## A2 — low-rank spatial cache adapter

각 위치의 cached token을 `z ∈ R^(768×H×W)`라고 할 때:

```text
h  = DWConv3x3(V(LN(z)))
z' = z + U(gelu(h))
```

- `V`: `768 → r` 1×1 projection
- `DWConv3x3`: rank channel별 작은 공간 문맥
- `U`: `r → 768` 1×1 projection
- `r`: primary `16`; sensitivity `8, 32`
- `U`를 0으로 초기화해 step 0의 출력이 정확히 A0와 같게 한다.

이것을 **Low-Rank Spatial Cache Adapter**라고 부른다. encoder weight를 바꾸는 LoRA와 구분한다.
primary는 현재 공개·로컬 자산과 연결되는 final spatial cache만 쓴다. early/mid-layer cache는
별도 저장·추출을 요구하므로 method가 살아난 뒤의 sensitivity이며 core claim에 필요하지 않다.

### 가장 중요한 identifiability guard

현재 P4 decoder는 첫 층에서 이미 `1×1 Conv 768→128`을 학습한다. adapter와 이 decoder를 동시에
자유롭게 학습하면 low-rank residual이 단순히 “더 큰 decoder”와 구별되지 않을 수 있다.

따라서 primary 비교를 다음처럼 고정한다.

1. **A0:** source-trained decoder 동결, adapter 없음.
2. **A1:** 같은 source checkpoint에서 decoder만 target support로 적응.
3. **A2-strict:** source decoder를 동결하고 adapter + scalar calibration만 적응.
4. **A2-nospatial control:** 같은 파라미터 예산의 `V→GELU→U`, depthwise spatial conv 없음.
5. **A2-joint sensitivity:** adapter와 decoder를 함께 적응하되 primary method evidence로 세지 않음.

A2-strict이 A1보다 좋아야 “cache representation을 고쳤다”는 해석이 가능하다. A2-joint만 좋아지면
method 주장은 약화하고 decoder capacity 결과로 보고한다.

## target-label 계약

세 종류의 label 질문을 절대 섞지 않는다.

| 질문 | source train label | target train label | target 평가 |
|---|---:|---:|---|
| 기존 M65 zero-target transfer | 100% | 0 | held-out region 전체 |
| 봉인된 source-label curve | 1/5/10/100% | 0 | held-out region 전체; full source val 별도 |
| **CacheTune target few-shot** | 우선 100% | `K={5,20,50}` tiles | support와 공간적으로 분리된 query |

target few-shot에서 full target validation label을 쓰지 않는다. epoch·learning rate·rank·fixed update
수·threshold는 이미 노출된 개발 지역에서 고정한다. target normalization을 unlabeled pool 전체에서
계산하면 `transductive, unlabeled-target-available`이라고 명시하고 inductive sensitivity를 함께 낸다.

support와 query는 단순 random tile split이 아니다.

- event/connected component를 먼저 묶는다.
- spatial block과 buffer를 둬 같은 산사태 또는 인접 patch가 양쪽에 들어가지 않게 한다.
- 같은 K와 같은 support ID를 A0–A4가 공유한다.
- support draw seed와 optimizer seed를 분리한다.
- target query는 model selection에 사용하지 않는다.

Sen12의 개발 후보는 이미 노출된 China/Chimanimani이지만, CPU feasibility audit에서 각 K의
양성·음성, component 수, 최소 buffer를 충족할 때만 사용한다. 충족하지 못하면 region 이름을
결과를 보지 않고 바꾼 뒤 manifest를 봉인한다.

## 단계별 실행 — 대량 GPU보다 method viability가 먼저다

### PT-0 · CPU 계약/runner

- target support/query spatial manifest 생성·SHA 봉인
- 기존 runner를 직접 훼손하지 않는 별도 adaptation runner
- 필수 인자: `subset_manifest`, `init_ckpt`, `freeze_policy`, `fixed_updates`, `target_manifest`
- `pos_weight`, normalization, loader exposure가 허용된 support만 보는지 synthetic unit test
- source decoder checkpoint와 code snapshot을 시작 시점에 봉인
- no GPU, no Korea test access

### PT-1 · exposed-region strict gate

primary screen:

```text
2 exposed regions × K {5,20} × {A1, A2-strict, A2-nospatial} × 3 support/optimizer pairs
= 36 runs
```

A0는 같은 source checkpoint의 고정 결과를 재사용한다. `r=16`은 결과를 보기 전에 고정한다.
PT-1은 method viability screen이며 논문 확증이 아니다.

### PT-2 · encoder ceiling

PT-1을 통과한 경우에만 rslearn에 이미 있는 APLA 또는 구현을 감사한 LoRA를 A3 comparator로 연다.
먼저 `K=20`의 두 개발 지역에서 수행한다. A3는 raw input read, encoder GPU, 새 embedding 생성,
새 cache bytes를 모두 비용에 포함한다. official full fine-tune는 A3 ceiling이 필요할 때만 연다.

### PT-3 · method-confirmatory matrix

PT-1/2 recipe를 동결한 뒤 Sen12의 나머지 지역에서 A0/A1/A2와 제한된 A3 ceiling을 수행한다.
이 데이터셋의 기존 P4 결과는 이미 열렸으므로 “완전히 untouched dataset”이라 부르지 않고,
**new-method preregistered evaluation**이라고 부른다.

### PT-4 · Korea final external transfer + 3-task system test

Sen12에서 모든 선택을 끝낸 뒤 Korea를 연다.

```text
한 번 계산한 Korea OLMoEarth cache
  ├─ land-cover adapter
  ├─ deforestation adapter
  └─ landslide adapter
```

- support cluster와 sealed query cluster를 공간적으로 분리한다.
- 같은 cache를 task 1/2/3개가 쓸 때 cold/warm 비용과 break-even task 수를 보고한다.
- v2 12-band materialization은 보존하되 Sen12 비교 primary는 같은 10-band-compatible view와
  OLMo v1 input contract를 먼저 사용한다.
- Korea를 hyperparameter 개발과 final external test에 동시에 쓰지 않는다.

### PT-5 · Nepal/Switzerland

- Nepal: K=0 operational demonstration. 피해 확정이나 method 확증으로 세지 않는다.
- Switzerland: 7-band 제품은 missing-band contract shift다. 동일 transfer라고 부르지 않고,
  band availability gate와 별도 adapter/abstention action을 평가한다.

## 사전 kill gates

### CacheTune method gate

PT-1에서 아래 둘 중 하나를 충족해야 한다.

1. A2-strict가 A1보다 primary metric `+0.01` 이상이며 두 개발 지역에서 같은 방향, 또는
2. A2-strict가 A1의 `−0.01` 이내 성능을 유지하면서 trainable parameter를 5배 이상 줄인다.

추가 조건:

- learning-curve AUC `+0.02` 또는 같은 성능에 필요한 target label 50% 절감은 강한 승급 근거다.
- rank 16이 rank 32/full-rank 없이는 작동하지 않으면 `low-rank` 주장을 철회한다.
- A2-joint만 이기면 cache-representation method가 아니라 decoder adaptation 결과로 강등한다.

### systems-value gate

A3의 A1 대비 gain이 양수일 때만 다음 비율을 계산한다.

```text
(A2 − A1) / (A3 − A1)
```

A2가 A3 gain의 80% 이상을 회수하면서 raw bytes read + encoder GPU time이 A3의 10% 이하이면 강한
systems claim이다. A3 gain이 0 이하이면 “80% 회수”는 정의하지 않고 A2/A1 절대 비교만 보고한다.

### MoE promotion gate

MoE는 처음부터 구현하지 않는다. source-region별 A2 adapter의 cross-transfer matrix를 먼저 만든다.

- FP-matched label-peeking adapter oracle이 single shared adapter보다 `+0.02` 이상
- 최소 3개 개발 region 또는 2개 이상 task에서 같은 현상
- seed 방향 일치와 region-level uncertainty 통과

이 조건을 통과할 때만 다음 sparse mixture를 연다.

```text
z' = z + Σ[e in TopK(pi)] pi_e · U_e(DWConv(V_e(LN(z))))
```

expert는 full encoder가 아니라 작은 adapter다. router는 held-out label을 보지 않고 target support
summary·contract metadata·embedding shift를 입력으로 쓰는 **region/support-set-level** selector다.
P2/P4 prediction fusion, tile label-free winner prediction, region-ID shortcut은 금지한다.

### action planner는 learned router보다 먼저다

A0–A4 frontier가 생기면 첫 산출물은 복잡한 router가 아니라 deterministic planner다.

```text
입력: target label K, raw 접근 가능 여부, cache version/contract,
      허용 GPU·latency·storage, 요구 risk/accuracy
출력: A0 / A1 / A2 / A3 / A4 / abstain-and-request-labels
```

각 action의 개발지역 lower confidence bound와 실측 비용을 사용해 요구 조건을 만족하는 가장 싼
action을 고른다. 어떤 cell에서도 승자가 바뀌지 않으면 learned selector는 만들지 않는다. label·task·
contract cell 사이에 재현 가능한 crossover가 있고 deterministic rule의 regret이 클 때만 supervised
ranker 또는 contextual bandit을 연다. multi-step cache-age 로그가 없으므로 PPO/RL은 현재 범위가 아니다.

### 선택 확장 — cache-driven active support acquisition

A2가 살아나면 같은 cache를 `무엇을 예측할지`뿐 아니라 `어떤 target tile을 먼저 라벨링할지`에도
쓸 수 있다. spatially separated unlabeled pool에서 embedding diversity(k-center)와 source-head
uncertainty를 결합해 K개 support를 요청하고, spatial random·uncertainty-only·raw-spectral
diversity와 비교한다.

- 같은 성능에 필요한 label을 spatial random보다 20% 이상 줄이거나 learning-curve AUC `+0.02`
  이상일 때만 확장 기여로 승격한다.
- final query block은 acquisition pool에도 노출하지 않는다.
- ground-truth positive/negative로 support 후보를 미리 층화하지 않는다.
- 같은 event의 여러 patch를 여러 독립 label로 세지 않는다.

이 gate를 통과하면 사업 문장이 `adapter를 싸게 학습한다`에서 `어디를 먼저 라벨링할지까지 cache가
줄여준다`로 확장된다. A2가 실패하면 이 가지도 열지 않는다.

## 평가표 — accuracy만으로 끝내지 않는다

| 축 | 반드시 기록할 값 |
|---|---|
| task risk | positive-tile macro IoU, exact AP, empty-tile FP, calibration |
| transfer | region-macro, worst-region, region-level/bootstrap interval |
| labels | support tile·component·positive pixel 수, annotation unit |
| train | GPU-second, peak VRAM, optimizer updates, trainable/total params |
| data/I/O | raw bytes read, cache bytes read/write, re-embedded samples |
| serving | inference latency, adapter bytes, 배포 artifact 수 |
| cache contract | encoder/release/input checksum, cache valid/invalid, abstention reason |
| system scale | task 수 K에 따른 cold/warm total cost와 break-even |

비용은 다음처럼 분리한다.

- **cold:** raw acquisition/read + OLMo extraction + training + deployment
- **warm:** 기존 cache 존재 + adapter/head training + serving
- **version change:** cache validation + migration/refresh + 재추출

“trainable params가 작다”만으로 싸다고 쓰지 않는다. encoder LoRA가 planetary cache를 무효화하는
비용과, A2가 cache를 재사용하는 I/O 차이를 실측한다.

## 최근 연구와 겹치는 곳·남는 곳

- [PANGAEA](https://arxiv.org/abs/2412.04204)와
  [Shaping Fine-Tuning of GeoFMs](https://proceedings.mlr.press/v292/castiglioni25a.html)는
  limited-label/frozen evaluation을 이미 다룬다. label curve만으로는 novelty가 아니다.
- [Parameter-Efficient Self-Supervised Geospatial Domain Adaptation, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Scheibenreif_Parameter_Efficient_Self-Supervised_Geospatial_Domain_Adaptation_CVPR_2024_paper.html),
  [Fine-tune Smarter, Not Harder](https://arxiv.org/abs/2504.17397),
  [DEFLECT, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Thoreau_Parameter-Efficient_Adaptation_of_Geospatial_Foundation_Models_through_Embedding_Deflection_ICCV_2025_paper.html)는
  GeoFM PEFT가 이미 혼잡함을 보여준다. OLMoEarth에 generic LoRA를 붙이는 것은 기여가 아니다.
- [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291),
  [Earth Embeddings](https://arxiv.org/abs/2608.03410),
  [OlmoEarth Embeddings](https://allenai.org/blog/olmoearth-embeddings)는 embedding을 재사용 가능한
  product로 다룬다. 본 연구는 그 product를 **무효화하지 않는 task/region post-training**을 묻는다.
- [MAPEX](https://arxiv.org/abs/2507.07527)와 low-rank expert mixture 선행 때문에 MoE 자체는
  novelty가 아니다. 실제 action/expert complementarity가 있을 때만 시스템 구성요소가 된다.
- rslearn의 OLMoEarth 경로는 fine-tuning과 `LayerDecayAdamW`를 지원하고 설치 환경에는 APLA
  callback이 있다. 이것은 A3 구현 경로이자 강한 baseline이지 새 기여가 아니다.

현재 조사 범위에서 `stored Earth embedding product의 cache validity를 명시적으로 보존하면서
target few-shot low-rank spatial adaptation과 encoder PEFT의 re-embedding cost를 한 frontier에서
비교`하는 조합은 직접 일치하는 선행을 찾지 못했다. “선행 없음”을 보장하는 문장은 아니며 제출 전
추가 novelty search와 reviewer-style nearest-work 표가 필요하다.

## 세 목표와 하나의 산출물

| 목표 | CacheTune이 남기는 것 |
|---|---|
| AI2 취업 | OLMoEarth cache contract, rslearn-native APLA comparator, sealed Slurm recipe, failure-aware deployment artifact |
| 박사/CVPR | geographic few-shot에서 cache-compatible adaptation과 label–compute–invalidation frontier |
| 비즈니스 | 새 국가/task마다 planetary reprocessing 없이 작은 adapter만 배포하는 운영 단위 |

제품 문장:

> **한 번 만든 planetary embedding cache를 버리지 않고, 새 지역과 새 재해에 몇 개의 라벨만으로
> post-train한다. 부족할 때만 더 비싼 encoder 적응이나 재학습으로 올라간다.**

논문 제목 후보:

> **Cache, Adapt, or Recompute? Budgeted Post-Training of Earth Embedding Products under Geographic Shift**

## 지금의 실행 결정

1. 기존 raw recipe audit는 strong A4 baseline을 위해 유지한다.
2. 432-run source-label manifest는 봉인 상태로 보존하지만 대량 GPU 결론을 먼저 열지 않는다.
3. CPU PT-0 spatial support/query audit와 별도 adaptation runner 명세를 먼저 닫는다.
4. exposed-region 36-run PT-1에서 A2가 A1과 구별되는지 판정한다.
5. 통과할 때만 encoder APLA/LoRA ceiling, method-confirmatory matrix, Korea 3-task로 확장한다.
6. MoE·active acquisition·bandit/RL은 각각 oracle headroom이 생긴 뒤의 promotion stage다.

이 순서는 복잡한 모델을 먼저 만드는 것이 아니라, **어느 engineering action이 실제로 필요한지**를
가장 싼 반증 실험부터 확인한다.
