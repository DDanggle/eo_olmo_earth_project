# Post-training 타당성 판정 — 무엇이 지금 되고 무엇이 안 되는가

작성 2026-08-26. 사용자 질문 "post training도 충분히 되냐"에 대한 판정.
**이번 사이클에서 LLM 학습은 하지 않음.** Apertus 대응표는 아이디어로만 기록함.

## 판정 요약

| 단계 | 내용 | 판정 | 전제 |
|---|---|---|---|
| L0 | contract validator + snapshot | **지금 가능** | 기존 audit/seal 코드 재사용 (Gym spec 참조) |
| L1 | OlmoEarth 지역 head/adapter SFT | **지금 가능** | E5 recipe 동결 후. 공식 경로(LayerDecayAdamW) 존재 |
| L2 | supervised action ranker | **matrix 후 가능** | E6/E8의 utility label. M40 oracle은 자격 없음 |
| L3 | contextual bandit (offline) | **matrix 후 가능** | 단일 단계 결정이므로 full RL 불필요 |
| L4 | offline RL (cache 나이 누적) | 후속 | multi-step 로그가 없음. v0 Gym은 단일 단계 |
| L5 | LLM tool-use SFT/LoRA (8B급) | **이번 사이클 제외** (사용자 결정) | L2 label 확정 + Gym 구현 후에만 의미 있음 |
| L6 | GeoFM continued pretraining / 70B full post-train | CSCS급 인프라 영역 | 이 저장소 범위 밖 |

## 왜 L2가 지금 안 되는가 — 오염된 label 문제

M40의 oracle은 (a) 지표 불일치(tile-IoU 선택 + micro 보고), (b) 잡음 바닥 미차감,
(c) FP율 불일치 상태에서 계산됐음. M41이 (a)(b)를 고쳤고 (c)는 확률맵(E5b) 대기 중임.
**이 상태의 "최고 action"을 SFT 정답으로 쓰면 seed 운과 오경보 축 차이를 학습하게 됨.**
E5 통과본 utility만 label 자격이 있음.

## 왜 full RL이 아니라 ranker→bandit 순서인가

현재 문제는 한 상태에서 한 action을 고르는 **단일 단계 결정**임. PPO류는 credit assignment가
필요한 다단계 문제의 도구이고, 단일 단계에서는 supervised ranker와 bandit이 같은 것을 더
적은 분산으로 학습함. cache 나이가 시간에 따라 누적되는 로그가 쌓인 뒤에야 L4가 의미를 가짐.

## Apertus 방법론 대응 (아이디어 기록 — 미실행)

| Apertus post-training | EarthRoute 대응 | 상태 |
|---|---|---|
| SFT dataset mixture | E6/E8 action matrix → EarthRoute-SFT v0 | 스키마만 (E6) |
| preference optimization | utility 차이가 잡음 바닥을 넘는 action 쌍만 chosen/rejected | 설계만 |
| RLVR | Gym verifier reward (전 항목 자동) | 명세만 (GYM_SPEC) |
| synthetic gym tasks | 시나리오 S1~S10 (전부 M-기록 근거) | 명세만 |
| reward calibration | Δ지표 − λ·FLOPs − 위반 페널티 | λ 후보 미등록 |
| tool-use eval | STAC/KMA 호출 + verifier 통과율 | 미구현 |

기술적 겹침이 실재함: 이 저장소의 seal·checkpoint·replay·cost audit 흐름은 post-training
저장소의 재개 가능한 실험 관리와 같은 부류임. 다만 겹침은 **주장이 아니라 이력서 재료**이고,
논문 기여는 L2~L3(라벨 없는 action utility 예측)에 있음.

## 이 판정이 틀릴 수 있는 지점

- E8에서 세 task의 action 순위가 같게 나오면 L2~L5 전체가 무의미해짐 (router 불필요).
  그 경우 L0~L1만 남고 논문 축은 계약 감사로 이동함.
- E9에서 out-of-sample regret이 다수결을 못 이기면 L2에서 중단함.
