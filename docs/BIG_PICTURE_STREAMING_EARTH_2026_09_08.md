# 큰 그림 — 저장된 지구 표현을 새 관측으로 갱신한다

> **현재 상태는 [2026-09-09 최신판](STREAMING_RESEARCH_UPDATE_2026_09_09.md)으로 이동했다.**
> 아래는 9/8의 설계·최초 NaN 실행 감사다. 이후 Kuro updater 재실행의 양성 수치와 남은
> decoder 검증 조건, 교차센서/이탈리아 결과는 새 문서가 설명한다. 원 기록은 보존한다.

작성: 2026-09-08. 사용자 요청에 따른 **큰 그림·설계 검토**다. 최신 로컬 HEAD `1edbacc`의
MS-115/116 장부, `streaming_update_train.py`, `t1_cost_measure.py`, KuroSiwo prereg 초안 및
공급자 원문을 읽었다. 이번 검토에서 서버 결과를 새로 다운로드해 전수 재계산하거나 비용을
재측정하지 않았다. 아래 수치는 장부 보고이고, 비용 측정 범위에 관한 지적은 코드 직접 확인이다.

**현재 집중 질문과 다음 작업에 대해서는 이 문서가 9/7 큰 그림과 9/8 00:03 인수인계를 대체한다.**
과거 결과·사전등록 gate는 바꾸지 않는다. 이 문서는 새 실험의 사전등록이나 실행 승인이 아니다.

## 1. 사용자에게 설명할 한 문장

> **OLMoEarth가 읽어 놓은 과거의 지구 정보를 버리지 않고, 새 위성 관측만 처리해서
> 현재의 지도를 갱신할 수 있는가? 정확도를 얼마나 유지하며 실제로 무엇을 절약하는가?**

예를 들어 같은 계곡의 영상이 계속 추가된다. 기존 방식은 누적 영상 전체를 매번 다시
OLMoEarth에 넣는다. 현재 학생 방식은 새 영상의 단일 시점 임베딩을 만들고, 저장 상태와
합쳐 작은 순환 갱신기(GRU)로 업데이트한다. 그 상태를 기존 판독기가 읽는다.

- 고정하는 것: 이번 실험의 OLMoEarth 인코더 가중치와 평가 판독기.
- 학습하는 것: source 지역의 전체 창 임베딩을 교사로 삼는 갱신기. 현재 기본 목적함수는 feature MSE.
- 줄이려는 것: 반복 요청 때 과거 원시 영상의 재인코딩. 새 영상 자체의 인코딩은 여전히 필요하다.
- 하지 않는 것: 미래 산사태 예측, 물리 시뮬레이션, 원본 OLMoEarth 가중치의 직접 개선,
  정확한 Transformer KV-cache 구현, 센서 관측/제품 게시 지연 제거.

넓게는 **동결된 EO 모델 위에 갱신 능력을 학습시키는 post-training/distillation**이다.
LLM의 RLHF나 OLMoEarth 자체를 재사전학습하는 것과는 다르다.

## 2. 기존 연구와의 연결 — 프로그램은 넓게, 이번 논문은 좁게

| 자산 | 답한/답할 질문 | 현재 역할 |
|---|---|---|
| Sen12·Solar frozen cache 및 few-shot | 다른 지역의 적은 라벨로도 표현을 재사용할 가치가 있는가? | 기존 empirical 기반. 조건부 결과와 철회는 유지 |
| family·입력·해상도 감사 | 어떤 표현/입력 조건에서 그 가치가 유지되는가? | 적용 범위와 대조군 선정 근거. 모든 실험을 본문 기여로 넣지 않음 |
| **MS-116 갱신형 캐시** | 새 관측을 받으면 과거 전체를 다시 읽지 않고 상태를 갱신할 수 있는가? | **이번 집중 질문의 개발 증거** |
| **KuroSiwo** | 새 사건 후 SAR 관측으로 홍수 지도를 갱신할 때도 방법이 작동하는가? | **다음 외부 과업 검증** |
| Korea shared-cache 3-task | 갱신 상태 하나가 서로 다른 판독기/과업에도 쓸모 있는가? | 후속 공유 비용·다중 과업 검증. 시간별 라벨 계약은 별도 확인 |
| FoldRefresh·release bridge | 세계가 아니라 모델 버전이 바뀌었을 때 호환성을 어떻게 유지하는가? | 별도 검증 자산. 시간 갱신의 성공으로 합산하지 않음 |
| Nepal 앱 | 이 정보를 사용자에게 어떻게 전달하고 검수할 것인가? | 별도 저장소의 응용·설명 자산. 이번 학습/확증으로 소급 편입하지 않음 |

장기 미션인 “환경 조직이 변화하는 지도를 적은 컴퓨트로 유지한다”는 그대로다.
이번 논문에서 모든 release·sensor·task·region shift를 한 번에 해결한다고 하지 않는다.
**모델 버전은 고정하고 관측이 추가되는 시간 축부터 닫는다.**

## 3. MS-116에서 실제로 진전된 것

4개 노출 지역에서 GRU가 단순 평균/EMA와 관측 없는 대조보다 훨씬 많이 teacher downstream을
회복한 것은 의미 있는 개발 신호다. 시험한 no-observation GRU·calibration 대조로 설명되는
회복은 작아, **현재 비교 안에서는 새 관측을 활용한 학습된 갱신의 가치**를 지지한다.
이것은 “모든 비학습법은 불가능” 또는 “분포 보정은 전혀 기여하지 않는다”의 증명이 아니다.

남겨야 할 조건:

1. **판독기 3개는 같은 과업의 3seed다.** 3과업도, 4×3×3개의 독립 지역도 아니다.
2. 대표 열은 판독기 seed1 기준이다. Hiroshima도 seed2 회복률은 .86이고 Chimanimani는
   .64–.77이다. “모든 판독기에서 90% 이상”으로 쓰지 않는다. 원래 gate의 실패는 보존한다.
3. New Zealand의 123%는 오류라고 삭제할 값이 아니다. `.466 > .433`으로 학생이 teacher의
   AP를 넘은 것이다. teacher는 기준이지 참성능 상한이 아니다. 회복률과 함께 절대 AP/gap을 낸다.
4. 현재 핵심 task score는 **마지막 c12**다. teacher의 c6/c8/c10과 feature를 맞춘다고 모든
   중간 시각의 정확한 사건 지도가 입증된 것은 아니다. 사건 후 라벨을 사건 전 정답처럼 쓰지 않는다.
5. 원 residual의 패배는 그 구조/recipe의 실패다. 표준 GRU의 성공을 독자 구조의 성공으로
   바꾸지 않는다. 표준 GRU를 앞으로의 강한 기준으로 유지한다.
6. `--aux-decoder-loss`는 현재 코드상 **frozen decoder logit MSE**다. 이미 Chimanimani에서는
   개선, Hiroshima에서는 악화했다. “판독 보존은 미실험” 또는 “모든 task-aware loss 실패”가 아니다.

MS-115의 실제 날짜 native 팔은 현재 장부에 7/8로 남아 있지만 6/8 승리 gate는 이미 도달
불가능하다. 최종 파일 집계 전에 native 8/8 완결을 독립 확인했다고 쓰지 않는다. T0/T0b의 실패도
시험한 readout/학습 설정의 실패이지, 시간 정보는 원리상 쓸모없거나 GRU에서만 유용하다는 증명이 아니다.

## 4. 비용에서 지금 바로 구분해야 하는 것

`code/t1_cost_measure.py`의 동작을 직접 읽었다.

| 보고 | 코드가 실제로 측정/계산한 범위 | 현재 허용 표현 |
|---|---|---|
| 0.172→0.074 s, 2.3배 | A는 cutoff 6/8/10/12의 창을 차례로 처리. B는 새 시점 **8개×4크롭을 한 번에 배치**한 후 4회 GRU. 원시 배열은 타이머 밖에서 먼저 로드 | 전체 새 관측을 확보한 상태의 배치 처리 compute 이득. 순차 도착 온라인 latency는 아직 미측정 |
| raw 11.8→2.6 MB, 4.5배 | `raw.nbytes / T × 시점 수`로 계산한 **논리적 입력량**. 실제 파일/네트워크 read 계측 아님 | 초기 상태가 이미 있을 때 update-only 입력량 36/8=4.5배 |
| 초기 포함 비용 | 양쪽 초기 4시점을 더하면 입력량 40/12 | 논리적 입력량은 초기 포함 **3.33배**. 4.5배와 workload를 나눔 |
| 최종 지도만 0.043 s | 12시점을 한 번 인코딩하는 별도 workload | 최종 한 번만 필요한 고객에게 streaming이 반드시 더 싸다는 근거 없음 |

시점별 인코딩이 독립이므로 전체를 미리 배치했다는 사실만으로 학생의 평가 입력에 미래가
섞였다고 판정하지 않는다. 문제는 **중간 도착 시각에 아직 없는 관측까지 묶어서 얻은 처리 효율을
그 시각의 서비스 지연으로 사용할 수 없다는 것**이다. 반복 경로와 일괄 backfill은 다른 workload다.

다음 비용 검증은 새 설정에서 다음 순서로 한다:

1. 초기 상태를 동일하게 준비하고, cutoff마다 새 2시점만 제공한다.
2. 새 관측 인코딩 → GRU → 판독 → 해당 시각 응답 완료를 계측한 뒤 다음 관측을 제공한다.
3. 여러 타일은 **그 시각에 이미 도착한 타일끼리** 묶을 수 있다. teacher에도 같은 batch/대기
   예산을 허용한다. 일괄 backfill throughput도 별도 보고한다.
4. 인코딩/갱신/판독/실제 raw 및 cache I/O, 초기 상태 구축, peak VRAM, update별 분포를 분해한다.
   하드웨어·외부 GPU 작업·warmup·샘플 목록을 기록한다. 현 `gpu_procs_at_start`는
   `torch.cuda.device_count()`라 GPU 프로세스 존재를 증명하지 못한다.
5. 벽시계 비용과 동일 시점의 task/teacher fidelity를 짝짓는다. 단순 시점 수 비율에서 GPU 속도를
   외삽하거나 최종 AP만으로 중간 지도 품질을 보장하지 않는다.

기존 시간 수치를 없애는 것이 아니라 **batch-compute 결과로 보존하고 온라인 주장은 따로 검증**한다.

## 5. KuroSiwo의 역할과 동결 전 수정 항목

공급자에 따르면 KuroSiwo는 사건 전 S1 2장·사건 후 1장과 홍수 라벨을 제공한다.
43은 전체 사건 수이며 원본 BlackBench는 10사건을 test로 지정한다. 현재 쓰는
GEO-Bench-2 재가공 split도 같은 독립성을 유지하는지는 별도 확인해야 한다.
[원본 데이터 설명](https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset),
[GEO-Bench-2 재가공본](https://huggingface.co/datasets/aialliance/kuro_siwo).

현재 `config/kurosiwo_streaming_prereg_v0_draft.json`은 DRAFT다. 아래는 동결 전 보완 제안이며
본 검토에서 JSON이나 실행 코드를 바꾸지 않았다.

**오후 상태 변경:** 초안은 `d78abeb`에서 frozen으로 등록되고 추출/학습까지 진행됐다.
따라서 아래 동결 전 권고를 이미 반영된 것처럼 읽지 않는다. 실제 반영 여부와 실행 무효 건은
이 문서 끝의 **§8 오후 재감사**가 최신이다. 기존 frozen 파일은 소급 수정하지 않는다.

| 보완 | 이유와 완료 조건 |
|---|---|
| **POST_ONLY 추가** | 새 영상 단독 표현 위에서 source로 학습한 판독기를 비교한다. 기존 3시점 head에 single을 넣는 비교만으로 대신하지 않는다. 이것이 충분하다면 과거 상태의 추가 가치가 작을 수 있다 |
| 사건·AOI·좌표 기반 분리 | actid/aoiid/공간 중첩을 train/val/test에서 확인한다. tile 수나 43개 전체 사건을 독립 test 수로 세지 않는다. primary는 event-macro와 사건별 결과, pooled AP는 함께 보고 |
| 실제 관측 날짜 | 초안의 `flood_date−24/−12/0일`은 사건일 기반 가상 날짜다. 원본 메타데이터로 취득일을 찾는다. 복구 불가이면 모델이 지원하는 시간정보 부재 처리/명시적 대체 계약과 sensitivity를 동결하고 실제 시각 실험이라 하지 않는다 |
| S1 단위·NoData·라벨 | 제공 loader의 rescale 여부를 먼저 확인하고, 유효한 선형 backscatter에만 dB 변환을 적용한다. NoData를 작은 값으로 바꿔 물 신호로 학습하지 않는다. 원본 0/1/2와 재가공 4class mapping을 혼동하지 않는다 |
| crop·head·gate 결정 | 192 중앙 crop 여부, binary flood head인지 multiclass인지, common evaluation mask, 선택 규칙, 비용 workload를 동결한다. teacher−stale gap이 작거나 음수인 사건의 회복률 처리도 미리 정한다 |

최소 비교는 `full-window teacher / stale / POST_ONLY / singles mean / GRU / noobs GRU`다.
EMA는 기존 코드 재사용이 가능한 단순 기준으로 유지할 수 있다. 원래 초안의 pooled-80% 성공
기준을 채택할지 수정할지는 **실행 전** 정하며, 이 문서의 event 보고 제안을 사후 새 gate로 쓰지 않는다.

### 이것을 어떤 transfer라고 부르는가

- KuroSiwo train에서 SAR 갱신기와 판독기를 새로 학습하고 미본 사건에 평가하면:
  **다른 센서·과업에서 같은 방법의 재현 + 사건 일반화**다.
- Sen12에서 학습한 갱신기 가중치를 그대로 SAR에 적용해야:
  **같은 가중치의 S2→S1 zero-shot transfer**다. 현재 초안은 이것이 아니다.
- 3시점은 한 번의 post update다. 따라서 KuroSiwo 하나로 수십 회 갱신의 오차 누적,
  계절 전이, 수개월 운영 안전성을 검증했다고 하지 않는다.
- cold-start 총량은 teacher 3시점, 학생 초기 2+새 1시점이다. 초기 cache가 없다면
  시점 수 자체의 절약은 없다. 초기 cache가 이미 존재할 때의 update-only 비용은 별도로 비교한다.

KuroSiwo는 적절한 다음 시험이지만, 단 한 과업으로 “실시간·센서 전이·장기 안정성·저비용”을
동시에 완결하는 만능 시험은 아니다.

## 6. 노벨티와 개선 목표를 어디에 둘 것인가

최근 [Temporal Sensitivity Analysis of Tessera Embeddings (2026-08-27 preprint)](https://arxiv.org/abs/2608.27175)는
동결 인코더의 시간 창을 줄여 다시 계산하고, 과업별 성능/시간 범위의 관계를 분석한다.
이는 **지구 임베딩의 시간적 유효성과 갱신 빈도**가 관련성 높은 문제임을 뒷받침한다.
그러나 “EO 임베딩의 시간 길이를 바꿔 보았다”만으로 새 기여라 하기 어려워진다.

영상에서는 [Deep Feature Flow (2017)](https://arxiv.org/abs/1611.07715)가 비싼 특징 계산을
줄이고 특징을 전파했다. 따라서 “특징 재사용 + 작은 모듈” 자체도 신규성으로 삼지 않는다.
이 문헌 대조는 포괄적 novelty search나 최초성 증명이 아니다.

우리가 검증할 구체적 차이는:

> **불규칙하게 들어오는 실제 EO 관측 아래, 과거 원시 영상 없이 저장 상태를 갱신하여
> frozen downstream의 현재 task utility를 보존하고, 같은 도착 조건의 전체 재인코딩보다
> 비용–품질 관계를 개선하는가?**

이는 아직 **목표 주장**이다. 표준 GRU 4지역 utility만으로 완성되지 않는다.
KuroSiwo와 온라인 비용을 먼저 닫은 후에도 개선 여지가 남는다면, 강한 GRU에서 출발해
**반복 갱신 오차 누적** 또는 **같은 상태의 미본 판독기/과업 보존** 중 실제로 관찰된 병목 하나를
개선한다. 주기적 full refresh·짧은 창 재인코딩·POST_ONLY를 비용 대조로 둔다.
기존 mixed logit-MSE 결과만으로 새 loss가 성공할 것이라 예고하지 않는다.

selector·MoE·새 이름을 붙이는 것을 목표로 하지 않는다. 적응형 reset 정책도 고정 주기
reset보다 나아질 여지가 실제 측정될 때만 연다. 실패한 과거 G0를 되살렸다고 하지 않는다.

CVPR 판정은 “GRU 하나 통과/실패”로 자동 결정하지 않는다. 현재는 명확한 운영 문제와
강한 개발 baseline을 확보한 단계다. 본회의 주장의 질은 **비자명한 개선/통찰, 독립 사건·과업
일반화, 실제 비용–품질 검증**의 결합으로 판단한다. 채택 확률이나 특정 학회 등급을 수치로 약속하지 않는다.

## 7. 작업 순서와 세 목표의 연결

1. **지금:** MS-116의 완료 산출물·decoder별 AP/gap을 고정 표로 정리한다. 이미 끝난 EMA/noobs/
   checkpoint 저장·판독 보존 실험을 새 후보처럼 다시 예약하지 않는다.
2. **다음 검증 묶음:** 도착 스케줄을 맞춘 비용 측정 계약 + 위 KuroSiwo 미결정 항목을 확정한다.
   기존 queue는 건드리지 않고 새 OUTROOT에서 실행하는 후속 작업으로 준비한다.
3. **그 다음:** source에서만 선택한 recipe로 미본 홍수 사건을 평가한다. teacher가 사건 라벨에
   유효한지, post-only보다 상태를 유지할 이유가 있는지, GRU가 그 utility를 보존하는지 순서대로 본다.
4. **후속 확장:** 누적 갱신 안정성 또는 Korea의 서로 다른 task head 보존을 검증한다.
   Korea는 정적인 3task cache sharing과 시간별 task update를 구분하고 봉인 계약을 지킨다.

| 우선 목표 | 이 경로에서 낼 산출물 |
|---|---|
| Ai2 취업 | OLMoEarth/rslearn에 연결 가능한 갱신 컴포넌트, 명확한 입력·상태 버전, 재현 recipe와 비용/실패 리포트. upstream 채택은 별도 외부 검증 |
| 박사·논문 | 지리 transfer에 더해 temporal update의 적용 범위와 실패 원인을 설명하는 독립 사건 평가 |
| 사업 | 새 관측 게시 후 지도 갱신까지 걸린 시간·반복 처리 비용·사용자 검수량. 아직 고객 가치/수익성 실측은 없음 |

추천 working headline: **Updating Earth Embeddings Without Reprocessing the Past**.
한국어로는 **“과거를 다시 읽지 않고, 지구 지도를 갱신한다.”**
현재 가능성의 중심은 이 한 문장이고, 아직 측정하지 않은 서비스 보증까지 넣지 않는다.

## 8. 2026-09-08 오후 재감사 — 실제 진전과 실행 무효를 분리한다

최신 로컬 HEAD `ac36ed5`, 서버 15:25–15:30 KST 조회 및 보존본 기준이다. 이번에는 서버에
접속해 **기본 갱신 36개·새 구조 22개 완료 JSON**을 직접 받아 재집계했다. teacher/stale AP와
보고 split count의 일치를 확인했고, 원시 예측 재채점이나 공간 CI 재계산은 하지 않았다.
재현: `code/audit_streaming_progress.py`; 근거: `artifacts/streaming_review_20260908_1530/summary.json`.

### 8.1 현재 상태

| 작업 | 확인한 결과 | 허용 판정 |
|---|---|---|
| MS-116 기본 GRU/EMA/noobs | 4지역×3arm×3seed=36 JSON 정상, 기존 대표 결과 재현 | 기존 streaming utility 유지; 이번 재집계는 판독기 seed1 |
| 새 갱신 구조 | Δt·spatial·xattn 중 22/36 완료, GPU1 runner 관측 | 완료된 3seed cell만 비교; 미완료를 음성으로 세지 않음 |
| KuroSiwo S1 추출 | 7,000 완료 로그, meta 7,000 unique IDs | 메타데이터/추출 완료; 전체 cache content 봉인까지 뜻하지 않음 |
| KuroSiwo 사건 분리 | train/val/test = 27/6/10 actid, 세 split 모든 쌍에서 ID·actid 중복 0 | 사건 ID 분리는 재확인. 다른 사건의 동일 공간 중첩까지 감사한 것은 아님 |
| KuroSiwo 판독기 | 3개 checkpoint 존재, validation best AP .66209/.66410/.68755로 출력 | 검증 입력의 NaN 오염을 뒤늦게 발견했으므로 AP/epoch 선택 재검토 필요. test 결과 아님 |
| **KuroSiwo 갱신기** | **GRU seed1 val NaN 30epoch, best epoch0인데 DONE/rc0** | **실행 무효. 방법의 외부 음성으로 세지 않음** |
| KuroSiwo 최종 평가 | eval JSON·DONE marker 없음, 조회 때 해당 chain/process 없음 | 현재 유효한 외부 갱신 결과 0. 중단 경위는 미확인 |
| 온라인 비용 | `t1_cost_measure.py` 로컬/서버 SHA 동일, 오전과 같은 코드 | 도착 조건을 맞춘 재측정은 아직 없음 |

### 8.2 새 구조는 어떻게 읽는가

표의 AP는 판독기 seed1에 대해 갱신기 3seed 평균이다. 서로 다른 실행의 수치를 섞지 않고
각 fold의 teacher/stale가 일치하는 기본 GRU와 비교했다.

| test 지역 | 기본 GRU AP | 시간 간격 GRU AP | AP 변화 | 회복률 변화 |
|---|---|---|---|---|
| Hiroshima | .52548 | .53772 | +.01224 | +2.28%p |
| Thrissur | .55715 | .56938 | +.01222 | +2.04%p |
| Chimanimani | .22426 | .23278 | +.00852 | +3.05%p |

**Δt 조건은 완료 세 지역의 평균을 소폭 올렸지만, 등록 조건(+5%p 회복을 3/4지역)은 아니다.**
spatial GRU는 완료 Hiroshima/Thrissur에서 AP −.03745/−.01721, local xattn은
−.13029/−.13818이다. 이 두 arm도 현재 완료 두 지역에서는 gate 기준을 못 넘었다.
따라서 기존 결과가 유지되는 한 세 후보 모두 남은 지역을 전승해도 3/4 승리에 도달하지 못한다.
이것은 **완료된 cell에서 계산한 승격 불가능 상한**이지, 남은 실행을 완료/음성으로 처리한 것이
아니다. 감사자는 runner를 중지하거나 gate를 낮추지 않았다.

실무적으로 Δt의 작은 일관된 평균 개선은 보존할 만하다. 그러나 CI·독립 지역 검증 없이
보편적 개선/새 방법 성공으로 쓰지 않는다. “GRU 외에는 방법이 없다”는 결론도 아니다.

### 8.3 KuroSiwo에서 먼저 고칠 결함

1. **비정상 수치의 성공 처리** — `kurosiwo_pipeline.py:61–78`: scale/loss/gradient/val에
   finite 검사와 best-state 유효성 검사가 없다. NaN이면 `v < best`가 항상 False라
   `best=(1e9,None,0)` 그대로인데 저장과 DONE 경로로 간다. 실제 seed1 로그에서 재현됐다.
2. **chain fail-open** — `kurosiwo_chain.sh:7–10`: 하위 실행 rc를 기록만 하고 다음 실행으로
   넘어가며 마지막에 무조건 DONE marker를 만든다. 최종 검증이 통과해야 marker를 만들도록
   고쳐야 한다. 현재 marker가 실제 생성됐다는 주장은 아니다.
3. **불완전 평가 허용** — `kurosiwo_pipeline.py:97–98`: 없는 updater checkpoint를 조용히
   제외한다. partial report라면 누락을 명시하고 전체 3×3 결과의 완료 판정은 거부해야 한다.
4. **회복률·FAR** — `:90`의 no-water 분모는 invalid 픽셀을 빼지 않는다. `:106`은
   teacher−stale가 0/음수여도 1e-9로 바꿔 거대한 회복률을 낼 수 있다. 유효 마스크를 분자/분모에
   같이 적용하고, 양의 teacher headroom이 없으면 회복률을 정의 불가로 보고해야 한다.
5. **재시작 안전성** — 추출기는 `single+teacher` 두 파일만으로 skip하고 metadata는 마지막에
   한 번 append한다. 중도 종료 후 재시작에서는 내용 파일과 metadata가 어긋날 수 있다.
   현재 보존본 meta가 7,000 unique라는 사실과 미래 resume 안전성을 구분한다.

별도 review 경로에서 GPU 없이 CPU 진단을 실행했다. **훈련 teacher 4,000개, 총
7,077,888,000개 값이 finite**이고, 실제 코드의 전체 `torch.std()` = `.4885376692`,
float64 chunk 계산 = `.4885376708`로 일치했다. 이 실행에서 scale 자체가 NaN이라는 설명은
배제된다. 원 실행은 batch loss/gradient를 저장하지 않았으므로 NaN의 최초 발생 위치는 아직
확정하지 못했다. 정상 scale이라고 optimizer 경로까지 정상이라는 뜻은 아니다.

**후속 전수 입력 진단으로 직접 원인을 찾았다.** train4,000/validation1,000의 teacher/stale/post
캐시를 전부 finite 검사한 결과, train은 정상이고 validation **`ks_04357` 한 타일**이 세 종류
모두 비정상이다. 이 타일의 export 원시 영상에 nonfinite 1,338개(시점별 NaN444/446/448)가
들어 있고, 현 `db()`의 clip/log는 NaN을 제거하지 않는다. 각 teacher/stale의 589,824개 값,
공간768토큰이 비정상이며 이 중 **327토큰은 유효 라벨 픽셀을 포함**한다. 중앙 crop 유효 픽셀은
20,687/36,864이므로 “통째로 빈 타일이라 제외”로 처리할 수 없다.

`vloss()`는 이 오염 teacher를 마스킹 없이 MSE에 넣어 모든 epoch의 val NaN을 만들 수 있다.
**저장 검증 입력 오염만으로 현재 validation 실패를 설명하는 충분한 경로**를 확인했다.
원 학습 trajectory/gradient는 저장되지 않았으므로 optimizer까지 항상 정상이라고 주장하지 않는다.
판독기의 exact AP도 finite score 검사가 없어 숫자가 출력됐다는 것만으로 정상 평가를 보장하지
않는다. 오염 입력이 유효 라벨 영역까지 번졌으므로 판독기 validation AP/epoch 선택도 재검토한다.

근거: `numeric_input_scan.json`, `invalid_validation_tile.json`(동일 snapshot 폴더).
test 입력/라벨은 이번 진단에서 검사하지 않았다. 원본 tortilla와 export 간 손상 발생 위치는
아직 대조하지 않았으므로 공급자 원본의 오류라고 지목하지 않는다.

복구 순서: 원본/export 해당 타일 확인 → 영상 finite/valid-mask 처리 규약 명시(임의 0·−30dB를
실관측으로 취급하지 않음) → 해당 입력의 teacher/stale/single을 **새 artifact revision**에서
재추출·전수 finite 검사 → decoder 검증/선택 및 updater 학습 재검증. 입력 검증은 모델 성능을
보기 전 모든 split에 적용하되, split/라벨을 결과에 유리하게 바꾸지 않는다. 이어서
**입력→출력→MSE→gradient→optimizer 후 parameter** finite 검사와 유효 checkpoint/완료 검사를
추가한다. 기존 캐시/실패 로그는 보존한다. 단순히 validation NaN을 0으로 바꾸면 안 된다.

### 8.4 오전 권고 반영 현황

- **반영:** 사건 ID 분리 감사, S1 dB 변환, 라벨 mapping/invalid 정의 확인, binary head·3seed.
- **미반영:** POST_ONLY 대조, 실제 취득일 확보, 도착 스케줄 비용, headroom 분모의 안전한 처리.
- **문서/구현 차이:** frozen JSON crop에 `DECISION PENDING`이 남지만 코드는 중앙192 crop을
  실행한다. 추출 코드는 0값을 −30dB로 넣고 실제 missing mask로 전달하지 않는다. “NoData를
  정상적으로 mask했다”와 같지 않으므로 후속 입력 품질 감도와 구분해야 한다.
- **보존 원칙:** 이미 실행된 frozen v0는 유지한다. 구현 결함 보정과 POST_ONLY 추가는 별도
  amendment로 기록하고, 원래 gate와 새 비교의 보고 위치를 명시한다.

### 8.5 큰 그림은 유지, 다음 행동만 더 구체화

현재 우선순위는 **KuroSiwo의 정상 학습 복구 → 유효한 외부 사건 평가 → POST_ONLY/공정 비용의
운영 가치 확인**이다. 이미 관측된 시간 간격 효과는 작은 개선 후보로 보존하고, 승격 gate를
넘지 못한 공간/attention 가지를 다음 과업으로 자동 확장하지 않는다.

즉 오늘의 정직한 업데이트는 “전부 잘 됨”도 “또 연구가 망함”도 아니다.
**OLMoEarth S1 추출·판독기 실행과 사건 분리는 진전했고, 외부 성능은 특정 검증 입력 오염을
복구한 뒤 판정해야 한다. Sen12의 갱신 utility는 유지된다.** Korea 라벨 봉인과 Nepal 앱은 그대로다.
