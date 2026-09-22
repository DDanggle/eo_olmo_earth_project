# 메타정보 조건부 공간 기억과 근거 기반 판단 갱신 — CVPR 구현 청사진

작성일: 2026-09-18. 상태: **설계 제안, 구현·학습·새 평가 미실행**.

현재 코드는 재사용 가능한 부품이지 이 가설의 성공 증거가 아니다. 기존 gate 실패, prereg,
실험 기록은 그대로 둔다. 이 문서는 `RESEARCH_DESIGN_TIME_AND_LANGUAGE_2026_09_18.md`나
실행 중인 실험의 계약을 대체하지 않는다. 전체 문헌 정독·재현·최초성 검증도 완료되지 않았다.

## 1. 논문 질문을 한 문장으로 고정

> 불규칙하고 품질이 다른 위성 관측이 들어올 때, VLM은 제한된 공간 기억으로
> 과거 사건의 근거를 유지하면서 현재 판단을 적절하게 수정·보류할 수 있는가?

사용자가 원하는 지역 VLM을 본체로 삼는다. 시간 보간, PDE, 전 지구 FM 재사전학습,
로봇 actuator, 법적·도덕적 환경파괴 판단은 첫 논문의 필수 요소가 아니다.
추론 때 매 관측마다 가중치를 다시 학습하는 대신, offline으로 updater/reader를 학습하고
test에서는 가중치를 고정한 채 기억을 갱신하는 계약부터 검증한다.

연결 경로는 `OlmoEarth 공간 토큰 → 메타 조건·좌표 대응 → 공간/사건 기억 → 근거 검색·접지 → VLM 답과 판단 갱신`이다.
논문 기여 후보는 이 경로 전체를 붙였다는 사실이 아니라, **품질에 따른 국소 쓰기와
사건 근거를 잃지 않는 제한 용량 기억을 연결하는 방법**, 그리고 그것을 검증하는 순차 평가다.

## 2. 무엇을 어느 선행연구와 연결하는가

| 연결할 기능 | 가장 직접적인 선행연구 | 우리가 추가 검증할 부분 |
|---|---|---|
| 영상 토큰에 관측 맥락을 조건으로 제공 | [Atomizer](https://arxiv.org/abs/2506.13542), [Copernicus-FM](https://arxiv.org/abs/2503.11849) | 조건이 최신 답의 shortcut이 아니라 국소 쓰기·보류에 기여하는가 |
| 시각 정보와 의미 정보를 기억하고 선택적으로 융합 | [MemoryVLA](https://arxiv.org/html/2508.19236) §3 | 유사 관측을 합칠 때 드문 사건의 전후 근거가 사라지는 EO 실패를 줄이는가 |
| 위치·시간·시각 기억을 질의로 검색 | [RAVEN](https://arxiv.org/html/2606.25206v1) §3–4 | 실제 지리좌표·불규칙 관측·관측 신뢰도·근거의 생성 시점을 함께 다루는가 |
| EO 시계열을 언어 질의에 정렬 | [TEOChat](https://arxiv.org/abs/2410.06234) | 정적 시계열 QA를 넘어, 같은 기억에 새 관측을 넣었을 때의 답 변화·망각을 평가 |
| 답과 영상 속 근거 영역을 연결 | [TerraScope](https://arxiv.org/html/2603.19039) §3.2, Appendix A | 긴 순차 입력에서 근거 영역뿐 아니라 근거 관측 ID·시기·판단 수정의 타당성을 평가 |

Atomizer의 시간·해상도·파장 조건, Copernicus-FM의 유연한 metadata encoding은 이미 선행 아이디어다.
MemoryVLA에는 perceptual/cognitive memory, gated fusion, 인접 유사 항목의 평균 병합이 있다.
RAVEN은 검색한 원영상도 VLM에 제공하며, 텍스트 검색에 정렬된 multimodal embedding을 사용한다.
이것이 OlmoEarth latent와 텍스트의 무학습 cosine 검색을 보장하지는 않는다.

TerraScope는 mask로 고른 시각 특징을 언어 생성에 다시 넣는다. 따라서 mask head를 붙이는 것만으로는 새롭지 않다.
저자들은 Appendix A에서 bi-temporal 분석의 한계, 긴 시계열 확장의 필요성과 환각 가능성을 직접 명시한다.
이는 연결할 연구 공백의 단서이지, 우리가 이 공백의 최초 해결자라는 증명은 아니다.

읽기 범위: 이번 방향 검토에서 MemoryVLA/RAVEN/TerraScope의 위 지정 본문 절을 확인했다.
Atomizer/Copernicus-FM/TEOChat은 초록·서지 확인 수준이며 이 청사진에서 학습 세부를 그대로 복제하지 않는다.
광범위한 문헌 목록과 읽기 상태는 [별도 감사](REGIONAL_CONTINUOUS_EO_LITERATURE_AUDIT_2026_09_18.md)에 있다.

## 3. 현재 자산에서 재사용할 것과 변경할 것

| 실제 파일 | 재사용 가능한 부분 | 새 실험에서 별도로 바꿔야 하는 부분 |
|---|---|---|
| `code/extract_olmo_streaming.py` | frozen OlmoEarth 단일 관측 공간 map, 기존 추출 경로 | 관측 ID·정확한 시각·QA mask·좌표/affine·encoder hash를 포함한 manifest와 cache 검증 |
| `code/streaming_update_train.py` | EMA, GRU, Δt-GRU, spatial GRU, local XAttn, 재귀 rollout | 두 관측 평균이 아니라 관측별 순차 갱신; metadata/quality; VLM·근거 loss와 reader 연결 |
| `code/task_aware_update_candidate.py` | finite/valid 입력 검사, 품질·시간 조건 residual, hard no-write | 미학습 engineering candidate이므로 baseline부터; 근거 보존이나 VLM 효용은 별도 검증 |
| `code/earthtalk_projector_train.py` | 768차원 EO→LLM projector, 출력 scale 안정화 | XY/시간/품질 조건, 공간/사건 memory reader, evidence pointer, 순차 QA 학습·평가 |
| `code/sentinel_qa_gen.py` | 날짜별 질문과 기존 mask 기반 약한 QA 생성 | prefix별 answerability·판단 상태·근거 관측/영역 annotation, strict causal split |
| `code/build_sen12_gp_contract.py` | timestamp/라벨/SCL 기본 계약 검사 | 정확한 token→지도 변환 검증, arrival schedule, 관측별 annotation 계약 |

중요한 코드 마찰:

- `streaming_update_train.py`는 새 두 관측을 `S[:, c-2:c].mean(1)`으로 합친다.
  관측의 순서·서로 다른 품질을 지운 후 들어오는 updater로 근거 관측별 판단 갱신을 주장할 수 없다.
- Δt는 현재 `.days`로 만들어진다. 새 계약에서는 초 단위 차이를 명시적 단위로 정규화한다.
  그렇다고 현재 자료가 시간 단위 관측을 제공하는 것은 아니다.
- 현재 추출/QA/projector의 12장 선택은 전체 15장의 SCL을 보고 가장 맑은 장을 고른다.
  이는 retrospective subset이지 실제 도착 시점에서 이용 가능한 정보만 쓰는 운영 프로토콜이 아니다.
  새 순차 실험은 미래 영상 품질로 과거 입력을 선택하지 않는다.
- 기존 projector는 32×32 map을 8×8로 평균 pooling하고, 마지막−첫 시점 차이를 넣는다.
  Q2 시간 구간 질문은 학습/평가 item filter에서 제외된다. 재귀 기억·근거 검색은 없다.
- 현재 `--text-only` arm은 LLM을 별도 학습하지 않는다. 동일 QA·동일 adaptation 예산으로 학습한
  metadata-only baseline을 새로 정의해야 projector의 추가 기여를 공정하게 평가할 수 있다.
- 기존 QA의 final event mask를 모든 prefix의 정답으로 복사하면 미래 라벨 누출이다.
  관측 수 세기는 prompt에 수가 제공되므로 영상 시간 추론의 주 지표로 쓰지 않는다.
- 계약의 center 좌표 범위 검사와 CRS 존재 확인은 충분한 지리정합 증명이 아니다.
  실제 CRS/affine/축 순서/원자료 pixel correspondence가 없으면 좌표 대응 기능은 fail-closed로 둔다.

기존 실험 파일을 즉시 덮어쓰지 않는다. 새 계약/manifest/entry point를 별도 버전으로 만들고
기존 결과와 cache provenance를 보존한다. checkpoint·band order·정규화·timestamp·crop·QA 정의가
검증되지 않은 cache는 shape가 맞는다는 이유만으로 재사용하지 않는다.

## 4. 모듈 A — 메타정보를 prompt뿐 아니라 기억 쓰기에 연결

관측당 최소 레코드:

`obs_id, aoi_id, bbox, crs, affine, acquisition_time, arrival_time, sensor,
bands, gsd, valid_mask, quality_map, token_cache_key, encoder_checkpoint_hash, source_provenance`.

획득 시각은 땅을 언제 관측했는지, 도착 시각은 모델이 그 관측을 언제 알 수 있었는지다.
첫 실험에서는 두 순서가 같은 자료로 시작해도 되지만, 실제 arrival 자료가 없다면 획득순 replay임을 명시한다.
그 후 지연 도착·중복 관측은 별도 운영 stress test로 다룬다. 과거 획득 영상이 늦게 도착했다고
최신 상태를 무조건 과거 상태로 덮어쓰면 안 된다.

첫 구현은 간단한 Fourier/embedding/MLP 조건을 선택한다. 메타 벡터를 EO 토큰에 결합하여
update gate와 reader에 넣고, LLM에는 날짜·위치·관측 ID 등 해석 가능한 필드를 함께 제공한다.
자연어 메타를 주는 것과 updater에 typed metadata를 주는 것을 별도 ablation으로 나눈다.
QA-derived quality는 관측의 사용 가능성이지 사건 확률/예측 불확실성이 아니다.

사건 발생일·최종 mask·post_index·미래 관측 clear fraction·답을 암시하는 파일명은 입력에서 제외한다.
추후 공공기록/기상 맥락을 붙여도 해당 도착 시점에 알려진 정보만 쓰고, 시각-only와 분리 보고한다.
frozen OlmoEarth가 지원하지 않는 센서를 metadata만 붙여 처리할 수 있다고 가정하지 않는다.

## 5. 모듈 B — 좌표에 고정된 현재 기억 + 제한 용량 사건 기억

현재 공간 기억은 공통 좌표계의 map으로, 사건 기억은 시간·위치·근거 ID가 남는 K개 항목으로 시작한다.
하나의 최신 latent만 계속 덮어쓰면 ‘지금 상태’와 ‘지난 사건’ 질문을 동시에 보장할 수 없기 때문이다.
사건 slot은 자유 생성한 문장이 아니라 출처가 있는 관측 특징/영역과 시기를 저장한다.
생성된 의미 요약은 hypothesis로 표시하며, 관측 사실의 대체물로 취급하지 않는다.

공간 대응은 우선 affine 기반 warp/overlap baseline으로 검증한다. 현재 10m 입력·patch4 map의
명목 token footprint는 40m이다. 20m shift는 반 token 이동이므로 단순 인덱스 shift로 정확히 처리되지 않는다.
resampling·경계 mask·원영상 재추출을 비교하고, 8×8 pooling된 기존 projector에 원래 40m 정밀도가
남는다고 주장하지 않는다. GNN/equivariant network는 이 대조의 한계를 확인한 뒤 후보로 추가한다.
위치 encoding이나 warp만으로 encoder 자체의 정확한 shift equivariance가 증명되지는 않는다.

국소 write의 최소 후보식:

`M_new(x) = M_old(x) + valid(x) * gate(M_old, Z_new, metadata, quality)(x) * delta(x)`.

invalid/비유한 입력은 산술 전에 처리하고, 유효하지 않은 위치의 시각 상태는 보존한다.
그러나 마지막 유효 관측 이후 경과 시간은 증가한다. 상태를 유지했다고 ‘변화 없음’이나
‘높은 확신’을 출력하도록 강제하지 않는다. 구름뿐인 prefix는 판단 보류가 정답일 수 있다.
이 gate 자체는 기존 GRU/quality candidate와 유사하므로 새로움의 근거로 과장하지 않는다.

K개의 사건 slot에서는 최근성만으로 FIFO 삭제하거나 유사도만으로 평균 병합하는 baseline과 비교한다.
방법 후보는 관측 품질과 변화 신호, 여러 학습 질문에서의 근거 필요성을 이용하여
사건의 전후 대조를 남기는 선택/압축이다. test에서 정답 event mask로 중요 slot을 고르지 않는다.
서로 다른 날짜의 특징을 병합하면 개별 날짜의 근거를 정확히 복원할 수 없으므로 provenance와
유효 시간 범위를 남기고, 복원 불가 정보를 문장으로 지어내지 않는다.

쓰기 정책은 질문과 독립적으로 운용하고 읽기만 질문 조건부로 시작한다.
질문 조건부 write를 도입하려면 task-specific memory임을 명시하고 다른 질문에 대한 망각을 측정한다.
중복 obs_id는 dedup하여 같은 관측의 반복 전송이 반복 근거처럼 계산되지 않게 한다.

## 6. 모듈 C — 기억을 읽는 VLM에 근거와 판단 상태를 함께 학습

reader는 질문·관심 영역·요청 시간 범위로 현재 map과 사건 slot을 고른다.
OlmoEarth와 언어의 직접 cosine 검색 대신 학습한 query/key 정렬 또는 cross-attention reader를 사용한다.
projector 입력에 공간 위치, 획득 시각, 품질, 관측 ID를 연결한다.

최소 출력 계약:

`answer, event_type, region_mask_or_bbox, time_interval, evidence_obs_ids,
judgment_status, calibrated_score, supersedes_claim_id`.

`judgment_status`는 보류/지지/반박·철회 등을 작업별로 정의한다. 확률을 낼 경우 독립 validation에서
proper scoring rule·calibration을 확인한다. 모델이 스스로 작성한 confidence 문장을 검증된 확률로 취급하지 않는다.
관측되지 않은 날짜를 정확한 발생일로 쓰지 말고 마지막 음성 관측–첫 양성 관측 사이 구간 등을 답한다.

판단 갱신기는 `AOI + 사건 종류 + 고정된 질의 시간 범위`로 식별한 기존 claim과 새 근거 특징을 읽고,
새 근거의 관계를 지지/반박/불충분으로, 갱신 행동을 유지/강화/수정·철회/보류로 예측하는 작은 head로 시작한다.
서로 다른 시간 범위의 주장은 다른 claim이므로 과거의 양성을 현재 음성으로 취소하지 않는다.
이 structured claim ledger는 시각 기억과 구분한다. 생성 문장만 스스로 읽고 옳다고 확정하는 loop는 피한다.
관계·행동 CE supervision은 prefix 근거 annotation과 rollout의 기존 판단에서 만들며,
실제 교정 사례가 없으면 수정 head의 실효성을 주장하지 않는다. 규칙 기반 ledger도 반드시 baseline에 둔다.

처음부터 긴 CoT를 요구하지 않는다. 근거 관측 pointer, 영역 head, 시간 구간 head를 감독하고
이들이 고른 관측/영역 token을 answer decoder가 실제로 읽도록 연결한다.
표준 LM CE, mask BCE/Dice, evidence multi-positive pointer loss, 시간 구간 loss를 사용한다.
이는 새로운 loss 이름을 만드는 제안이 아니며, 핵심은 memory rollout과 실제 근거 연결이다.

그럴듯한 mask를 함께 출력하는 것만으로 답의 근거 충실성이 증명되지는 않는다.
근거 제거/무관 근거 치환/유효 반대 근거 추가 시 답과 보류 확률이 어떻게 변하는지 통제한다.
복수의 대체 근거가 있는 경우 하나를 지워도 답이 유지될 수 있으므로 gold 근거 집합을 함께 고려한다.
합성 조작 결과는 실제 사건에서의 올바른 판단 수정과 별도 표로 보고한다.

## 7. 학습은 세 단계로, test에서는 가중치 고정

1. **정적 grounding/alignment 확인.** EO encoder와 LLM을 고정하고 metadata adapter,
   projector, reader/근거 head를 학습한다. 전후 영상으로 무엇/어디/어느 관측을 답하는지부터 검증한다.
   이것이 실패하면 기억 모듈의 효과와 언어 정렬 실패를 구분할 수 없다.
2. **순차 memory 학습.** 관측 하나씩 들어오는 student rollout에서 updater와 reader/projector를
   prefix QA+근거/판단 행동 loss로 학습한다. 과거의 고정된 사건 질문을 다시 물어 근거 보존 효용을 평가한다.
   frozen readout/logit preservation은 보조 baseline이지 언어·새 task 보존의 보증이 아니다.
   full-window teacher latent MSE를 유일한 목표로 쓰지 않는다.
3. **선택적 LLM LoRA.** 정렬·구조화 능력이 병목이라는 validation 증거가 있을 때 추가한다.
   주요 대조군에도 동일 LLM adaptation 예산을 준다. EO encoder adaptation은 필수 시작점이 아니다.

loss 가중치·slot K·retrieval 수·token 수·threshold는 validation에서 정하고 test 전에 봉인한다.
hard slot 선택이 학습 때 soft selector와 다르면 그 train/test 차이를 명시하고 검증한다.
seed·episode 수·optimizer step·표본 노출·라벨 비용을 대조군과 함께 기록한다.
2 GPU에 들어가는지는 context length·precision·batch·LoRA 범위를 확인한 뒤 말한다.
기존의 ‘2 GPU/10주 충분’ 문장을 이 새 설계의 확정 일정으로 재사용하지 않는다.

## 8. 가장 큰 업데이트는 데이터: 전후 QA를 순차 근거 QA로 바꾸기

한 episode는 같은 AOI의 실제 관측열과 여러 도착 cutoff, 질문, prefix별 판단·근거 annotation이다.
기존 Sen12 사건 mask/전후 시각은 pilot 재료지만, 모든 prefix가 언제 답변 가능한지까지 알려주지 않는다.
새 annotation은 최종 사건의 존재뿐 아니라 cutoff에서 가능한 주장과 보류해야 할 주장을 구분한다.

반드시 서로 다른 세 상황을 구분한다:

- **땅이 변함:** ‘지금 침수되어 있는가’가 no→yes. 현상 상태 변화이지 잘못된 믿음의 교정만은 아니다.
- **증거가 늘어남:** 고정된 과거 구간의 사건에 대해 unknown→supported. 초기 보류가 오답은 아니다.
- **기존 해석이 교정됨:** 같은 고정된 사건 명제에 대한 unsupported/잘못된 주장이 반대 근거로 수정됨.

실제 교정 사례가 없으면 ‘belief correction’ 주장은 줄이고 evidence accumulation/grounded update로 표현한다.
애매한 첫 영상에 일부러 오답을 강제해 합성 belief revision을 만드는 방식은 실측 성공으로 세지 않는다.
확정된 과거 사실에 대한 무관 새 관측은 답을 바꾸지 않아야 하지만, 현재 상태 질문은 실제 변화에 따라 바뀔 수 있다.

구성해야 할 사례: 명확한 변화/무변화, 구름·그림자·계절·수확 등 혼동, 새 유효 증거,
무관 새 관측, 드문 과거 사건을 다시 묻기. 후속 관측으로도 원인 판단이 불가능하면 보류를 유지한다.
평가 annotation에는 당시 이용 가능한 prefix 자료만 보여 답변 가능성과 근거를 판정하고,
사후 물리적 사건 확인은 별도 gold/provenance로 저장한다. 최종 gold가 있다고 초기 답이 항상 가능하지는 않다.

첫 현상은 하나로 한정한다. 현재 산사태 자산은 코드 진단용으로 쓰고, 최종 현상을 벌목/산림손실로
선택한다면 dated 근거·영역 label·혼동 사례 검수를 새로 확보해야 한다. 넓은 ‘환경파괴’ label로 시작하지 않는다.
학습 지역 내부에서만 fine-tune한 결과는 지역 적응 실험이며, 외부 AOI/다른 기간에서의 무학습 성능을 따로 낸다.
같은 사건·중첩 타일·동일 궤도 근접 장면은 split 사이에 섞지 않는다. 이미 살펴본 지역을 새 sealed test라 부르지 않는다.

## 9. CVPR용 핵심 대조와 평가

최소 2×2는 memory 유무 × updater의 typed metadata 조건 유무다. prompt의 기본 날짜/위치와
좌표 정합은 모든 arm에서 유지하여, ‘메타 없음’이 잘못된 지리정합 대조가 되지 않게 한다.
이 대조만으로 부족하므로 다음을 함께 둔다:

- 최신 관측-only와 동일 용량 running mean/GRU/Δt-GRU/국소 XAttn.
- 같은 K/budget의 FIFO와 MemoryVLA식 유사도 병합, 제안된 근거 보존 memory.
- 학습한 metadata-only, readout+구조화 템플릿: visual memory와 LLM의 추가 가치를 분리.
- 동일 관측 범위·입력 표현에 맞춘 TEOChat 계열, 가능하면 TerraScope식 grounded reader.
- RAVEN식 원영상 retrieval: 비용과 원영상 접근 범위를 명시한 강한 외부 시스템 대조.

원영상 기반 시스템과 latent 시스템은 입력 정보가 다르므로 matched representation 대조와
system-level 비용 대조를 구분한다. 전체 과거를 다시 읽는 reference는 resource-matched baseline이 아니라
더 많은 비용을 쓰는 reference다. 항상 성능 상한이라고 단정하지 않는다.
추론 때만 memory를 zeroing한 ablation과 처음부터 no-memory로 학습한 baseline도 구분한다.

주 지표는 다음을 묶어 읽는다:

| 질문 | 지표/평가 단위 |
|---|---|
| 답이 맞고 실제 근거가 있는가 | QA + 영역 IoU/box overlap + evidence observation precision/recall; 세 조건의 joint success |
| 새 증거에 적절하게 반응하는가 | answerable prefix에서 잘못된 주장→올바른 주장, unknown→supported를 분리 |
| 불필요하게 답을 바꾸지 않는가 | 동일한 고정 명제·정답 불변 episode의 unwarranted flip rate |
| 예전 사건을 기억하는가 | 고정된 과거 구간 질의의 grounding/QA 성능 vs memory 길이·K |
| 보류가 의미 있는가 | risk–coverage, calibration/proper score, 음성에서의 false alarm |
| 운영 비용이 줄어드는가 | batch1 encode+write+retrieve+decode+I/O latency, peak GPU 및 persistent memory bytes |

‘올바른 수정’은 단순 답 변화 횟수가 아니다. 전후 gold·answerability·근거 유효성을 갖춘 paired case로 정의한다.
관측 간 구간만 known이면 탐지 지연을 정확한 발생시각의 일수로 과장하지 않고 interval-aware로 보고한다.
statistical CI는 질문 행의 iid bootstrap보다 독립 사건/AOI cluster와 seed 반복을 고려한다.
단일 지역 case study만으로 보편적 EO memory 원리를 주장하지 않는다.

예산은 작업 map+K slot+색인+provenance+raw archive의 존재/접근을 모두 명시한다.
원영상 링크만 저장해도 query 때 전체 archive를 읽을 수 있다면 총 시스템이 bounded memory인 것은 아니다.
main bounded arm은 폐기된 증거를 test에 외부 archive에서 다시 읽지 않으며, retrieval arm은 별도 비용을 센다.
메모리 단위는 slot 수만이 아니라 byte/token으로 맞춘다.

## 10. 실제 업데이트 순서와 새로운 kill gate

| 순서 | 구현/계약 업데이트 | 끝났다는 증거 |
|---|---|---|
| P0 | 관측 manifest와 spatial/arrival 계약; 새 prefix 근거 QA; cache lineage | 미래 품질 선택 금지·ID dedup·좌표 대응·timestamp 및 label 접근 검사 |
| P1 | 기존 projector에 위치/시각/ID를 연결하고 근거 head/reader | 정적 답+근거가 학습한 metadata-only/템플릿보다 개선; 독립 dev 평가 |
| P2 | 관측별 chronological loop; 현재 map+K 사건 memory; 단순 recurrent baselines | 동일 budget에서 memory 필요성과 long-history forgetting의 실제 사례 |
| P3 | 품질 조건 local write+근거 보존 선택/압축; prefix 근거 loss | 강한 단순 memory 대비 joint grounding/update 이득, invalid/무관 입력 안정성 |
| P4 | sealed 외부 AOI/기간 및 새 shift/cloud/gap stress; 시스템 비용 | 방법 효과의 재현·일반화·budget trade-off, 근거 제거/치환 결과 |

각 단계는 다음 단계 진입 전 실패 위치를 분리하기 위한 제안이다. 기존 test를 반복 개발에 썼다면 dev로
분류하고 새 final test를 정한다. 새 평가의 효과 크기/허용 열화/memory 비용 기준은 데이터·validation을
보고 **confirmatory 실행 전에** prereg한다. 기존 latent IoU의 +.03 gate를 언어 판단 지표로 옮겨 쓰지 않는다.

방법 주장 철회 조건:

- trained metadata-only/템플릿과 비교해 visual grounding의 추가 가치가 없다.
- matched latest-only/단순 recurrent memory 대비 과거 근거 보존·적절한 갱신에 추가 가치가 없다.
- 성능 차이가 다른 encoder/더 많은 raw 접근/더 큰 LLM adaptation·표본 노출에서만 나온다.
- 생성 근거의 품질은 좋아도 답이 실제 근거 입력을 이용한다는 통제 증거가 없다.
- 외부 사건/AOI에서 이득이 재현되지 않는데 보편적 방법으로 주장한다.

이 경우 P0의 데이터/평가 계약 가치와 method 성공을 분리해 보고한다.
논문에 남길 핵심은 EO-specific memory failure → 방법 → 순차 grounded update 평가 → 비용/강건성이다.
robot memory, metadata FM, temporal EO VLM, pixel grounding의 결합은 출발점이며 acceptance 보장은 아니다.

## 11. 이번 작업의 범위

문헌·현재 코드 확인과 청사진 작성만 했다. 서버 접속/GPU 실행/실험 코드 변경/기존 prereg·gate 변경,
새 라벨 개봉, data download, commit/push는 하지 않았다. 다음 구현 시에는 위 P0 계약을 먼저 확정한다.
