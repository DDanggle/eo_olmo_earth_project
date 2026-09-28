# 다음 GPU1 실험의 실행 계약: 독립 검토

2026-09-27. 요청된 현재 상태 문서, 상태 JSON, `code/oe6_pastis_input_v0/`의 준비·감사·검산·후보 선택 코드를 읽었다. 추가로 이전 GPU 실행 상태와 기존 라벨 예산안을 대조했다. **이 문서는 제안이며 실행·학습·다운로드·저장소 변경은 하지 않았다.** 아래 시간은 완료 예상이 아니라 작업을 무한히 늘리지 않기 위한 제안 상한이다. 처리량을 측정한 뒤 실제 run config에 동결해야 한다.

## 1. 현재 위치를 한 문장으로

**정상 PASTIS 부분 자료에서 원 입력 두 사례의 날짜·밴드·정규화를 검증했고, 부모 지역을 분리한 80개 메타데이터 후보를 골랐지만, 감독 episode·실제 trainer 저장복원·학습된 비교 기준선은 아직 없다.**

확인 근거:

- `docs/OE6_PASTIS_INPUT_PROGRESS_20260927.md`
- `config/oe6_execution_and_crossdomain_status_20260927.json`
- `config/oe5_execution_status_20260927.json`
- `code/oe6_pastis_input_v0/prepare_two_cases_v2.py`, `verify_prepared.py`, `prepare_development_manifest.py`, `audit_existing_v1.py`, `pastis_generator_pinned.py`

0000은 공식 SHA 일치, 0001은 불일치라 사용 금지, 0002는 없다. 정상 0000에는 기존 benchmark train 911개가 있다. 이번에는 그중 train 부모 두 곳에서 64개, development 부모 한 곳에서 16개를 골랐다. t30uxv는 이번 단계에서 개봉하지 않았을 뿐, 과거·사전학습 미노출 test라고 인증되지 않았다.

`prepare_two_cases_v2.py`는 감사에서 뽑은 **첫 두 행만** 처리하고 각 이미지의 최대 빈도 crop를 target으로 만든다. 80개 후보를 일반적으로 처리하는 extractor나 support/query sampler가 아니다. `verify_prepared.py`도 두 사례의 산술 검산이며 새로운 모든 사례의 입력 품질을 인증하지 않는다. 이를 그대로 반복해 교정 전이 episode가 준비됐다고 세면 안 된다.

## 2. 먼저 고정할 범위

첫 실행의 과제는 **연간 crop 참조 mask를 이용한 조건부 영역 찾기**다. 건강·고사·개화·변화 시작일·홍수 동역학·기후 영향은 이번 정답에 없다. 인간 전문가 교정은 아직 없으므로 공개 mask에서 만든 support는 `simulated/public-label support`로 기록한다.

입력은 native S2 10개 실측 밴드와 공식 PASTIS 보완 2개다. B01←B02, B09←B8A와 `band_observed` sidecar를 그대로 명시한다. sidecar를 encoder가 읽는 per-band mask로 오해하지 않는다. 정규화는 raw int16 → 공식 COMPUTED 한 번이며 영상과 날짜는 같은 인덱스로 선택한다.

정확한 footprint와 cloud mask가 없는 것은 **현재 개발을 전부 막는 조건은 아니지만 주장을 제한한다.** centroid와 parent tile로 분리한 개발 비교, 원 픽셀 격자에서의 mask 평가는 가능하다. 엄밀한 경계 중복 감사, 실제 지표 면적, 맑음 인증, 독립 공간 일반화의 확증은 아직 불가능하다.

## 3. 단계 A — 두 사례 검증을 재실행 가능한 계약으로 마무리

이미 통과한 입력 산술을 반복하여 성과로 더하지 않는다. 현재 결과를 manifest 기반 준비기의 regression fixture로 가져온다.

**산출물 제안:** `input_recipe.json`, `two_case_contract_receipt.json`, 원 파일/loader/정규화·band recipe SHA 목록. 기존 결과 위치를 참조하고 변경된 부분만 검사한다.

**통과 기준:**

1. 두 NPZ의 실제 hash가 기록과 일치하고 source patch/parent/원 S2 H5 payload까지 역추적된다.
2. 날짜×프레임 1:1, 정렬·중복·범위 검사, 0-based month 변환이 재현된다. legacy 날짜 오류를 우회하는 준비기가 사용된다.
3. 10→12밴드 재배열·보완, COMPUTED 한 번, semantic 첫 평면의 class 의미와 instance map이 일치한다. `void=19`와 class0 처리 규칙을 loss/metric 모두에 명시한다.
4. 현재 눈으로 고른 날짜는 두 사례의 공학용 선택으로 유지한다. 학습/평가의 일반적인 clear-date 정책으로 승격하지 않는다.
5. 모델 입력/출력 shape, raw dtype, normalization 및 selected-date hashes를 실제 batch에서 출력할 방법을 마련한다.

실패하면 변환만 수정하고 source 파일·기존 결과는 보존한다. 40GB 전체 SHA 검사를 이유 없이 다시 돌리지 않는다. 해당 shard가 변경됐거나 provenance가 깨졌으면 그 파일부터 다시 확인한다.

## 4. 단계 B — 80개를 실제 raw·label 묶음으로 준비

**필요한 새 코드:** frozen candidate manifest를 읽는 extractor, temporal sampler, public-label episode builder, 독립 verifier. `train.iloc[:2]`와 이미지별 최대 class 선택을 일반 학습 정책으로 사용하지 않는다.

**산출물 제안:**

- `manifest.jsonl`: patch/parent/role, 원 packed reference와 payload hash, 원 취득일 전부, 선택 날짜/인덱스, 실측/보완 밴드, semantic/instance provenance, 실제 출력 경로/hash.
- `quality_policy.json`: nodata·불완전 관측·구름 정보 부재·날짜 선택 규칙과 제외 기준.
- `preparation_receipt.json`: 요청 64/16, 성공/실패/제외 수와 이유. 후보 80을 준비 성공 80으로 미리 기록하지 않는다.
- `episode_manifest.jsonl`: 질문 대상 class, support/query patch·객체·부모 지역, 동일 출처/객체 제외, 공개 라벨 support 출처, query 정답 경로, 문자열·mask·loss valid grid 계약.
- `preparation_verification.json`와 고정 표본 montage: 입력 산술, label 범위, 중복, 날짜, nonfinite/nodata 분포를 별도 코드로 검산.

**관측 정책 권고:** 첫 비교는 retrospective annual crop task임을 선언하고 원 날짜 전체에서 정해진 균등 규칙으로 공통 T를 선택한다. T=8을 throughput 후보로 두되 profile 결과에 따라 공통 T=4로 줄일 수 있다. T=2만으로 crop 판독성이 충분하다고 가정하지 않는다. 정답/모델 예측을 보고 각 이미지에서 맑은 날짜를 고르지 않는다. 구름을 unknown으로 기록하고 QA 표본은 오류 발견에만 사용한다. 유효한 cloud mask를 추가로 확보하면 별도 버전으로 변경한다. 이런 전체 연도 선택은 causal arrival 평가가 아니다.

**nodata 권고:** 원 자료에서 보인 -10000의 의미와 per-band/all-band sentinel 계약을 먼저 고정한다. 유한하다는 이유로 정상 반사도라고 취급하지 않는다. class label valid, observed band availability, missing pixels, unknown cloud quality를 분리한다. per-band missing 처리 기능이 없는 경로에는 보수적인 input validity 또는 관측 제외 정책을 적용하고 실패/제외 장부를 남긴다. 일반적인 음수 반사도 전체를 임의로 지우지 않는다.

**통과 기준:**

1. 정상 shard만 사용하고 source stat/hash·packed offset bounds가 확인된다. 0001과0002를 읽은 흔적이 없다.
2. 모든 성공 사례가 영상/날짜/semantic/instance를 실제 개봉·검산했다. 원 label의 연간 의미를 QA에 보존한다.
3. train parent={t32ulu,t31tfj}, dev parent={t31tfm}; patch/payload 중복 없고 예비 parent는 개봉하지 않는다. footprint 독립성 미인증은 결과 metadata에 남긴다.
4. class 후보/질문 생성/혼동 쌍은 train 자료에서 정하고, dev 결과를 보고 쉬운 class만 고르지 않는다. dev에 target 부재 사례가 있으면 제거하지 말고 사전 정의한 no-target 규칙으로 다룬다.
5. support와 query가 같은 patch/객체가 아닌지 검증한다. 같은 이미지를 여러 질문으로 늘렸다고 독립 관측 수를 늘리지 않는다.
6. 실제 성공 수·class coverage가 목표 과제를 만들 수 있는지 보고한다. 부족하면 원래 hash 순서의 다음 후보로 대체하되 이유/정책을 기록하고 전체 source/label budget을 센다.

일부 입출력 오류가 생기면 전체 자료 다운로드부터 시작하지 않고 해당 payload/변환을 조사한다. 공식 test 확보가 없어도 이 단계의 개발 실험은 가능하다. 전체 benchmark 점수라고 부르지는 않는다.

## 5. 단계 C — GPU1 처리량·2-case overfit·checkpoint

현재 통과한 것은 **물리 GPU1의 H200에서 FP32 실제 VLM 1-step gradient**다. H100이라는 이름으로 기록하지 않는다. 그때 입력은 작은 smoke이고 현재128×128 입력의 비용을 보장하지 않는다. patch4이면128×128×2시점에서 EO 격자는2,048개이며 T=8이면8,192개다. 실제 shape를 찍어서 확인하고 full-grid를 그대로 Qwen에 넣는 실험과 dense EO readout 참고 실험을 구분한다.

**산출물 제안:** full reader weight/config identity manifest, 실제 supervised trainer, `runtime_profile.json`, optimizer·scheduler·RNG·sampler를 포함한 checkpoint, `overfit_and_resume_receipt.json`.

**실행 순서:**

1. GPU1 UUID/name·점유·용량을 확인하고, 다른 사람 프로세스를 종료하지 않는다. local CUDA0으로 remap하더라도 physical1/UUID를 기록한다.
2. microbatch1, 공통 M=64의 learned resampler, FP32로 forward/backward 몇 step의 실측 메모리·처리량을 얻는다. T·grid 크기·RGB token·EO token을 모두 기록한다. BF16는 기존 cache parity 실패가 미해결이므로 “FP32 성공했으니 해결”로 간주하지 않는다.
3. 준비된 두 공학 사례와 실제 source-derived 질문/mask로 overfit한다. 일반 문장 `The image shows land cover.`만 반복한 손실은 감독 판독 학습으로 세지 않는다.
4. checkpoint 저장 → 새 프로세스에서 reload → 동일 batch eval → optimizer state까지 재개한 다음 update를 원 연속 실행과 비교한다.

**통과 기준:**

- encoder/adapter/실제 mask head 등 선언한 trainable 모듈에 finite nonzero gradient와 실제 weight 변화가 있고 frozen reader는 optimizer에서 제외된다. frozen reader라도 입력 gradient를 끊는 `no_grad`를 사용하지 않는다.
- 두 사례의 foreground 목표를 외울 수 있는지 확인한다. CE만 줄거나 background로만 수렴하면 통과하지 않는다. mask head 해상도에 맞는 reference와 threshold를 실행 전에 고정한다. 예: coarse target에서 foreground IoU≥0.90를 engineering gate 후보로 둘 수 있지만 원128×128 경계와 혼동하지 않는다. 이것은 일반화 결과가 아니다.
- 이미지별 여러 source-valid 질문을 구성할 수 있다면 같은 이미지에서 target 조건을 바꾼 mask가 바뀌는지 확인한다. 서로 다른 class prompt 두 개만 외운 결과로 영상 의존성을 주장하지 않는다.
- save/reload 뒤 weights·vocab·input recipe hash 일치, eval logits/mask의 사전 고정 수치 tolerance, 재개 step·optimizer/RNG/sampler 일치. tolerance 실패 시 원인을 보고하고 결과를 본 뒤 완화하지 않는다.
- BF16를 사용할 필요가 생기면 별도 original-RGB/EO, full-vs-cache 및 gradient 검사로 해결한다. 학습 자체에서 KV reuse를 꺼 비용/정확도를 먼저 확인할 수 있으나, 그것은 BF16 cache 이슈 해결과 별개다.

입력 profile OOM이면 T/M/microbatch·checkpointing 등 공통 recipe를 줄여 다시 profile한다. 한 비교군만 더 적은 관측을 받아서는 안 된다. full-grid LLM 경로가 안 맞으면 dense readout 참고로 명시하고 동일 budget 비교군으로 꾸미지 않는다.

## 6. 단계 D — 먼저 충분히 학습한 일반 기준선을 만든다

첫 development 비교는 방법을 많이 붙이지 않고 다음을 남긴다.

1. 원 OlmoEarth 격자의 동일 용량 readout: 실제 라벨 정보가 원 특징에서 읽히는지 확인하는 참고.
2. 공통 M learned resampler + 같은 dense head/질문/reader: 제안 방법이 이겨야 할 일반 모델.
3. 현재 평균 pooling은 저비용 진단 대조로만 유지한다. 이것 하나를 이겨 연구 방법을 증명했다고 하지 않는다.

각 군의 EO/RGB 관측·질문·support·mask 감독·native replay와 trainable 범위를 맞춘다. full grid가 더 많은 reader token을 쓰면 상한 참고이다. 같은 M이 같은 FLOPs라는 뜻이 아니므로 encoder/head/reader 시간, peak memory, 실제 optimizer updates, 관측/label 노출을 별도로 보고한다.

**산출물 제안:** 고정 run recipe·manifest hash, learning curve, split/class별 IoU·혼동 FPR·source-grounded answer 지표, best/last checkpoint, 질문-only/영상 교체 대조, 실제 사용 compute/label 장부.

**충분히 학습했다는 통과 조건:** 작은 공통 LR 후보를 동등한 예산으로 검토하고 최적화 장애가 없으며 train curve/사전 patience 기준이 평가 간격 여러 번 안정적이어야 한다. 제안 hard cap에 걸려 아직 계속 개선 중이면 `budget_limited_baseline`이지 충분히 학습한 baseline이 아니다. 고정 step 수만으로 학습 충분성을 선언하지 않는다. 첫seed는 debugging/development이며 효과의 불확실성을 제시할 때 seed/support draw를 확대한다.

개발 parent가 하나이므로 “여러 지역에서 반복됐다”는 결론은 불가능하다. 클래스별 원 patch/parcel 수를 함께 보고하고 pixel 수·질문 수를 지역 반복수처럼 쓰지 않는다. 질문-only나 단순 class prior가 비슷하게 풀면 관측 의존성/episode 설계를 먼저 고친다.

## 7. 단계 E — 다음 연구 비교의 순서

기준선이 동작하고 오류 유형이 확보된 뒤 아래를 각각 실행한다. 전부 동시에 추가하지 않는다.

- **정보 경로:** ordinary resampler 대 공간·시간 구별 reader. 같은 정보·같은 dense head·같은 감독 예산.
- **학습 목적:** 일반 공동학습 대 교정/관측 감독. 구조×목적2×2로 분리한다. official native 목적의 추가 학습 arm과 untouched OlmoEarth 기준을 함께 유지해 단순 추가학습 효과와 구분한다.
- **재사용성:** 언어 reader를 제거하거나 새 reader로 교체한 고정 EO 평가. 현재 두 공학 사례/80개 개발 결과만으로 OlmoEarth foundation model 개선을 주장하지 않는다.
- **선택적 확장:** bounded observation loop, discrete codebook, 실제 KV reuse, 물리 시뮬레이션은 드러난 오류와 비용이 정당화될 때 한 가지씩 추가한다. 약한 pooling을 고친 결과에 loop/tokenizer 효과를 섞지 않는다.

확증 평가로 넘어가기 전에 새 region/event의 과거 노출·전처리·원 footprint와 독립 split을 확보해야 한다. Kuro는 과거 test 결과 관찰 이력이 있으므로 공식 test 명칭만으로 새 holdout으로 삼지 않는다. 전문가 교정 비용 주장은 기존100만원 계획의 시간 측정 pilot 이후 실제 인간 기록으로 검증한다. 공개 label에서 만든 support 수는 인간 소요시간과 같지 않다.

## 8. 제안 자원 상한과 중단 조건

아래는 **초기 실행을 안전하게 관찰하기 위한 제안 제한**이다. GPU 처리량이나 학습 완료 시간을 약속하지 않는다. 모두 같은 GPU1 순차 실행이며 다른 GPU로 자동 확장하지 않는다. 이미 사용 중인 GPU1을 선점하기 위해 타 작업을 종료하지 않는다.

| 단계 | 첫 배치 자원 제안 | 상한 도달 시 행동 |
|---|---|---|
| A 입력 계약 재검증 | CPU1–2, RAM8GiB, 새 산출물1GiB, 추가 다운로드0 | source 변경 없으면 기존 audit 참조. 불일치한 부분만 수리 |
| B 80개 raw 준비 | CPU2이하, RAM16GiB, 새 산출물10GiB, 자료는 기존0000만;30분 단위 watchdog receipt | 성공/실패 manifest까지 보존하고 미완료분만 재개. 무제한 병렬 추출 금지 |
| C0 실제 shape/profile | GPU1 하나, microbatch1, FP32, 최대10 optimizer step 또는30분 중 먼저; peak allocated 한도는 시작 시 GPU 총용량의80% 이하로 설정 | OOM/NaN/다른 점유 발견 시 해당 worker만 정상 종료. 공통 T/M recipe 조정 후 재검증 |
| C1 두 사례 overfit/저장복원 | GPU1, 최대200 optimizer step 또는2 GPU시간 중 먼저, mutable checkpoint last/best/resume검사용 각1개 | 단순히 step 부족인지 gradient/target 오류인지 curve로 구분. 자동 확대하지 않음 |
| D 기준선 초기 비교 | 첫 두 핵심 arm에 각 최대4 GPU시간 또는1,000 optimizer step 중 먼저; 합계8 GPU시간을 첫 검토점 | 미수렴이면 충분한 baseline으로 부르지 않고 throughput·curve 기반 다음 budget 결정. LR 후보를 추가할 경우 모든 군에 같은 탐색비용 기록 |
| E 연구2×2 | D에서 확인한 공통 recipe·실측 비용을 이용해 각 cell4 GPU시간 이하, 전체16 GPU시간 이내를 첫 개발 검토점으로 제안 | 순차 실행, 최초seed개발에 한정. 멀티seed·새 지역·분야 확대는 실제 성공/비용 근거 뒤 별도 계획 |

디스크에는 frozen Qwen/원 OlmoEarth를 매 checkpoint마다 복제하지 않고 검증된 immutable weights hash를 참조한다. optimizer를 포함한 mutable checkpoint는 예상 byte 수를 실제 tensor shape로 산출한 뒤 시작하며 여유 디스크보다 큰 run은 시작하지 않는다. crash 재개는 같은 recipe/hash가 맞을 때만 허용한다. 주기 로그·실행 PID·device UUID·step·peak memory·정상 종료 상태를 남긴다.

사람/API 예산은 기존안 그대로이며 GPU/엔지니어링 비용은100만원에 포함돼 있지 않다. 이 검토는 인력 연락·발주·API 지출을 승인하거나 실행하지 않았다.

## 9. 바로 다음 행동 하나

**manifest-driven 80개 raw/label 준비기와 고정 관측 정책을 작성하고, 실제 준비 성공 수와 episode 목록을 검산한다.** 동시에 full Qwen identity와 checkpoint/resume 코드 준비는 가능하다. 이 두 deliverable가 나온 뒤 GPU1에서 C0/C1을 실행한다. 지금 즉시 큰 방법 학습이나 911개 전체·새 자료 다운로드부터 시작할 근거는 없다.

이 순서는 작은 실험만 하자는 뜻이 아니다. 큰 주장의 최소 근거를 `올바른 감독 입력 → 실제 학습과 복원 → 강한 기준선 → 공정한 방법 비교 → 새 환경·독립 EO 재사용` 순으로 쌓기 위한 실행 계약이다.
