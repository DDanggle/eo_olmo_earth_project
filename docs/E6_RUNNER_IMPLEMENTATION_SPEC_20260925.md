# E6 작은 head runner 구현 명세 제안

2026-09-25. **아직 구현·실행·사전등록을 하지 않은 명세 제안**이다. 기존 `e6_head_model_v0.py`는 수정하지 않는다. 실제 repo의 E6 설계 문서, E5 source/prereg와 준비 manifest만 읽었다. E5 outputs/scores/학습 로그/완료 산출물 및 실제 prepared tensor·item·batch 파일 내용은 이번 검토에서 읽지 않았다. 아래 SHA는 준비 manifest의 선언값이며, 향후 prepare 단계가 실제 파일 바이트를 재검증해야 한다.

권고 기본값: `full=[A,B,B−A]` head 하나 × seed 1/2/3, 4,234 train의 동일 저장 batch로 각 1,590 update/12,702 exposure, 최종 checkpoint만 1,755 test에 적용한다. 새 head 예측은 총 5,265개다. 비교 대상은 결과와 무관하게 **E5 `full/native`**로 고정하고, 작은 head의 예제 예산 비교라는 범위를 유지한다. 새 runner가 E5 네 arm 중 좋은 것을 고르거나 T0 복구 날짜를 끼워 넣으면 다른 실험이다.

## 1. 입력 계보와 준비 파일

서버 parent 경로 제안은 `/home/work/data/olmoearth/e5_equal_budget_v0`, 새 출력은 `/home/work/data/olmoearth/e6_no_llm_head_v0`이다. 새 출력 디렉터리는 exclusive creation만 허용하며 resume/overwrite하지 않는다. E5 prepared manifest SHA는 다음으로 고정한다.

`e22fb584c95f2bac1916fd8d3c49234e993a156f80d38ac481140e12c3e7819f`

| byte-identical 재사용 파일 | 준비 manifest의 SHA256 |
|---|---|
| `items.jsonl` | `e8058fd5193374a1c32968b063e6b67032d8a4f2bc0bf659e490e09816a67a3a` |
| `pairs.npy` | `24ea523ac4342eac70fbecc5a67eacd07e3a171141343bbd6dee335a181bfd0c` |
| `ordered_ids.json` | `d92bc1591bcaa64c9f721eb106bf762f49ee6a578f83633bf5b8af536b9ffb5c` |
| `batches.json` | `38e411014eb25dd067d5740df15f00826cb2dd95acedea0dd386b76251b89fbe` |
| `eval_sets.json` | `46da34eae3674932fe8cdf696dc47f43c4e08e61ecadefe9d17e553901ee94f4` |
| E5 `prereg.json` | `fc7866feebbf1e9fcd4e93de5154bf9ffa141f434e70dc4731390cf18dfd4696` |
| E5 `input_audit.json` | `4ee66503ce2a4356b8fd654aeb53b7a5ceb9224c8f2ccca0329b62531ad3adeb` |

추천은 위 파일을 새 bundle의 `input_snapshot/`으로 복사한 뒤 모든 byte SHA를 다시 확인하는 것이다. `pairs.npy` payload만 약 2.35GB이므로 streaming copy/hash와 read-only mmap을 사용한다. hard link는 원본과 수정 상태를 공유하므로 기본값으로 쓰지 않는다. `parent_snapshot`의 C0/C1/E3/E4 metadata도 E5 manifest에 명시된 SHA로 복사한다. E5 manifest와 E5 code snapshot은 비교 계보로 보존하되 **E5 runner의 `verify_files`를 통째로 호출하지 않는다**. 그 함수는 LLM 파일 검증과 E5 실행 위치까지 강제한다. E6는 LLM 파일을 읽거나 모델을 로드할 필요가 없다.

E6 prepare는 다음을 별도로 동결한다.

- 자체 명세 JSON, head/prepare/train/scorer/launcher 코드 snapshot 및 SHA; 실행은 동결된 snapshot에서만 허용.
- 위 E5 입력 복사본, parent manifest와 parent metadata SHA, 실제 복사 전후 source stat/SHA.
- `metadata_vectors.npy`: float32 `(5989,8)`, `metadata_manifest.jsonl`: 각 global row의 `id,pair_index,phen,dates,encoded_values`만. source label은 이 feature archive에 넣지 않는다.
- 세 CPU 초기 head state와 canonical tensor hash를 **어느 seed의 훈련도 시작하기 전에** 저장. PyTorch/runtime version, head source hash, 초기값 manifest SHA를 기록한다. 새 head의 초기값을 E5 projector와 같다고 주장하지 않는다.

`input_audit`는 E5 당시 no_exclusions/3,756 cache/209개 bit-exact pool 확인의 계보 자료다. E6에서 실제 읽는 것은 이미 pooled된 `pairs.npy`이므로, 원 3,756 cache나 원 영상들을 새로 읽었다고 기록하면 안 된다. 현재 입력 pair SHA를 재검증했다는 범위와 과거 encoder 계보를 구분한다.

## 2. item/array/partition 계약

`items.jsonl`은 5,989개 고유 ID의 고정 line order다. `pairs.npy`는 float32 `(5989,2,64,768)`이며, 각 item의 `pair_index`는 이 line order의 **global zero-based index**다. train이 앞 4,234행이라는 가정을 금지한다. 반드시 `id→global_index`를 만든 뒤 저장 batch ID를 해당 index로 변환해야 한다.

| partition | flood pos / neg / hard_neg | landslide pos / neg | 합계 |
|---|---:|---:|---:|
| train | 1,066 / 1,066 / 1,066 | 518 / 518 | 4,234 |
| test | 457 / 457 / 457 | 192 / 192 | 1,755 |

검증 조건:

- `ordered_ids.json`의 키는 정확히 `train,test`; 각 리스트는 `items`에서 해당 partition을 필터한 원래 순서와 같아야 한다. ID/tile/홍수 사건/산사태 region-cluster의 train/test 겹침은 없다. 사건 support는 홍수 27/10, 산사태 7/2다.
- `answer`는 yes/no이며 `kind=pos→yes`, `neg/hard_neg→no`와 일치한다. 이 일치는 data integrity 검사다. 학습 target은 원 `answer`에서만 만들며 `kind`, flood fraction, quality에서 새 정답을 만들지 않는다.
- 홍수 `neg`는 `slots=[pre_1,pre_2]`, pos/hard_neg는 `[pre_2,post]`; 원 frozen indices와 두 날짜 순서는 바꾸지 않는다. 동일 tile의 pos/neg 완결성 및 hard-negative 별도 tile 계약을 유지한다.
- pair 전체와 float32 `B−A`가 finite인지 준비 때 전수 확인한다. 캐시 결측, dtype/shape/날짜 오류는 전부 실행 무효이며 재선정하지 않는다.
- train 배치는 `np.array(pairs[global_indices],dtype=np.float32,copy=True,order='C')`처럼 복사해 Tensor로 옮긴다. read-only mmap을 직접 writable Tensor로 간주하거나 inplace 수정하지 않는다. 한 배치의 CPU/GPU 입력은 `(batch,2,64,768)`이고 head 내부에서 full 192토큰을 만든다.

평가 membership도 재선정 없이 다음의 exact key와 ID 목록을 재사용한다.

| eval key | support |
|---|---|
| `all_test` | 1,755 test ID 전체 |
| `primary_same_prompt` | C1 902개, flood pos445/hard457, 8사건 |
| `paired_flood` | 원 pos457/pre-pre neg457, 10사건 |
| `hard_negative_flood` | 457개 source hard negative |
| `landslide` | 384개, 2지역 |
| `e3_subset` | 원 209개, 보존만 하고 새 primary로 쓰지 않음 |

C1 quality와 support metadata를 재검산할 때 원 예측값은 사용하지 않는다. 결과가 좋은 사건만 남기거나 클래스 한쪽이 없는 층을 새로 버리지 않는다.

## 3. 날짜와 metadata 입력

훈련과 inference 모두 같은 동결 `encode_metadata(phenomena,dates)`를 사용한다. 각 item의 `phen`과 원 두 날짜만 전달하고, source item dict 자체를 head에 넘기지 않는다. ISO `YYYY-MM-DD`, 실제 달력 유효성, `date_A<date_B`를 전수 검증한다. sensor 정보는 기존 질문에서 현상과 1:1 대응하므로 별도 feature를 추가하지 않는다.

8차원은 `flood/landslide one-hot`과 A/B 각각의 `((year−2000)/100,(month−1)/11,(day−1)/30)`다. train 통계 standardization, 날짜 범위에 따른 clipping, 누락값 imputation을 새로 넣지 않는다. head의 forward는 이미 encoded된 Tensor를 받으므로 runner가 임의 numeric vector를 만들지 않도록, 준비 archive와 encoder 재계산이 일치하는지 검증해야 한다.

**홍수 날짜는 E5가 사용한 사건일 기반 근사이며 encoder에도 시간 정보가 들어가 있다.** T0에서 찾아낸 원저자 source_date는 이 E6에 대체 입력하지 않는다. 날짜 복구가 옳더라도 E5와 E6에 서로 다른 날짜를 주는 순간 현재 주대조가 바뀐다. 이 head는 순수 영상-only baseline이 아니며, C1 동일 문구·날짜 층의 기준을 유지해야 한다. ID/event/kind/quality/정답/지도 좌표는 feature에 넣지 않는다.

## 4. 정확한 batch·update·loss 계약

`batches.json`은 `{'1': [epoch0,epoch1,epoch2], '2':..., '3':...}`이며 각 epoch는 batch ID 리스트의 리스트다. E5의 생성 규칙은 seed마다 `rng=np.random.default_rng(seed)`를 **한 번 만들고** 세 번 연속 `permutation(4234)`을 호출하는 것이다. 매 epoch RNG를 초기화하면 다른 순서가 되므로 금지한다.

runner는 저장 목록을 실행하되, 준비 검증은 이 규칙으로 전체 목록을 독립 재생성해 **순서까지 정확히 대조**한다. 매 epoch마다 4,234 ID가 중복 없이 한 번씩 등장하고, 529개 batch×8 + 마지막 batch×2 = 530 update다. 세 epoch 합계 1,590 update, 12,702 exposure, 세 seed 전체 4,770 update/38,106 exposure를 assert한다. `drop_last`, sampler 재정렬, class 균형 sampling, gradient accumulation, augmentation, 추가 warmup update는 없다.

권고 학습 설정은 원 제안 그대로 유지한다.

- AdamW, lr=1e-4, weight_decay=.01, betas=(.9,.999), eps=1e-8, fresh optimizer per seed. `amsgrad=False`, `maximize=False`, `foreach=False`, `fused=False`를 명시하면 구현 경로가 분명하다. 이 마지막 두 flag는 E5의 PyTorch 자동 선택과 다를 수 있으므로 동일 수치 optimizer 구현이라고 주장하지 않는다.
- float32 parameters/input/metadata/logit/target, AMP·GradScaler 없음. target은 `1.0 if source_answer=='yes' else 0.0`, `BCEWithLogitsLoss(reduction='mean')`에 전달한다. label smoothing, class weight, pos_weight, logit calibration 없음.
- `optimizer.zero_grad(set_to_none=True) → head(pairs,metadata) → BCE → backward → finite gradient check → optimizer.step → finite parameter check`를 정확히 한 번 수행한다. gradient clipping은 기본값 없음.
- PyTorch deterministic algorithms=True, TF32 matmul/cuDNN=False, float32 matmul precision='highest', cuDNN benchmark=False, `CUBLAS_WORKSPACE_CONFIG=:4096:8`을 CUDA 초기화 전에 명시하는 것을 권고한다. 지원하지 않는 연산은 조용히 비결정적으로 전환하지 말고 실행 무효로 남긴다. E5와 수치 backend/precision까지 동일하다는 주장은 하지 않는다.
- seed 1→2→3 순서로 세 최종 checkpoint를 모두 만든 뒤 test inference를 수행하는 구성이 간단하다. test BA를 보고 뒤 seed의 설정을 바꾸는 경로를 없앤다. per-epoch test나 best epoch 선택은 없다.

E5는 frozen LLM의 vocabulary와 EOS를 포함하는 생성 token loss이고, E6는 한 개 logit의 binary BCE다. label 노출 수/update/batch 순서는 같지만 loss의 token별 가중, pretraining, 용량, FLOPs, 수렴 난이도와 디코딩 비용은 같지 않다. 따라서 “동일 학습 예제 예산의 시스템 비교”가 정확한 표현이다. 작은 head가 뒤처져도 LLM의 필수성이나 언어 지식의 인과 효과를 바로 입증하지 않는다.

## 5. checkpoint·로그·예측 계약

각 seed는 `models/seed{seed}_head_full/`에 `initial.pt`, `training_contract.json`, `steps.jsonl`, `head.pt`, `training_completed.json`, `predictions.jsonl`, `completed.json`을 만든다. 모델·initial file SHA와 `state_hash`의 canonical tensor SHA를 모두 기록한다. optimizer state의 parameter별 step도 1,590인지 검증한다. 마지막 model state는 `torch.load(...,map_location='cpu',weights_only=True)`로 재로드하고 tensor hash가 같을 때만 inference에 사용한다. 평가 전후 state/file SHA도 같아야 한다.

각 step에는 seed/1-based step/0-based epoch와 batch_index/실제 batch IDs/global indices/exposure count/loss/finite checks/elapsed를 기록한다. 마지막 batch 2개의 mean loss를 전체 epoch loss에서 batch 8과 같은 무게로 단순 평균하면 문항 평균과 다르므로, 보고할 epoch BCE는 `sum(batch_loss*batch_n)/4234`로 정한다. per-step 값과 예제 노출 기록이 원본 감사 자료다. 훈련 중 정확도는 모델 선택에 쓰지 않는다.

평가는 고정 `ordered_ids.test` 순서, batch8(마지막3), eval mode + no_grad로 1,755개 전부 실행하는 것을 제안한다. inference batch 크기를 train 예산과 혼동하지 않는다. 모든 예측 logit은 finite, probability는 finite `[0,1]`, decision은 **logit≥0이면 yes**로 고정한다. sigmoid가 float32에서 0/1로 포화되는 것은 허용한다. 확률과 sigmoid(logit)은 명시한 절대 오차 1e-6 이내인지 감사하되 분류 threshold는 바꾸지 않는다.

E6 row 제안:

```text
run_id='E6-NL-v0', seed:int, model_arm='full_head', eval_arm='native',
id, pair_index(global), tile, cluster, phen, kind, dates, slots,
source_gold(original yes/no), transformed_gold:null,
output_kind='binary_logit', logit:finite float, probability:finite float,
prediction:'yes'|'no', parse_status:'not_applicable'
```

`answer_raw='yes'` 같은 생성 문장을 꾸며 저장하지 않는다. head에는 생성 parser가 없으므로 E5의 parse failure와 다른 구조다. source_gold 등 provenance는 inference가 끝난 뒤 output assembler가 붙이며 head/predict에는 전달하지 않는다. 세 seed의 `(seed,id)` exact coverage 5,265, source metadata·global index·원 gold 일치, duplicate/unknown/train-ID 누출 0을 확인한다. 집계 이전에 세 seed의 파일과 aggregate `predictions.jsonl` 내용이 정확히 같은지 확인한다.

## 6. 채점 재사용 판단

**E5 최상위 `score_run`을 그대로 호출하지 않는 것이 권고안이다.** 해당 함수는 네 model arm과 full_no_delta, 26,325개 응답, 생성 raw parser 및 pair-vs-full 판정을 강제한다. E6를 가짜 E5 arm에 끼워 넣거나 같은 답을 여러 arm으로 복제해서 통과시키면 provenance와 대조가 훼손된다.

작은 자체 E6 scorer를 만들고 다음 정의를 명시적으로 구현한다. frozen E5 helper들은 참고 oracle로만 사용해 합성 fixture에서 일치시키고, 실제 E6 독립 postrun auditor는 production scorer를 import하지 않는다. 참고 source SHA는 `e5_scoring_v0.py=2eb0be5e7fb8ac5d6534716a25c9a07c8f046b56ea8a55c7f2912b51b7e6af7d`다.

1. 주집계: C1 902 ID만 사용. `(str(cluster),tuple(dates),tuple(slots))`를 키로 층을 정하고, 층별 `BA=.5*(mean_positive(decision=='yes')+mean_hard(decision=='no'))`. 층을 사건 내 동등 평균, 8개 사건을 동등 평균한다.
2. E5 비교 raw 응답은 나중의 독립 comparator 단계에서만 읽는다. E5 전체 완료/유효성과 별도 감사 통과를 확인하고, 결과 선택 없이 full/native의 모든 3×1,755개를 고정 SHA로 연결한다. E5 row의 raw→parsed 규칙, source identity, global index, 각 현상 parse≤.01을 재검증한다. 이 검토에서는 이 파일들을 읽지 않았다.
3. E5의 unparsed `null`은 positive와 negative 양쪽에서 오답이다. 특히 negative specificity를 `1−FPR`로 계산하면 null을 정답 처리하므로 **specificity=mean(parsed=='no')**를 직접 계산해야 한다.
4. 각 seed의 유일한 primary contrast는 `Δ=BA(E5 full/native)−BA(E6 full_head)`. 사건별 delta를 **문자열 event ID 오름차순**으로 정렬한다. 매 seed `default_rng(20260925)`에서 `(5000,8)` 정수 resample index를 만들고 paired-event mean의 NumPy linear .025/.975 quantile을 계산한다. event order도 재현 계약이다.
5. 세 seed는 같은 사건의 반복 fitting이다. 24개 독립 사건처럼 합치지 않는다. seed별 point/CI·모든 사건별 값·클래스/층 분모를 공개한다. seed 평균이 필요하면 기술통계로만 별도 표시한다.
6. Secondary: 원 paired flood914의 10사건 BA, hard457의 사건/pooled FPR와 E5 parse counts, 산사태384의 두 지역 BA(지역2개 CI 없음). source hard negatives와 pre-pre negatives는 합치지 않는다. head probability AUROC/AP는 미래 선택 사항이며 이번 최소 scorer의 주대조에 필요 없다.

현재 기본 권고는 `scientific_verdict='descriptive_comparison_only'`와 연속 Δ/CI다. 원 E6 설계 문서의 .60/.05 및 2/3-seed 문구는 미채택 선택지였으므로 자동으로 공식 판정 기준으로 승격시키지 않는다. 운영상 그 규칙을 채택하려면 결과 이전에 별도 명세에 확정하고, 좁은 양의 VLM 차이와 허용 margin 내 head 보존이 동시에 성립할 수 있음을 그대로 보고한다. “차이가 유의하지 않음”은 동등성 판정이 아니다.

## 7. 실행 한도·실패·감사

아래는 구현 가능한 **운영 기본값 제안**이며 wall-time 예측이나 효율성 측정 결과가 아니다.

- 한 번에 GPU0 하나, head 한 seed씩 순차 실행. CPU threads=2, data workers=0, batch8 고정. 다른 장치 자동 fallback/자동 batch 축소 없음.
- CPU 준비 상한10분, 각 seed train+eval 누적 상한20분, 전체 활성 chain 상한60분. queue 대기는 별도6시간. 시작 시 output용 여유 disk≥4GiB 확인; 이 값은 약2.35GB pair 복사와 작은 checkpoint/log를 위한 운영 예산이다.
- 모델의 CUDA max_memory_allocated/reserved를 기록하고 allocated 상한2GiB를 제안한다. 이를 OOM-safe 예약이나 외부 context 포함 nvidia-smi 총량과 혼동하지 않는다. 초과/OOM이면 설정을 바꿔 재시도하지 않고 무효 보존한다. batch8의 이 작은 head에 충분한지는 실제 실행환경 preflight에서 확인할 운영 사항이다.
- 기존 E5 cooperative lock 순서 `.eo_e3_pair_dependence.lock`, `.eo_reader_gpu0.lock`, `.eo_reader_gpu1.lock`, `.eo_e5_equal_budget.lock` 뒤에 새 `.eo_e6_no_llm.lock`을 잡는 방식이 호환성이 높다. 실제 GPU0 UUID/compute process를 확인하고 빈 GPU에서만 실행한다. inherited `pass_fds`와 소유 FD/inode 검증으로 부모 종료 때도 자식 동안 잠금을 유지한다. 타 작업을 종료하지 않는다.
- E6 prepare/run은 E5 output을 고치지 않는다. E5가 아직 실행 중이면 head 학습 launch를 기다린다. E5 비교 gate는 코드상 `status='completed'`와 유효한 완료/audit 계보를 요구해야 하며 `inference_completed`만으로 통과시키면 안 된다.
- source/plan/입력/초기 state 변조, nonfinite loss/grad/params/logit, ID 누락/중복, 예산 불일치, deadline/OOM/exception은 실행 무효다. 실패 phase와 partial artifact를 보존하고 이미 완료한 seed를 교체하지 않는다. resume/추가 epoch/threshold 변경은 다음 version의 새 계획이 필요하다.

최종 감사는 exact 5,265 예측, 3개 최종 model, 4,770 step/38,106 exposure, 저장 batch/초기 state/최종 checkpoint hash, 메타데이터 인코딩과 원본 item/feature hash, 독립 BA/CI/FPR 계산을 포함한다. 완료 상태를 `head_execution_valid`와 `comparison_valid`로 나누면 E5 comparator 불가 상태를 head 실행 실패나 VLM 우위로 오해하지 않는다. comparator가 아직 유효하지 않으면 `comparison_valid=null, comparison_status='pending_reference_validation'`으로 남긴다. log만으로 모든 GPU 연산이 실제 수행됐다는 암호학적 증명까지 했다고 표현하지 않는다.

## 8. 후속 구현자가 실행 전에 확정할 항목

| 선택 사항 | 권고 기본값 / 이유 |
|---|---|
| 원 E6 판정 후보 채택 여부 | 연속 effect/CI만; 새로운 성공 문턱 없이 최소 질문에 답함 |
| prepared pairs의 저장 방식 | 새 input_snapshot에 byte-identical copy; hard-link보다 원본 독립성이 분명함 |
| 생성 loss와 다른 목적함수 | unweighted mean BCE 그대로; 동일 노출 예산이지 동일 loss라는 주장을 하지 않음 |
| optimizer/backend 세부 | 위 explicit AdamW/float32/deterministic 제안; 실행 전에 버전/flag 동결 |
| 실제 E5 comparator 파일 pins | E5 완료/독립 감사 후 full/native를 자동 선택하고 그대로 hash; 성능에 따른 arm 선택 없음 |
| deadline/device/disk 정책 | 위 60분/20분/10분, GPU0, disk4GiB를 운영 상한으로 확정; 실패 후 조용한 연장 없음 |
| inference batching | batch8, 원 ordered test 순서; 저장 logit/decision은 전수 보존 |
| 효율성 주장 | 이번에는 자원 사용량 보고만. 양쪽 공통 latency protocol 없이는 속도배수 주장하지 않음 |

이 문서는 runner 구현에 필요한 경계를 제시한다. 현재 head draft, E5 입력/명세/출력, T0 날짜 자료는 모두 그대로 둔다. 모델 자체가 좋아 보이는지에 따라 세부 선택을 바꾸지 않고, 최종 source/config/input/initial state를 실제 학습 전에 고정하는 것이 다음 작업이다.

검토 문서 SHA: repo `docs/E6_NO_LLM_BASELINE_DESIGN_REVIEW_20260925.md` = `8d929786253b4fb67438fc78cea7711e348d1852472a899665fc120d4f3f7e5e`. 현재 head draft SHA = `8cb61fafb4ed73b395678cbe10d6d81fee8c0a9c77b575aaf5498ce1d6b033f6`. 모델 테스트 결과는 이 명세에 다시 보고하지 않는다.

메인 통합 결정(이 문서 전달 직전): 주결과는 continuous Δ만 채택, 성공 경계/arm 선택 없음. E6 row의 고정 필드는 model_arm=full_head, eval_arm=native, prediction=yes/no이며, main prepare가 E5 full/native의 reference_rows.jsonl 5,265개와 reference_audit.json을 동결한다. train runner는 이를 비교 단계에서만 사용하며 E5 실행 코드를 import하지 않는다. per-model 상한은20분으로 확정되었고 전체60분이다.
