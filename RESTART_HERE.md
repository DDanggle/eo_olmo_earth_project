# OLMoEarth 연구 재시작 지점

> ## 2026-09-17 — CVPR 재집중, frozen 민감성 진단 실행
>
> PR·Ai2 지원 작업은 종료. 논문 방향 검토 `docs/EARTHBRIDGE_PROPOSAL_REVIEW_2026_09_17.md` §3·§5~7,
> 자산 목록 `docs/CVPR_ASSET_INVENTORY_2026_09_17.md`. 서버 `frozen_sensitivity_v0/` 체인 결과
> 결과 MS-122~124: 격자·월·연도·구름·결측 모두 결정 수준 강건, 손실은 사건 후 증거 제거뿐(2/2). A·C post-training 축 Sen12에서 폐기.
> 다음 분기는 검토문 §10: 갱신형 캐시+continuity benchmark 복귀(추천) 또는 PASTIS 재다운로드 후 시간 민감 과업 진단.
>
> **9/18 갱신:** 두 트랙 설계 `docs/RESEARCH_DESIGN_TIME_AND_LANGUAGE_2026_09_18.md`(§5 문헌검증·관문). MS-125: 사건 후 관측 1장 21%·2장 64~75%·3장 87~96%(2/2 폴드), G-T0 통과.
> 논문 몸통 = 갱신형 캐시(R7/R8) + 잠재 보간 벤치마크(T2) + 강건성·지연 열(MS-122~125). 다음 관문 G-T1(같은 달 구분), G-T2(보간 벤치마크 v0).
> GPU 두 장 허용(CLAUDE.md 4b 갱신). 새 sealed 파일 변경 없음.

> ## 2026-09-13 — PR 최종 준비 완료, 제출·새 실험은 하지 않음
>
> [최종 준비표](docs/PR_REENTRY_2026_09_10.md): sample 정적 검사 PASS, 최신 upstream에서도 schema 불일치 유지.
> SCL은 두 compositor + 실제 재격자 회귀 테스트, LFMC는 당시 config/log/full hash가 남았다.
> 기존 meeting cards의 T9~T14를 교정했다. Markdown만 변경했고 HTML/공개 페이지는 미동기화다.
> 공식 채용 마감 9/16 확인. PR 완료를 지원 조건으로 두지 않는다.
> 연구·제품 코드, GPU 작업, 외부 게시를 변경하지 않았다. 아래 연구 상태는 당시 검토 기록이다.

> ## 2026-09-10 — Korea KR-4 검증 / PR 준비 위치
>
> [한국 감사](docs/KOREA_KR4_AUDIT_2026_09_10.md): 원 보고서 수치 확인, random 주 규칙 불통과 유지.
> test C05 한 타일64칩은 cache 20220924 / raw·label 20220427 cutoff 비대칭.
> FULL batch32/16으로 노출2배, init 후 seed 설정·raw padding·mIoU 정의도 다음 revision에서 보완한다.
> K-shot은 이미 같은 support/step/batch다. 단순 raw step 확대를 exposure matching이라 부르지 않는다.
> 한국 label/test 성능은 이제 개봉됐다. 신규 head 세 개의 static 재사용이며 source-head 전이와
> 공유 streaming은 아직 없다(single_fp16=0). 원 v1은 보존하고 실행 전 새 규격을 마련한다.
> [PR 재진입 목록](docs/PR_REENTRY_2026_09_10.md): 첫 영문 초안·branch 확인. 최신 upstream 검증은 별도.
> 이번 감사는 원격 읽기·CPU 집계·문서만 수행했고 GPU/production/PR 제출을 변경하지 않았다.

> ## 2026-09-09 — 다음 학습 준비: JEPA / encoder post-training
>
> **설계 입구:** [추가 학습 준비안](docs/POSTTRAINING_JEPA_UPDATE_2026_09_09.md).
> 현재 수치판은 아래 MS-117~120 검토다. 새 성능/확증 결과는 추가하지 않았다.
> 방향: frozen readout을 유지하는 **encoder frozen/LoRA × forecast loss 유무** 비교.
> 교차센서 정렬·surprise·decoder joint tuning은 원인이 섞이지 않도록 별도 가지로 둔다.
> “관측0 실패=재해 예측불가”, “surprise=재해”, “S1 날짜정렬 원인배제”는 채택하지 않는다.
> draft JSON은 `launch_allowed=false`; 날짜/clean validation/기억대조/λ·gate·외부holdout을
> 실행 전 동결해야 한다. 이번 준비는 서버/GPU/production/prereg/Korea 봉인을 건드리지 않았다.

> ## 2026-09-09 — MS-117~120 최신 검토부터 읽기
>
> **현재 입구:** [최신 연구판](docs/STREAMING_RESEARCH_UPDATE_2026_09_09.md).
> Kuro의 어제 false-DONE은 이후 updater 재실행으로 수치상 회복됐다.9개 updater의 finite
> validation/skip0,3개 eval의12-arm 완결 및 원 산술 gate 통과를 확인했다. 단 val/test 각1개
> NaN 제외 + 제외 전 decoder 선택을 유지한 상태이므로 clean 확증 보류. 기존 AP를 폐기하지 않는다.
>
> **핵심 남은 질문:** GRU가 새 영상만 읽어도 되는 것이 아니라 과거 캐시를 실제 활용하는가?
> POST_ONLY/NO_MEMORY 대조가 미실행이다. 같은 가중치 S2→S1 zero-shot·공유3task도 미검증이다.
> 구조36/36 gate미달, 교차센서2지역만 완료(나머지 test0/val0), 이탈리아 크기층화까지 완료됐다.
> 이탈리아 raw 미실행이며 “해상도/IoU만의 문제”라는 인과 해석은 보류한다.
>
> **다음 제안:** validation/coverage 복구 → 기억 기여 대조 → 도착 스케줄 비용 → DEN 시간별 평가
> → Korea3task. 원 prereg·active queue는 변경하지 않았고 새 GPU 실행도 하지 않았다.
> 과거의 여러 SSOT/다음 순서는 날짜별 이력이다. 실험 추가 시 별도 amendment가 필요하다.

> ## 2026-09-08 오후 — 서버 재점검: 외부 갱신 실험은 현재 실행 무효
>
> **확인:** KuroSiwo S1 7,000 cache·판독기3개 완료. 사건 27/6/10개, 모든 split 쌍의
> sample/activation 중복0 재확인. 판독기 validation AP .662/.664/.688, **test 결과 아님**.
> **추가 발견:** 이 AP도 검증 입력 NaN이 유효 영역에 번진 상태라 재검토해야 한다.
>
> **P0:** GRU seed1은 val NaN 30epoch인데 DONE/rc0를 남겼다. 유효한 updater·최종 평가 없음.
> 방법 실패로 세지 말고 finite 검사·유효 checkpoint·fail-closed chain을 갖춘 별도 수정본으로
> 재현해야 한다. 훈련 teacher 전수 finite·전체 std .488538은 CPU에서 확인했으므로 scale 자체는
> 정상이다. 후속 전수 검사에서 validation **ks_04357**의 원시 NaN1,338개와 오염 cache를
> 확인했다(유효 라벨 포함327토큰). 데이터 전처리/캐시 해당 revision을 복구한 뒤 검증/학습을
> 재검토한다. 빈 타일로 삭제하거나 NaN을0손실로 바꾸지 않는다. 기존 산출물은 보존한다.
>
> **T1 구조:** 보존본22/36, 기본대조36/36. Δt는 3지역 평균 AP +.009~+.012지만 +5%p/3-of-4
> gate에는 못 미친다. 공간/attention도 완료 두 지역에서 미달이라 세 arm 모두 승격 gate는
> 남은 지역만으로 도달 불가다. 미완료를 음성으로 기록하거나 runner를 감사자가 중단하지 않았다.
>
> **다음:** 정상 S1 갱신 실행 복구 → 외부 사건 평가 → POST_ONLY와 온라인 비용 검증.
> frozen v0는 이미 실행됐으므로 오전의 "동결 전 권고"는 이제 별도 amendment 제안이다.
> 최신 근거/결함/해석은 [큰 그림 §8](docs/BIG_PICTURE_STREAMING_EARTH_2026_09_08.md#8-2026-09-08-오후-재감사--실제-진전과-실행-무효를-분리한다).

> ## 2026-09-08 오전 — 현재 큰 그림과 다음 검증
>
> **집중 질문:** OLMoEarth의 저장 상태를 새 관측으로 갱신해 과거 전체 재인코딩 없이
> downstream 지도를 유지할 수 있는가? 이전 cache/few-shot은 기반, MS-116은 시간 갱신 개발,
> KuroSiwo는 다음 외부 사건 시험, Korea 3-task는 후속 공유 상태 검증이다.
>
> **최신 장부:** MS-116은 4노출 지역·판독기 3seed·갱신기 3seed로 확대됐다. EMA/noobs/calib,
> checkpoint 저장 및 logit-MSE 보존 손실도 실행됐다. 아래 00:03의 9/18·미완료 표시는 과거 snapshot이다.
>
> **비용 코드 검토:** 2.3배는 새 8시점을 한 번에 묶은 메모리 상주 batch-compute 결과이지
> 순차 도착 온라인 latency가 아니다. 4.5배는 update-only 논리 입력량이며 실측 I/O가 아니다
> (초기 포함 3.33배). 다음 계측은 cutoff별 실제 가용 관측으로 맞춘다.
>
> **다음:** KuroSiwo 초안의 POST_ONLY·사건 분리·실제 취득일·단위/mask·head/crop을 동결 전
> 보완한다. SAR에서 updater를 재학습하는 것은 방법의 외부 재현이지 같은 가중치의 S2→S1
> zero-shot이 아니다. 현재 초안·서버 queue·GPU·Korea 봉인은 이번 검토에서 변경하지 않았다.
>
> **현재 설명/설계 입구:** [큰 그림과 KuroSiwo 체크리스트](docs/BIG_PICTURE_STREAMING_EARTH_2026_09_08.md).
> 이 블록은 최신 장부·코드 검토이며 서버 전수 재계산은 아니다. 아래 인수인계와 기존 gate는 이력으로 보존한다.

> ## 2026-09-08 00:03 KST — T1 streaming utility 독립 감사
>
> **최신 확인**: Hiroshima GRU 3seed AP `.525479`, full teacher `.549624`, frozen c4 `.013356`.
> teacher 격차 95.50% 회복, 절대 AP gap `.024145`. 두 지역 residual은 54.10%/24.49%로
> 원래 방법의 90%-양지역 필요조건 실패. 전체 9/18 완료이며 Chimanimani GRU·양쪽 EMA는 미완료.
>
> **현재 SSOT 보충**: [`docs/T1_GRU_UTILITY_AUDIT_2026_09_07.md`](docs/T1_GRU_UTILITY_AUDIT_2026_09_07.md).
> GRU354만/residual472만은 strict parameter-matched가 아니고 36→12는 초기비용 비대칭이다.
> 실제 GPU speedup 미측정. e4→e12 head contract 보정과 새 관측의 기여를 분리해야 한다.
> EMA는 scalar를 학습하는 baseline이다. T1은 exposed development이며 decoder seed1 하나다.
>
> **다음**: 원계약 T1 종료 → 별도 실행에서 updater/score/ID/snapshot 저장 → no-new-input 대조 →
> 같은 GRU의 task-fidelity loss·실제 latency → 외부 stream/지역 검증. 새 설계 JSON은
> `config/t1_evidence_and_fidelity_review_v1_draft.json`이며 **미등록·미실행**이다.
> 과거 gate/실행경로는 덮어쓰지 않았고 Korea label은 그대로 sealed. 아래 상태는 이전 인수인계다.
>
> ## 2026-09-07 KST 최신 감사 — MS-113 하향 정정 + patch-2 fail-closed
>
> **상태 한 줄**: MS-114의 field-adaptation 방법 가지는 kill-gate 실패로 닫혔다. MS-113의
> “라벨 5장 포화/추가 라벨 낭비”도 철회한다(K=20 `.317` > pool `.299`, non-nested support,
> query-label FP threshold). 살아 있는 사실은 “현재 head/recipe에서 K=5와 pool의 aggregate gap이
> 작고 method gate가 열리지 않았다”까지다.
>
> **다음 실험**: `docs/MS113_114_PATCH2_AUDIT_2026_09_07.md`의 4-arm cache-contract screen.
> 첫 patch-2 추출은 decoder 0건 전에 중단했다. validator의 64×64 shape 결함과 audit 실패 후 계속
> 실행하는 runner 결함을 수정했으며, GPU1이 비었을 때만 새 `resolution_contract_v2` OUTROOT에서
> P4_NATIVE_CONTROL / P4_UPSAMPLE2 / P2_NATIVE / P2_AVGPOOL2를 실행한다. Sen12는 development이며 Korea/외부 task에서만
> 확인한다.
>
> **새 계약**: `config/label_efficiency_curve_prereg_v0.json` ·
> `config/second_fm_cache_prereg_v1_draft.json` addendum v1d. Korea label은 계속 sealed다.
> 서버 파일 SHA 7/7 일치 후 GPU1 waiter PID `349969`을 걸었다(60초 poll, 24시간 제한). 상태는
> `./bin/nx sh 'tail -20 /home/work/data/olmoearth/logs/gpu1_waiter.log'`로 확인한다.
>
> ## 2026-09-06 23:55 KST 최신 감사 — MS-112 정정 + 외부 검증 계약 v1
>
> **상태 한 줄**: core empirical result(Sen12+Solar cache reuse/few-shot)은 유지된다. 그러나
> MS-112의 `G0-A 승자 3종/G0-B .085 통과` 해석은 **철회**한다. v3 action matrix가 비직사각이고
> cache-contract 행이 단일 seed였으며, 계산기가 누락 action을 교집합에서 조용히 제외했다.
> 3행동×3seed를 명시해 재감사하면 `INCOMPLETE_ACTION_MATRIX_DIAGNOSTIC_ONLY`, G0=False다.
>
> **현재 SSOT**: `docs/MS112_CVPR_AND_KOREA_AUDIT_2026_09_06.md` ·
> `config/geobench_cache_action_prereg_v1.json` ·
> `config/korea_shared_cache_3task_prereg_v1_amendment.json`.
> 아래 20:30 인수인계의 v0/MS-111 설명은 provenance로 보존하지만 실험 지시는 이 최신 블록이
> 대체한다.
>
> **다음 임계경로**: (1) DEN cache 단일-writer 전수 감사 → (2) external task별 anchor/head/cost
> 동결 → (3) S-support 직사각 행렬(CACHED/ADAPT/RAW×3seed) → (4) 독립 task 수준 G0 재판정.
> G0가 실제로 통과할 때만 selector를 학습한다. Korea label은 transient error 6건과 selection-bias
> gate를 닫고 v1 amendment를 commit하기 전까지 열지 않는다.
>
> ## 2026-09-06 20:30 KST 인수인계 (다른 컴퓨터에서 이어받기)
>
> **상태 한 줄**: EarthCache = "새 EO 과업에 라벨 없이 cache 재사용/적응/재계산을 비용·성능으로 고른다".
> G0(필요조건 게이트) 계약을 IIA-safe로 확정. 지금은 **action matrix를 채우기 위한 GEO-Bench 데이터 확보 단계**.
>
> **상태 한 번에**: `./bin/nx sh 'bash /home/work/data/olmoearth/code/status.sh'`
>
> **방향(2026-09-06 확정, 다시 안 바꿈)**: `docs/EARTHCACHE_ROADMAP_G0_FIRST.md`가 SSOT.
> 3단계 = [G0 필요조건] → 통과 시 [S 선택기·CVPR main] / 불통과 시 [A EarthCacheBench 특성화] → [E 한국 3-task 추가 외부].
> 계약 = `config/geobench_cache_action_prereg_v0.json`. 계산기 = `code/geobench_action_headroom.py`(테스트 9/9, IIA 증명).
>
> **G0 재판정(MS-111-v2, 시드별 행·실측 적응비용·고정 anchor)**: G0-A 불통과(robust 승자 = HEAD_ADAPT 하나, Solar는 시드 간 역전), G0-B headroom .017 < .02 불통과. 개발 2과업에서는 선택기 불필요. 진짜 REEMBED 미측정.
>
> **G0 이전 판정(개발 2과업, MS-111-정정)**: G0-A 이질성 신호 있음(Sen12→HEAD_ADAPT +.036, Solar→CACHED_HEAD +.009)
> 단 시드 1개. G0-B 가치는 **계산 불가**(고정 anchor·진짜 REEMBED·실측 비용 없음). G0_pass=False.
> 초판의 headroom .013/.500은 **정규화 인공물이라 폐기**. RAW_FINETUNE(=raw 재학습)과 진짜 REEMBED(=인코더 재실행) 구분됨.
>
> **GEO-Bench 데이터 (geobench2/<ds>/, .venv-geobench, HF 공식 다운로더 사용)**
> - ✅ fotw(검증OK, RGB+NIR 4밴드) · DynamicEarthNet(검증OK, s2 10밴드+planet) · benv2/biomassters(preflight OK, 우리 10밴드 보유)
> - 🔄 benv2→biomassters 다운로드 중: `logs/dl_cls_reg.log`, `code/run_geobench_cls_reg.sh`. 완료 마커 `logs/{benv2,biomassters}_DONE.json`.
> - 🔄 DEN OlmoEarth 캐시 `olmo_den/`: GPU1에서 추출 중(`logs/x_olmo_den.log`, 16,000칩; 이전 실패 원인 = 캐시 없는 소스에서 타일 id를 못 찾던 버그, 수정 커밋).
> - ❌ BioMassters 0000 파트 sha256 불일치(PASTIS와 같은 유형).
> - ❌ **PASTIS 0001 파트: HF 공식 도구로도 sha256 불일치**(got 3f1e98e3 vs want 7d0463a6). GeoBench 쪽 sha256str 오류로 판단 — 우리가 못 고침. PASTIS는 보류, 나머지로 진행.
>
> **다음 순서(로드맵 §8)**
> 1. benv2·biomassters 확보 완료 확인(위 마커) → 계약 감사(chipping 128px, head_type, 시간선택)만 하고 protocol freeze.
> 2. **각 과업에 고정 anchor 선언**(lower=고정 supervised baseline, upper=full-label 천장) — G0-B 계산의 전제.
> 3. GPU 비면 action matrix: CACHED_HEAD/HEAD_ADAPT 먼저(값쌈), RAW_FINETUNE·진짜 REEMBED는 sequential stopping.
>    3~4 과업 + anchor + 실측비용 채워지면 G0 재실행 → 트랙 S/A 분기.
>
> **GPU 규약 4b**: GPU1만, 남의 프로세스 있으면 중단. `code/gpu1_waiter.sh <chain>`이 GPU 비는 순간 자동 기동.
>   지금 GPU0/1 타 사용자 점유 잦음. Solar 2nd-FM(MS-109)은 완료.
>
> **재발 방지**: `code/preflight.py`(비싼 단계 전 5초 계약검사) · `code/test_chain_rc_pattern.py`(rc 버그 린터) ·
>   "다운로드와 검증을 한 사슬에 묶지 않는다" · 대형 LFS는 자작 다운로더 말고 HF 공식 도구.
>

## 한 문장 연구 질문

> **세계·과업·모델 release가 바뀔 때, 저장된 Earth embedding과 downstream head를 그대로 쓰고,
> head만 적응하고, 표현을 migration하고, 재임베딩하거나 라벨을 더 요청할 시점을 정확도와
> label·GPU·raw-I/O·cache invalidation 비용으로 결정할 수 있는가?**

세 축을 별도 논문으로 벌리지 않는다.

- **A — release migration**: OlmoEarth v1→v1.2에서 old cache/index/head를 살리는가.
- **B — product validity**: Clay·AlphaEarth 등 다른 embedding product에서도 같은 결정 문제가
  성립하는가.
- **C — safe action**: support label과 contract만 보고 A0 reuse/A1 head-adapt/A3 re-embed/
  REQUEST를 고르는가.

설계 SSOT는 `docs/ABC_EMBEDDING_CONTINUITY_2026_09_04.md`다.

## 지금까지 닫힌 양성 결과

### Task 1 — Sen12Landslides, 8개 geographic holdout

- source-only frozen OlmoEarth cache + decoder(P4): region-macro `.2722`.
- raw UNet3D(P2) `.1966`, raw U-TAE(P3) `.1834`; 최고 raw 대비 7/8 지역 우위.
- target tile K=5/20에서 A1 cache-head adaptation은 raw full A4w를 8/8, parameter-matched
  A4h를 방향 기준 **7/8** 이겼다(MS-96/97). fixed-exposure에서도 A1>A4w 8/8이다.
- 그러나 K=5에서 A1>A0는 5/8뿐이다. `라벨 5장이면 항상 적응`은 금지한다.

### Task 2 — Solar Farm, 8개 UTM-zone fold group

- A0 cache no-adapt `.591`, stratified A1 K=5 `.582`, K=20 `.609`.
- raw A4w K=5/20 `.240/.245`, A4h `.257/.291`; cache pathway가 raw adaptation을 8/8
  이겼다(MS-98/99).
- random K=5에서 support 12/24가 positive tile 0장이었고 A1이 `.426`으로 붕괴했다.
- tie-correct AP는 A0 `.9252`가 A1 K=5/20 `.8651/.9044`보다 높다. 따라서 action은 task뿐
  아니라 support 구성과 배포 utility/threshold 계약에 의존한다.
- Solar fold는 독립 지역 8개가 아니라 UTM-zone 기반 group이다. Sen12의 `8지역`과 같은
  일반화 단위라고 부르지 않는다.

### Model shift

- M1: 같은 scene의 OlmoEarth v1/v1.2 token identity R@1은 양방향 0. Procrustes·affine ridge도
  등록된 retrieval compatibility gate를 실패했다. pooled CKA `.979`는 task continuity 증거가 아니다.
- M85는 v1/v1.2 radar/optical 성능 비교이지 cache migration 실험이 아니다.

## 닫힌 음성 방향 — 이름을 바꿔 되살리지 않는다

- CacheTune A2 low-rank spatial residual: MS-94에서 A1보다 `.05–.08` 낮아 stop rule 발동.
- prediction fusion·GeoContextGate: MS-90B/91/92 종료. FP-matched oracle headroom도 거의 0.
- label-free winner router와 block routing: 등록 gate 실패.
- MoE: action complementarity와 untouched selector 성공 전에는 열지 않는다.

## 2026-09-04 Clay v0 실행 경계

서버에서 확인된 사실:

- `clay_cache`: 6,834/6,834, 실패 0, Clay v1.5, 1024-d.
- 그러나 native `16×16`을 bilinear로 `32×32`에 확대한 cache다. audit의
  `all_gates_pass=true`는 파일 완결성만 보증하며 비교 공정성을 보증하지 않는다.
- source decoder chain과 few-shot wait chain이 실행 중이었다. 확증 실행 중 서버 코드 push 금지
  규칙에 따라 이 검토에서는 서버 코드를 건드리지 않았다.
- 이 v0 결과는 **interpolated deployment-adapter exploratory baseline**으로만 보존한다.
  B1 common-physical 또는 B2 native confirmatory 결과로 쓰지 않는다.
- current Clay few-shot FP-matched IoU는 Clay A0의 FP budget을 쓰고, 비교하려는 historical raw
  A4는 OlmoEarth A0 budget을 썼다. 서로 다른 작동점이므로 report 간 primary IoU 비교는 금지한다.
  threshold-free tie-correct AP만 탐색 비교할 수 있다.

## 아직 주장할 수 없는 것

- A1이 A0보다 항상 낫다.
- Clay 하나로 product-agnostic 원리가 증명됐다.
- v1→v1.2 bridge가 old task head/index를 보존한다. 아직 downstream migration은 0회다.
- AlphaEarth 연간 embedding으로 사건 전후/실시간 변화를 측정한다.
- support label 없이 self-training하면 정확도가 개선된다.
- Korea sealed target 또는 독립 Task-3에서 safe action policy가 통과했다.
- OLMoEarth가 모든 GeoFM보다 보편적으로 우월하다.

## 다음 실행 순서

1. **P0 증거 복구(CPU)**: Solar random support ID/양성 수/SHA, WGS84 cross-CRS distance,
   exact-query A4w0, 서버 48-run 원시 report/snapshot을 로컬 봉인한다.
2. **A release migration**: full-12-band Solar의 exposed 2fold에서 v1/v1.2 exact-scene bridge가
   old-head AP·fixed-threshold IoU를 보존하는지 screen한다. 새 query raw read는 정상 비용이고,
   피하는 것은 과거 archive 전체의 raw backfill이다.
3. **B-v1 Clay**: v0 종료 뒤 native 16×16 smoke를 새로 봉인한다. B1은 80 m·16×16·256-d
   compact cache, B2는 native product로 분리하고 같은 report/threshold에서 raw를 재평가한다.
4. **C A3 ceiling + deterministic guardrail**: exposed 4 unit, K=20에서 공식
   LayerDecayAdamW와 q/v LoRA sensitivity를 측정한다. positive 0이면 A0/REQUEST, 그 외에도
   leave-one-tile-out 하한이 0보다 클 때만 A1을 허용한다.
5. **untouched first-look**: policy와 threshold를 동결한 뒤 독립 Task-3 또는 Korea sealed target을
   한 번만 연다.

AlphaEarth는 Solar/static mapping B2 뒤에만 둔다. 연간 64-d product이므로 event Sen12와 같은
시간 gate에 넣지 않는다. Prithvi는 Clay 뒤 contract-shift sensitivity다.

## 읽는 순서

1. 이 파일
2. `docs/ABC_EMBEDDING_CONTINUITY_2026_09_04.md`
3. `docs/ASSET_INVENTORY.md`
4. `docs/CRITICAL_PATH.md`
5. `MEASURED_FINDINGS.md`
6. `docs/PAPER_NARRATIVE_2026_08_31.md` — 이전 narrative, 현재 A/B/C 문서가 실행 방향을 대체
7. `GOAL.md` 마지막 Worklog

기계 판독 사전등록 파일(파일명에는 `draft`가 남았으나 `efce8b8`에 실험 전 커밋됨):

- `config/release_migration_prereg_draft_v0.json`
- `config/second_fm_cache_prereg_v1_draft.json`
- `config/safe_cache_action_prereg_draft_v0.json`

결과 뒤 발견된 gate/metric 결함은 원 파일을 조용히 고치지 않고 dated amendment와 재생성 report로
남긴다. 결과가 나온 뒤 문구·gate를 바꾸면 사전등록이 아니다.

공식 Ai2 checkout `..`에는 사용자 수정이 남아 있다. 연구 재시작 작업에서
`olmoearth_run_data/forest_loss_driver/{dataset.json,model.yaml}`과 `.pnpm-store/`를 건드리지 않는다.

> **2026-09-05 18:40**: `nepal-live-twin`(DDanggle/eo-rasuwa)은 이 작업공간 관리 대상에서 제외함. 여기서 푸시·동기화하지 않음.
>
> **2026-09-05 18:30 추가**: 서버 체인 `code/arch_axes_chain.sh`(`logs/arch_axes.log`, 끝 표시 `ARCH_AXES_DONE`) — 아키텍처 축 7캐시(olmo_nano/tiny/base_half, galileo_nano/tiny/base_half, clay_in256_half) → 8폴드 디코더. 표: `code/bv1_summary.py`에 캐시 이름 추가해 실행. 가설·판독 규칙: `config/second_fm_cache_prereg_v1_draft.json` addendum_v1b.
