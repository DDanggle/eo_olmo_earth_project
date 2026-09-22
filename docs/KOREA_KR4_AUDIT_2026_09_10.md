# Korea KR-4 독립 감사 — 공유 캐시는 작동, 비교·전이·비용 주장은 조건부

검토일: 2026-09-10 KST. 연구 저장소 기준 `0421ffe`. 범위는 서버 원 보고서·manifest·현재 코드의
읽기 및 CPU 산술 검증이다. 새 학습, GPU 추론, label raster 다운로드/재채점, PR 제출은 하지 않았다.

## 1. 판정과 큰 그림

**실제 진전은 있다.** 한 물리적 OLMoEarth 캐시를 세 과업에서 재사용했고, 현재 구현의
토지피복·산사태 점수는 raw보다 높다. 그러나 **공정 비교 완료, Sen12 head의 한국 전이,
공유 상태의 시간 갱신, 세 과업 end-to-end 비용 절감**까지 확인된 것은 아니다.

현재 연결은 다음과 같다.

```
한국 다중시점 영상 → frozen OLMoEarth full-window cache 1개
                    ├─ 한국 라벨로 새 토지피복 head 학습
                    ├─ 한국 라벨로 새 벌목지 head 학습
                    └─ 한국 라벨로 새 산사태 head 학습

아직 안 한 연결: 새 영상 → 단일시점 embedding → 공유 updater → 기존 세 head 동시 갱신
아직 안 한 전이: Sen12에서 학습한 head → 한국 무적응 / 동일 head K-shot 적응
```

한국은 이제 **라벨과 성능을 본 외부 데이터셋**이다. 한국에서 설계·튜닝을 계속할 수 있지만
동일 test를 다시 “untouched 최초 확증”이라고 부르지 않는다. v1 실패 판정과 산출물은 보존한다.

## 2. 원 보고서에서 확인한 수치

| 과업 / 현재 구현 지표 | FULL_CACHE | FULL_RAW | 절대 차이 |
|---|---:|---:|---:|
| 토지피복 cluster-macro mIoU¹ | .202677 | .137058 | +.065619 |
| 산사태 pooled AP | .060021 | .001504 | +.058517 |
| 벌목지 pooled AP | .001437 | .000200 | +.001237 |

FULL 참조는 각각 **한 번 학습한 모델**이다. 토지피복은 이 모델 쌍에서 7/7 test 군집 모두
cache가 높다. 이는 7개 독립 학습 재현이나 seed-robust 확증과 다르다.

- train/val/test = 25,152 / 4,992 / 7,232칩. 합계 37,376칩, 584개 원본 공간 타일.
- 토지피복 K=20 random: .157529 vs .130886, 정확한 값으로도 cache 3/3 승.
  K=5는 평균 .150418 vs .128533이지만 1승·1무·1패다.
- 산사태 K=20 positive-aware: .115290 vs .002458, cache 3/3 승.
  **총 20개 정답 칩 = 양성 10 + 음성 10**이다. “정답 10장만 사용”은 틀리다.
  cache 세 값은 .1822/.1032/.0604로 변동도 크다. FULL_CACHE .060보다 평균이 높지만
  다른 support 구성·학습 조건·단일 FULL 참조이므로 “더 많은 라벨이 해롭다/10장에서 포화”는 아니다.
- 산사태 random K=5/20은 각 3개 draw 모두 양성 칩 0개. v1 주 규칙 불통과는 유지한다.
- positive-aware는 전체 train mask에서 양성 칩 목록을 만든 뒤 뽑는다. 알려진 양성 목록이
  있는 조건부 학습 실험이지, 양성을 찾는 비용까지 포함한 순수 20-chip annotation 예산은 아니다.
- 양성 support가 없었던 것은 확인됐다. 그것이 주 규칙 실패의 **유일한 원인**이라는 주장은 미검증이다.

¹ 지표 구현의 GT-present 클래스 조건은 §3에서 구분한다. 위 수치는 해당 v1 정의의 값이다.

## 3. 다음 학습 전에 해결할 실제 결함과 비교 조건

### P1-A. test 64칩의 cache/raw 시간 범위가 다르다

`SA0300000001`(test C05)의 마지막 라벨 `20220924`는 손상 파일이다. 파이프라인은 마지막
존재 라벨 `20220427`로 물러나고 raw 입력도 그 날짜까지 잘라낸다. 하지만 사전 추출된 cache는
라벨을 읽지 않아 전체 날짜, 즉 **20220924까지** 사용한다.

- manifest·라벨 오류 인벤토리로 이 원본 타일의 64칩을 식별했다.
- 서버에서 대표 칩 `SA0300000001_r0c0`의 마지막 mask가 20220427인 것도 확인했다.
- 이는 미래정보/입력 범위 비대칭이다. 이로 인한 전체 성능 차이의 크기는 아직 재채점하지 않았다.
- 근거: `code/korea_3task_pipeline.py:19`(label cutoff), `:33`(raw cutoff),
  `code/extract_korea_cache.py:43`(전체 keys 사용).
- 조치 제안: 현재 라벨 날짜 이하의 동일 시점 집합으로 두 arm을 맞춘 별도 cache revision을 만든다.
  기존 v1은 보존하고 전체/해당 타일 제외 sensitivity도 병기한다. 제외만으로 본 문제를 숨기지 않는다.

### P1-B. FULL은 같은 스텝이지 같은 샘플 노출 예산이 아니다

`korea_3task_pipeline.py:136,139`: 4,000 steps × cache batch32 / raw batch16.
따라서 학습 샘플 노출은 **128,000 vs 64,000**, 2배다. 사전학습, 입력 밴드, 구조, wall time도 다르다.
문구를 “같은 step 수의 제한된 raw 참조”로 좁힌다. 이 차이가 우위를 전부 설명한다는 뜻도 아니다.

반대로 **K-shot은 이미 같은 support·300 steps·min(K,8) batch**다. RAW_K 스텝을 늘리는 것은
“노출을 같게 만드는 수정”이 아니라 추가 최적화/수렴 감도 분석이다. 동등한 튜닝 기회를 정의해야 한다.
raw는 새 RawUNetT(시점별 2D encoder + temporal max/mean + decoder)이며 공식 UNet3D가 아니다.
cache는 10-band primary view(B01/B09 MISSING), raw는 12밴드·별도 정규화여서 순수 표현만의 인과 비교도 아니다.

### P1-C. seed 기록이 가중치 초기화를 통제하지 않는다

모델은 caller에서 생성하고, `train()` 진입 후에야 `torch.manual_seed(seed)`를 호출한다
(`:68`, `:136`, `:149`). support draw는 독립적으로 고정되지만 가중치 초기화는 앞 arm의 RNG
소비·실행 순서에 의존한다. “완전히 재현 가능한 3-seed 학습”을 보증하지 못한다.
다음 revision에서는 모델 생성 전에 seed를 고정하고 arm 순서/재시작 불변성을 테스트한다.

### P1-D. raw의 가변 시계열 padding이 실제 관측처럼 평균에 들어간다

`np.pad(..., mode='edge')`로 마지막 관측을 반복하고 `RawUNetT.tp`가 mask 없는 평균을 낸다
(`:59`, `:80`, `:96`). 예: `[1,3]`의 평균2가 `[1,3,3,3]`에서는2.5다.
훈련에서 서로 다른 길이의 시계열을 섞으면 의미가 달라진다. mask-aware temporal pooling 또는
명시적인 length grouping이 필요하다. 현재 test batching에서 실제 점수가 얼마나 변했는지는 미측정이다.

### P2. 지표와 결과 보존도 보강해야 한다

- **토지피복 mIoU:** `:109`의 `if t.sum()>0`는 각 군집에서 GT에 없는 클래스의 FP-only IoU를
  평균에서 제외한다. union-present mIoU와 다르다. 실제 evaluate 함수를 CPU로 실행한 2픽셀
  반례에서 v1=.5, union-present=.25. v1을 지우지 말고 지표명을 명확히 하고 표준 정의를 함께 재평가한다.
- 코드70/80은 GT로 제외하므로 토지피복은 “벌목/산사태 정답 영역 밖의 7-class 평가”다.
  세 과업이 완전히 독립인 것처럼 설명하지 않는다.
- binary exact AP의 tie/no-positive 처리는 소형 테스트에서 맞았다. 그러나 원 per-pixel 예측을
  다시 계산한 검증은 아니다. 원 등록의 FP-matched IoU·task interval은 보고서에 없다.
- `korea_3task_summary.py`는 반올림 후 seed 승수를 센다. 새 감사기는 원 float와 seed ID로 짝짓는다.
- K-shot checkpoints/per-pixel probabilities는 해당 run 폴더에 저장돼 있지 않다. 현재 source와
  서버 source SHA 일치는 확인했지만 **실행 전 봉인 사본/당시 실행 코드의 증명은 아니다**.
- cache audit의 `all_gates_pass`는 존재 개수·skip0 검사다. 전체 dtype/shape/finite/내용 해시를
  검증했다는 뜻은 아니다. 이번 감사도 37,376개 array 전수 내용 검사까지 하지는 않았다.

## 4. 희소성과 비용을 정확히 읽기

### 희소성

마지막 라벨 기준 test 양성은 산사태79,098픽셀(0.06676%), 벌목11,952픽셀(0.01009%).
전체 날짜를 합친 인벤토리의 양성률은 K-shot 지원 모집단의 양성 칩 비율이 아니다.

- 산사태 양성은 C05/C08/C12 세 군집에만 있으며 C08+C12에 약97.8%가 몰려 있다.
- 벌목 양성은 다섯 군집에 있다. AP는 계산 가능하므로 “측정 자체가 불가능”은 과장이다.
  낮은 AP·집중도·불확실성 때문에 실용적 탐지 능력의 입증이 약하다고 표현한다.
- raw 산사태 AP .001504는 query 기저율 .000668보다 높다. “전혀 학습 못함”보다 “현재 recipe에서
  성능이 매우 낮음”이 정확하다. 40배 AP 비율은 재현율/실용성 40배가 아니다.
- “40m라 소형 클래스가 원리적으로 붕괴”도 여기서는 원인 가설이다. 라벨 정렬·클래스 불균형·raw
  비교·threshold·학습을 분리하지 않았으므로 토큰 크기만으로 인과를 확정하지 않는다.
- 무양성 support 확률은 칩 수 N과 **양성 칩 수 M**에서 `C(N-M,K)/C(N,K)`로 계산해야 한다.
  픽셀 양성률 .06%를 칩 성공확률에 넣으면 틀린다. 현장 screening 비용도 별도 기록한다.

### 비용

원 보고서 시간을 더한 **부분 성분 비교**는 다음과 같다. end-to-end 계측을 새로 한 것이 아니다.

| 성분 | 초 |
|---|---:|
| cache 추출 보고시간(1회) | 2,668.413 |
| 세 FULL_CACHE head 학습 합 | 1,283.303 |
| 위 두 항의 합 | 3,951.716 |
| 세 FULL_RAW 학습 합 | 2,043.273 |

이 부분 합만으로도 cold cache 경로는 raw 학습 합의 **1.93배**다. 따라서 K-shot 학습 구간의
약2.5배 속도를 “한국 세 과업 전체 비용 절감”으로 확장하지 않는다. warm cache 재사용이 유리할
가능성과 실제 손익분기는 별도다. raw도 공통 ingestion을 한 번만 계산하는 공정 시스템 비교가 필요하다.

`gpu_s`는 CUDA synchronize를 포함한 host wall time이며 GPU kernel time 전용 값이 아니다.
small-support preload의 bytes가 누적되지 않아 **모든 K-shot train_bytes_read=0**이다.
이는 raw 읽기가 없다는 증거가 아니다. raw bytes는 반복 접근한 논리 array slice 용량이지 물리 디스크 I/O도 아니다.
선언 shape로 계산한 fp16 payload는 58.787 GB / 54.75 GiB이고 실제 파일/디스크 용량은 이번에 재측정하지 않았다.

## 5. 다음 실행 순서 — 새 아이디어보다 비교 계약부터

1. **v1 보존 후 v2 실행 규격:** 공통 cutoff·밴드·mask·정규화 계약, seed-before-init,
   raw padding, 명시적 지표, checkpoint/예측/실행 전 source snapshot 저장을 먼저 준비한다.
2. **좁은 재검증:** 기존 support ID를 고정한 디버깅 replay와 full exposure-matched sensitivity.
   새 support를 뽑아 성능을 올리는 것보다 원 결과가 어느 수정에 민감한지 먼저 본다.
3. **라벨 효율을 별도 질문으로:** uniform random과 명시적으로 양성을 확보하는 현장 조건을
   둘 다 평가한다. 후자는 “라벨 밀도에 맞춘 random”이라는 이름으로 기존 random 실패를 대체하지 않는다.
4. **Q3 전이:** Sen12 checkpoint 로드·공통 출력/클래스 계약 검증 후 무적응/동일 head 적응을 분리한다.
   지금 코드는 Korea에서 head를 새로 만든다. pretrained encoder 전이와 source-task head 전이를 구별한다.
5. **한국 공유 streaming:** 정렬된 시점별 cache와 cutoff teacher를 만든 뒤 *동일 업데이트 상태*를
   세 고정 head가 소비하게 한다. stale/full-reencode/POST_ONLY/GRU를 공통 날짜·칩으로 비교하고
   과업별 품질 저하와 raw-read/latency를 함께 잰다. 시점 1개 타일은 update 평가 불가로 별도 집계한다.
   Korea 성능을 본 뒤 설계한 실험임을 표시하고 일반화 확증은 미본 사건/시간에서 수행한다.

서버 2026-09-10 15:08 KST: GPU1 0 MiB, GPU0는 다른 작업 사용 중. 한국 작업 실행 없음은
앞선 프로세스 조회에서 확인. `korea_cache_v1/single_fp16` 파일은 **0개**였다.
즉 “한국에서 세 head가 같은 갱신 상태를 이미 썼다”는 결과는 아직 없다.
이번 감사에서 학습/중단/자동감시/서버 코드 수정은 하지 않았다.

## 6. OLMoEarth PR 준비 노트 위치

- [PR_DOSSIER.md](../PR_DOSSIER.md): 제출 전 증상·원인·패치·검증·우선순위의 본 문서.
- [PR_REVIEW_NOTES.md](../PR_REVIEW_NOTES.md): 제출 큐·예상 리뷰 답변·제출 후 기록.
- [pr_bodies/01_sample_schema.md](../pr_bodies/01_sample_schema.md): 첫 sample-schema PR 영문 본문.
- [ISSUE_DRAFT_lfmc.md](../ISSUE_DRAFT_lfmc.md): LFMC checkpoint 이슈 영문 초안.
- [외부 데이터 onboarding 및 PR 감사](OLMO_EXTERNAL_DATA_ONBOARDING_AND_PR_AUDIT_2026_08_26.md).

문서 마지막 upstream 검증일은 **2026-08-26**. 당시 첫 후보는 `es_* → oe_*` sample schema,
로컬 branch `fix/sample-annotation-oe-schema`, commits `5e044ee`/`21b658a`다.
PR_REVIEW_NOTES에는 **실제 제출 URL이 기록돼 있지 않다**. 오늘 GitHub 전체 현황을 재검색한
것은 아니므로 현재도 중복이 없다는 뜻이 아니다. 제출 전 최신 upstream/중복/replay를 재확인한다.
이번에 찾은 KR-4 trainer 결함은 우리 연구 코드 문제이지 Ai2 upstream 결함으로 보내지 않는다.

## 7. 재현과 근거 보존

소형 원본 report/manifest/inventory/current source:
`artifacts/korea_review_20260910/`. 원시 영상·label raster·credentials는 내려받지 않았다.

| 입력 | SHA-256 |
|---|---|
| run_v1/report.json | `00fbde970f1999c6a4865cc52971d03a7f7d3e66ff08bf84be21fa2e95dff0f4` |
| run_v1_fullraw/report.json | `f6661711c193117f016798dd5d1bee822bf3944b61d8ccbb52fe6bc2ec524337` |
| korea_chip_manifest.jsonl | `873e635a580a6bd8c8a589a336f81c33bea92bda57917aa779abeae797f33a5d` |
| 현재 local/server trainer | `cb32955a1a3b9fb2928dbb0b30501246ad04786e7a48ce4f3e5609ad90c310bb` |

해시는 내려받은 현재 산출물의 정체성을 고정하며 **원 실행 전 봉인을 소급 증명하지 않는다**.

```bash
python3 code/audit_korea_3task_results.py --root artifacts/korea_review_20260910 --output artifacts/korea_review_20260910/audit_summary.json
python3 -m unittest discover -s tests -p 'test_korea_report_audit.py'
```

독립 집계는 raw float seed pairing, support 동일성/크기, FULL/K step 수, 데이터 분모,
known-missing cutoff, 부분 비용합을 검사한다. 테스트9개는 모두 통과했다.
**예측을 재생성해 AP를 다시 채점한 검증/원시 georegistration 전수 감사/새 성능 결과는 아니다.**
