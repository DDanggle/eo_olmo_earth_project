# OlmoEarth upstream 기여 — 최종 준비·판정표

최종 검토: **2026-09-13**. 파일명은 기존 링크를 위해 유지한다.
**미제출:** 이 검토에서 PR/issue·메일·지원서 제출, fork/push, GPU 실행, 제품 코드 변경은 하지 않았다.

> **결론:** 첫 제출은 sample schema, 가장 좋은 좁은 기술 PR은 SCL scoring,
> 가장 좋은 재현 질문은 LFMC다. 큰 연구 방향은 별도다.
> “버그를 많이 찾았다”보다 **외부 사용자의 문제를 재현하고, 원인을 좁혀, 검증 가능한 수정으로 돌려준다**가 핵심이다.
>
> 이 판단은 공개 코드·공식 채용 요건에 근거한 외부 검토이지 Ai2의 실제 채용/머지 결정이 아니다.
> 9/13 이전 문구 중 과장된 판정은 §4에서 정정한다.
> 상세 이력은 [PR_DOSSIER](../PR_DOSSIER.md), 비교 근거는 [9/11 EO 비교](PR_PRIORITIES_AND_EO_GAPS_2026_09_11.md).

## 1. 세 관점으로 보면 무엇이 강한가

| 관점 | 보는 질문 | 지금 보여줄 자산 | 아직 대신하지 못하는 것 |
|---|---|---|---|
| 유지보수자 | 내가 재현하고 안전하게 고칠 수 있는가? | sample 한 파일 수정, SCL 두 scoring 경로, 명확한 LFMC 확인 요청 | 실제 회귀 테스트, 최신 실행 로그, 변경 영향 확인 |
| Applied Research Engineer | 파트너 요구를 데이터·모델·평가·배포로 연결하는가? | 제주 입력 품질 문제 → 재현/수정; 네팔 frozen embedding 적용과 한계; trainer 감사 | 현장 사용자의 실제 의사결정 개선, 운영 지표, 타인이 실행할 수 있는 예제 |
| 모델 연구자 | 성능 차이의 원인을 통제했고 일반화되는가? | release/band/time 계약 분석, 고정 판독기 streaming 실험의 설계와 반증 | 새 학습법의 우월성, 교차센서 불변성, causal mechanism, 독립 확증 |

공식 [OlmoEarth Senior Research Engineer 공고](https://job-boards.greenhouse.io/thealleninstitute/jobs/8140098)는
파트너별 모델 적용, 임베딩·추가 모달리티 활용, rslearn 개선, PyTorch 학습 루프·task head 개발을 명시한다.
현재 자산은 이 **응용 연구 엔지니어링 연결부**와 맞는다. PR 몇 개가 senior 역량이나 채용 결과를 보장하지는 않는다.
연구 논문의 novelty와 작은 PR의 유용성은 다른 기준이다.

## 2. 우선순위 — 가치와 준비도를 분리한다

괄호 번호는 dossier ID다. 아래 순번이나 GitHub issue 번호와 혼동하지 않는다.

| 우선순위 | 후보·대상 | 의미 있는 이유 | 현재 판정·제출 전 조건 |
|---|---|---|---|
| **P0** | sample `es_* → oe_*` (#1), projects | bundled training-window 준비 예제의 schema 실패를 좁게 고침 | **첫 PR 준비도 가장 높음.** 로컬 커밋·영문 본문·과거 Linux 6-window 실행. 아래 §3A |
| **P1** | SCL scoring nearest (#10a), rslearn | class ID를 연속량처럼 보간하는 경로를 바로잡음. 조용한 선택 오류 가능 | **가장 좋은 기술 PR 후보, 아직 완성 아님.** FirstValid/BestClear 둘 다 + 실제 재격자 회귀 테스트 |
| **P1 병행** | LFMC 951.9 vs 문서 580.6 (#2), projects | 공개 평가의 재현 경로를 확인하는 가치 있는 질문 | **Issue, 원인 미확정.** 원 실행 config·split IDs·ckpt full hash·로그 묶음 확인 후 |
| **P2** | 환경 호환성 (#13), projects/rslearn | 지원되지 않는 release/config의 모호한 실패를 줄임 | 공개 설정으로 실패 재현 → 지원 조합 예제/명확한 오류 제안. “lock이 옛날”만으로 버그 아님 |
| **P2** | forest 외부 접근·URL 미설정 (#3/#6) | 공개 예제의 인증 요구·실행 조건·fail-fast를 명확히 함 | 현재 runner 최소 재현 후 docs/issue. 내부 소스를 PC로 바꾸는 로컬 패치는 upstream 기본값으로 제출하지 않음 |
| **P2** | SCL 보조자산 (#10b), rslearn | 필요한 SCL을 PC 경로에 어떻게 준비하는지 알려줌 | **완전한 설정 예제부터.** 의존성 자동 등록 API는 수요 확인 후 별도 RFC |
| **P2 제안** | embedding 입력·품질 receipt | 사용자가 “어떤 관측이 이 벡터에 들어갔나”를 확인할 수 있음 | 기존 metadata·QA를 연결하는 작은 예제. 새 표준·플랫폼 발명으로 부르지 않음 |
| **P3** | partial-band (#12) | 한국/다른 S2 제품의 입력 제약을 실제로 이해하는 데 중요 | 지원 정책 질문 + 공개 최소 재현. per-band mask를 임의로 설계하거나 zero-fill을 동등 입력으로 주장하지 않음 |
| **백로그** | macOS hang·Python 상한·상대경로·timestep 안내 (#7/#8/#9 등) | 반복 재현되면 사용자 지원 가치가 있음 | “무의미/팀 일만 늘림” 아님. 환경·공개 경로·설계 의도 확인 전에는 우선순위 낮춤 |
| **연구 별도** | streaming GRU·JEPA/LoRA·한국 3-task | 모델 적용 범위를 넓힐 수 있는 실험 | 성능/비용/독립 검증·API 요구 확인 전 제품 PR에 섞지 않음 |

**P1 두 건은 병행 준비 가능하다.** sample 답변이 와야 다음 검증을 시작할 이유는 없다.
제출은 각각 독립적이고 작은 내용으로, 연구 수치 묶음을 덤으로 붙이지 않는다.

## 3. 제출 직전 확인 — 무엇이 실제로 준비됐나

### A. sample: 작은 완성품을 먼저

- 외부 저장소 branch `fix/sample-annotation-oe-schema`, HEAD `21b658a98e2b49c6f6c6ace413239d0f20c8a9c5`.
- **커밋 diff** `origin/main..HEAD`: sample 한 파일, +36/−24. 작업트리에는 별도 forest 두 파일 변경이 있으므로 섞어 stage하지 않는다.
- 9/13 정적 검사 PASS: feature 6개, category `[1,3,1,4,3,4]`, geometry/task ID 보존, legacy key 없음.
- runner 0.1.12/0.1.14 wheel 필드 검증은 9/10~11 기록. Linux runner 0.1.14의 6-window E2E는 **8월 기록**, 오늘 재실행한 것이 아니다.
- [영문 PR 본문](../pr_bodies/01_sample_schema.md). “전체 quick-start 첫 명령” 대신 해당 training-window 예제를 특정한다.
- 남은 준비: 공개 가능한 과거 Linux 로그 첨부 또는 현재 Linux 1회 replay. macOS hang은 이 data-only 수정과 별도다.
- 남은 게시 절차: 중복 최종 확인 → 사용자 승인하에 fork/push/PR. “이미 보냈다”는 실제 URL이 생긴 뒤에만 말한다.

### B. SCL: 패치 크기보다 회귀 테스트가 핵심

[FirstValid L115](https://github.com/allenai/rslearn/blob/c47952f44811532b9f6e3a8561bd104fe53aa1a6/rslearn/dataset/sentinel2_scl.py#L115)와
[BestClear L279](https://github.com/allenai/rslearn/blob/c47952f44811532b9f6e3a8561bd104fe53aa1a6/rslearn/dataset/sentinel2_scl.py#L279)는
SCL(uint8) 읽기에 layer resampling을 전달한다. layer 기본값은 bilinear다.
그러나 **SCL을 통한 multi-item scoring에서 실제 재격자가 생기는 경우**가 대상이지 모든 사용자가 영향받는 것은 아니다.

제안 제목: `Use nearest-neighbor resampling for Sentinel-2 SCL scoring`.

- [ ] 두 compositor의 **SCL score 읽기만** nearest로 지정.
- [ ] 20→10 m 또는 half-pixel offset fixture: 실제 보간이 발생하고 기존 코드에서 실패해야 함.
- [ ] class 집합·clear/cloud score·두 후보 순위/선택이 기대와 일치.
- [ ] 최종 반사도 출력에는 기존 bilinear 유지.
- [ ] 동일 격자·nodata·동률·SCL 없음·single-item 현행 동작 회귀 확인.
- [ ] 새 테스트 + 기존 [SCL unit tests](https://github.com/allenai/rslearn/blob/c47952f44811532b9f6e3a8561bd104fe53aa1a6/tests/unit/dataset/test_sentinel2_scl.py) 실행 로그.

[로컬 adapter](../code/scl_compositor.py)는 BestClear 우회 구현이지 두 클래스 upstream 패치/테스트 완료본이 아니다.
제주 전후 그림은 동기를 보여주지만 **nearest 단독 정확도 향상**의 증명은 아니다.

### C. LFMC: 재현 요청이지 원인 판결문이 아니다

[영문 Issue 초안](../ISSUE_DRAFT_lfmc.md)의 수치는 과거 실행이다.
문서 test MSE **580.6**, 공개 ckpt 평가 **951.9**, 별도 backbone 기반 task fine-tuning **558.8**.
가까운 점수를 새로 얻었다고 원 checkpoint나 데이터가 잘못됐다는 원인 분리가 되지 않는다.

- [ ] 당시 평가한 파일의 full SHA-256과 아래 공개 LFS hash 대조.
- [ ] 실제 실행 config와 원 config diff, 버전·당시 master commit, 평가 명령·로그 제공.
- [ ] val/test window IDs·개수·집계 정의·normalization 확인.
- [ ] epoch 33 선택 기준 설명; test에 맞춰 선택하지 않았는지 기록으로 확인.
- [ ] 공개 첨부의 자격증명·개인정보·권한 제한 데이터 제거.

최초 요청은 “재업로드”가 아니라 **580.6에 대응하는 checkpoint/config/split/평가 절차 확인**이다.
현지 실행 자료를 아직 묶지 못했다면 보내기 전에 보강한다. 다른 fine-tuned weight 공유도 권한·출처 확인 후 제안한다.

## 4. 이번 최종 검토에서 고친 과장

| 이전 문구·추론 | 최종 판단 |
|---|---|
| “SCL 기본 설정 사용자 전원 영향” | 실제 재격자 + 해당 scoring 경로에 한정. 영향 사용자 수 미측정 |
| “LFMC 재학습으로 데이터·레시피 정상 입증” | 다른 run에서 근접 점수 획득. 원 run/업로드/데이터 원인 미확정 |
| “sample 리뷰 비용 0” | 낮은 리뷰 부담. 검증·호환성 검토는 여전히 필요 |
| “팀이 이미 아니까 forest/SCL 문서 이슈 무의미” | 팀의 인지·우선순위는 모름. 외부 실행 마찰은 유효하되 범위를 좁혀 제안 |
| “MISSING mask+0으로 v1.2의 B01/B09 부재 해결” | v1.2 단일 band group에서는 밴드별 결측과 그룹 결측이 다름. zero-fill은 계약 밖 imputation일 수 있음 |
| “shared embedding이므로 S1/S2 Δz 분포가 같아야 함” | 공동 latent 공간은 관측 불변성·동일 거리 분포 보증이 아님. 센서·날짜·궤도·정규화 통제가 먼저 |
| “10 m 모드가 없으니 만들어달라” | Studio는 이미 10/20/40/80 m export를 안내. 로컬 patch 설정과의 동등성은 별도 확인 |
| “Nano=Large 겹침 → Nano로 충분” | 제한된 검사 결과. 성능·비용·불확실성의 task별 비교가 필요 |
| “PR 없으면 말뿐인 사람, 지원 전에 반드시 제출” | 실제 팀 판단을 알 수 없음. 지원/후속 연락과 PR 완료를 분리 |
| “end-to-end 예제가 전혀 없다” | 기존 LFMC·forest 및 Studio 사례가 있음. 우리의 특정 검증 워크플로에서 못 찾은 부분을 명시 |

이전 9/13 초안은 research repo commit `8f2d29e993ea0cc860c21743c2ea70695682b705`에 남아 있다.
오류를 숨기지 않되, 새 준비본에 상충하는 지침을 계속 병기하지 않는다.

## 5. 모델러가 소유해야 할 일 / 팀과 같이 연구할 일

### 모델러가 먼저 책임질 것

1. **입력 동등성:** 실제 관측 시각, S1 단위, bands/order, masking, pooling, quantization.
   “같은 shape”는 “동등한 정보”가 아니다. 설정상의 지원과 실제 소비를 구분한다.
2. **공정 평가:** support/test 공간 분리, seed-before-init, validation supervision, 샘플 노출·비용·임계값.
   한국 KR-4의 시간 창 비대칭·노출량·seed·mIoU 문제는 우리 코드 문제다.
   [KR-4 감사](KOREA_KR4_AUDIT_2026_09_10.md)의 원 판정은 새 공정 실행 없이 뒤집지 않는다.
3. **변화 점수 보정:** Δz는 피해 확률이 아니다. 지역·계절·시간 간격·센서·품질별 placebo와 외부 라벨,
   공간 상관·다중 비교를 다뤄야 한다. 보편적 “정상 Δz 숫자”를 모델 공급자에게 요구하지 않는다.
4. **작은 물체·희소 라벨:** 출력 격자를 촘촘히 만드는 것과 새로운 세부 정보를 얻는 것은 다르다.
   고해상도 영상·decoder·경계 라벨의 기여를 분리한다. 양성 support 선택 비용도 라벨 예산에 포함한다.
5. **결과 책임:** 한국 shared static cache는 source-head transfer나 shared streaming 확증과 다르다.
   실패 gate·참고 분석·탐색 지역을 논문의 긍정적 헤드라인으로 바꾸지 않는다.

### 팀과 논의할 만한 T9~T14 — 요구가 아니라 검증 가능한 질문

| 카드 | 올바른 질문 | 먼저 만들 증거·형식 |
|---|---|---|
| T9 품질 | 기존 SCL/valid mask·유효 관측량을 embedding과 같이 전달할 수 있는가? | SCL 수정과 별개인 QA receipt 예제. learned confidence는 향후 연구이며 외부 QA를 없애지 않음 |
| T10 눈금 | 시간·센서·quality 조건부 calibration 예제를 함께 만들 수 있는가? | placebo/threshold 방법 + held-out 오경보 평가. checkpoint별 고정 정상값 요청 아님 |
| T11 계절 | 계절 정보를 쓰면 seasonal placebo보다 검토 순위가 나아지는가? | 별도 연구 가설. 예측 가능/분리 가능 확정 아님 |
| T12 센서 | 동일 조건의 cross-sensor retrieval·downstream transfer는 어느 정도 보존되는가? | 날짜·궤도·단위·missingness 통제부터. 분포 차이만으로 버그 이슈를 열지 않음 |
| T13 공간 | 기존 10/20/40/80 m 옵션이 small-object 성능과 비용을 어떻게 바꾸는가? | Studio/local 입력·readout 동등성 + 크기별 성능. 이미 있는 모드를 재요청하지 않음 |
| T14 크기 | Nano/Tiny/Base 중 어느 것이 이 task의 정확도·비용 요구를 충족하는가? | 동일 support/test·입력·metric의 Pareto 표. 다운로드 수는 선택 근거 아님 |

[영문 미팅 카드](Ai2_Meeting_Cards_DG.md)의 해당 기술 문장도 교정했다.
**별도 HTML/공개 페이지는 이번 범위에서 동기화하지 않았다.** 다른 실험 수치나 개인 이력은 재검증하지 않았다.

## 6. 다른 EO 대비 기여 범위 — 없는 기능을 만들었다고 하지 않기

- TorchGeo의 범주형 nearest 처리 원칙은 참고할 수 있다. SCL PR의 가치는 **새 알고리즘**이 아니라 rslearn의 특정 경로를 맞추는 것이다.
- Cloud Score+의 QA, STAC의 처리 계보, AlphaEarth의 version metadata는 receipt 설계의 참고다.
  OlmoEarth도 이미 QA·STAC·metadata 기반이 있으므로 연결 예제를 먼저 제안한다.
- OlmoEarth는 이미 유사 검색·few-shot·change detection·SFT·다중 해상도 export를 안내한다.
  우리의 기여는 해당 기능의 최초 구현이 아니라 **어떤 조건에서 실제로 믿고 쓸 수 있는지 평가하는 사례**다.
- rslearn의 기존 EmbeddingCache는 window/crop 메모리 캐시다. 연구의 영속·시간 갱신 상태와 다르지만,
  그 차이가 곧 새 streaming 방법의 우월성 증명이 되는 것은 아니다.

출처·구체 비교: [9/11 비교표 §5](PR_PRIORITIES_AND_EO_GAPS_2026_09_11.md#5-다른-eo에-있는-것과-olmoearth의-정확한-빈틈),
[공식 embedding 기능 안내](https://allenai.org/blog/olmoearth-embeddings).

## 7. 팀에 전달할 최종 패키지

**3분 설명 순서**

1. **30초 — 문제:** 제주/네팔에서 모델을 쓰면서 입력 품질·예제 호환·평가 재현의 마찰을 만났다.
2. **60초 — 구현:** sample schema의 작고 검증 가능한 수정. SCL은 현재 소스 경로와 필요한 회귀 테스트를 보여준다.
3. **60초 — 연구 판단:** LFMC 불일치는 해결 전 확인 요청이다. 네팔 Δz는 검토 후보이지 피해 판정이 아니다.
4. **30초 — 다음 질문:** 파트너에게 가장 중요한 마찰 하나를 골라 reproducible example로 함께 줄일 수 있는가?

**영어 소개 — 아직 게시 전인 현재 상태에 맞춤**

> I used OlmoEarth in workflows in Jeju and Nepal. I traced input-quality and reproducibility issues, and prepared a small sample-schema fix. I’m also preparing a targeted SCL regression test and a checkpoint-reproduction report. I’d like to turn these findings into a workflow another user can reproduce.

**제안 1개만 추가한다면:** 제주 1 window의 RGB/QA → 실제 입력 시각·band·weights receipt → embedding 해석 예제.
[제주 그림](../artifacts/figures/v7_rgb_pairs.png)과 [원 보고서](../artifacts/results/v7_summary.json)는
한 window·4기간의 복합 입력 처리 결과다. 95.64% bad-proxy 감소를 탐지 정확도나 nearest 단독 효과로 설명하지 않는다.

한국 예제는 KR-4 검증 문제를 닫은 뒤 후보로 둔다. **AI-Hub는 다운로드 가능과 외부 재배포 가능이 다르다.**
원시 자료를 upstream에 복사하지 말고 데이터별 이용조건 확인 후 사용자 다운로드 절차·manifest·합성 fixture 중심으로 구성한다.
네팔 공개 데이터도 source별 라이선스와 외부 라벨의 해석 한계를 확인한다.

**지원 준비는 PR과 독립:** 공식 공고의 마감은 **2026-09-16**, 근무지는 Seattle이다.
마감 시각/시간대는 확인한 본문에 없으므로 마지막 날까지 기다리지 않는 편이 안전하다.
PR merge나 reviewer 답변을 지원 조건으로 두지 않는다. 실제 연락 내용·약속은 사용자 확인 전 추정하지 않는다.
CV·후속 메일에는 `prepared / submitted / merged`를 사실대로 구분한다.

## 8. 이번에 실제 확인한 범위

- 9/13 공개 API: projects main `23a3d7b799ba1fbb0c9138cb1444166ae1d3dd0a`, rslearn master `c47952f44811532b9f6e3a8561bd104fe53aa1a6` 유지.
- projects open issue 13 / PR 4(#37/#42/#43/#64); 목록상 sample schema·LFMC 수치 불일치 직접 중복 없음.
  closed 이슈 전체 검색/현재 계정 작성 권한 검증까지 했다는 뜻은 아니다.
- upstream sample 6개 legacy properties, SCL 두 read와 기본 bilinear, LFMC 문서 580.6 직접 확인.
- HF `model.ckpt`: 1,139,505,083 B; LFS SHA-256
  `20064f6a0a7a70acc1d5e304bc8050bbe07e86891a60806ee585fffbeb86f92f`;
  file lastCommit `0c27933f49c5ec99444783845481bf7b972af812` (2025-10-30).
- 오늘 LFMC 가중치 재다운로드/평가, SCL runtime test, sample Linux CLI replay, 과거 네팔/한국 결과 재계산은 **하지 않았다**.
- 현재 로컬 sample 정적 검사·커밋 diff 확인만 추가했다. 제품 작업트리와 연구 산출물은 보존했다.

공개 근거:
[projects main](https://github.com/allenai/olmoearth_projects/tree/23a3d7b799ba1fbb0c9138cb1444166ae1d3dd0a),
[open issues/PR](https://api.github.com/repos/allenai/olmoearth_projects/issues?state=open&per_page=100),
[rslearn SCL](https://github.com/allenai/rslearn/blob/c47952f44811532b9f6e3a8561bd104fe53aa1a6/rslearn/dataset/sentinel2_scl.py),
[layer defaults](https://github.com/allenai/rslearn/blob/c47952f44811532b9f6e3a8561bd104fe53aa1a6/rslearn/config/dataset.py#L643),
[LFMC card](https://github.com/allenai/olmoearth_projects/blob/23a3d7b799ba1fbb0c9138cb1444166ae1d3dd0a/docs/lfmc.md),
[HF file metadata](https://huggingface.co/api/models/allenai/OlmoEarth-v1-FT-LFMC-Base/tree/main?expand=true),
[공식 채용 공고](https://job-boards.greenhouse.io/thealleninstitute/jobs/8140098).

## 2026-09-16 제출 직전 점검 (마감일)

- upstream `origin/main` 23a3d7b 변동 없음. sample 파일 legacy `es_` 키 24개 그대로. 새 PR #67(monocrop, 09-15)은 무관.
- 브랜치 `fix/sample-annotation-oe-schema` = origin/main + 2 commits, 1 file(+36 −24). 작업 트리의 forest_loss_driver 수정 2개는 커밋에 없어 PR 에 섞이지 않음.
- 브랜치의 GeoJSON 재검증: 6 features, 키 = oe_annotations_task_id · oe_end_time · oe_labels · oe_start_time, `es_` 없음.
- fork `DDanggle/olmoearth_projects` 아직 없음. 제출 절차는 fork → push → PR 3단계.
- PR 본문 끝에 Claude Code 생성 표기 줄을 넣어 둠. 지원용 첫 PR 이므로 남길지 지울지는 사용자 결정(지우려면 마지막 줄 삭제).

## 2026-09-16 최종 확정 — 사용자 결정 반영

제출 목록은 **3건**으로 확정함. 나머지는 의견 수준이라 제출하지 않음.

| 순서 | 항목 | 형식 | 상태 |
|---|---|---|---|
| 1 | sample annotation `es_* → oe_*` | PR (olmoearth_projects) | 브랜치·본문 완료. 커밋 트레일러·본문 footer 처리 결정 대기 → 즉시 제출 |
| 2 | LFMC 공개 ckpt 불일치 | Issue (olmoearth_projects) | 본문 최종본 완료 → 즉시 제출 |
| 3 | SCL 범주형 scoring 에 nearest 강제 | PR (rslearn) | 결함 위치 확인(v0.1.14 `sentinel2_scl.py`, 두 compositor 모두 layer resampling 을 SCL 읽기에 전달). 패치·테스트 미작성 |

- 4(forest_loss_driver 무한 재시도)·5(모델 카드 제안)는 **백로그로 내림**. Ai2 입장에서 새 정보가 아니거나 의견이라 판단.
- 6·7 은 보류 유지.
- 3번 버그 범위: SCL compositor 는 rslearn PR #598(2026-04 머지)로 추가됐고, 추가 시점부터 최신 release v0.1.14 까지 같은 코드 경로. OlmoEarth 모델 버전(v1/v1.1/v1.2)과는 무관 — 이건 데이터 전처리(장면 선택) 단계의 버그라 어떤 모델을 쓰든 입력 영상 선택에 영향을 줌.
