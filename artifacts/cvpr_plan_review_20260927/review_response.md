# 독립 검토 종합 및 수정 대응 — 2026-09-27

범위: 문헌·설계 검토. 실험 복제, 모델 성능 검증, 학회 심사 예측이 아니다. Root가 작성한 종합이며 reviewer 원문은 아래에 별도 보존했다.

## 결론

**연구 방향과 준비 작업은 진행. 제한된 방법 반증 pilot은 자료·실행 계약을 확인한 뒤 진행. 100k 본 학습과 CVPR 채택 가능성50% 이상이라는 판단은 유보.** 세 reviewer의 조건부 의견을 확률이나 투표로 환산하지 않는다.

| 독립 역할 | 1차 의견 | 실제 종합 draft 검토 | 반영 후 판단 |
|---|---|---|---|
| `/root/cvpr_prior_art_redteam` | TerraScope/SPEX/SegGPT 등 대비 기능 통합만으로 신규성 부족 | P/G 구별, 실제 VLM 경로가 모호 | 기준선·반증 준비 권장; P-v0는 generic contrastive와 가까워 신규성 미확정 |
| `/root/eo_vlm_benchmarks` | shared-field 개념 결합 후보, specialist 대조 필수 | primary 출력/누출/replay/gate/A-B 명세 보완 | 원·후속·최종 메모를 별도 보존; 실행 결과를 검증한 것은 아님 |
| `/root/olmoearth_core_fit` | 규모·예산·공식 baseline과 label계약 필요 | 21+12 비용, 4tile 통계, CPT overlap, stage 정의 문제 | 큰 설계 모순 수정 확인; actual data/runtime/throughput 전 main GO 아님 |

## 핵심 수정 대응

| 비판 | 계획 변경 | 여전히 필요한 증거 |
|---|---|---|
| mask→area→caption을 여러 능력 전이로 중복 계산 | primary는 새 target×관계의 실제 VLM 출력; 수치·설명은 utility | specialist+연산보다 나은 이유/효과 |
| 개념 field/contrast는 generic meta일 수 있음 | G/P 동일data·forward·negative, 독립tuple loss vslistwise 결합 차이표/의사코드 | 실행 코드·최적화 통제·ablation; 신규성 여전히 미확정 |
| VLM이 설명 부품일 수 있음 | oracle structured query / 동일VLM parser+specialist / wholeEO–VLM의3경로 | 실제 입력/출력 receipt, 자연어 조건·grounding 오류분해 |
| 최신 OlmoEarth와 과거Tiny 혼동 | 주후보v1.2-Base·revision/sourcepin; v1.2recipe별도감사 | 다운로드·로더parity·원본기준선 아직 없음 |
| 신규CPT에도query/support영역노출 가능 | 신규A/B전체footprint/date제외·mapontology/parent/negative누출감사 | source별실제join/derivation검사 |
| PASTIS4tile/MADOSsparse로광역평가과장 | 조건부4지역분석, sparse범위만평가, completegallery필수 | 독립지역·개념·support 수와 power |
| C와B, frozen범위 모호 | A/Btrainable/reset/carry 명시; frozen은A+B원본encoder고정 | config/runtimeparameter감사 |
| 강한generic/frozen실행비용누락 | 15main+6frozen=21core;6scale+6reader=12aux;baseline별도 | 모든단계처리량실측·10/10budgetgate |
| source mask/anchor를몰래query입력에줄수있음 | fixedwindowgallery·targetGT경계비제공·useranchor동일 | 실제episodeJSON/sourcecontract검사 |
| specialist이겨야한다면서최종gate엔없음 | final순차contrast후보G/S/specialist명시 | test전multiplicity/효과/모집단동결 |
| 1M질문과1M독립관측혼동 | source/공간/관측/라벨/질문/독립cluster별도계수 | 실제확보량아직미정 |
| 사람120으로작은효과를증명할수없음 | 100만원은reference의미·관측가능성·설명audit | timed20pilot·인력/단가확정 |

## 원 검토 기록

- [선행 원 리뷰](prior_art_review.md) / [종합 draft 후속](prior_art_draft_review.md)
- [방법 원 리뷰](method_review.md) / [종합 draft 후속](method_draft_review.md) / [최종 확인](method_final_note.md)
- [실행 가능성 원 리뷰](feasibility_review.md) / [종합 draft 후속](feasibility_draft_review.md) / [최종 확인](feasibility_final_note.md)
- [문헌 출처](sources.json) / [공개 자료 metadata 실측](public_asset_metadata.json)

각 리뷰는 작성 시점의 draft를 대상으로 한다. 후속 수정은 본 대응표에 기록하며 원 리뷰를 유리하게 고치지 않는다. 최종 메모의 검토 SHA는 당시 버전이고 publication SHA는 별도 manifest에 기록한다.

## 이번에 실행한 범위

로컬 source/기존 실험 기록 검토, 공식 문헌 조사, 공개 dataset HEAD·HF metadata GET, 문서/config 작성과 정합성 검사. 새 GPU 학습, corpus/weights 다운로드, 서버 전송, 유료 API, 채용, 논문 제출은 실행하지 않았다.

다음 판단 자료는 추가적인 낙관적 계획이 아니라 **실제 원본 기준선·자료 정합·VLM 연결·generic/specialist 반증 결과**다. 규모와 응용 목표는 유지하지만 현재 증거가 허용하는 투자 단계는 준비와 한정 실험이다.
