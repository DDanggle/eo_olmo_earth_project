# 실행 가능성 최종 확인 메모

확인일: 2026-09-27. 대상: `CVPR_2027_OLMOEARTH_RESEARCH_PLAN_20260927.md`, 확인 시 SHA-256 `7561c8ac7ae0ce8b373b61444f1549c9891ad887d2e252458b70dc680c0abe38`.

**이전 후속 검토에서 지적한 큰 계획상의 모순은 수정본에서 해소됐다. 계획 수립과 한정된 준비·반증 pilot에 조건부 GO를 유지한다. 실행 준비 완료나 100k/21+12 실행의 자원 확보를 승인한 것은 아니다.**

확인한 수정:

1. A/B의 encoder 학습 범위와 frozen 대조가 구별되고, C가 B에서 instruction을 받는다는 점도 명시됐다. 배운 concept/field 이월과 fresh readout 감사가 분리됐다.
2. 신규 A/B 학습에서 locked query/test support의 footprint·buffer·관련 날짜를 제외하고, 좌표 미확인 record 및 지도 ontology를 통해 들어오는 held-out 개념 감독도 감사하도록 바뀌었다.
3. PASTIS의 네 광역 tile에 조건부인 분석과 넓은 지역 일반화를 구별했다. 소수 tile의 CI를 광역 일반화 통과 기준으로 쓰지 않는다.
4. 10/10 결정에 전체 A/B·EO 평가·specialist·새 reader·IO·재실행 여유의 실측 예산을 포함했다. 21+12가 일정에 들어간다는 미확인 주장은 없다.
5. v1.2와 v1 recipe, 공개 weights와 trainer checkpoint 형식, optimizer restart가 구별됐다. 데이터 단위와 human120의 제한도 유지됐다.

실행 전에 여전히 확정할 항목은 실제 source/split/gallery·개념별 support 수, 손실 세부 정의와 가중치, module reset/carry manifest, 10밴드 입력 계약, runtime parity, 단계별 처리량·메모리·디스크 비용이다. 특히 VLM이 후보 ID 순서를 생성하는 primary로 구체화했으므로 **후보 pool 크기·후보 선택 규칙·context/token 예산을 실측하고 고정**해야 한다. 제한된 candidate set의 reranking을 무제한 gallery 검색으로 해석하지 않으며, candidate selection recall도 별도로 기록한다.

다음 판단 자료는 또 다른 계획 문서보다 **고정된 소량 real-data 로더/gradient/저장복원 receipt와 I/G/P/S·specialist 반증 결과**여야 한다. 본 학습 GO는 그 결과와 10/10 자원 gate에서 결정한다. 채택 확률은 평가하지 않았다.
