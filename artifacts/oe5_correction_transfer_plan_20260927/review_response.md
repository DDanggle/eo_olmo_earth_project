# 독립 검토와 수정 대응

2026-09-27. 모델 기반 독립 검토이며 전문가 라벨·실험 결과·채택 확률 평가가 아니다.

| 검토 | 발견 | 반영 |
|---|---|---|
| recent_eo_vlm_gap | TerraScope뿐 아니라 Earth-OneVision, textual inversion, TMPA, Think2Seg 등이 직접 선행. 양성+대조 few-shot 자체도 이미 연구됨 | 2026-06/07 선행과 입력 표현·교정 전이 평가의 차이를 명시. 최초 주장 제외 |
| unified_mechanism_redteam | context-conditioned prototype/weighting은 일반 meta-learning과 구별 필요 | 교정의 전이효과를 먼저 관찰. utility 감독을 쓸 경우 동일 감독의 generic router 대조 |
| first_transfer_data_audit | PASTIS 10밴드/원 타일·날짜 계약, S4A L1C, CropGlobe feature 자료의 한계 | 원 입력 감사·누락 밴드·정규화·전처리 교락을 선행 관문으로 고정 |
| 초안 후속 검토 | 언어 정렬 없는 prototype에는 K=0이 정의되지 않음 | prototype은 같은 K의 random support 기준; 유효 zero-shot VLM만 K=0 비교 |
| 초안 후속 검토 | PASTIS 원 tile 4개로 독립 dev 3개와 다지역 train·test를 동시에 만들 수 없음 | frozen 현상 관찰 3개 비봉인 tile / 학습 비교 2train+1dev / 1봉인 tile로 단계 구분. 실제 ID 배분은 미완료 |
| 초안 후속 검토 | 각 query의 유사·상이 환경 support가 실제로 존재하지 않을 수 있음 | 사전 feasibility 검사, 성립하지 않으면 not_estimable |
| 초안 후속 검토 | library 100개에서 K=5 선택은 라벨 5개 비용이 아님 | 후보 library·원 학습·추가 target 주석·인시를 분리하고 같은 pool 비교 |
| 초안 후속 검토 | native replay에서 보류 개념을 다시 감독할 수 있음 | replay ontology/caption/negative/footprint/date 감사와 새 개념 표현 제한 |

원 결과·기존 사전등록은 수정하지 않았다. 새 JSON은 실행 가능한 확증 config가 아니라 미충족 선행조건이 명시된 개발 계획이다. 새 GPU 학습은 시작하지 않았다. 두 기존 controller의 terminal failure status를 읽어 보존했으며 모두 GPU worker 실행 이전의 idle/dependency 종료다.
