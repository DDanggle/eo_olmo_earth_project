# OE7 독립 검토 반영

2026-09-27. 이번 산출물은 다음 실험의 개발 계획이며 새 GPU 학습 결과가 아니다.

| 검토 | 발견한 문제 | v6 반영 |
|---|---|---|
| 통계 1차 | 공개 mask를 사람 교정으로 해석, K 분모 불명확, dev 지역1개로 CI 주장 위험 | pair 단위와 K0 범위, source/target support, 개발 AUC, cluster 단위와 power 미정 명시 |
| 통계 최종 | K마다 cohort 변화, source bank 학습 노출, test120 비용 과소계상, timed replay 과장 | 공통 query/class/weight; 평가 bank 별도; support 제작 비용·상각·작업량; 직접 workflow 시험 구분 |
| 방법 1차 | pair loss도 기존 metric learning, 선택기의 미개봉 영상 접근, 날짜 정책이 학습 차이로 혼입 | loss 식과 시작값 제안; 신규성 미확정; 동일 training subset·공통 selector; 전체 처리량 장부 |
| 방법 최종 | same Qwen만으로 reader 귀속 불가, specialist 인터페이스 불공정 | 동일 저장 mask/count/source replay; support head 또는 named-target 범위 제한 |
| 실행 | 두 사례 전용 preparer와80개 준비기는 다름; 실제 격자/VRAM 확대; trainer 복원 없음 | P0 새 extractor/episode builder; P1 profile·overfit·full state resume; BF16 미해결 구분 |

에이전트의 4후보·1초기 관측 제안 대신 이전 v5에 연결한 8후보·2초기·최대2추가를 개발 기본안으로 택했다. 정확한 날짜 인덱스와 처리량은 raw audit/profile 후 모델 비교 결과 전에 고정한다. 모든 후보 one-shot이 더 저렴한 경우 실제 비용 경쟁자로 포함한다.

리뷰 초안의 full-grid를 '상한'이라고 부른 표현은 최종 문서에서 비용이 큰 참조로 제한했다. 그 모델이 항상 더 정확하다는 보장은 없다. 자원 초안은 P1 2h, P2 초기12h, P3 추가16h로 정리했으며 확정된 충분 학습량이나 전체 프로젝트 비용이 아니다. 네 칸의 평가를 네 encoder 학습으로 중복 계산하지 않는다.

신규성을 보장하는 미구현 목적함수를 만들어 확정하지 않았다. 첫 pair 감독은 가설 검증이며, generic contrastive와 같으면 동일 조건으로 통합한다. 별도 보존 목적은 실제 실패와 표준 distillation 대조 이후 결정할 연구 과제다.

보존 원문: statistical_review.md, method_review.md, execution_review.md, final_statistical_review.md, final_method_review.md. 원문 간 자원/관측 수 제안 차이는 위 통합 결정을 따른다. 서버·학습·외부 인력 연락·유료 API 실행은 이번 작업에 없다.
