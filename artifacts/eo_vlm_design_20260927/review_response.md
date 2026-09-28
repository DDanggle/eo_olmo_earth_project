# 2026-09-27 — 문제·자료·아키텍처 독립 검토 대응

문헌과 공개자료를 두 에이전트가 병렬 조사했다. 문헌 담당자는 제안 구조와 대조 설계를 다시 검토했다. 원문은 `prior_art.md`, `data.md`, `final_design_redteam.md`에 보존한다. 외부 동료 심사나 성능 재현을 완료했다는 뜻은 아니다.

| 지적 | 반영 |
|---|---|
| 양성−혼동 유사도·지역 gate·episodic learning 자체는 익숙한 조합 | 신규성 가설로만 기록. 같은 정보와 utility를 받은 강한 generic/prototype/router를 2×2에 포함 |
| PASTIS 연간 작물은 미지의 생태 개념·계절 변화 정답이 아님 | 첫 episode를 정해진 혼동 쌍의 지역 간 판독으로 정의. 대상 신규성은 클래스/사전학습 노출 감사 이후 |
| utility 학습의 hidden label/지역 누출 가능 | geography/year/parent 단위 cross-fitting. 후보 library·검색·utility·model selection의 라벨/계산비용 합산 |
| 공간 판독기만 잘해도 VLM 기여처럼 보일 수 있음 | 동일 저장 dense 출력→두 reader 대조와 별도 고정 EO 평가 추가 |
| 현재 16-token 평균만 이기는 것은 약한 비교 | 같은 token learned resampler, 공간/시간 평균, 더 많은 token의 단순 baseline 포함. 실제 GPU시간·용량까지 보고 |
| 공개 전문가 mask와 새 전문가 교정은 다름 | Kuro의 기존 5인 검수는 source label 품질의 근거로만 사용. 자연어 교정/새 사용자 비용은 별도 측정 |
| 데이터 조사자가 제시한 100인시/600episode는 기존 예산과 다름 | 채택하지 않음. 기존 100만원/64인시 가정과 train200/dev60/test120, 20개 시간 측정 pilot 유지 |
| Kuro의 공식 test를 이 프로젝트가 이미 관찰 | 9/9 기록을 연결. 이전 관찰 사건은 development로 취급하고 새 confirmatory 평가의 노출 감사 명시 |

직접 코드/receipt에 근거한 CPU 진단은 128→16 평균의 각 bin이 공간4개×날짜2개를 묶는 것을 확인했다. 합성 post-encoder 특징의 날짜/위치 구분 가능성만 시험했고 실제 이미지 판독, 새 방법 성능, OlmoEarth 자체 결함은 측정하지 않았다. 원 격자 정보가 실제 정답에 쓰이는지 먼저 비교하도록 했다.

이번 결과: 구현·자료 확인을 진행할 근거와 대조 설계 확보. 연구 성능·독창성·채택 확률의 확정은 없음. 새 GPU 실행, 대규모 다운로드, 유료 API 호출은 이번 갱신에서 수행하지 않았다.
