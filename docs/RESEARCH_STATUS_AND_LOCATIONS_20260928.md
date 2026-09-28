# 2026-09-28 연구 현황과 파일 위치

09:22 KST 서버 재확인 기준. 사용자의 요청으로 예약 작업을 중지했다. OlmoEarth 1시간 heartbeat는 PAUSED이며, 확인된 다른 두 예약도 이미 PAUSED여서 활성 예약은0개다. 예약 기록은 삭제하지 않고 보존했다. 추가 실험은 자동으로 시작하지 않는다. P2 자체 프로세스와 GPU1 계산 프로세스는 없고, 마지막 연결 검사는 완료 상태다.

**현재 연구의 목표와 도달점**

목표는 지역·계절 설명과 소수의 양성/혼동 예시를 받아, 다른 지역에서 원하는 대상을 찾고 위치·측정·근거를 답하는 OlmoEarth 기반 VLM이다. 현재 확인한 것은 실제 EO/VLM 연결, 기준 모델 학습, 설명 입력이 위치 예측과 EO 역전파로 이어지는 경로다. 지역 지식의 성능 효과, 실제 전문가 교정의 효율, 새 지역 전이, 독립 EO 개선은 아직 입증하지 못했다.

**지금까지 진행한 일**

| 단계 | 실제 수행 내용 | 현재 해석 |
|---|---|---|
| 원래 GCM·SN7/D1 | 동동 판독12에피소드·172장 완료. 변화3/무변화4/근거부족5 | 두 번째 판독자가 없어 합의 관문 미완료. 이후 미완료 보관으로 정리했으며 기억 병목 주장을 되살리지 않음 |
| EO 입력 진단 | 영상 교체·제거, 단일 시점, 질문/날짜 편향 등 통제 실험과 별도 논문 골격 | 영상 정보를 사용한다는 것과 변화·시간을 이해한다는 것을 분리하는 진단 노선. 현재 OE 방법 성능표와 혼합하지 않음 |
| OE1 초기 질의응답 학습 | OlmoEarth Tiny+고정 Olmo3-7B, 실제 S2 train512패치/dev128패치. frozen68.36%, joint72.27%, blind66.80% | joint가 frozen보다3.91%p 높았지만 질문 편향·단일seed·개발평가라는 한계. 인코더 새 사전학습 방법의 입증은 아님 |
| native 모델·PASTIS 준비 | OlmoEarth v1.2 Base와 Qwen3-VL-8B 연결. PASTIS80패치×8관측,48train/16support-bank/16dev와 합성 교정 입력 준비 | 원영상·밴드·날짜·정규화·라벨·학습/재개 검사 기반 확보. 한 개발지역이며 사람 교정은0 |
| P2 B0/B2 기준 비교 | 고정 OlmoEarth와 일반 공동학습을 같은 입력·update 수로 비교.3seed×2조건 중5회 완료 |12시간 상한 종료로 전체 미완료. 완료5회 모두 학습 안정성 기준 미충족 |
| 확장 입력 | 포도밭·사료용 콩과384항목,48개의 고유훈련 query,221mask와20개 검토 영상, 출처 있는 설명 조건 준비 |384개 독립 영상이 아님. 공개 설명 요약은 실제 전문가 교정으로 세지 않음. 신규 평가/평균 개선 결과 없음 |
| 실제 설명→mask 검사 | 동결한 한 train K1 사례에서 실제 EO/Qwen 가중치 실행. EO212/head10텐서에 유한한 비영 기울기, 저장·복원 오차0 |연결 가능성 확인. optimizer update0이며 설명의 정확도 개선·의미 이해·생성 설명 결과는 아님 |

B0는 인코더를 고정하고 연결부/영역 head를 학습한다. B2는 인코더도 일반 공동학습한다. 이 비교는 제안한 새 방법 대 강한 기준선의 최종 비교가 아니다.

| seed | B0 최종 IoU K-AUC | B2 최종 IoU K-AUC | 차이 |
|---|---:|---:|---:|
|270927|0.300996|0.341754|+0.040758|
|270928|0.296005|0.349704|+0.053699|
|270929|0.274924|미완료|계산하지 않음|

IoU K-AUC는 교정 쌍1/2/4/8개에서의 영역 IoU를 곡선으로 요약한 값이다. 완료한 두 쌍의 평균 차이는 약4.72%p지만, 학습 안정성 미충족과 세 번째 쌍 미완료 때문에 충분히 학습한 우월성을 주장하지 않는다. 겨울밀 실패는 계속 원 평가에 포함한다. 동일 데이터/update 수이며 동일 GPU 계산량 비교는 아니다.

원 P2는43,200.395초에 종료됐고, 마지막 B2는 로그1,876/2,304회였다. 부분 체크포인트와1,536회 중간 점수를 최종 결과로 사용하지 않는다. 전체3seed 판정은 실행하지 않았다. 최종 원예측1,920개를 포함한2,056파일을 회수해 해시·독립 집계를 검증했다.

별도 설명 연결 검사는47.450초에 완료돼 누적 pilot 예산에48초를 반영했다. 미정산 예약은0이고 자동 예산 잔여7,152초도 지금은 사용하지 않는다. 이름만 vs 설명 포함의 logit 차이0.014232, 설명 제거 시0, 교환 설명도 차이가 발생했다. 문자열 변화에 반응했다는 사실만으로 농업 지식을 올바르게 활용했다고 해석할 수 없다. 이 adapter의 Qwen은 텍스트만 읽으며 EO/RGB 입력을 받아 문장을 생성하는 경로는 이번 검사 범위가 아니다.

**논문으로 남은 핵심 증거**

실제 전문가 교정 비용·라벨 효율, 새로운 지역 전이, 작물명만 조건을 넘는 설명의 추가 효과, 다른 reader와 독립 EO 평가에서도 남는 표현의 이점이 필요하다. 현재는 일반 기준선과 학습 경로를 검증한 단계이며 방법 신규성이나 CVPR 논문 완성도를 확정할 근거가 부족하다. 향후 순서는 학습/optimizer/RNG 새 프로세스 재개 검사와 공정한 설명 학습 비교였지만, 사용자 재개 요청 전에는 실행하지 않는다.

**폴더는 두 개이며 용도가 다르다**

| 위치 | 역할 |
|---|---|
| /Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project | 실제 연구 저장소. 연구 문서·실험 소스·config·회수 결과·라벨링 패키지 |
| /Users/dongdong/DongDong/ai_projects/earth_paper/olmoearth | 현재 앱에서 열린 공부용 폴더. OlmoEarth 원논문 PDF, 한국어 학습 노트, Nano CPU 실습 |
| /home/work/data/olmoearth/ | 서버의 실제 입력·가중치·실행 사본·checkpoint. 로컬 폴더가 아님 |

연구 저장소의 Git 원격은 DDanggle/eo_olmo_earth_project, 공부 자료의 원격은 DDanggle/earth-ai-paper다. 이번 중지·정리 작업에서 Git commit/push는 하지 않았다. 최근 실행 코드와 결과에는 미커밋/미추적 파일이 있으므로 Git 원격과 로컬의 완전한 동기화를 전제하면 안 된다.

**논문·연구 문서를 찾는 순서**

| 문서 | 구분 |
|---|---|
| [이번 현황 요약](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/docs/RESEARCH_STATUS_AND_LOCATIONS_20260928.md) | 현재 상태와 위치를 보는 시작점 |
| [최신 실험 기록](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/docs/OE10_P2_EXECUTION_20260927.md) | 실제 P2 종료·후속 연결 결과의 근거 |
| [CVPR 연구계획](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/docs/CVPR_2027_OLMOEARTH_RESEARCH_PLAN_20260927.md) | 큰 문제와 주장·규모 계획. 실측 완료 보고가 아님 |
| [구체적 실험설계 v6](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/docs/OLMOEARTH_NEXT_EXPERIMENT_DESIGN_20260927.md) | 비교군·입력·교정·전이 실험 설계. 당시 진행상태는 최신 실행 기록으로 보완 |
| [독립 연구계획 리뷰](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/docs/OLMOEARTH_VLM_PLAN_INDEPENDENT_REVIEW_2026_09_27.md) | 설계 과잉·B0·사람 검수·노출 감사 등에 대한 지적 |
| [과거 EO-VLM 논문 골격](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/docs/PAPER_EO_VLM_SKELETON_20260926.md) | What Do Earth-Observation VLMs Actually Read? 진단 노선의 골격. 현재 교정 전이 방법의 완성 원고가 아님 |
| [이전 GCM 연구계획](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/docs/PLAN_GROUNDED_CHANGE_MEMORY_2026_09_22.md) | Grounded Change Memory for Earth Observation Streams 노선의 계획. 완성 원고가 아님 |
| [D1 판독 기록](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/docs/D1_H_STATUS_20260924.md) | 원래 GCM/SN7 노선의 A 판독과 미완료 이력 |
| [OlmoEarth 원논문 PDF](/Users/dongdong/DongDong/ai_projects/earth_paper/olmoearth/OlmoEarth_2511.13655v1.pdf) | Ai2 원논문 참고 자료. 우리 연구 원고가 아님 |
| [코드 학습 자료](/Users/dongdong/DongDong/ai_projects/earth_paper/olmoearth/index.html) | 공부방·Nano 실습 진입점 |

확인한 두 폴더에서 현재 OE 방법 연구의 제출용 LaTeX 원고나 완성된 논문 PDF는 찾지 못했다. 논문 골격·연구계획·실험기록을 각각 보유한 상태다. 과거 골격은 상단 정정과 오래된 본문이 함께 있어 그대로 최종 주장이나 최신 결과로 인용하지 않는다.

**코드·결과·라벨 파일**

| 내용 | 위치 |
|---|---|
| B0/B2 동결 실행 코드 | [code/oe10_p2_v3](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/oe10_p2_v3) |
| 설명 조건부 mask 코드 | [code/oe10_text_mask_v2](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/oe10_text_mask_v2) |
| 원 비교 최종 회수 묶음 | [review_export_terminal_20260928_0845_v0](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/artifacts/oe10_p2_20260927/review_export_terminal_20260928_0845_v0) |
| 설명 연결 검사 회수 묶음 | [review_export_connection_20260928_0856_v0](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/artifacts/oe10_connection_launcher_20260928/review_export_connection_20260928_0856_v0) |
| 추가 작물 훈련 입력 계약 | [runtime_v2](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/artifacts/oe10_expansion_catalog_20260928/oe10_expansion_catalog_v0/runtime_v2) |
| 동동 D1 판독 원본 | [sn7v05-198e4ff03e26f004_dongdong.json](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/labeling_pack/visible_contract_v05_20260922/exports/sn7v05-198e4ff03e26f004_dongdong.json) |
| 예약 중지·서버 재확인 | [oe10_user_pause_20260928](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/artifacts/oe10_user_pause_20260928) |

서버의 주요 경로는 아래와 같다. 실제 위성 배열 전체와 모델 가중치는 전부 로컬에 내려받은 상태가 아니다.

- P2 checkpoint/로그: /home/work/data/olmoearth/oe10_p2_v0/training_v0
- 설명 연결 결과/서버 예산 원장: /home/work/data/olmoearth/oe10_text_mask_identity_v2
- PASTIS 준비 배열: /home/work/data/olmoearth/oe8_pastis_prepare_v0/prepared_v0
- 추가 작물 훈련 입력: /home/work/data/olmoearth/oe10_expansion_catalog_v0/runtime_v2
- OlmoEarth v1.2 원 모델: /home/work/data/olmoearth/oe4_native_v12_v0/models/OlmoEarth-v1_2-Base
- Qwen 가중치: /home/work/data/olmoearth/models/Qwen3-VL-8B-Instruct

서버 접근은 연구 저장소의 ./bin/nx를 사용한다. 현재 별도 새 학습을 시작하지 않았으며, 이미 종료된 우리 작업 이외의 다른 사람 작업은 건드리지 않았다.
