# OLMoEarth 연구 재시작 지점

> **09/28 09:25 KST — 사용자 요청으로 예약 중지.** OlmoEarth heartbeat는 PAUSED이며 자동 연구 진행을 중지했다. 확인된 예약3개 모두 PAUSED. P2는5/6완료·상한 종료, 후속 연결 검사는 완료, GPU1 자체 작업/미정산 예약0. 결과는 보존했으며 추가 실행은 재개 요청 후 진행한다. [현재까지의 정리·논문/폴더 위치](docs/RESEARCH_STATUS_AND_LOCATIONS_20260928.md).

> **09/28 08:58 KST — P2 상한 종료·실제 연결 검사 통과.** 원 비교는 5/6 완료로 미완료 종료했으며 완료5회 모두 학습 안정성 미충족이다. 두 완료쌍 평균차이 B2 +0.047228은 확정적 개선 주장이 아니다. 원예측1,920개 회수·종료 독립감사 완료. 별도 한 사례에서 설명→mask와 EO 역전파212텐서·저장복원 오차0을 실제 확인했다. 새 학습 update0, GPU 점유47.45초·예산 차감48초/7,200초. 다음은 학습·새 프로세스 재개 검사이며 설명의 성능 효과는 아직 미검증이다. [실행·한계](docs/OE10_P2_EXECUTION_20260927.md).

> **09/28 06:59 KST — 5회 완료 검산·후속 실행기 CPU 준비 완료.** 새 B0 최종0.274924이며 완료5회 모두 학습안정성 미충족이다. 마지막B2 진행 중(06:38 384/2304,06:56 running 확인), 원12시간 상한 유지. 원예측1,920개 회수·독립검산 완료. 후속 실행기 서버CPU17검사 통과·실제 사전 검사에서 중복 GPU 실행 차단 확인. 새GPU/예약/실가중치 연결검사0. 다음은 P2종료감사 후 한사례 연결검사다. [실행·한계](docs/OE10_P2_EXECUTION_20260927.md).

> **09/28 06:00 KST — 두 번째 비교쌍 검산·후속 연결 입력 동결.** 4/6완료: B0 .296005 / B2 .349704(두번째seed,+.053699),두 쌍 평균차이+.047228. 완료4개모두학습안정성미충족이며 겨울밀·K증가이득 부재를 유지한다. 원예측1,536개 회수/독립검산 완료.05:38직접확인다섯B0 1152/2304. 별도text→mask v2 입력identity·유한기울기CPU검사와한사례실연결조건동결,실가중치forward/새GPU/예약0. 다음은P2종료감사 후 bounded 연결검사다. [실행·한계](docs/OE10_P2_EXECUTION_20260927.md).

> **09/28 04:50 KST — 설명 입력·별도 연결부 CPU 검사 완료.** 포도밭/콩과384항목·48영상에 설명 조건을 연결하고 경계/대체 모델19검사를 통과했다. 실제 tokenizer58/44/267토큰, 고유문구7개다. 실제 EO/Qwen 가중치 forward·새 성능·새 GPU는0이며 후속 실행 identity관문이 남았다. P2는04:39확인3/6완료·두번째B2 1920/2304. [근거·한계](docs/OE10_P2_EXECUTION_20260927.md).


> **9/28 03:55 KST — 추가 훈련 입력과 품질 검토 영상 준비 완료.** 포도밭·콩과384항목(48영상의방향/K변형)·221mask·실제CPU loader를 검증했고20개 검토PNG/빈응답을 회수했다. 정식입력은 runtime_v2, 사람응답/새GPU0이다. 03:48확인상3/6완료·두번째B2 log1333/2304. 다음은 두 작물 설명카드와 설명→mask 경로 준비. [실행·정정 이력](docs/OE10_P2_EXECUTION_20260927.md).


> **9/28 02:47 KST — 3/6 실행 검산·확장 입력 준비.** B2 두 번째 seed 진행 중이며 새 비교쌍은 아직 없다. 새 B0 최종0.296005·학습안정성 미충족. 포도밭·사료용콩과 입력30개/객체294개를 CPU 검산했고 두 대상 모두48train query에서 K8지원 가능하다. 구름·연무·bank지역편중과 경계객체 의존성은 미해결이다. 추가GPU0. [실행·한계 기록](docs/OE10_P2_EXECUTION_20260927.md) · [검토 영상](artifacts/oe10_expansion_inputs_20260928/audit_v0/contact_sheet.png).


> **9/28 01:39 KST - Hourly research continuation:** olmoearth-vlm is ACTIVE every60min in this thread, replacing the missing oe10-b0-b2 monitor. Follow [durable execution queue](config/oe10_hourly_research_context_20260928.json): current P2 audit, new-target inputs, sourced-context model integration, bounded pilot. Latest live B0 seed270928 reached1702/2304; no new paired score. Existing results and12h cap remain frozen.


> **09/28 01:05 KST 첫 비교 쌍 완료·독립 검산.** 최종 IoU K-AUC는 B0 0.300996 / B2 0.341754(차이 +0.040758). 양팔 모두 기존 학습 안정성 미충족으로 전체 우월성 판정은 미정이며 다음 seed가 진행 중이다. 겨울밀 K8 IoU 0.007845·16행 중 14행 빈 예측이 주요 실패다. 최종 원확률 768개를 포함한 860파일을 회수해 해시 검증·독립 재집계를 완료했고 CPU 확률 진단도 보존했다. 기존 학습·예산·판정 변경이나 새 GPU 실행은 없다. [실행 기록](docs/OE10_P2_EXECUTION_20260927.md).

> **9/28 평가 확장 선호 반영:** 겨울밀만 우선 고치는 데 개발을 묶지 않는다. 원 주평가는 겨울밀을 포함해 유지하고, 추가 대상은 자료 가용량으로 검토한다. 포도밭·사료용 콩과 등이 후보이며 아직 새 평가/학습은 없다. [가용성·보고 범위](docs/OE10_P2_EXECUTION_20260927.md).

> **09/28 01:17 KST 후속 입력 준비:** 겨울밀 등4작물의 출처 있는 설명을 기존훈련2304episode에 연결하고20개 검수후보·경계검사6개를 준비했다. 실제전문가응답/새학습0. 현재모델은설명을mask에직접쓰지않으므로다음은별도연결검사다. [준비 기록](docs/OE10_P2_EXECUTION_20260927.md).

> **9/27 OE10 B0/B2 본 비교 실행 중 — [실행 기록](docs/OE10_P2_EXECUTION_20260927.md).** GPU1에서 두 모델×3시드×2,304update를20:42KST에 시작했다. 최대K8·공식native replay·별도프로세스 재개 검증과80입력/역할감사를 완료했다. 수치 안정성 수정 후 B2 재개 차이3.576e-7로 기존1e-6관문 통과, 반복K8약9.07초. 결과 전에 학습·평가 일정을 동결했으며 전체12시간 상한이다. 23:16KST 첫B0는2,304update 완료(AUC0.300996)했지만 마지막3창 안정성 기준은 미충족이다. 첫B2는1,152/2,304update(AUC0.308668 중간값)로 진행 중이다. 두 모델 비교·전체관문·논문 기여는 미판정이다. 30분 추적을 등록했고 유의미한 변화만 보고한다. 아래는 이전 이력이다.

> **9/27 OE9 리뷰 반영·P1 완료 — [실행 결과](docs/OE9_P1_EXECUTION_20260927.md).** v6에서 설계를 동결하고 B0/B2 수치 판정·Kuro 감사 제한·D1 미완료 보관을 반영했다. 실제80입력 loader 검사와 GPU1 두 학습 사례의 head fit(IoU .954/.861), 실제Qwen→EO 역전파·동일worker 저장복원이 통과했다. 이 수치는 훈련 사례이며 P2 비교·새방법·일반화 결과가 아니다. 사람pilot v1의 정보 부족(대상/혼동3건)을 보존하고 층화20건 v2를 준비했다. 실제사람응답0. GPU작업은 종료했고 다음은 native replay를 포함한 공통 B0/B2 trainer와 결과 전 학습 일정 고정이다. 아래는 이전 이력이다.

> **9/27 OE8 입력 준비 완료 — [실행 결과·제약](docs/OE8_PASTIS_INPUT_PREPARATION_20260927.md).** PASTIS80개×8관측을 실제 추출하고 원자료80개/240payload를 독립 재검산했다. 48train/16source-bank/16dev, 합성 교정 train2,304/dev672구성까지 생성·감사 완료. 겨울보리 support 상한7로 전체4class는K1/2/4, 공통3class96query-pair만K8까지 가능하다. 구름 존재·dev부모1개·실제 사람 교정 및 runtime loader 미검증을 유지한다. 전체입력은 서버, 약59.4MB 검토 묶음은 로컬에 회수·hash검증; CPU작업 종료, 이번 새GPU학습0. 다음은 실제 loader/저장복원→GPU1 profile·두 사례 overfit→강한 기준선이다. 아래 이전 상태는 당시 이력이다.

> **9/27 OE7 다음 실험 설계 v6 — [실험 순서·비교군·비용·판정](docs/OLMOEARTH_NEXT_EXPERIMENT_DESIGN_20260927.md).** 세 독립 검토와 두 후속 검토를 반영해 입력80개 준비→실제 trainer→강한 기준선→표현×관측 선택2×2→독립 평가로 구체화했다. 교정 쌍1/2/4/8, 같은 support/query, 숨은 관측 비용, 실제 사람 시간, EO/새 reader 재사용을 분리한다. 현재 두 입력·80메타후보·제안 학습0 상태는 그대로이며, 이번 갱신은 설계/문서만이다. 신규성·본 평가 사전등록은 미확정. 이전 내용은 당시 이력이다.

> **9/27 OE6 실제 입력 준비·타 분야 확장 — [실행 기록](docs/OE6_PASTIS_INPUT_PROGRESS_20260927.md) · [설계 v5](docs/OLMOEARTH_CROSSDOMAIN_LOOP_PLAN_20260927.md).** PASTIS 약40GB hash 검사로 정상1/불일치1/누락1 확인. 정상 shard의911개 중 원사례2개를 변환하고 영상/날짜 불일치 수정·정규화 독립검산·육안 QA 완료.64train/16dev는 지역을 분리한 메타데이터 후보이며 아직 입력 준비 전이다. 로봇·시각토큰·물리 모델을 조사하고 캐시 의존성10개 테스트 통과. 비교 GPU 학습·tokenizer/loop 성능은 아직 없다. 다음은80개 입력과 관측 품질 정책·실제 trainer 저장복원 준비다. 이전 기록은 당시 이력이다.

> **9/27 연구 설계 v4 — [문제·자료·아키텍처 업데이트](docs/OLMOEARTH_VLM_ARCHITECTURE_DATA_PLAN_20260927.md).** 전문가의 양성·혼동 교정을 새 환경에 재사용하는 OlmoEarth VLM을 중심으로, 근접 선행 8개와 공개 자료를 대조하고 독립 검토를 반영했다. 현재 128→16 token 평균이 공간 4개×날짜 2개를 묶는 것을 CPU 연산 진단으로 확인했다. 실제 영상의 의미 손실이나 방법 성능은 아직 미측정이다. PASTIS·Kuro를 중심으로 같은 정보의 강한 일반 모델과 구조/학습 2×2를 비교하며, 이미 관찰한 Kuro test 사건은 개발 자료로 취급한다. 새 GPU 학습·다운로드·유료 호출 없이 계획과 진단을 갱신했다. 다음 실행은 PASTIS 원 입력 최대 2사례와 Kuro 사건 노출 감사다. 직전 GPU1 연결 검사 완료 범위는 아래 기록과 별도로 유지한다.

> **9/27 15:43 KST GPU1 완료 — [실제 VLM 실행 결과와 다음 실험](docs/OE4_GPU1_VLM_PROGRESS_20260927.md).** native8/32step 학습·저장복원과 실제 Qwen3-VL8B FP32 연결/역전파 검사 통과. 언어손실로 OlmoEarth211개·연결부4개tensor에gradient, 추적가중치변경과 원RGB경로보존 확인. BF16캐시검사실패는보존하며 FP32통과와구분한다. VLM checkpoint저장복원·새지역전이·성능개선은미검증. 이번GPU작업종료·결과/로그회수완료. 다음은PASTIS공개라벨입력2사례계약확인이다. 아래이전대기/실패표시는당시이력이다.

> **9/27 11:54 KST 갱신 — [통합 문제와 첫 교정 전이 실험](docs/OLMOEARTH_TRANSFER_FIRST_EXPERIMENT_20260927.md).** 최근 EO–VLM 선행과 독립 검토를 반영해, 먼저 같은 교정의 환경별 도움/피해를 확인한 뒤 동일 정보의 G1/P1을 비교한다. PASTIS native 입력·지역 분할·교정 후보 library 비용은 실행 전 감사 대상이다. 현재 JSON은 초안이며 실제 episode 0개, 새 GPU 학습 없음. 기존 native v2는 유휴 GPU 90분 대기 초과로 04:09 KST 종료, VLM v1은 선행 실패로 종료했다. 둘 모두 대기 중이 아니며 이번 시도에서 GPU worker 성능 결과 없음. 원 종료 상태와 검토 대응은 `artifacts/oe5_correction_transfer_plan_20260927/`에 보존했다.

> **9/27 OE4 실제 자료·학습 경로 실행 — [신규성 후보와 실행 상태](docs/OE4_NOVELTY_AND_NATIVE_VLM_EXECUTION_20260927.md).** 신규성9축·primary source21개와 독립 검토를 정리했다. 공식v1.2-Base·17.9GB corpus 확보,3,996H5/999저장좌표 그룹 감사,train64/dev8 입력준비 완료. 실제CPU1step에서 encoder213개 tensor gradient·실제가중치변경·저장복원 통과. 공개loss설정 복원 오류를 수정하고 원 실패를 보존했다. GPU0/1은 기존작업 점유로 native8/32step과실제Qwen3-VL검사는 한정 대기 중이며 GPU완료·신규방법효과는 아직 없다(02:45KST).

> **9/27 CVPR 상세 계획·독립 검토 — [OlmoEarth 기반 VLM 연구계획](docs/CVPR_2027_OLMOEARTH_RESEARCH_PLAN_20260927.md).** 최신v1.2-Base+실제8B VLM을 중심으로 새 support 개념의 자연어 검색·영역·관계 재사용을 검증한다. mask→면적→설명을 독립 능력으로 중복 계산하지 않는다. 3개 독립 원 리뷰·draft 후속 검토와 수정 대응을 보존했다. 준비/한정 반증 실험은 조건부 진행, 방법 신규성·100k 본 학습·CVPR50% 채택 확률은 미확정이다. 계획 비용은21 core pipelines+12 보조+baseline별도이며10/10 처리량·자료 관문 전 실행을 약속하지 않는다. 이번에는 공개1k H5 약17.9GB와v1.2 revision의 metadata만 확인했고 새 다운로드·GPU학습·서버전송·유료호출은 없다. [검토 대응](artifacts/cvpr_plan_review_20260927/review_response.md).

> **9/27 방향 확장 — [OlmoEarth 모델·학습·응용 프로그램](docs/OLMOEARTH_CAPABILITY_PROGRAM_20260927.md).** 최종 목표는 영역 찾기·측정·비교·설명을 공유하고 새 개념에 적은 교정으로 적응하는 모델이다. Encoder 단독 평가는 기여 귀속과 재사용성 검증으로 둔다. 동의한 첫 작업은 원본 EO 기준선과 공식 목적 추가 학습의 공정 비교 준비다. 공식 두 과제의 source/CLI를 확인하고 서버 26개 경로를 점검했다. 가중치 로더 연결·의존성·자료 계약이 남아 있으며, 이번에는 새 성능 결과나 학습이 없다.

> **9/27 OE2 실제 문제 감사·수정 — [결과와 다음 실험](docs/OE2_PROBLEM_AUDIT_AND_FIXES_20260927.md).** train 클래스 prior만으로68.36%(frozen과 동점), 동일질문 양답26층 평균은blind50/frozen57.37/joint66.99%다(사후·68문항). 채점 동률75행을 재현하고 다음 실행용 사본을 수정했다. 640개면적라벨 계약은0개통과·기존격자충돌36배. 실제 frozenTiny20장 CPU 진단 완료: 원공간/16/64토큰 MAE .1245/.1553/.1458. VLM 성능·새 사전학습 기여는 미입증. 로그·그림·코드는 로컬/서버에 보존했고 이번 CPU 작업은 종료했다. 원결과/사전등록은 유지한다.

> **9/26 OE2 후속 — [TerraScope 대비·첫 VLM 실행 순서](docs/OE2_TERRASCOPE_COMPARISON_AND_FIRST_VLM_20260926.md).** SPEX·MS-CLIP·분광 뷰 VLM 선행을 반영해 신규성 범위를 좁혔다. 기존 Qwen3-VL-8B 가중치 파일과 H200 메모리 상태를 읽기 확인했고, 20개 train 사례·40문항의 메타데이터 후보를 준비했다(raw 로컬0, 추론불가). 첫 단계는 실제 VLM 연결 검사이며 새 학습/추론 결과는 없다. 동일 연결부 P−L 비교→새 reader 전이·EO 성능 유지가 방법 연구의 결정 실험이다. 기존 진단 논문 골격과 원 사전등록은 유지한다.

> **9/26 최신 설계 — [OE2 다중 AI 감독·OlmoEarth 표현 학습](docs/OE2_MULTIMODEL_SUPERVISION_AND_REPRESENTATION_PLAN_20260926.md).** OpenRouter는 학습 자료 생성·검수 후보이며, 논문 주장은 동일 감독의 방법 비교와 새 VLM으로의 encoder 전이로 검증한다. 총100만 원 계획=판독72만+전문16만+API상한5만+예비7만 원. train200/dev60/독립2인test120 목표와 큰 corpus 계획은 유지한다. 실행 전 설계이며 새 API 호출·학습·전송 결과는 없다. 아래 예산12만 원 예비비 기록은 이번에 API5만/예비7만으로 세분했다.

> **9/26 예산 반영 — [100만 원 AI 보조 라벨링 계획](docs/EO_VLM_LABELING_BUDGET_1MKRW_20260926.md).** 이전5,000개 전수 정밀 라벨 제안은 이력으로 남긴다. 계획 단가 기준 검수자60인시+전문가4인시+예비12만 원. 목표는train200/dev60/독립2인test120개이며20개 유료pilot 실측 전의 가정이다. 채용·견적 확정·발주·새 학습은 아직 없다. 대규모 공개/AI 보조 학습 목표는 유지한다.

> **9/26 확장 설계 — [대규모 EO VLM·전문가 라벨 계획](docs/EO_VLM_SCALE_AND_EXPERT_DATA_20260926.md).** 10k/100k/1M 관측 묶음과5,000개 정밀 라벨을 목표로 재설계했다. 공개 현장·전문 매핑·서술 자료9종의 출처 장부와 공정 비교·비용 계획을 정리했다. 목표 수량이며 확보량이 아니다. 이번 갱신은 문서·출처 조사까지이며 새 데이터 수집·GPU 학습 결과는 없다.

> **9/26 OE1 완료 — [GPU 결과·전송 상태·다음 비교](docs/OE1_BENTXT_FEASIBILITY_20260926.md).** 실제 S2 train512/dev128 패치, 고정 Olmo3-7B, seed17·3조건×256step. 주 BA frozen .6836 / joint .7227 / blind .6680. joint encoder 203개 텐서 변화, 복원 logits 일치,1,740응답 독립 재채점 통과. 19:48 KST 종료 뒤 GPU 계산 프로세스 없음. 실행 자료는 `artifacts/oe1_bentxt_v0_20260926/`, 가중치는 서버 `oe1_bentxt_v0/runs/pilot_v1/training/`. 다음은 새 연결부로 encoder 효과 분리·질문 편향·날짜 통제다. 아직 시작하지 않았다. 보조 75%의 BF16 동률 처리 차이를 다음 실행 전에 통일한다. localhost8774는 E5 화면이며 OE1 화면이 아니다.

> **9/26 상위 설계 — [OlmoEarth 사전학습·사용자 VLM 적응 실험과 CVPR 범위](docs/OLMOEARTH_VLM_EXPERIMENT_PLAN_20260926.md).** 첫 실행 가능성은 위 OE1에서 확인했다. 동일 감독 사전학습 비교·새 지역 적응·근거/EO 성능 보존은 남아 있다. 아래 날짜별 기록은 당시 상태이며 E5·D1의 판정을 바꾸지 않는다.

> **9/25: [E5 완료·독립 검산](docs/E5_EQUAL_BUDGET_RESULTS_20260925.md), 주판정 mixed_or_inconclusive.**
> 12모델·26,325응답·19,080학습step를 검산했다. full BA .7712/.8284/.9094, pair .7533/.8009/.7821. 주대조의 성능 보존·차이 이점 규칙 모두0/3이다.
> full에서 평가 시 D 제거는5,265/5,265개 모두no지만 pair 재학습은 원 라벨 구분을 회복했다. 사전 지정 보조 관측이며 시간 추론·기억 효과·새 사건 일반화의 증거가 아니다.
> [T0](docs/T0_SOURCE_DATE_AUDIT_RESULTS_20260925.md)는7,000타일의 문서상 날짜를 연결했다. E5 primary 두 날짜 모두 일치는26/902, 8사건 중6사건은36–288일 간격이다. 기존 합성 날짜·EO 캐시는 바꾸지 않았다.
> [E6](docs/E6_EXECUTION_STATUS_20260925.md)는 결과 전 명세·실행 코드·57개 합성 검사와 독립 감사기18개 검사를 준비했다. E5 완료·감사 관문은 충족했으나 서버 전송이 자동 승인 심사에서 두 번 거절돼 명시적 사용자 승인 대기다. E6 실제 성능 결과는 없다.
> localhost8774의 최신 화면(v12)은 E5 집계와914개 타일의13,710개 저장 답을 연결했다. 전체 원 응답·실제 질문을 검산하고, 두 사례의 날짜·품질·예측을 브라우저에서 확인했다. [근거 연결](docs/EO_E5_CASE_CONNECTION_20260925.md) · [E0→E5 주장 범위](docs/EO_READER_CLAIM_CHAIN_20260925.md). D1 H는 A 단독·합의 미정이다. 아래 기록은 각 시점의 이력이다.

> **9/25 현재: [E4 완료·독립 검산](docs/E4_DELTA_PROBE_RESULTS_20260925.md).**
> 같은209문항에서2,508응답. D-only 홍수 BA .8421/.8250/.7858, 원 답 일치95.76–96.36%지만 충분성1/3seed로 **판정 유보**다.
> 차이 부호/성분 순서 변경에 강하게 민감하다. 차이 블록 의존성의 탐색 근거이며, 시간 이해·과거 불필요·기억 효과의 증명은 아니다.
> 지역 화면 v6은 E2 914타일과 E3/E4 108타일의 저장 답을 연결했고 원본대조·실제API·브라우저검증을 마쳤다.
> 다음은 네 입력 형식의 동일예산 학습 비교(E5 제안). E4는 종료했고 새 학습은 아직 시작하지 않았다. D1 H는 A 단독·합의 미정.
> 아래 단계별 상태는 각 시점의 이력이다.

> **9/25 현재: [E3 완료·독립 검산](docs/E3_PAIR_DEPENDENCE_RESULTS_20260925.md).**
> 209문항·3,777응답, 원 입력 재현 100%. 차이 토큰 D=B−A가 0인 다섯 조건은 **2,676/2,676개 모두 no**다.
> A/B를 둘 다 남긴 차이 제거도 동일하므로 시간 이해·과거 필요성의 증명이 아니다. 등록 민감성 판정은 그대로 유지한다.
> C1은 같은 문구·품질 안의 EO 입력 구분 근거(BA .8035 대 .5000); E3와 표본이 달라 수치 향상으로 비교하지 않는다.
> 지역 근거 화면 v5에 E2 914타일과 E3 108타일의 저장 답을 연결했다. E4는 D만 보존·부호 반전·좌표 순열 진단을 다음으로 준비한다.
> D1 H는 여전히 A 단독·합의 미정이다. 아래 날짜별 실행/대기 기록은 당시 이력이다.

> **9/25 최신: [C1 동일 문구·날짜·품질 진단](docs/C1_SAME_PROMPT_RESULTS_20260925.md).**
> 기존 E2 답변을 8사건·침수445/비침수457개에서 재집계: reader BA .8035, blind .5000,
> 차이 +.3035 (탐색적 사건 bootstrap 구간 [.1808,.4122]). 날짜만으로는 설명하기 어려운 입력 구분 근거다.
> 과거 영상의 필요성·시간 추론·기억 효과·새 사건 일반화는 미검증. E3는 GPU0 운영 개정 v1에서 실행 중이다 (2026-09-25 05:00 UTC 확인).
> 지역별 화면에는 914타일의 저장된 E2 답을 참조 라벨과 따로 연결한다. D1 H는 A 단독·합의 미정이다.

> **9/25 최신: [C0 CPU 기준선 완료·독립 검산 통과](docs/C0_LINEAR_VIEW_RESULTS_20260925.md).**
> 이후→두 영상: 무침수오탐221→4/457, 홍수검출414→150/457. BA 우열은미정이다.
> E3는별도GPU대기이며모델결과없음. localhost8774 ‘연구 실험’에결과와한계를연결했다.

> **2026-09-25 — [E2 감사·E3 시점 제거 진단·C0 CPU 기준선](docs/E3_PAIR_DEPENDENCE_PROGRESS_20260925.md).**
> E2의 등록 통과는 재현되지만 홍수10사건은 과거 연구에서 노출됐고 산사태는Hiroshima186/Indonesia6으로 편중됐다.
> E3는1,106캐시 정상 확인·209문항·3,777생성으로 동결 후 GPU대기 중. E3 모델 결과는 아직 없다.
> 별도 C0는 이전/이후/두 영상의 선형 판별 정보를 점검한다. D1 합의·시간 추론·기억 필요성은 여전히 미검증이다.


> **2026-09-25 — [홍수·경작지 실제 공간 연결](docs/EO_FLOOD_CROPLAND_CONNECTION_20260925.md).**
> Larkana train 칩1개의 원 좌표·마스크와 WorldCover2020을 연결: 관측 교집합 약366.5ha.
> localhost8774에서 SAR·교집합·1계산/6,999미계산 범위 확인 가능. 피해·VLM 성능 결과는 아니다.
> 실제 SAR 취득일은 미확인. D1은 A 단독 완료·합의 미정으로 보류, 연구 방향은 아래 EO embedding reader/E0–E1 문서를 따른다.


> ## 2026-09-25 — 방향 재정의: 위성 임베딩을 실제로 읽는 VLM
>
> **[DIRECTION_EO_EMBEDDING_READER_2026_09_25.md](docs/DIRECTION_EO_EMBEDDING_READER_2026_09_25.md)가 현재 기준이다.**
> SN7 D1은 판독자 A만 완료(미정)로 보류. 첫 실험 E0 = 기존 EarthTalk(MS-131)에 임베딩 교체·제거 통제
> (`config/earthtalk_content_controls_prereg_v0.json`, `code/earthtalk_content_controls_v0.py`). 학습 없음, GPU 약 20분.


> ## 2026-09-24 저녁 — D1 단계 H 진행 중
>
> **한 장 정리: [D1_H_STATUS_20260924.md](docs/D1_H_STATUS_20260924.md).** 연습 패키지·판독 화면·판독 전 기록 준비 완료,
> 판독자 A 연습 완료. 남은 것: 판독자 B 확보 → 연습 비교 → 기록 커밋 → 실전 → finalize.


> ## 2026-09-24 — CVPR·VLM 재검토: D1 실행 관문·채점 수리 필요
>
> **최신 검토: [CVPR·VLM 연구 업데이트](docs/CVPR_VLM_RESEARCH_UPDATE_2026_09_24.md).**
> MS-155 실패와 기억 병목·SFT 선행 주장 철회는 유지한다. D1 H는 두 사람 독립 판독 ≤10인시,
> 전체 12개 중 완전 합의 최소 8개 + 세 범주 존재가 기존 기준이다.
> 합성 감사에서 합의 3/12도 R 모델 loader에 진입하는 H 관문 누락과,
> 완벽한 reader도 탈락시키는 real/metadata 정답 혼합을 재현했다. 시간만 다른 swap의
> class-only 채점과 답변 파일 완결성도 보완해야 한다. **L/R 판정 실행 전에 구현 수리와
> 필요한 사전 결과 amendment가 우선**이다. H 판독 준비 자체를 금지하는 것은 아니다.
> 원 prereg·운영 runner·기존 MS 판정은 이번에 변경하지 않았다. L 중간값은 inconclusive로 둔다.
> ‘여섯 판 내내 규칙 불변’ 대신 ‘기존 실패를 보존하고 새 통제를 버전별로 기록’으로 서술한다.
> 사람 gold·새 모델 실험·서버 상태 조회는 하지 않았다. 아래 9/23 인수인계는 위치·이력 자료다.
>
> **9/24 두 번째 세션 — H 준비 완료(로컬 검증).** 검수 패키지를 서버에서 받아
> `labeling_pack/visible_contract_v05_20260922`(65MB)로 확보하고 감사했다:
> pack_id `sn7v05-198e4ff03e26f004`(사전등록과 동일), 12 episode·6 AOI·172영상,
> 영상 해시·빌드 소스 해시 모두 일치. 테스트 23개·L selftest 12·R selftest 10 통과,
> 원본 5개 파일 SHA-256은 9/24 감사 기록과 동일, 합성 D1 감사 반례 4개도 이 세션에서
> 그대로 재현했다. **이제 남은 것은 사람 결정 두 개다: (1) 검수자 2명 확보 — 단독이면
> 합의율 관문을 계산할 수 없어 논문에 단일 검수자 한계 + 계약 개정 기록이 필요,
> (2) 판독 시기 — 총 ≤10인시는 상한이며 12 episode를 각자 한 번 보면 끝난다.**
> 판독은 로컬 http.server로 바로 시작 가능(절차는 9/23 인수인계 §5). L·R 판정 실행 전
> 수리 A–D와 amendment 선행 조건은 그대로다. 사람 판독 산출물과 H 판정은 아직 없다.


> ## 2026-09-23 — 다른 기기 인수인계 · 결정 실험 D1 대기
>
> **먼저 읽을 것: [인수인계 문서](docs/HANDOFF_2026_09_23.md).** 저장소·서버 지도, 새 기기 준비 절차,
> 검수 패키지 재생성, D1 실행 명령, 하지 말아야 할 것까지 한 문서에 정리했다.
>
> 연구 질문은 유지한다: 관측이 쌓일 때 제한된 기억으로 무엇이 변했고 언제 확인됐고 지금 무엇을
> 모르는지를 근거와 함께 답하기. **다만 "기억이 병목임을 확인했다"는 전제는 내려놓았다**(MS-155).
> MS-154(v0.3)의 privileged 수치는 위치 교락으로 무효이며 인용하지 않는다. "SFT 선행" 결론도 철회했다.
>
> 다음은 결정 실험 D1 하나다. 사전등록 `config/decision_experiment_d1_prereg_v0.json`,
> 계획 `docs/PLAN_GROUNDED_CHANGE_MEMORY_2026_09_22.md` §9b.
> 단계 H(사람 판독 ≤10인시) → L(근거 손실, 모델 없음, `code/sn7_evidence_loss_v0.py`) →
> R(내용 통제 reader 재실행, `code/sn7_d1_reader_run_v0.py`). H 실패 시 L·R은 돌리지 않는다.
> 결과 네 갈래에 따라 기억 방법 진입 / 벤치마크 축소 / reader 정렬 유료 선택 / 노선 종료를 정한다.
>
> 서버에 진행 중인 우리 작업은 없다. 새 학습·새 데이터셋·새 아키텍처를 동시에 붙이지 않는다.

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
