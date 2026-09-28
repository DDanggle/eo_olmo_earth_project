# OE10 — B0/B2 실제 비교 실행

> **09/28 사용자 요청: 예약 작업 PAUSED.** 아래 실험 결과를 보존하고 자동 후속 실행을 중지했다. [현황과 위치 정리](RESEARCH_STATUS_AND_LOCATIONS_20260928.md).

> **09/28 08:58 KST 최신 상태:** 원 12시간 상한에서 5/6 완료·전체 비교 미완료로 종료했고 최종 회수·독립 감사를 마쳤다. 완료 5회 모두 학습 안정성 미충족이다. 별도 GPU1 한 사례 연결 검사는 통과했다(실제 47.45초·예산 차감48초, optimizer update0). 설명 입력→mask 및 EO 역전파 경로 확인이며 성능 개선 근거는 아니다. 아래는 시점별 이력이다.

2026-09-27. 설계 v6와 OE9의 성능 판정 규칙을 유지한다. 이 문서는 실행 기록이며 새 방법 제안서가 아니다.

GPU1(H200)에서 **고정 OlmoEarth B0와 일반 공동학습 B2의 본 비교를 시작했다.** 시작 시각은 9월 27일 20:42:43 KST다. 두 모델 × 3개 시드, 모델당 2,304회 업데이트를 순차 실행한다. 현재 결과는 진행 중이며, 두 모델의 완주 결과·학습 충분성 검산 전에는 성능 향상이나 동률을 판정하지 않는다.

## 실행 전 실제로 확인한 것

- 원본128×128 입력, query 2날짜와 각 support 8날짜를 유지한다. K=8에서는 positive8개와 counterexample8개 모두 처리한다.
- 공식 OlmoEarth 목적의 replay를 연결했다. 기존 native train64에서 32×32·2날짜·batch2를 사용하고, 원 crop 해시를 재검산했다. native dev8은 열지 않았다. 이것은 작은 replay 보조 항이며 공식 전체 사전학습의 재현이 아니다.
- 실제 Qwen3-VL-8B를 고정하고 언어 손실이 연결부와 B2 인코더까지 전달되는지 확인했다. 학습은 mask BCE+Dice + 0.05×언어 CE + native loss다.
- 두 모델 모두 별도 프로세스에서 optimizer·모델·RNG를 복원해 다음 단계를 재현했다. 최종 v3의 B2 가중치 최대 차이는 **3.576×10⁻⁷**, 예측·손실 차이는0이다. 원래의 절대 허용치10⁻⁶을 유지했다.
- K=8 반복 학습 시간은 B0 약1.99초, B2 약9.07초였다. B2 최대 할당 메모리는52.22GiB였다. 이 값은 두 학습 사례의 공학 측정이며 전체 학습 시간·수렴 보장이 아니다.
- 정확한 실행 소스에서 전체80입력 및 모델 입력 경계·source bank 분리·누락 관측 검사7항목을 CPU로 검증했다. query 정답을 평가 모델에 전달하지 않는 API/자료 흐름 검사이며 OS 차원의 접근 차단 증명은 아니다.

### 재현성 문제와 수정 이력

| 실행 | 관측 결과 | 처리 |
|---|---|---|
| engineering_v0 | B2 가중치 차이6.51×10⁻⁶으로 실패. 예측·손실은 동일 | 실패 보존, 본 학습 미실행 |
| engineering_v1 | 두 모델 가중치까지 정확히 재현. B2 반복 단계50.73초 | 전역 결정적 연산의 높은 비용 기록 |
| engineering_v2 | 속도9.07초 회복, 위치 표현 한 텐서의 차이7.45×10⁻⁶으로 실패 | 나머지 최대1.19×10⁻⁷. 실패 보존 |
| engineering_v3 | 양쪽 모두 기존 허용치 통과, B2 속도9.07초 유지 | 본 비교에 채택 |

v3는 AdamW epsilon을 두 모델 모두10⁻⁶으로 지정했다. 작은 기울기의 수치 차이를 증폭하는 정도를 줄이는 optimizer 설정 변경이며, 비결정적 커널 자체를 해결했다는 뜻은 아니다. 인코더나 위치 표현을 얼려 검사를 통과시킨 것이 아니다. 이 선택은 **첫 development 예측 전에** 완료했으며 새 논문 기여로 주장하지 않는다.

## 결과 전에 동결한 비교

[실행 프로토콜](../config/oe10_p2_execution_protocol_20260927.json)은 20:42:29 KST에 고정했다. SHA256은 `778505eed8d5f5318f48b4f21d318283b95f100d8d5d7e5a0d8ceacfb4ecc1b3`이다.

| 항목 | 고정값 |
|---|---|
| 반복 | seed270927 / 270928 / 270929, 각 B0와 B2 |
| 학습량 | 각2,304 update, 전체 catalog 정확히1회 |
| 실제 지리 자료 | train48patch. 48×12방향 class-pair×K4가2,304episode이며 독립지역2,304개가 아니다 |
| 학습 순서 | 각4update에 K=1/2/4/8 한 번씩, paired seed의 양팔 순서 동일 |
| 평가 | 매384update, common96×K4=384예측 저장·독립 채점 |
| 주 결과 | 마지막2,304update checkpoint. 중간 최고점 선택 금지 |
| optimizer | AdamW, eps10⁻⁶, weight decay0.01, clip1.0 |
| learning rate | encoder10⁻⁵ / native decoder10⁻⁴ / head·connector10⁻³ |
| schedule | warmup96, cosine으로 기본 LR의0.1까지 감소. 사전 recipe1개 |
| precision | FP32, 동일 Qwen 고정 |
| 계산 상한 | run당 최대4시간, 전체6run 최대12시간. 초기 자원 배분 초안을 처리량에 맞춰 결과 전에 구체화 |

B0는 인코더와 native 모델 전체를 고정하고 같은 head·connector를 학습한다. B2는 native 모델의 같은 online encoder를 공유하며 EO와 head·connector를 공동학습한다. B2 encoder의 모든 파라미터를 학습 가능하게 하는 기존 EpisodeModel 동작을 명시한다. 공식 target encoder는 EMA=1로 고정한다. Qwen 가중치는 양팔 모두 고정한다.

두 모델은 같은 native 자료와 downstream 정보를 받는다. B0의 native loss는 no_grad로 계산하며 가중치를 바꾸지 않는다. B0는 고정 특징을 재사용할 수 있고 B2의 학습 특징은 매번 다시 계산한다. **동일 데이터·update 수 비교**이며 동일 FLOPs 비교가 아니다. 초기 특징 계산, 재계산, CPU I/O를 포함한 GPU 점유 시간과 평가 시간을 기록한다. 평가 시간은 전체 시간에 포함되는 부분집합이다.

## 어떤 주장까지 판정할 수 있는가

주 지표는 고정 common96의 K별 target-present mask IoU와 정규화 AUC다. class1·2·3만 포함하며16개 dev query patch를 재사용한다. 대상 부재32개 base의 잘못된 표시 면적·사례 비율도 검사한다.

기존 [OE9 관문](../config/oe9_review_execution_gates_20260927.json)을 그대로 적용한다. 세 시드 모두 AUC 차이+0.02 이상이어야 반복된 실용적 개발 이점의 후보가 된다. 부재 오탐, 학습 충분성, 마지막 세 평가창의 안정성 등도 통과해야 한다. 덜 학습된 실행, 자원 상한으로 잘린 실행, 불안정한 곡선은 동률·실패의 근거로 삼지 않는다.

이 평가는 `with_language=False`로 mask를 채점한다. 실제 VLM을 통한 학습 신호가 포함되지만 **VLM의 생성 설명 품질 평가가 아니다.** 전문가 교정은 현재 공개 라벨에서 만든 합성 support다. 사람pilot 응답0, 다른 reader 재사용, 독립 EO 성능, 두 번째 개발 지역은 아직 미검증이다. B0/B2 결과만으로 신규성이나 CVPR 채택 가능성을 확정하지 않는다. D1은 미완료 보관 상태이고 Kuro 실행은 여전히 감사 전 금지다.

## 실행·감사 위치와 추적

- 실제 연구 저장소: `/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project`
- 서버 실행: `/home/work/data/olmoearth/oe10_p2_v0/training_v0/`
- 고정 학습 소스: [code/oe10_p2_v3](../code/oe10_p2_v3/)
- 완료한 저장·재개 검사: 서버 `engineering_v3/`. 앞선 v0~v2 기록도 보존했다.
- 공정성 근거: 서버 `fairness_audit_v3.json`, `fairness_audit_v3_evidence.json`
- 최신 CPU 감사·회수 코드: [code/oe10_audit_v2](../code/oe10_audit_v2/). 초기 optimizer 감사기의 파라미터 대응 오류를 수정한 버전이며 학습 소스는 바꾸지 않았다.
- 로컬 결과 모음: [artifacts/oe10_p2_20260927](../artifacts/oe10_p2_20260927/). 모델 가중치는 서버에 남긴다.
- 추적 automation: `oe10-b0-b2`, 30분 간격. 변화가 없으면 알림을 보내지 않고, 실패·첫 paired 결과·전체 완료를 확인한다. 전체 감사·보고 후 중지한다.

완료 시 `summarize_runs.py`가 실제 checkpoint·예측·scorer·source·로그 해시를 확인하고, 여섯 run이 모두 검증된 경우에만 기존 gate의 입력을 만든다. 검증되지 않은 공정성 항목을 자동으로 true로 채우지 않는다. 이어 `evaluate_p2_gate.py`로 판정하고 최종 원 예측과 작은 기록을 회수한다. 실패한 경우 관측된 오류부터 처리하며 새로운 아키텍처나 판정 규칙으로 결과를 맞추지 않는다.

## 첫 실제 중간 평가 — 최종 비교 아님

B0 seed270927의384/2304 update에서 실제384개 확률마스크의 독립 채점을 완료했다. target-present IoU의 K-AUC는 **0.15435**다. K1/2/4/8 IoU는 각각0.14740 / 0.13715 / 0.16607 / 0.15140이며, 대상 부재32개 base의 잘못된 표시 면적·사례 비율은 네 K 모두0이었다. 현재 첫 점수는 낮다. 한 팔의 초기 평가이므로 학습 충분성, B2 대비 개선, 동률을 판정할 수 없다. 고정된 나머지 학습과 평가를 계속한다. 이 수치를 보고 학습 조건·표본·판정 규칙을 바꾸지 않았다.

[첫 원 예측 채점 결과](../artifacts/oe10_p2_20260927/review_export_v2/training_v0/B0_270927_train/score_step_000384.json). 원384개 probability NPZ와checkpoint는서버에있으며,이로컬묶음에는채점·해시·실행기록을회수했다.

### 21:16 KST 자동 추적

controller와 GPU1 worker 생존을 확인했다. B0 seed270927은1,536/2,304 update이며, receipt의 중간 AUC는384/768/1152/1536 단계에서0.15435/0.10294/0.22520/0.24215다. 아직 첫 비교 쌍이 완성되지 않았으므로 향상·수렴 판정은 하지 않는다. 실패나 사용자 조치가 없어 별도 알림 없이 기존 학습을 유지한다. 이번 확인에서 원 예측 재채점이나 실행 조건 변경은 없었다. [추적 기록](../artifacts/oe10_p2_20260927/heartbeat_20260927T121645Z.json).

### 21:47 KST 자동 추적

21:47KST: 첫 B0 seed270927은2,304update·exit0으로 완료했고 최종 AUC는0.300996이다. 마지막3창 범위0.058848이 허용0.005를 초과하여 학습 안정성 필요조건은 미충족이다. 완주는 충분학습이나 우열의 증거가 아니다. B2는GPU1에서 실행 중이며 receipt96/최근log116update,유한loss와PID1421757생존을 확인했다. 첫pair미완료·전체gate미실행. 실행조건/예산/자료/소스 변경 없이 계속하며 새 GPU작업은 시작하지 않았다.

이 안정성 결과는 저장 receipt 곡선과 기존 기준의 필요조건 대조이며 전체 collector/gate 판정이 아니다. 코드의 `budget_cap_reached_while_improving`은 고정 update 예산에서 상승 중이라는 뜻으로, 실제4시간 timeout 발생과 구분한다. [추적 원자료](../artifacts/oe10_p2_20260927/heartbeat_20260927T124716Z/summary.json).

### 22:16 KST 자동 추적

첫B2가GPU1에서 정상 진행 중이다(receipt456/최근log457update). 384step의 AUC는0.193819이며 채점JSON을 회수해receipt SHA와 일치함을 확인했다. 아직 첫pair완료 전 중간 결과이며 원예측 재채점·전체관문 판정은 수행하지 않았다. 앞서 기록한B0 학습 안정성 미충족은 유지된다. 학습조건·상한 변경 및 새 실행 없이 추적한다. [추적 및 회수 기록](../artifacts/oe10_p2_20260927/heartbeat_20260927T131652Z/summary.json).

### 22:46 KST 자동 추적

첫B2는receipt792/최근log799update로 계속 진행 중이다. 768step AUC0.154909의 채점JSON을nx로 회수해 해시를 검증했다. 중간 점수의 하락만으로 우열·실패를 판정하거나 조건을 변경하지 않았다. 첫pair미완료, 앞선B0 안정성 미충족 유지. 실행 오류·필요 조치가 없어 별도 알림 없이 추적한다. [기록](../artifacts/oe10_p2_20260927/heartbeat_20260927T134659Z/summary.json).

### 23:16 KST 자동 추적

첫B2는1,152/2,304update이며GPU1에서계속진행중이다. 1,152step AUC0.308668의 채점JSON을회수해해시를검증했다. 학습중간값이므로B0최종값과의우열판정은하지않는다. 첫pair미완료·기존B0안정성미충족유지·전체gate미실행. 조건변경없이추적한다. [기록](../artifacts/oe10_p2_20260927/heartbeat_20260927T141658Z/summary.json).

23:46KST 추적: 첫B2가1,536update 평가 경계에 도달했고GPU1 worker 생존을 확인했다. 이 시점에 기록된 최신 점수는1,152step의0.308668이며 새 점수·완료pair·실행 오류는 없다. 조건 변경 없이 추적한다. [상태 기록](../artifacts/oe10_p2_20260927/heartbeat_20260927T144653Z/summary.json).

## 9/28 사용자 요청 후 연결·결과 재확인

00:18KST server controller1419400/worker1421757이 기존PID 그대로생존했다. B2 receipt1,896/log1,899update를읽었고, 회수묶음의후속log는1,919였다. 실행중파일을순차복사하므로같은시점의원자적snapshot이라고주장하지않는다. 학습재시작이나새GPU실행은필요하지않았다.

같은1,536update에서B0 K-AUC0.2421479612/B2 0.3238788779(차이+0.0817309167)다. 단일seed중간평가이며B0최종0.300996과섞어최종우열을판정하지않는다. B0마지막3창안정성미충족은유지한다.

최신v2 CPU collector가B0최종checkpoint와6창의예측/점수/로그/학습순서를검산해완료1run을확인했다. 결과schema는partial이며미완료B2와미시작4run/전체controller미완료를기록했다. 가중치역직렬화나원마스크재추론을새로한것은아니다.

[회수묶음](../artifacts/oe10_p2_20260927/review_export_20260928_resume_v0)의467개파일37,368,053bytes를독립검토자가검증했다. B0최종384개확률NPZ도예측목록·에피소드점수와일치했다. [독립검증기록](../artifacts/oe10_p2_20260927/resume_export_independent_verification_20260928.json). 모델가중치와중간NPZ는서버에유지한다.

기존30분heartbeat카드를조회했고중복자동화는만들지않았다. view도구는카드만반환하므로이조회만으로scheduler상태를별도인증하지않는다. 다음은기존첫pair완주후동일감사,나머지시드추적이다. 실행소스·학습조건·판정규칙·전체12시간상한은변경하지않았다.

## 9/28 00:25 KST 연구적 의미 재검토

현재GPU1 firstB2 receipt1944/log1951update,같은프로세스생존. 최신1920step B0 AUC0.292462/B2 0.336834(차이+0.044372)로앞선1536step차이+0.081731보다작아졌다. [최신점수](../artifacts/oe10_p2_20260927/B2_270927_score_step_001920_20260928.json) SHA검증및독립검토완료.

독립검토에서class2 IoU는K별약0.02144/0.00124/0.00112/0.00166으로낮고,전체K1 IoU0.34744보다K8 0.33876이낮다. 여러대상에대한균일한성능이나교정수증가의이점을현재평균값으로주장하지않는다. 마스크는EO+support의별도DenseHead가예측하고현재평가는with_language=False이므로,Qwen언어손실의고유기여는대조군없이는분리불가다. 단일시드공동학습초기신호는있지만전문가교정효율/VLM이해/새방법기여는미입증이다. B0안정성미충족도유지한다.

예약view는카드만반환했다. 로컬legacy automation파일/DB에동일id가없다는사실만으로thread heartbeat정지또는중복을판정하지않는다. 지금까지대화에들어온6번의heartbeat는30분간격이었고,GPU학습은별도controller에서정상진행했다. 예약중복/상태는이반환값만으로확정못하며새예약·기존예약수정은없다. [검토기록](../artifacts/oe10_p2_20260927/meaning_independent_review_20260928.json).

## 9/28 00:40 KST 구현·감독·입력 진단

사용자의성능기대에대해두독립agent와로컬CPU진단으로설정부터재확인했다. 현재조건에서공동학습점수는상승했고,학습전체단절을보여주는증거는없다. native+mask+.05CE 합산과실제encoder/head/connector기울기는기록상정상이다. 겹치는sed출력으로두번보인encode checkpoint는원소스에한번뿐이다.

확인된제약: 영역head는K양성/K대조를각각평균1벡터로받고,전체K토큰은언어연결부에들어가지만현재mask평가가그경로를거치지않는다. 날짜는EOencoder에전달된뒤시간평균되므로시간정보가전부소실됐다고단정하지않는다. query2/support8은의도된동결조건이다. devquery2날짜는2019-01-28/2019-07-12이며연간작물정답의판독가능성을보장하지않는다. 지역지식/자유전문가교정/근거서술학습은이기준선에들어있지않다.

[라벨·놓침CPU감사](../artifacts/oe10_p2_20260927/label_readout_audit_20260928.json): trainclass2양성22patch,class3는23patch로비슷하다. class2가있는dev8patch의16episode행중각K에서14행이빈예측이다(16독립지역아님). 따라서단순양성patch수부족만으로차이를설명할수없다. K별cohort와supportprefix는독립검산상일치했다.

언어정답은정수면적과고정region_1문구다. 2304train행중996행이0%이며그중48행은작은양성을반올림해0%로표현한다. 이숫자는연결가능성을보여주는작은감독의한계이지신규원인판정이아니다. B2양성로그의99.29%에서전체기울기가clip되었지만손실별norm/방향이없어어느손실이원인인지단정하지않는다.

[통합진단](../artifacts/oe10_p2_20260927/representation_loss_input_diagnostic_20260928.json). 실행중조건·판정·표본·예산을변경하지않았고새GPU실행없음. 현재결과를OlmoEarth/VLM접근전체의실패로해석하지않으며,이제약을풀면반드시오른다고약속하지도않는다.


## 09/28 01:05 KST — 첫 비교 쌍 완료와 절대 성능 진단

첫 seed270927의 B0/B2가 모두 2,304 update·exit0으로 완료됐다. 기존 CPU collector는 두 run의 checkpoint 파일 해시, 각 6개 평가창의 예측·점수·로그를 검증했다. 다음 B0 seed270928은 기존 controller에서 진행 중이다. 전체 6개 실행·P2 판정은 아직 미완료다.

| 주 결과 | B0: EO 고정 | B2: 공동학습 |
|---|---:|---:|
| 최종 IoU K-AUC | 0.300996 | 0.341754 |
| 마지막 3개 평가창 범위 | 0.058848 | 0.017875 |
| 기존 안정성 기준 ≤0.005 | 미충족 | 미충족 |

관측 차이는 **+0.040758**이다. 양팔 모두 안정성 조건을 충족하지 못했으므로 충분히 학습한 모델 사이의 확정 우월성으로 해석하지 않는다. B2 마지막 한 구간 증가폭이 0.005보다 작아도 마지막 세 창 범위 조건의 통과가 아니다. 대상 부재 오탐 면적·사례율은 모든 K에서 B2가 B0 이하였지만, 대상이 있는데 놓치는 문제는 별개다.

[새 회수 묶음](../artifacts/oe10_p2_20260927/review_export_20260928_first_pair_v0)에 최종 원확률 768개를 포함한 860파일·63,019,576bytes를 회수했다. root와 독립 agent 모두 전체 해시·크기를 검증했고, agent가 원점수 행에서 IoU와 계층 집계·K-AUC를 재계산했다. [독립 검산](../artifacts/oe10_p2_20260927/first_pair_independent_verification_20260928.json). 가중치는 서버에 남는다.

### 무엇부터 개선할 것인가

예시 8쌍(K8)에서 최종 B2 클래스별 IoU는 초지 **0.568971**, 연질 겨울밀 **0.007845**, 옥수수 **0.448658**다. 겨울밀이 있는 8개 query의 counter 변형 16행 중 14행이 빈 예측이다. 16개의 독립 지역이 아니다.

[CPU 확률 감사](../artifacts/oe10_p2_20260927/review_export_20260928_first_pair_v0/probability_audit_B2_270927_20260928.json)는 고정 최종 예측과 이미 노출된 dev 정답만 사용한다. 양팔 384개씩의 NPZ·label 해시와 원 임계값(확률>0.5)의 픽셀 수를 확인했다. ROC/AP 동률 처리는 독립 1,140개 소형 사례 대조에서 수치 오차 범위 내 일치했다.

겨울밀 K8은 B0 AP **0.388343**, B2 AP **0.395829**로 영역 순위 구분 신호가 남아 있지만, 원 임계값의 IoU는 B0 **0.021329**보다 B2 **0.007845**가 낮다. B2의 query별 양성 확률 중앙값을 평균한 값은 **0.005504**다. 전체 양성 픽셀의 중앙값은 아니다. AP를 IoU 개선으로 바꿔 읽지 않는다. 순위 신호만으로 보정 오류나 특정 손실의 인과를 확정하지 않는다. 낮은 임계값은 대상 부재 오탐을 늘릴 수 있다. 임계값 탐색이나 주지표 교체는 없었다.

기존 비교 뒤 우선순위는 (1) 충분히 학습한 공통 기준선과 훈련 자료에서 정한 양성 학습·출력 정책으로 겨울밀 누락을 줄이는 비교, (2) 같은 checkpoint·입력·예산에서 개별 예시를 유지하는 head와 현재 역할 평균 head 비교, (3) 입력 두 날짜로 연간 작물 정답을 읽을 수 있는지 별도 확인이다. 다음 실행 조건은 결과 전에 따로 동결하며 이번 여섯 run과 기존 관문은 바꾸지 않는다.

현재 세 클래스 동일가중에서 다른 점수가 그대로이고 겨울밀 K8 IoU만 0.30이 된다면 전체 K8 IoU는 약 **0.4392**다. 실패 집중도를 설명하는 산술 예시이며 달성 예상이나 관측 성능이 아니다. 입력 확대·양성 정책·예시 구조를 한 번에 바꾸지 않는다. 원예측 진단까지 실행했고 **개선 학습을 새로 시작한 것은 아니다.**

첫 쌍의 실제 실행시간은 약4.24시간이다. 같은 속도면 세 쌍 약12.73시간으로 기존12시간상한을 초과할 수 있다. 마지막 실행의 예산 종료 가능성을 추적하며 상한을 늘리거나 여섯 실행 완주를 보장하지 않는다. 사람 교정0·단일 개발지역·VLM 설명평가 부재는 유지된다.


## 09/28 01:17 KST — 작물 전문지식 설명의 입력 준비

사용자 제안에 따라 공개 농업 전문자료를 확인하고 겨울밀·초지·옥수수·겨울보리의 설명 초안을 준비했다. **실제 전문가가 현재 영상을 판독한 교정 데이터나 새 학습 결과는 아니다.** 실행 중인 v3 및 기존 판정은 그대로 유지한다. 설계 v7이나 새로운 GPU 실행은 만들지 않았다.

- [지식카드 4개](../config/oe10_agronomy_context_cards_20260928.json): 출처6개, 적용 지역·시기와 한계, 관측에서 확인할 가설, 사람 미검수 표시.
- [입력 묶음 v1](../artifacts/oe10_context_prepare_20260928_v1): 기존 훈련2304 episode/48 query에 support 주석을 통해 설명카드 연결. 설명2304개를 독립 생성하거나 새 정답2304개를 확보한 것이 아니다.
- [검수 후보20개](../artifacts/oe10_context_prepare_20260928_v1/review_queue_20.json): 겨울밀을 요청 대상으로 둔10개와 혼동 대상으로 둔10개, 서로 다른 train query만 선정. query 정답은 읽지 않아 양성/음성 여부를 확정하지 않는다. 현재 이미지 패키징 전이며 실제 응답0개다.
- [생성 코드](../code/oe10_context_prepare_v0/build_train_context.py)와 정보 경계 검사6개 통과. 독립 agent도 입력/산출물 해시·48train범위·정답 비노출을 확인했다.

### 지식으로 제공할 수 있는 것

[ARVALIS의 밀 발달 설명](https://www.arvalis.fr/infos-techniques/date-de-semis-du-ble-differencier-levolution-des-parcelles)은 지역·품종·파종일에 따른 생육 시점 차이를 다룬다. [겨울곡물의 저온 반응 설명](https://www.arvalis.fr/infos-techniques/la-vernalisation-un-passage-oblige-pour-fleurir)은 일반 생리 배경이며 위성영상에서 해당 필지의 저온 노출이나 개화 준비를 직접 관측했다는 뜻이 아니다.

[AHDB의 겨울보리 지침](https://ahdb.org.uk/knowledge-library/establishment-in-barley-germination-and-emergence-gs0-gs2)은 가을 정착과 환경에 따른 발달 차이를 설명하지만 영국의 정확한 달력을 프랑스 필지에 옮기지 않는다. [ARVALIS의 2024 옥수수 사례](https://www.arvalis.fr/espace-presse/estimer-la-date-de-floraison-pour-anticiper-la-recolte-du-mais-fourrage)는 연도별 발달 시점 변동의 예이며 2019 정답이 아니다.

[INRAE/CESBIO·UREP 자료](https://www.cesbio.cnrs.fr/wp-content/uploads/2024/11/stage_prairie_fauvel_pottier.pdf)는 초지의 여름 식생지수 감소가 자연 계절성·수분 스트레스·예초/방목에서 모두 발생할 수 있음을 설명한다. 이 출처는 연구 과제 소개이며 검증된 분류 성능 논문이 아니다. [JRC 작물 달력 보고서](https://publications.jrc.ec.europa.eu/repository/handle/JRC112670)는 관측된 생육 계절 자체가 작물 고유 식별자가 아니라는 배경 근거로만 사용한다.

### 실제 학습에 필요한 연결

현재 EpisodeModel은 고정 prompt만 허용하고 mask를 EO/support에서 먼저 계산한다. 이후 Qwen 출력은 mask로 돌아오지 않는다. 전문가 문장만 추가해 현재 평가마스크가 바뀐다고 주장할 수 없다. 학습 중 언어손실이 인코더를 바꾸는 간접 경로와 추론 중 텍스트를 읽어 mask를 바꾸는 직접 경로를 구분해야 한다.

다음 작은 구현에서는 **같은 영상·support에서 텍스트만 바꿔 mask logits가 달라지고 mask 손실이 텍스트 경로와 EO까지 전달되는지** 먼저 확인한다. 단순 텍스트 조건부 readout은 전문가 지식 사용의 기준선으로 취급한다. 정보가 추가된 효과와 새로운 학습 방법의 신규성은 별개다.

| 조건 | 준비한 입력 | 비교 의미 |
|---|---|---|
| A | 일반 지시·동일 관측·동일 support | 새 공통 구조의 기준 |
| B | A + support의 작물 이름 | 이름이라는 추가 감독의 효과 |
| C | B + 올바르게 연결한 설명 | 핵심 비교 C−B: 설명의 추가 효과 |
| D | C 모델에서 설명만 교환/제거 | 추가 학습군 없이 내용 의존·오류 민감성 진단 |

이 표는 준비한 텍스트 조건이며 실행한 GPU 비교가 아니다. D는 작물 이름과 설명의 명백한 모순을 만들 수 있어 농업 지식의 깊은 이해를 증명하지 않는다. A/B/C에 동일한 새 연결 구조와 영상·support·관측 예산을 적용하고 실제 토큰/계산 예산을 동결해야 한다. 현재 토큰 예산·모델 연결·기울기 검사는 미완료다. 카드에 있는 작물명은 support의 공개 주석에서 가져온 추가 정보이며, 기존 익명 과업의 원점수와 그대로 비교하지 않는다.

실제 전문가 응답은 '양성/혼동 필지·근거 날짜·관찰한 차이·판단 불가능한 부분'으로 받는다. 일반 지식으로 날짜별 실제 상태나 설명 정답을 자동 채우지 않는다. 반복 개선은 훈련 자료에서 하고, 지금 실패를 본 dev는 개발 결과로 남긴다. 최종 주장에는 미개봉 지역 평가가 필요하며 겨울밀 회복과 함께 다른 작물·대상 부재 오탐을 확인한다.

이번 완료 범위는 조사·입력 생성·CPU 경계 검사·독립 검토다. 서버 전송·추가 GPU 학습·사람 검수는 수행하지 않았다. 다음은 기존 P2 종료/감사 뒤 별도 입력 계약의 연결 검사이며, 조건·표본·예산은 새 결과를 보기 전에 동결한다.


## 09/28 01:21 KST — 겨울밀 실패를 유지하며 평가 범위 확장

사용자는 겨울밀 개선에만 집중하기보다 다른 대상을 추가하고 겨울밀의 낮은 성능은 공개하는 방향을 제안했다. **개발 우선순위를 다른 대상으로 넓힐 수 있다. 기존 동결 평가의 겨울밀 포함 평균은 유지하고, 겨울밀 제외 결과는 사후 부분집합으로 별도 보고한다.** 전체 평균에서 제외했다는 사실을 표시해도 그 상승 자체가 모델 개선이 되지는 않는다.

같은 저장 예측을 재집계한 첫 seed의 결과는 다음과 같다. 새 학습·추론·임계값 변경은 없었다.

| 범위 | B0 K-AUC | B2 K-AUC | 상태 |
|---|---:|---:|---|
| 원래 세 대상 전체 | 0.300996 | 0.341754 | 동결 주 결과 |
| 겨울밀을 요청 대상에서만 제외 | 0.442049 | 0.507816 | 사후 초지·옥수수 부분집합; 겨울밀 counterexample은 포함 |
| 대상·counter 양쪽에서 겨울밀 제외 | 0.452506 | 0.511864 | 다른 사후 부분집합 |

두 팔 모두 학습 안정성 미충족이고 한 seed의 개발 결과다. 위 부분집합 수치를 주평가 대신 쓰거나 기존0.342에서0.508로 모델이 개선됐다고 쓰지 않는다. 대신 현재 잘 작동하는 범위와 실패하는 범위를 함께 설명하는 데 사용한다.

[추가 대상 가용성 CPU 감사](../artifacts/oe10_p2_20260927/class_expansion_audit_20260928.json)는 이미 준비된 train/support bank의 적격 객체 목록만 이용했다. 새 대상의 dev 정답이나 성능을 선택 기준으로 사용하지 않았다. 아래는 기존 입력 안에서 확인한 후보이며 새 입력 수량이 아니다.

| 대상 | 적격 train patch | support-bank patch | patch당 최대2개 적용 후 support 수 |
|---|---:|---:|---:|
| 포도밭 | 16 | 4 | 8 |
| 사료용 콩과 작물 | 15 | 7 | 10 |
| 과일·채소·꽃 혼합 범주 | 12 | 4 | 8 |
| 겨울 듀럼밀 | 11 | 6 | 12 |
| 겨울 트리티케일 | 9 | 5 | 10 |
| 겨울 유채 | 7 | 6 | 10 |
| 과수원 | 6 | 5 | 8 |

최소 train4patch와 bank8객체라는 **수량 사전검토**를 통과한7개다. 실제 새 평가의 관측 판독 가능성·정답 품질·개발지역 확보·query 제외 후 train support 수·필지 독립성은 아직 검사 전이다. 포도밭·사료용 콩과는 train자료 수가 상대적으로 많은 우선 검토 후보다. 혼합 원예 범주는 여러 개념을 묶은 라벨이므로 단일한 새 대상처럼 해석하지 않는다. 성공할 것으로 보고 고른 목록이 아니고 평가 대상을 최종 동결한 것도 아니다.

후속 보고는 (1) 기존 겨울밀 포함 결과와 실패표, (2) 사전에 확정한 신규 대상에서 양팔을 동일하게 평가한 결과, (3) 명시된 대상 목록·가중치의 확장 평균으로 나눈다. 기존 세 대상 평균과 다른 대상들이 추가된 평균을 직접 비교해 학습 효과로 세지 않는다. 실제 응용을 좁히면 제목·주장도 그 범위를 따른다. 겨울밀 교정 작업은 자료 준비 상태로 보존하고 추가 개발 우선순위를 다른 대상에도 배분한다. 실행 중6run·원 gate·학습 예산은 변경하지 않았다.


## 09/28 01:39 KST - Hourly continuation registered

The existing oe10-b0-b2 update failed because the app reported that it no longer exists. Created one replacement heartbeat, olmoearth-vlm, in this thread at a 60-minute interval; the tool returned ACTIVE. No duplicate old monitor was retained. The next scheduler firing has not yet been independently observed.

The durable queue and cumulative pilot budget are in config/oe10_hourly_research_context_20260928.json. Preserve the ongoing frozen P2 comparison and its 12-hour cap. Continue CPU input preparation for grapevine and leguminous fodder, then verify a separate text-to-mask and EO-gradient path with name-only versus name-plus-sourced-facts controls. Additional GPU feasibility work can start only after the current run ends and is audited, with an initial 30-minute per-job / 2 GPU-hour cumulative operating cap including retries and inference. This cap is an initial automation operating choice, not a claimed full-research budget. Existing scores, winter-wheat failures, and unopened final regions remain preserved.

Live readback at approximately 01:38 KST: two runs completed, B0 seed270928 training log reached update1702/2304; controller1419400 and worker1435655 were present. This is execution progress, not another paired result. The configured nx connection emitted a changed-host-key warning but returned data; no SSH trust configuration was modified. New GPU jobs in this turn: 0. Automation receipt and live output: artifacts/oe10_p2_20260927/hourly_automation_setup_20260928.json.


## 09/28 02:47 KST — 시간별 첫 실행: 3run 감사·확장 입력 30개 검산

아직 새 대상 성능이나 두 번째 paired seed 결과는 없다. 새 B0 seed270928 최종 K-AUC는 **0.296005**이며 마지막 세 평가 범위 **0.085882 > 0.005**로 기존 학습 안정성 기준을 충족하지 않았다. 원 실험은 3/6 완료·검증, B2 seed270928 진행 중이다. 02:39경 export 기준 receipt504update·train log518update이며 controller/worker가 실행 중임을 확인했다. 새 GPU 작업은 시작하지 않았고 원12시간 상한·표본·조건·소스를 보존했다.

audit_snapshot/oe10_audit_v2/summarize_runs.py로 완료3run의 checkpoint 파일 해시·예측·로그를 검산했다. review_export_20260928_0240_v0의1,256파일95,206,251bytes·최종NPZ1,152개를 회수했고 전파일SHA/크기오류0이다. 독립 agent가 새 B0의384개 원 점수행을 intersection/union부터 query→class→parent 평균·K-AUC까지 재집계해 일치했다. checkpoint 재로딩·재추론을 새로 한 것은 아니다. 6run 완료 전 전체 gate는 실행하지 않았다.

### 추가 대상의 실제 입력 준비

별도 CPU 코드 code/oe10_expansion_inputs_v0/audit_inputs.py로 훈련·source bank의 **입력30개·객체294개**를 확인했다. 입력/라벨/마스크SHA, 밴드·정규화·날짜, instance→semantic→support-mask 결합, 경계 접촉·bbox·부모타일을 대조했다. 개발 packet과 미개봉 지역은 열지 않았다. 원본 tortilla shard를 다시 추출한 감사가 아니라, 앞서 추출한 packet의 해시와 실제 배열을 검산한 것이다. 잘못된 정규화·mask·dev접근·경로이탈·query제외를 검사하는6개 경계 테스트가 통과했다.

| 대상 | 적격 train 객체/patch | query 제외 후 patch당 최대2개 capacity | bank 객체/patch/capacity |
|---|---:|---:|---:|
| 포도밭 | 183 / 16 | 30–32 | 38 / 4 / 8 |
| 사료용 콩과 | 60 / 15 | 23–25 | 13 / 7 / 10 |

두 대상 모두 기존48개 train query 각각에서 query patch를 제외하고 K8 support를 구성할 수 있다. 이는 새 클래스 성능이나 독립 필지8개 확보를 의미하지 않는다. **정정:** 포도밭 bank의8은 patch당 최대2개 제한 후 수량이다. 원 객체38개가 있어 같은 patch 안의 대체 객체는 있고, 추가 patch 여유는 없다. 포도밭 bank는 t31tfj 한 부모 타일에만 있다. 콩과 bank는13객체 중7개가 영상 경계에 닿으며 경계 객체를 제외한다고 가정하면 capacity6으로 내려간다. 이것은 민감도 진단이며 기존 선택 규칙을 바꾸지 않았다.

[검토 영상판](../artifacts/oe10_expansion_inputs_20260928/audit_v0/contact_sheet.png)은 class×역할별 사전 규칙으로 고른4사례, 모두t31tfj다. 구름·연무가 보이고 patch20063의2019-05-15 RGB 패널은 고정표시범위에서 하얗게 포화된다. 포화를 구름 확정 라벨로 세지 않으며 정량 구름 비율도 추정하지 않았다. 결측 없음과 연간 작물 라벨 순도1.0은 그 날짜의 영상 판독 가능성을 보증하지 않는다. 날짜·샘플 제외나 재선정은 하지 않았다. 실제 전문가 검수0, 전체 필지ID/정확 footprint 부재도 유지한다.

입력 감사5파일1,252,217bytes와 그림을 회수·해시 검증했다. 독립 agent의 metadata·support capacity·경계 민감도 재집계와 일치했고, 전송 후 보호4소스의mtime/SHA와 실행 snapshot9파일SHA를 검사했다. 결과는 artifacts/oe10_expansion_inputs_20260928/에 보존했다.

다음 시간별 단계는 검증한 입력에서 **별도 train-only 확장 episode catalog와 관측 품질 검토 묶음**을 만드는 것이다. source bank를 encoder 학습에 사용하지 않고, 실제 설명→mask 경로 구현은 별도 버전에서 이어간다. 신규 dev 대상·가중치 동결과 미개봉 지역 평가, 실제 전문가 응답, VLM 설명의 정확성 평가는 아직 남았다. 추가 자동 GPU 예산 사용0/7,200초이며 기존 P2 종료·감사 전 추가 GPU를 시작하지 않는다.


## 09/28 03:55 KST — 확장 훈련 입력·20개 관측 품질 검토 묶음 완료

새 모델 학습이나 추가 대상 성능 결과는 아직 없다. **384개 항목은 독립 영상384개가 아니라 train 영상48개 × 대상/대조 두 방향 × K1/2/4/8이다.** 실제 전문가 응답은0이고 기존20개 전문가 일관성 pilot을 대체하지 않는다.

원본 builder/loader와 입력 metadata hash를 고정하고 포도밭·사료용콩과 훈련 목록을 별도로 생성했다. 원 train243개 적격 객체 중 고유221mask가 선택됐다. 같은 query 전체 patch 제외, class별 patch당2개 제한, K prefix, 역방향 동일 객체 swap을 검산했고 sourcebank 훈련 참조는0이다. query 정답 내용을 바꿔도 public 목록이 바뀌지 않으며5개 경계 검사가 통과했다. 기존 P2 표본·날짜·코드·예산은 변경하지 않았다.

서버CPU에서 실제 query48개 배열과 훈련용 정답384행을 연결했다. public 목록 hash를 먼저 고정한 후 **train 정답 파일48개만** 열었고, dev·미개봉 최종 지역은 열지 않았다. 두 훈련 부모 타일에서 K8 실제 loader를 호출해 query2관측·support128관측 인스턴스 반환 및 훈련 정답 접근을 확인했다. 모델/GPU forward는 수행하지 않았다. 훈련 항목 중 대상 존재136·부재248이며 이를 새 성능으로 해석하지 않는다.

별도 품질 검토 묶음은 target8/14 × parent t31tfj/t32ulu의4층에서5개씩, 전역 query중복 없이 hash로 선택한20사례다. query 초기2날짜, positive/counter 각8날짜와 경계·원mask를 PNG에 표시했다. 실제 입력31개·mask36개를 검증했으며 query 정답은 검토 그림에서 숨겼다. 작물명은 제공하므로 blind 작물 식별 정확도 평가가 아니다. 기본 응답은 전부 unreviewed다. 네 층의 대표 그림을 직접 확인했고 구름·연무가 있는 support 관측도 그대로 남겼다. 자료를 품질로 자동 제외하지 않았다.

정식 훈련 입력 경로는 **artifacts/oe10_expansion_catalog_20260928/oe10_expansion_catalog_v0/runtime_v2**이고 서버는 /home/work/data/olmoearth/oe10_expansion_catalog_v0/runtime_v2 이다. [20개 품질 검토 지침](../artifacts/oe10_expansion_catalog_20260928/oe10_expansion_catalog_v0/quality_review_v0/PROTOCOL.md)과 PNG·빈 응답 양식을 함께 회수했다. runtime_v2의232파일2,825,084bytes와 quality_review_v0의24파일12,585,425bytes 모두SHA/크기가 일치한다.

감사 중 두 전달 오류를 발견해 이력을 보존했다. runtime_v0는 reference 단계의 gold/raw 접근0 기록이 남았고, AppleDouble 전송 metadata229개를 export manifest에 포함해 회수본 전체검증이 실패했다. v1에서 실제 train gold/raw unique48 접근으로 메타데이터를 정정했고 v2에서 AppleDouble만 분리했다. 원 v0/v1은 보존하며 catalog/정답/mask의 연구 내용은 바뀌지 않았다. 이후 단계는 v2와 quality_review_v0만 사용한다.

처음 서버 읽기가 Connection closed/refused로 실패했지만 관리상 GPU 세션은 RUNNING이었다. 로컬 터널만 재연결하고 **nx tunnel up 및 후속 nx 명령을 동일 exec shell 안에서 실행**하자 접근이 회복됐다. 원격 학습 프로세스는 종료하지 않았다. 03:48 KST 현재 3/6완료 상태, B2 seed270928 receipt1320·log1333/2304, 자체 controller/worker 생존을 확인했다. 마지막 intermediate 평가는1152step K-AUC0.319568이며 최종 비교값이 아니다. 보호4소스mtime/hash 및 P2 snapshot9파일hash 불변을 확인했다.

다음은 새 두 작물의 출처 있는 설명카드(현재 없음)와 별도 설명→mask 경로를 준비하는 단계다. 작물명만 제공한 조건과 설명을 추가한 조건을 같은 구조에서 비교할 수 있어야 한다. P2 종료·감사 전 새 GPU 작업은 시작하지 않으며 추가 자동 GPU 사용은0/7,200초다. 새 지역 전이·VLM 설명 정확성·실제 전문가 교정 효과는 아직 미검증이다.


## 09/28 04:50 KST — 설명 조건 입력·별도 mask 연결부 CPU 검증 완료

**실제 OlmoEarth/Qwen 가중치의 새 forward·학습·성능 결과는 아직 없다.** 이번 완료 범위는 출처 있는 텍스트 입력, 별도 공통 기준선 구현, 실제 tokenizer와 작은 대체 모델의 계산 경로 검사다. 기존 P2 실행은 그대로 보존했다.

- 포도밭·사료용 콩과 카드2개를 기관 원문6개에서 준비했다. AI가 공개 자료를 요약한 것이며 실제 전문가 응답0이다. 포도밭 바닥 식생과 포도 잎을 동일시하지 않고, 콩과 전체를 알팔파 하나로 좁히지 않았다. [카드·근거](../config/oe10_expansion_agronomy_cards_20260928.json).
- 별도 훈련 목록384항목/48query에 일반 문구·작물명·작물명+설명·설명 제거·설명 교환 입력을 연결했다. 작물명은 support 주석에서만 얻고 query 정답·ID·출처 경로는 모델 본문에 넣지 않는다. 1,920개 조건별 입력을 독립 재구성해 일치했다. 제거 조건은 이름만 조건과 정확히 같고 교환은 이름을 고정한다.
- 새 code/oe10_text_mask_v0는 frozen Qwen의 텍스트 표현을 EO 위치 예측 head에 직접 넣는 **일반 조건부 기준선**이다. Qwen이 위성영상/EO token을 읽거나 설명을 생성하는 경로는 이번 adapter에서 실행하지 않는다. 기존 P2의 connector·language CE와 다른 경로이므로 원 P2 점수와 비교해 설명 효과로 주장하지 않는다.
- 로컬 입력 경계7개와 서버 CPU 계약/작은 모델 검사12개, 총19개가 통과했다. 텍스트·영상·support 역할 변경 의존성, 정답만 바꿨을 때 logits 불변, mask 손실→B2 encoder 기울기/B0 동결, 캐시 갱신, padding·저장복원 identity를 검사했다. 검토 중 빠진 frozen EO/정규화 identity를 추가했다. 실제 EO/Qwen 가중치 기울기는 여전히 미검증이다.
- 실제 Qwen tokenizer에서 일반58/이름44/설명267/제거44/교환267토큰이며1,024상한에서 절단0이다. 조건별 텍스트 연산량이 같다는 주장은 하지 않는다. 설치된 Qwen3VLModel.forward 소스를 회수해 호출 형태를 확인했지만 실제 가중치 forward를 실행하지 않았다.
- 문맥은 고유7개뿐이며 설명 조건은 대상 방향별2개다. 향후 이 작은 pilot의 차이만으로 농업 지식 이해와 고정 클래스 구분 코드 효과를 구별할 수 없다. 새 개념·표현·지역 전이 검증은 별도다.

서버 별도 루트는 /home/work/data/olmoearth/oe10_text_mask_prepare_v0 이다.22파일67,016bytes 전송 묶음으로 준비했고 CPU 결과4파일13,750bytes를 회수해 해시 오류0을 확인했다. 원 보호4소스mtime/SHA와 P2 snapshot9파일SHA가 불변이다. 결과는 [검증 기록](../artifacts/oe10_text_mask_prepare_20260928/verification_20260928.json)에 있다. 준비용 GPU CLI는 아직 실행 가능 판정을 받지 않았다. 실행 전 episode contract/catalog/scoring의 외부 고정SHA를 추가하고 실제 파일 identity·사전 조건·누적 예산 예약을 검증해야 한다. 훈련 pilot에는 optimizer/RNG/새 프로세스 재개도 필요하다.

P2 최신 직접 확인은 **04:39 KST:3/6완료, 두 번째B2 receipt/log1920/2304, 자체controller/worker생존**이다. 최근 완료 평가는1536step의중간값0.336939이며 최종값이 아니다. 기존3완료receipt·최종score SHA는 이전 감사와 같아서 동일한 전체export를 반복하지 않았다. 첫 상태 조회는 보존 스크립트 경로를 code로 잘못 지정해 파일없음으로 실패했고 code_snapshot/oe10_expansion_catalog_v0로 바로잡아 정상 회수했다. 실행 장애나 GPU 재시작은 아니다.

다음 한 단계는 새 완료run이 생기면 먼저 감사하고, P2 종료·감사 전에는 위 실행 identity/사전조건만 CPU로 완성하는 것이다. 이후 예약된 작은 실제 가중치 연결 검사로 넘어간다. 원12시간 상한과 추가 자동GPU누적0/7,200초를 유지한다. 기존 겨울밀 포함 주평가·실패표는 그대로이며 한 개발지역·합성 교정·실제 사람0·VLM 설명 평가부재·학습 안정성 부족도 유지한다.


## 09/28 06:00 KST — 두 번째 비교쌍 감사·실제 연결 검사 입력 고정

**완료한 네 run 모두 기존 학습 안정성 기준을 통과하지 못했다.** 두 seed에서 B2 점수가 높아지는 방향은 재현됐지만, 충분히 학습한 기준선 대비 우월성·전문가 교정 효율·VLM 설명 개선은 아직 입증하지 못했다. 원 겨울밀 포함 평가와 12시간 상한을 유지한다.

| seed | B0 최종 K-AUC | B2 최종 K-AUC | B2−B0 | 양팔 안정성 |
|---|---:|---:|---:|---|
| 270927 | 0.300996 | 0.341754 | +0.040758 | 모두 미충족 |
| 270928 | 0.296005 | 0.349704 | +0.053699 | 모두 미충족 |

K-AUC는 지원 교정 수 K1/2/4/8에 따른 대상 IoU를 적분해 요약한 원 동결 지표다. 두 쌍 평균 차이 +0.047228은 기술 통계이며, 세 seed 전체 판정이나 불확실성 구간이 아니다. 같은 데이터 노출·update 수를 비교했으며 GPU 시간은 동일하지 않다. 새 B2의 마지막 세 평가 값은 0.336939→0.354747→0.349704이고 범위 0.017808 > 0.005다. 가장 높은 중간 checkpoint를 고르지 않고 고정 최종값을 유지했다.

두 번째 B2에서도 겨울밀 K8 IoU는 0.006398, 대상 존재16행(고유query8개) 중12행은 빈 예측이다. 첫 B2의14행보다 빈 예측은 줄었지만 겨울밀 IoU가 개선된 것은 아니다. 두 번째 B2 전체 K1 IoU 0.361952→K8 0.348248로 교정을 더 준 이점도 없다. 독립 재집계에서 두 seed의 대상 부재 오탐 면적·사례율은 모든 K에서 B0보다 늘지 않았다. 이 진단만으로 실패 원인을 확정하지 않는다.

### 회수·검산 및 실행 상태

audit_snapshot/oe10_audit_v2/summarize_runs.py로 4/6 완료 run의 checkpoint 파일 SHA·예측·로그를 검산했다. review_export_20260928_0540_v0에서 최종 원확률 NPZ1,536개를 포함한1,653파일127,940,279bytes를 회수해 SHA/크기 오류0을 확인했다. 독립 agent는 모든 NPZ의128×128/float32/유한값/[0,1] 범위를 확인하고24개 평가의 저장 intersection/union부터 query→class→parent→K-AUC를 다시 계산해 일치했다. 정답 배열에서 intersection을 새로 만들거나 checkpoint를 다시 로딩·추론한 검사는 아니다.

직접 live 확인 시각은 **05:38 KST**이며 controller1419400·다섯 번째 B0_270929 worker1456659가 살아 있고 receipt/log1152/2304였다. 이 기록을06:00현재의 step이라고 표시하지 않는다. 전체 gate는6run 검증 전이라 실행하지 않았다. 원 상한은09/28 08:42:43KST이고 마지막 run이 미완료면 미완료로 남긴다. [독립 재집계](../artifacts/oe10_p2_20260927/second_pair_independent_review_20260928.json) · [회수 검증](../artifacts/oe10_p2_20260927/hourly_verification_20260928_0542.json).

### 다음 실제 연결 검사 준비 완료 범위

새 code/oe10_text_mask_v1에서 실행 소스·전체 episode export의 외부 SHA 검사를 모델 import 앞에 추가했다. 232파일 입력과 contract/catalog/scoring/mask를 확인하고 경로 이탈·누락·중복·추가 파일을 거부한다. 새 경계13개와 기존 계약4개 검사가 통과했다. 독립 검토에서 무한대 기울기도 단순 nonzero 검사를 통과할 수 있음을 발견해 **v2에 스칼라 손실/모든 존재하는 기울기의 유한성 검사**를 추가했다. 서버CPU 실제 torch 검사8개가 통과했고 모델·head·기존 연산 코드는 그대로다. v0/v1을 덮어쓰지 않았다.

CPU에서 원 EO weights/config·선택된 공식 소스6개, Qwen 모델/tokenizer자산14개와 다운로드 cache JSON1개, 준비 manifest·설명·episode 입력 identity를 검산했다. 첫 준비는 예상14파일 외 cache JSON이 있어 중단됐고 실패 기록을 보존했다. 파일 내용/해시를 확인해 보조 metadata 하나만 기대 목록에 추가한 뒤 통과했다. 원14자산의 해시나 발견 규칙을 변경하지 않았다. 전체 공식 소스/설치 의존성까지 봉인한 것은 아니다.

[실제 연결 검사 조건](../config/oe10_text_mask_connection_protocol_20260928.json)을 새 실제 가중치 실행 전에 로컬에 동결했다. 코드 v2 source SHA e9f393b4e648931603b2130c00de157dd1bc2333e8340050b507dd0cf58b93ad, 입력 identity SHA1254678033e298052e2f418e172a83d59a932c8a78aa9fded79e28ebdbfa6263을 외부 기준으로 쓴다. class8 양성인 train K1 항목 중 ID 사전순 첫 항목054338dbc857484e70310553:k1을 점수/정답 내용 없이 고정했다. B2·seed280928·동일head·5텍스트조건·optimizer update0이며, 설명 추가 시 mask 수치 의존성, 설명 제거 시 이름만 조건과 일치, 유한한 head/EO 역전파, 동일프로세스 저장복원 오차≤1e-6을 검사한다. 설명의 의미를 이해했다는 검사는 아니다.

v2 소스·identity는 서버의 별도 루트 oe10_text_mask_identity_v2에 전송됐고 identity 결과2파일9,408bytes를 회수해 두 차례 독립 해시 검산했다. 새로운 연결 protocol 파일은 아직 로컬뿐이며 실행 예약도 하지 않았다. 보호4소스mtime/SHA·P2 snapshot9파일·동결protocol SHA가 전후 동일하다. [회수·검증](../artifacts/oe10_connection_identity_20260928/local_recovery_verification_v1.json).

**새 실제 모델 forward·새 GPU 사용·예약은 모두0**이다. 다음은 원 P2 진행 중에는 중복 실행 방지·누적 예산 예약·GPU1 idle 검사·외부 timeout을 담당하는 launcher만 준비하고, P2 종료/감사 후 한 번의 연결 검사로 넘어가는 것이다. 회당1,800초·누적7,200초 상한을 유지한다. 단일 개발지역·합성 교정·사람응답0·생성 설명 평가부재·충분한 학습 미확보를 그대로 보고하며, 이 공통 조건부 기준선을 방법 신규성이나 CVPR 성공 근거로 세지 않는다.


## 09/28 06:59 KST — 5회 실행 감사 완료·후속 실행기 CPU 검증

**완료한 5회 모두 기존 학습 안정성 기준을 충족하지 못했다.** 세 번째 B0 최종 K-AUC는 **0.274924**, 마지막 세 평가 범위는 **0.069244 > 0.005**다. 겨울밀 K8 IoU는0이며 양성16행 중13행이 빈 예측이다. 마지막 B2는 아직 진행 중이므로 이 B0와 중간 B2 점수를 최종 비교로 짝짓지 않는다. 기존 두 비교쌍의 수치와 겨울밀 포함 주평가를 그대로 유지한다.

06:38 KST 직접 확인에서 controller1419400과 마지막 B2_270929 worker1457846가 살아 있었고, receipt/log는384/2304였다. **06:56 KST 후속 사전 검사에서도 P2 running과 GPU1 사용 중을 확인**했다. 뒤의 확인은 새 update 수를 측정한 것이 아니다. 원 전체 상한09/28 08:42:43KST를 유지하며, 시간 부족으로 끝나면 미완료로 기록한다. 6회 검증 전 전체 gate를 실행하지 않았다.

동결 audit_v2로 완료5회를 검산하고 review_export_20260928_0640_v0의 **2,048파일156,498,242bytes·최종 원확률1,920개**를 회수했다. 전체SHA/크기 오류0, 독립검토에서 모든NPZ의형태·자료형·유한성·범위와30개 평가 재집계가 일치했다. 원 정답으로 교집합을 다시 계산하거나 checkpoint를 재추론한 것은 아니다. 기존4회·두 비교쌍의 검산 결과도 이전 기록과 일치한다. [독립 검산](../artifacts/oe10_p2_20260927/five_runs_independent_review_20260928.json).

### 이번에 준비한 실제 실행 장치

별도 code/oe10_connection_launcher_v0를 구현·독립검토·동결했다. P2 실제 종료 및 감사 증거, 종료 상태의해시·보호소스 무결성, 자체 작업 종료, GPU1 UUID/유휴 상태를 확인한다. 고유 작업ID와 파일 잠금으로 중복 실행을 막고, 예약·실제 점유시간을 서버 장부에 원자적으로 기록한다. 회당1,800초·누적7,200초이며 미정산/중단 예약은 전액 남기고 자동 재초기화하지 않는다.

취소나 시간 초과에는 자체 프로세스 그룹만 정리하며, 실행기 비정상 종료에 대비해 고정된 단일 Python worker의 부모 종료 처리도 추가했다. 시간 제한 안에 정리를 확인하지 못하면 미확정 상태와 전체 예약을 보존한다. 세부 범위와 Linux의 상속 제한은 [실행기 설명](../code/oe10_connection_launcher_v0/README.md)에 명시했다. 이 코드는 기존 모델·자료·학습·판정값을 바꾸지 않는다.

로컬 검사15개와 Linux 전용2개를 합쳐 **서버CPU17개 모두 통과**했다. 초기 서버 검사에서는 임시 폴더가 생산 경로 안에 들어가 경계 검사1개가 예상과 다른 단계에서 거절되는 문제가 있었다. 실패 cpu_verification_v0를 보존하고, 독립검토 후 CPU 검사기의 임시 폴더만 허용 경로 밖으로 옮긴 v1에서 같은17개 검사를 통과했다. 동결 실행기·테스트·모델 코드는 수정하지 않았다.

실제 서버에서 읽기 전용 사전 검사는 **P2 진행 중·GPU1 사용 중을 이유로 실행을 거부**했다. 운영 장부·잠금·초기화 증거·작업 폴더는 검사 전후 모두 부재다. 새 GPU 사용과 예약은0이며 실제 EO/Qwen 가중치 forward도 아직 없다. 성공한 운영 실행이나 모델 성능 검사로 세지 않는다. CPU 결과 v1의4파일11,457bytes 및 보존한 실패 v0의4파일12,381bytes를 회수해 해시 검증했다. 전송 후 보호4소스mtime/SHA, P2 실행사본9파일과 원protocol SHA가 불변이다. 로컬 터널 연결 거부1회는 터널만 재연결해 해결했으며 원격 학습은 재시작하지 않았다.

실행기 manifest SHA는3bc700fb400e4d878afb1907b6061936b89b474493753c5f8f0162187090fd12다. 실제 서버는 /home/work/data/olmoearth/oe10_text_mask_identity_v2/code_snapshot/oe10_connection_launcher_v0이며 동결 연결protocol과 초기 예산snapshot도 같은 루트의config에 전송했다. protocol 내부의 server_protocol_transferred:false는 생성 당시 기록으로 보존하고, 현재 전송 상태는 실행config에서 별도로 관리한다. [서버CPU·회수 검증](../artifacts/oe10_connection_launcher_20260928/final_verification_v1.json).

다음은 **P2 종료 확인 → 완료분 감사/미완료 여부 기록 → 종료 감사 증거의 외부SHA 고정·독립검토 → 준비된 실행기로 한 번의 실제 연결 검사**다. 서버 예산 장부를 실행 원장으로 쓰고 로컬 장부에 회수·반영하며, 기존 장부를 덮어쓰지 않는다. P2가 진행 중인 동안 준비한 입력·카드·CPU 검사를 이유 없이 반복하지 않는다. 한 개발지역·합성 교정·사람응답0·생성 설명 평가부재·학습 안정성 부족은 그대로다. 이번 실행기 검증은 연구 효과나 CVPR 기여의 근거가 아니다.


## 09/28 07:38 KST — 변경 없는 진행 확인

5/6 완료 상태이며 마지막 B2_270929의 receipt는1,104, 실제 로그는1,119/2,304다. 자체 controller1419400·worker1457846 생존과 완료5개의 receipt/최종score SHA 불변을 확인했다. 새 완료 결과가 없어 기존 감사/export·준비/CPU 검사를 반복하지 않았다. 원08:42:43KST 상한, 추가GPU/예약0, 기존 한계와 미판정 상태를 유지한다. 근거: artifacts/oe10_p2_20260927/live_status_20260928_0738.json. 다음은 실제 종료 후 감사이며 별도 실행이나 조건 변경은 없다.

## 09/28 08:53 KST — 원12시간 상한 종료·최종 회수·별도 연결 검사 시작

**P2는 미완료로 종료됐다.** 08:42:44KST에 controller가 worker timeout으로 종료했고, 실제 전체경과시간은43,200.395초다. 원12시간 상한의정리오버헤드는약0.395초이며 재시작·연장하지 않았다. 완료5개와미완료B2_270929를분리한다. 미완료receipt는1,872, 실제로그마지막은1,876/2,304, 마지막채점은1,536이다. 중간AUC0.346858을세번째B0최종값과짝짓지않는다. 남은latest.pt는원해시와일치하지만내용재추론·재개는하지않았다.

| seed | B0 final K-AUC | B2 final K-AUC | final paired difference |
|---|---:|---:|---:|
|270927|0.300996|0.341754|+0.040758|
|270928|0.296005|0.349704|+0.053699|
|270929|0.274924|미완료|계산안함|

두완료쌍의평균차이는+0.047228(약4.72%p)이다. 완료5회모두마지막3평가범위<=0.005 조건을충족하지못했다. 동일데이터/update비교이며동일GPU시간비교가아니다. 6회검증이안되어전체gate를실행하지않았고, 일반공동학습의충분한학습후우월성·교정효율·새방법효과를주장하지않는다. 겨울밀실패는원주평가에계속남는다.

동결audit_v2로완료5개의checkpoint·원예측·로그를검산하고, 새review_export_terminal_20260928_0845_v0를회수했다. **2,056파일161,365,609bytes, 최종확률NPZ1,920개**, 전체SHA/크기오류0이다. 독립검토의30평가집계와확률배열형태/유한성/범위검사도통과했다. 원정답으로IoU교집합을재계산하거나checkpoint를재추론한것은아니다. [최종회수검증](../artifacts/oe10_p2_20260927/terminal_export_verification_20260928.json) · [독립결과검토](../artifacts/oe10_p2_20260927/terminal_completed_runs_independent_review_20260928.json).

별도읽기전용종료감사는controller/worker종료,보호4소스mtime/SHA,실행사본9파일,원protocol,완료5개와부분checkpoint기록을대조했다. source-level fairness7개증거확인은전체13개공정성판정과구분한다. attestation SHA는ab387bddceb0600ac828fb2345a9df5ca488dbd25cea4533b76265348e39e3ea이며독립리뷰를완료했다. **종료무결성통과는과학관문통과가아니다.** [종료감사독립검토](../artifacts/oe10_terminal_audit_20260928/independent_review_20260928.json).

후속실행기의v0는유휴GPU사용률이숫자라고가정했다. 실제GPU1은memory0·computePID없음·utilization=[Not Found]여서v0를보존하고v1에조회불가처리만추가했다. 정확한허용표기에서memory가0이고PID가없을때만허용하며미지값/비유한값/점유는거부한다. 모델·표본·조건·판정·예산·기존장부경로는불변이다. 독립검토와서버CPU22검사통과후사전검사ready=true를확인했다. v1manifest SHA3f00b5fd7e9669e4149c1c933d39da64d673fa92418b3432bf9776c4cca850a4.

08:53:15KST에별도connection_20260928_01을시작했다. 최대1,800초를누적7,200초장부에예약했다. 원OlmoEarth+고정Qwen의text-only표현이mask에영향을주는지,mask손실이EO/head에유한기울기를주는지,동일프로세스저장복원이맞는지검사한다. 사전고정train K1 한사례,optimizerupdate0,평가/생성없음이다. 성공하더라도의미이해·작물설명효과·일반화·신규성근거가아니다. 종료후실제점유시간을회수해예산에반영한다.

## 09/28 08:58 KST — 실제 설명 입력→mask 연결 검사 통과

**아직 설명 학습의 효과나 새 방법의 성능 결과는 없다.** 이번에는 사전 고정한 train K1 한 사례에서 실제 OlmoEarth와 Qwen 가중치를 사용해 연결 자체를 검사했다. 새 평가 지역이나 정답을 개봉하지 않았고, optimizer update와 생성 설명 평가는0이다.

connection_20260928_01은 08:53:15–08:54:03 KST, 실제47.450초에 종료됐다. 준비·가중치 확인·모델 실행을 포함한 점유시간을 올림해 서버 누적 장부에48초 차감했고, 미정산 예약0·잔여7,152초다. 자체 worker1461250 종료와 GPU1 compute PID 없음도 확인했다. GPU0의 다른 작업은 그대로였다.

| 검사 | 실제 결과 | 의미 |
|---|---:|---|
| 설명 포함 vs 작물명만 최대 logit 차이 |0.014232|텍스트가 예측 계산에 연결됨|
| 설명 제거 vs 작물명만 차이 |0|동일 입력 통제 일치|
| 교환 설명 vs 작물명만 차이 |0.020865|교환에도 반응; 올바른 지식 사용의 증거는 아님|
| 비영 기울기 텐서 |EO212 / head10 / Qwen0 / 기존 connector0|동결·학습 경로가 명세와 일치|
| 손실·존재하는 모든 기울기 |유한|한 사례 역전파 수치 검사 통과|
| 같은 프로세스 저장·복원 오차 |0|가중치 복원 일치; optimizer/RNG·새 프로세스 재개는 미검증|

일반 문구58, 작물명만44, 설명 포함267토큰을 실제 처리했다. 공개 자료 요약은 사람 교정으로 세지 않으며 사람 응답은0이다. 이 구조는 **Qwen이 텍스트만 읽고 그 표현을 EO mask head에 전달하는 일반 조건부 기준선**이다. Qwen이 EO/RGB를 보고 설명을 생성하는 경로를 이번 adapter에서 검사한 것은 아니다. 임의 초기 head가 문자열 차이에 반응하는 것만으로 농업 지식 이해나 성능 향상을 말할 수 없다.

review_export_connection_20260928_0856_v0의36파일96,310bytes를 회수해 전체 SHA/크기 오류0을 확인했다. checkpoint 가중치는 서버에 남기고 작은 결과·환경·예산·로그를 회수했다. [회수 검증](../artifacts/oe10_connection_launcher_20260928/connection_01_recovery_verification.json) · [실제 모델 검사 기록](../artifacts/oe10_connection_launcher_20260928/review_export_connection_20260928_0856_v0/connection_runs_v0/connection_20260928_01/model_output/receipt.json).

다음 한 단계는 이 연결부의 **실제 학습과 optimizer/RNG를 포함한 새 프로세스 저장·복원 검사**다. 결과 전에 조건·입력·코드·시간을 고정한 작은 pilot으로 진행한다. 이후 같은 구조·자료·학습량에서 일반 문구 / 작물명만 / 출처 있는 설명을 비교하며, 설명의 추가 효과는 작물명만 조건을 이겨야 한다. 제거·교환 통제도 유지한다. 이번 한 사례 검사를 반복하거나 원래 상한 종료된 P2를 재시작하지 않는다. 단일 개발지역, 합성 support, 사람0, 언어 설명 평가 부재, 충분한 학습·전이 미검증이라는 한계는 그대로다.

독립 검토에서도36개 export 파일, 동결 사례/5조건, 모델·입력 identity, launcher/예산 장부의48초 정산이 일치했다. 원 logit·gradient 텐서를 재추론한 검사는 아니며 동결 코드와 실제 실행 기록 대조다. [연결 결과 독립 검토](../artifacts/oe10_connection_launcher_20260928/independent_review_connection_01.json).
