# PASTIS 80개 입력 준비·교정 episode 생성 결과

2026-09-27. **80개 원 입력 준비와 독립 원자료 검증을 완료했다. 새 방법의 성능이나 신규성을 입증한 결과는 아니다.** 구름 판독 가능성·정확한 필지 독립성·실제 모델 loader와 trainer 검증은 남아 있다. [실험 설계 v6](OLMOEARTH_NEXT_EXPERIMENT_DESIGN_20260927.md)의 P0를 실제 수행한 기록이다.

**실제로 완료한 범위**

| 항목 | 결과 |
|---|---|
| 입력 추출 | 동결 후보80개 모두 성공, 실패0, 준비51.50초 |
| 관측 | 각8개, 총640 patch-date 입력; 원 날짜와 영상 공동 인덱스 |
| 모델 입력 | 128×128×8×12, 실제 측정10밴드와 공식 규칙 보완2밴드 구분 |
| 감독 분리 | 영상 NPZ와 semantic/instance 정답 NPZ를 물리적으로 분리 |
| 학습 역할 | 실제 학습 pool48 / 평가용 source-support bank16 / 개발 query16 |
| 독립 검증 | 80개 모두 원 catalog→nested 참조→S2/semantic/instance H5 재대조 통과,240 payload |
| 입출력 결측 | 선택된80개에서 운영 sentinel -10000 검출0. 구름 없음의 증거는 아님 |
| 교정 후보 객체 | train pool1,282 / source bank456; 공개 mask로 만든 합성 교정 |
| 실제 질문 묶음 | train2,304 / development672; 반복 구성 수이며 독립 영상 수가 아님 |
| 모델 학습 | 이번 단계 GPU0회, 새 학습 결과·새 가중치 없음 |

기존 정상 shard0000의911개 train 목록 안에서만 준비했다. 0001 불일치 파일, 누락0002, 예비 지역t30uxv의 원 관측/정답은 열지 않았다. 원 전체파일 hash는 직전 OE6 검증을 재사용하고 크기·나노초 mtime 연속성을 확인했으며, 이번에는240개 사용 payload를 각각 다시 hash하고 실제 배열을 재검산했다. 전체 약20GB hash를 새로 수행한 것으로 세지 않는다.

**선택 정책은 영상·모델 결과를 보기 전에 고정했다**

원 T개 관측에서 `floor(i*(T−1)/7), i=0..7`로 양 끝을 포함한8개를 선택한다. 초기 모델 관측 후보 위치는2·5, 고정 추가 관측은0·7이다. 이것은 과거 연간 시계열을 읽는 개발 과제이며 실시간 미래 예측이 아니다. 구름이 적은 날짜를 사람이 골라 바꾸지 않았다.

원 int16에서 공식 COMPUTED 식을 한 번 적용했고 clipping은 하지 않았다. B01←B02, B09←B8A 보완은 실제 밴드로 표시하지 않는다. 모든 원 배열·날짜·정규화·라벨·void를 별도 검증기가 다시 계산했다. 원자료의 다른 음수는 자동으로 결측 처리하지 않는다. -10000 처리 정책은 보수적 운영 규칙이며 원 위성 제품의 보정/QA 재획득을 대신하지 않는다.

입력 품질과 학습 권한을 별도 필드로 나눴다. `strict_no_missing_input_eligible`은 입력 결측 여부, `supervised_training_allowed`는 train_pool 소속 여부이며 `clean_training_eligible`은 두 조건의 AND다. bank/dev가 기존 후보의 `role=train` 또는 결측 없음만 보고 학습에 들어가는 오류를 막기 위해 authoritative `training_partition`을 기록했다. [배열 축·권한 명세](../artifacts/oe8_pastis_prepare_20260927/review_bundle_v0/io_schema.json).

**실제 확인된 제약: 구름과 교정 수**

원영상 contact sheet의 page00/04/08/09,32개 patch·256개 관측을 육안 확인했다. 학습·개발 양쪽에서 지표를 가리는 구름/밝은 불투명 관측이 보인다. 예를 들어40474의2019-06-19,30031의2018-11-04는 지표가 크게 가려져 있다. 이는 assistant의 입력 QA이며 전문가의 cloud mask나 판독 가능성 정답은 아니다. 정상 수치와 맑은 관측을 구분해야 한다.

[학습 사례 미리보기](../artifacts/oe8_pastis_prepare_20260927/review_bundle_v0/qa/page_00.png) · [개발 사례 미리보기](../artifacts/oe8_pastis_prepare_20260927/review_bundle_v0/qa/page_08.png).

학습48개에서 객체가 존재하는 patch 수만으로 고른4개 class는 meadow(초지),corn(옥수수),soft_winter_wheat(연질 겨울밀),winter_barley(겨울보리)다. 새로운 class·나라·건강 상태를 학습한 것이 아니다. 대상/비교 방향을 바꾼12개 pair를 만들었다. 정답은 연간 crop mask이며 각 날짜에서 읽힐 수 있다는 보장은 없다.

| 교정 쌍 K | train 구성 수 | development 구성 수 |
|---|---:|---:|
| 1 |576|192|
| 2 |576|192|
| 4 |576|192|
| 8 |576|96|

겨울보리 bank에는 적격 객체8개가5개 patch에 있지만, 같은 patch/class당 최대2객체 규칙을 적용하면7개만 사용 가능하다. 같은 예시를 복제하거나 bank를 사후 재선정하지 않았다. 따라서 **4개 class 전체는 K1/2/4, K8까지의 공통 곡선은 겨울보리를 제외한3개 class 사이96개 query-pair에 한정**된다. K1/2/4의192개 모집단과 K8의96개 모집단을 한 곡선에 혼합하면 안 된다. K8 공통96개의 모든 K를 따로 집계하고, 전체192개의 lower-K 곡선을 별도 보고한다.

dev672구성 중 target-present397/target-absent275다. 이것은 반복K를 포함한 구성 수다. 공통96개에서는 각 K마다 present64/absent32, 전체192개 lower-K에서는111/81이다. target이 없는 영상도 제거하지 않았고 empty mask의 성공으로 IoU 평균을 부풀리지 않도록 scorer용 target 존재 여부를 별도 저장했다. [coverage](../artifacts/oe8_pastis_prepare_20260927/review_bundle_v0/episodes/coverage.json) · [episode 계약](../artifacts/oe8_pastis_prepare_20260927/review_bundle_v0/episodes/episode_contract.json).

**교정과 평가 정답이 새지 않도록 처리했다**

- source bank는 원64후보 안에서 부모별 hash순8개씩 선택했다. bank의 정답을 supervised encoder 학습에 사용하는 것은 금지했다.
- class 선택은 train_pool48에서만 수행했다. dev 정답을 열기 전에 공개 질문 목록을 저장하고 hash를 고정한 뒤 scorer용 정답을 만들었다.
- 양성/혼동 support는 query와 다른 patch의 서로 다른 객체이며 K가 증가하면 동일 순서의 prefix를 사용한다. support 원 instance 전체가8관측에서 sentinel 결측이 없는지 검사했다. cloud 품질은 별도다.
- 공개 질문에는 query class 정답을 넣지 않고, class 이름을 숨긴 동일 예시 기반 질문을 쓴다. ID·경로·hash·class mapping은 검색/평가용이며 모델 prompt로 주면 안 된다.
- 준비 NPZ에는 후보8개가 모두 들어 있다. **향후 모델 loader가 획득한 날짜만 encoder에 전달하는 접근 제어를 구현해야 한다.** 이 단계에서 실제 runtime의 미획득 영상 접근 차단을 검증한 것은 아니다.
- support도8관측을 갖는다. support 처리·검색·관측 수와 encoder 갱신 뒤 재계산 비용을 계산 장부에 포함해야 한다.

bank/train 간 같은 부모384쌍의 centroid 거리는 최소3,615m이고2km 미만은0쌍이다. 이는 가까운 중복 위험을 확인하는 보조 진단이며 정확한 footprint나 전역 필지 ID의 비중첩 인증은 아니다. dev 부모는1개라 독립 지역 일반화 통계는 아직 만들 수 없다. [공간 진단](../artifacts/oe8_pastis_prepare_20260927/review_bundle_v0/centroid_separation_diagnostic.json).

**검증과 보관 위치**

변환 경계조건8개, 독립 verifier의 실패 주입8개, episode builder의 합성9개 검사를 수행했다. 이는 모델 정확도 시험이 아니다. [원자료 독립 검증](../artifacts/oe8_pastis_prepare_20260927/review_bundle_v0/source_verification_v0.json)은80개/240원 payload에서 통과했다. [별도 episode 감사](../artifacts/oe8_pastis_prepare_20260927/episode_independent_audit.json)는 train2,304/dev672개의 역할·중첩K·지원 수·모집단과1,738개 support mask를 검산했다. 해당 감사의 원 라벨 재검산은 로컬 대표3개에 있는82 support mask·90 scoring행에 한정하며, 전체80개 원자료 검증과 구분한다. 감사기 자체의 실패 주입2개도 통과했다.

[추출기 검토](../artifacts/oe8_pastis_prepare_20260927/extractor_review.md)에서 발견한 source 연속성 상수·학습 권한 표시 두 문제는 실행 전에 고쳤다. [episode 정적 검토](../artifacts/oe8_pastis_prepare_20260927/episode_builder_review.md)에서는 실제 실행 source와 동일한 hash를 확인하고 공개 질문의 정답 유출·bank 학습 혼입의 치명적 경로를 찾지 못했다. runtime loader 접근 제한 검증을 대신하는 판정은 아니다.

서버 `/home/work/data/olmoearth/oe8_pastis_prepare_v0/prepared_v0`에 전체 입력546MiB 표시 크기와 정답을 보관한다. source snapshot,로그,episode,검증 결과도 같은 run root에 있다. 로컬에는59,416,566바이트의 검토 묶음을 회수했고1,777개 내부 파일 hash가 모두 일치했다. 대표 입력3개(40411/40139/30031),전체 metadata·질문·support mask·미리보기와 검사 결과를 포함한다. 전체80개 배열이 로컬에 내려온 것은 아니다.

기존 보호 source4개의 code 사본과 존재하는 root 사본2개,총6파일의 hash·size·mtime를 전후 대조해 모두 불변이었다. 이번 CPU 추출·검증·episode 작업은 종료했다. 새 데이터셋 다운로드·GPU 학습·유료 API·인력 연락은 없었다. 코드와 산출물은 연구 저장소에 보존했으며 git commit/push나 외부 게시를 수행하지 않았다.

**깐깐한 연구 판단과 바로 다음 단계**

이 준비는 제한된 교정을 이용한 비교를 실제로 수행할 수 있다는 근거다. 표준 대조학습·예시 분할과 구별되는 새 학습 원리의 증거는 아직 없다. 입력 오류를 고쳤거나 구름 사례를 발견했다는 사실을 방법 신규성으로 바꾸지 않는다. 현재 support는 공개 mask에서 만든 것으로, 실제 사람이 일관되게 교정할 수 있는지와 소요 시간은 아직 측정하지 않았다.

별도로 작성된 [외부 교수 관점 검토](OLMOEARTH_VLM_PLAN_INDEPENDENT_REVIEW_2026_09_27.md)의 기준선 우선·20개 사람 교정 pilot·독립 지역 확대 지적을 다음 관문으로 추적한다. 해당 검토의 당시2입력 상태는 이번80개 준비로 갱신됐고, 실제 평가 support는 dev 부모 내부가 아니라 별도 두 train 부모의 source bank에서 온다. 그래도 dev 부모가1개인 제한은 남는다. 이 문서는 새 아키텍처 계획을 확장한 v7이 아니라 v6 P0의 실행 기록이다. 사람 채용/연락·비용 집행·D1 판정 변경은 수행하지 않았다.

다음은 실제 loader의 역할/관측 접근 제어와 Qwen identity·trainer 저장복원을 구현하고 GPU1에서 처리량 및 두 사례 과적합을 확인하는 P1이다. 이후 충분히 학습한 일반 기준선과 비교한다. 모든4개 class에서 K8까지 비교하려면 모델 결과를 보기 전에 bank 확장 정책을 별도 revision으로 정하고 필요한 추가 원자료를 준비해야 한다. 현재의 lower-K 비교와3개 class 공통 곡선은 그대로 수행할 수 있다.
