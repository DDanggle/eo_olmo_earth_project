# 외부 검토 반영과 실제 P1 실행

2026-09-27. **새 방법의 우월성·새 지역 전이는 아직 미측정이다.** 이번 결과는 v6을 확장한 새 설계가 아니라 입력 경계·두 사례 학습·실제 VLM 경로·저장복원의 실행 증거다. B0/B2 비교표는 아직 없다.

**리뷰를 실행 규칙으로 반영했다**

[외부 검토 원문](OLMOEARTH_VLM_PLAN_INDEPENDENT_REVIEW_2026_09_27.md)과 [v6 계획](OLMOEARTH_NEXT_EXPERIMENT_DESIGN_20260927.md)은 보존했다. [동결 실행 관문](../config/oe9_review_execution_gates_20260927.json)에 다음을 반영했다.

| 지적 | 적용 |
|---|---|
| 설계 동결 | P2 이전 v7·새 tokenizer/RL/물리/KV 방법 행을 만들지 않는다. 현재 학습형 resampler는 v6에 있던 일반 연결부 구현이다. |
| B0부터 검증 | 고정 encoder B0와 일반 공동학습 B2의 동일 common96, K1/2/4/8 AUC를 비교한다. 고정 seed270927/270928/270929 모두 +0.02 이상이면 개발상 반복 개선, 모두 절댓값0.02 미만이면 이 범위에서 encoder 학습 이점 주장을 보류한다. 나머지는 미결 또는 반복 악화로 구분한다. |
| 충분히 학습했는가 | 미수렴·정보 접근 불일치·부재 대상 오탐 악화가 있으면 수치가 좋아도 승격하지 않는다. step/LR/동일 노출·실측 비용 일정은 완성된 P2 trainer와 profile로 결과 전에 추가 고정한다. |
| 사람 교정 | 20개 시간측정·독립 A/B 판독 패키지를 앞당겨 준비한다. 공개 mask 합성 support를 실제 전문가 교정으로 세지 않는다. |
| Kuro | 사건/과거 test 노출·원 날짜·전처리·정답 의미·분할 감사 전 신규 실행을 허용하지 않는다. |
| 지역 독립성 | 현재 dev부모1개. 지역 전이 주장 승격 전에 두 번째 독립 지역과 노출 감사를 요구한다. t30uxv를 자동으로 봉인 test라 부르지 않는다. |
| EO 강/약 주장 | 약한 주장은 EO 비열등+새 reader 재사용 이점, 강한 주장은 감사된 독립 EO에서 원본/공식 추가학습/일반 joint 대비 개선이다. linear probe/full finetuning은 별도 평가 방식이며 강/약 주장과 동일시하지 않는다. |
| 연결부가 해결하면 | 연결부 개선으로 귀속하고 새 encoder 학습 원리로 포장하지 않는다. |
| D1/SN7 | 새 자원 투입을 중단한 미완료 보관 노선으로 관리한다. H의 A단독·합의 미정 및 원 사전등록은 그대로이며 성공/실패 판정을 새로 만들지 않는다. 현재 투자 노선은 OE다. |

0.02는 데이터에서 추정한 효과폭이나 유의성 경계가 아니라 **P2 결과 전 선언한 개발 투자 판단의 실용 기준**이다. 3개 학습 seed와 한 지역으로 통계적 동등성·지역 일반화·CVPR 채택 확률을 산출하지 않는다. 부재 대상에서는 각 seed/K마다 평균 오탐 면적 증가≤0.005, 오탐 사례율 증가≤0.02도 함께 요구한다. 이 숫자는 보편적 안전 기준이 아니다. gate validator의 합성 경계 검사18개가 통과했지만 실제 P2 점수를 입력한 결과는 없다.

리뷰 당시의 입력2개 상태는 [OE8](OE8_PASTIS_INPUT_PREPARATION_20260927.md)에서80개로 갱신됐다. 실제 dev support는 t31tfm 내부가 아니라 다른 두 학습 부모의 별도 bank에서 온다. 따라서 동일 dev 장면 내 support/query라는 우려는 현재 구성과 다르지만, 평가 지역 하나라는 제한은 여전하다.

**실제 입력 경계 검사**

`episode_loader.py`는 모델용 배열/동일 질문과 감사용 ID/경로를 나눠 반환한다. query는 획득 위치2·5만 복사하며 numpy view의 `.base`를 통해 나머지 날짜에 접근하지 못하게 했다. support는 계약대로8시점을 제공하고 비용에 포함한다. 영상 NPZ 전체8시점은 CPU에서 읽으므로 원자료 I/O 절감은 주장하지 않는다.

정답/채점 디렉터리를 제거해도 추론 load가 동작하고, 개발/bank를 학습 정답으로 요청하거나 역할·입력·질문을 바꾸면 거부한다. 지역 ID·class mapping·정답은 모델 입력에 포함하지 않는다. 이는 Python API 경계이며 OS 샌드박스 인증이 아니다. 로컬 핵심 검사11개와 서버의 실제80 packet/2,976 episode metadata/64 query load 검사가 통과했다. 구름·전역 필지 독립성 인증과는 별개다.

**GPU1에서 수행한 P1 검사**

실행은 물리 GPU1 H200, UUID `GPU-8b485982-e8bb-004a-a10a-e37637e5e2bb`, FP32/no-cache/no-TF32다. 원128×128 입력, query2시점과 양성/혼동 support각8시점을 사용했다. train에서 면적 비율0.10–0.50인 K1 두 사례를 사전 hash순으로 선택했으며 dev 점수는 계산하지 않았다.

| 검사 | 실제 결과 | 해석 |
|---|---|---|
| 고정 OlmoEarth 특징+영역 head 192 update | 학습 사례 IoU 0.95448 / 0.86101 | 같은 두 사례를 반복 학습해 맞추는 검사. held-out 성능이나 encoder 개선 수치가 아님 |
| 원 native token shape | query32×32×2×1×768, support32×32×8×1×768 | 128×128 공간 입력과 날짜 축을 유지한 실제 출력 |
| 실제 Qwen 경로 | mask+0.05×CE에서 encoder211/head6/connector10 tensor에 유한 nonzero gradient | 일반 공동 경로가 동작한다는 검사 |
| 언어 손실 단독 검사 | encoder의 sentinel2 embedding anchor gradient norm0.03714 | Qwen 가중치를 고정해도 언어 손실이 EO 입력·encoder까지 흐름 |
| 실제 encoder 변경 | 추적 가중치 변경 확인 | 새 의미 판독 능력의 증명은 아님 |
| 저장 후 동일 다음 step | loss·전체 mask·마지막 위치 VLM logit 차이0, 전체 trainable state 최대 차이3.8743e−7 | 사전 허용1e−6 안. 전체 문장 모든 위치의 logit이나 가중치 전체 bitwise 동일이라고 보고하지 않음 |
| 자원 | worker54.03초, peak allocated52.84GiB, reserved53.45GiB | 로드/hash/학습/검사 포함. 장기 학습 처리량 추정치는 아님 |

checkpoint에는 encoder·영역 head·일반 연결부·optimizer·Python/NumPy/Torch/CUDA RNG·다음 사례/step·설정·source/자료 ID를 저장했다. 고정 Qwen은 전체 weight shard SHA256으로 참조한다. 같은 worker에서 저장 상태를 다시 로드해 연속 실행의 다음 step과 비교했다. 새 프로세스의 일반 `--resume` 실행기까지 검증한 것은 아니며 P2 trainer에서 추가해야 한다. checkpoint SHA256은 `ab226ee49318bf5fe0489a11a75c4c792e4e7ddfc0fe94ec334b1bab95d939ef`이고 크기는1,235,902,343바이트다. BF16 캐시의 이전 실패를 이번 FP32/no-cache 결과로 해결했다고 하지 않는다.

수치는 두 단계를 구분해야 한다. 192회는 **원 encoder의 고정 특징 위에서 head만 학습**한 것이고, 이후 실제 encoder+Qwen 역전파는 step1·reference step2·resume step2의3회다. 저장 예측도 joint step 이전 head 결과다. 계산 장부는 EO forward15회/patch-date encode90회/VLM forward3회다. 이번 구현에는 native replay가 없으므로 **B2 완성본·충분히 학습한 일반 기준선으로 부르지 않는다.**

별도 CPU 감사기는 원 input/label/support와 사례 선택 규칙을 다시 대조했고, IoU를1363/1428=0.95448179,6238/7245=0.86100759로 재계산했다. checkpoint 파일 SHA도 독립 확인했다. [독립 검산](../artifacts/oe9_review_p1_20260927/p1_review_export_v0/independent_audit.json) · [원영상/정답/예측 QA](../artifacts/oe9_review_p1_20260927/p1_review_export_v0/p1_training_qa.png) · [실제 worker 기록](../artifacts/oe9_review_p1_20260927/p1_review_export_v0/receipt.json). QA에서는 첫 사례의 작은 경계 오류와 두 번째 사례의 분산된 경계 오탐·누락을 확인했다. 그림의 raw/3000 표시는 고정 시각화 변환이며 보정된 반사도 측정이라는 의미가 아니다.

**사람에게 배포하기 전에 발견한 표본 문제**

최초 v1은 query20개와 후보 필지·대상 쌍을 class-blind hash로 선택했다. 실제 결과는 target2/counterexample1/neither11/reference_unavailable6이었다. 대상과 혼동 대상을 구별하는 정보가3건뿐이라 `pilot_design_insufficient_informative_cases`로 판정했다. 사람 응답은0건이며 사람의 판독 실패가 아니다. [v1 원 결과](../artifacts/oe9_review_p1_20260927/human_pilot_v1_record/pending_status.json)를 보존한다.

v2는 사람 응답을 수집하기 전에 공개 **학습** 정답으로 대상·혼동·대조 층을 구성한다. 모델 예측을 표본 선택에 사용하지 않는다. 표본 설계 수정이지 모델 결과에 맞춘 평가 재선정이 아니다. 실제 배포본의 quota·원 ID·정답 mapping은 private에 두고 A/B 판독 화면에는 제공하지 않는다. 이20개는 일반 환경의 자연 빈도를 대표하는 정확도 benchmark가 아니라, 후보 구별과 판독 시간의 전제 검사다. 직접 영역을 새로 그리거나 전문가 교정 후 모델 향상을 측정하는 실험과 구분한다.

실제 v2는 **원래20개 query를 전부 유지**하고 후보 필지와 교정 쌍의 구성을 바꿔 준비했다. 배포 패키지ID는 `1f7f6d87da3143f309972630`이며 960개 RGB/NIR 그림 hash를 로컬에서 검증했다. 지리적 균형보다 원20개 유지를 우선한 사전 규칙에 따라 두 train부모의 구성은9/11이고, 이를 독립 지역 평가로 해석하지 않는다. [검수 화면](../artifacts/oe9_review_p1_20260927/human_pilot_v2/reviewer_package/index.html) · [검수자에게 전달할 ZIP](../artifacts/oe9_review_p1_20260927/human_pilot_v2_reviewers.zip) · [운영 안내](../artifacts/oe9_review_p1_20260927/human_pilot_v2/reviewer_package/PROTOCOL.md). 판독자에게는 public 패키지만 전달하며 private 디렉터리나 부모 디렉터리를 서비스하지 않는다.

최종 상태는 `awaiting_two_complete_independent_reviews`다. **실제 사람 응답0건, 합의율 미측정, 사람 연락·비용 집행0**이다. 패키지에는 대상/혼동/둘 다 아님/불확실/관측 불가의5개 선택, 근거 날짜, 작업 시간, A/B 독립 응답 내보내기를 넣었다. 합성UI의 브라우저 검증은 사람 판독으로 세지 않는다. 기존 UI 코드가 동일한 v2는 표본 구성 검증4개와 실제960개 자산/공개·비공개 경계 검산을 추가했다.

**보관과 종료 상태**

서버 run root는 `/home/work/data/olmoearth/oe9_review_p1_v0`다. 큰 checkpoint는 서버에 남기고 검산 결과·학습 곡선·예측·QA 및 검수 패키지를 로컬 연구 저장소에 회수했다. CPU/GPU 준비 작업은 모두 종료했고 최종 확인에서 GPU0/1 사용 메모리는 각각0MiB였다. 보호4개 source의 hash·size·mtime와 실제 P1 snapshot은 불변이다. git commit/push·외부 게시·유료 API 호출은 수행하지 않았다.

**남은 한 단계**

다음은 P2용 공통 trainer에 기존 공식 native replay를 결합하고, B0의 고정 특징 사전 계산 비용을 포함한 동일 감독·관측·튜닝 예산을 고정하는 일이다. 먼저 충분히 학습한 B0/B2의 개발 점수를 얻고, 그 결과가 요구하는 오류만 개선한다. 두 사례의 높은 IoU를 새 방법 성공으로 삼아 대규모 학습이나 새 아키텍처로 확대하지 않는다.
