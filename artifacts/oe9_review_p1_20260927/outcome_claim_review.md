# OE9 P1 결과 보고의 독립 claim·수치 검토

2026-09-27. 검토 대상은 `OE9_P1_EXECUTION_20260927.md` 초안과 실제 `p1_review_export_v0/receipt.json`, `independent_audit.json`, `controller_status.json`이다. loader 감사 및 human v1 `pending_status.json`/`build_contract.json`도 대조했다. 새 모델·GPU·checkpoint 복원 실행은 하지 않았다.

**판정:** 실질적인 숫자 불일치나 P1을 P2/신규성/일반화 성공으로 확대하는 서술은 발견하지 못했다. 아래 logit 범위 표현1곳을 좁히면 된다.

## 필요한 표현 수정1개

worker는 `result.logits[:, -1]`만 보관해 복원 전후를 비교한다. 따라서 표의 **“loss/mask/logit 차이0”**는 **“loss·전체 mask·마지막 위치의 VLM logit 차이0”**로 고치는 것이 정확하다. 모든 sequence 위치의 logit을 비교한 것은 아니다. 원 `vlm_logit_max_abs_difference` 수치0 자체는 맞다. 새 프로세스 resume 미검증이라는 현재 제한도 그대로 유지한다.

## 실제 자료와 일치한 수치

| 항목 | 원 자료 | 보고서 판정 |
|---|---|---|
| worker 시간 |54.0339734964초|54.03초로 적절히 반올림|
| controller 시간 |55.2026836127초|worker54.03초와 혼동하지 않음|
| 학습 IoU1 |worker0.9544817805, 독립 정수계수1363/1428=0.9544817927|0.95448 및 독립 검산 표기 일치|
| 학습 IoU2 |worker0.8610075712, 독립 정수계수6238/7245=0.8610075914|0.86101 및 독립 검산 표기 일치|
| head 반복 |192 update|frozen EO 특징 위 head 학습으로 구분|
| live joint VLM |step1/reference step2/resume step2의3 forward/backward|독립3단계 일반화 학습이라고 하지 않음|
| gradient |encoder211/head6/connector10, 모두 유한 nonzero|일치|
| CE→encoder anchor norm |0.0371358282864|0.03714로 적절히 반올림|
| continuation state 차이 |3.8743019104e−7|3.8743e−7, 허용1e−6 이하 표기 일치|
| peak allocated/reserved |52.8365197182/53.44921875GiB|52.84/53.45GiB 표기 일치|
| EO/VLM 계산 장부 |EO forward15, patch-date90, VLM forward3|일치|
| checkpoint 크기 |1,235,902,343바이트|일치|
| checkpoint SHA |`ab226ee49318bf5fe0489a11a75c4c792e4e7ddfc0fe94ec334b1bab95d939ef`|worker/독립 감사/보고서 일치|
| loader 감사 |80 packet,2304train+672dev=2976 metadata,48train+16dev=64 query load|일치|
| human v1 분포 |target2/counterexample1/neither11/reference_unavailable6|20개로 합계 일치|
| human v1 참여/응답 |contacted0, complete records0|사람 실패가 아닌 표본 구성 부족으로 구분|

## 주장의 범위가 적절한 부분

- 보고서는 두 사례의 frozen-head 학습 IoU와 실제 joint 단계의 gradient/continuation을 명확히 분리한다. 저장 예측도 joint update 이전 head 결과라고 적었다.
- native replay 미포함, B0/B2 비교 없음, dev 점수 미측정, 새 지역 전이 미확인, 새 방법 우월성 미입증이 명시돼 있다.
- 동일 worker 안의 module/optimizer/RNG 저장복원을 확인한 것이며 새 프로세스의 일반 resume 실행기를 검증하지 않았다고 적었다.
- independent audit는 저장 예측/원 input·label/support 및 checkpoint **파일 hash** 확인이다. 해당 감사기의 `checkpoint_torch_loaded=false`, `continuation_independently_reexecuted=false`와 충돌하는 주장은 보고서에 없다.
- full trainable state가 bitwise 같다는 주장을 하지 않고 실제 최대 차이와 허용폭을 공개한다.
- human v1 구성 부족을 사람의 판독 실패 또는 효과 부재로 해석하지 않는다. v2는 응답·모델 성능을 보기 전 학습 정답으로 구성하는 층화 시간측정 pilot이며 아직 사람 결과로 보고하지 않는다.
- 0.02 및 absent FP 경계는 P2 결과 전 개발 투자 기준으로 설명하며, 통계적 동등성·한 지역 이상의 일반화·채택 확률로 확대하지 않는다.

이 검토는 보고서와 회수 자료의 일치 확인이다. 자체적으로 GPU 결과를 재실행하거나 새 실험을 제안하지 않았다.
