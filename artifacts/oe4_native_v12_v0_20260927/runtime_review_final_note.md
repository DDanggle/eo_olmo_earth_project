# OE4 최종 runtime 수정본 독립 확인

2026-09-27. 원 검토 `runtime_independent_review.md`는 보존한다. 아래는 코드 정적 재검토이며 서버 실행 결과 검증이 아니다.

실제 파일 SHA256을 다시 계산하여 요청된 고정본과 일치함을 확인했다:

`5a038e059364bac95375861b1b3369c4e6c8f261a10047f13803bd8f49bf3eee`

**판정: 예정된 CPU prepare 및 유휴 GPU 확보 후 bounded smoke 실행에 동의한다. 원 리뷰의 주요 세 지적은 코드에서 적절하게 보완되었다.**

1. `runtime_smoke.py:249–259`: 공식 trainer처럼 online encoder에만 band dropout을 활성화한다. online/target 실제 rate를 receipt에 기록하고 target rate=0을 강제한다. 이후 `model.eval()`에서는 dropout이 비활성화되므로 dev-before/after 및 reload 비교와 양립한다.
2. `runtime_smoke.py:319–320, 374–383`: base와 InfoNCE의 양수 batch/step 수를 각각 기록한다. 일부 objective가 0이면 두 objective 학습 통과로 부르지 말라는 해석을 남기며, 최종 상태도 `PASS_bounded_encoder_update`로 축소했다. 양수 loss가 그 loss 단독의 유효 gradient를 증명하는 것은 아니지만 현재 smoke의 좁은 주장에는 충분하다. 기여 sample 수까지 세는 상세 계측은 후속 연구용 보완이다.
3. `runtime_smoke.py:322–331, 393–434`: 저장한 checkpoint를 실제 공식 loader로 재로딩한다. 같은 observation·mask seed·eval mode에서 S2 전체 token tensor와 각 dev batch의 total/base/contrastive loss를 비교하고, 허용 오차를 벗어나면 실패한다. 새 모델에도 동일 SDPA backend 설정을 적용했다. reload 시 online dropout active rate가 기본 0으로 돌아가도 이번 비교는 eval mode이므로 문제없다. optimizer state 부재와 inference snapshot이라는 제한도 유지한다.

원 리뷰의 운영상 제한은 그대로다. GPU 유휴 여부는 실행기가 확인해야 하며 900초 guard는 강제 전체-process timeout이 아니다. 32개 manifest·8 steps·batch2 설정의 실제 학습 노출은 16개다. 실제 forward/backward 성공, 저장본 parity, 공식 목적의 긍정적 활동 여부는 실행 receipt를 본 뒤에만 확정한다. 공식 대규모 학습 재현 또는 downstream 개선을 주장할 근거는 이 smoke에서 나오지 않는다.
