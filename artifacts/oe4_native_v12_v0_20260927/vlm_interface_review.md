# EO–Qwen3-VL 인터페이스 독립 검토

2026-09-27. 서버 실행 없이 `vlm_interface_smoke.py`와 `vlm_interface_contract.md`를 정적 검토했다. 원본 script는 수정하지 않았다. 실제 EO provider는 아직 검토 대상에 포함되지 않았다.

검토본 SHA256: script `5aceb52ecfedec2821fc6a181bda74b68d09b4714ee7fb5473e28fb72e142d75`, contract `7c6ec45808a82557e46bbd2c9315679e329f35b0ea95cba2be57221aa27eac1a`.

**판정: 연결 방식 자체를 막을 결함은 찾지 못했다. 그러나 DeepStack 보존 검사에는 이전 capture를 재사용하여 잘못 통과할 수 있는 P1 결함이 있다. 이를 고친 뒤 actual provider의 FP32/native-data 계약을 확인하면 bounded one-step 실행에 동의한다. 이 검토는 실제 forward/backward 성공을 보증하지 않는다.**

## P1 — 새 visual capture 없이 보존 검사를 통과할 수 있다

위치: `vlm_interface_smoke.py:284–290, 333–342`.

`capture_native_visual`은 `visual_pos_masks`가 없으면 아무 기록도 남기지 않는다. 그런데 `generate()` 후 `visual_captures[-1]`을 바로 가져와 baseline과 비교한다. generation prefill에서 visual 경로가 호출되지 않거나 capture가 누락되면 리스트 마지막 값은 직전 inactive/native pass의 값이다. 그 값은 baseline과 같으므로 검사하려던 “EO prefill에서도 native RGB/DeepStack을 보존했다”가 실제로 관측되지 않은 상태에서 통과할 수 있다.

필요한 수정:

- baseline/inactive/generation 단계마다 시작 index를 저장하고 해당 단계의 새 capture만 소비한다.
- 2-token cached generation에서는 prefill visual capture가 정확히 1개이고 cached decode에서는 새 visual capture가 없음을 확인한다.
- 새 prefill capture에 image embedding, 각 DeepStack feature, expected layer 수와 visual token 수가 존재하는지 확인한 뒤 baseline과 비교한다.
- visual mask의 true position도 저장한다. 현재 삽입점은 모든 image token 뒤여서 native visual 위치가 유지되어야 하므로, 그 위치가 input_ids의 image token 위치와 정확히 같은지 assert할 수 있다.

기존 경로가 실제로 잘못 동작한다는 뜻은 아니다. 현재 테스트가 그 고장을 빠뜨릴 수 있다는 finding이다.

## P2 — cache tolerance는 제한된 수치적 검사이며 위치 정확성 전체 증명이 아니다

위치: `vlm_interface_smoke.py:77–78, 348–360`.

동일 첫 generated token으로 cached step과 full recomputation을 비교하는 설계는 적절하다. 다만 기본 BF16 `atol=.15, rtol=.01`과 한 번의 top-1 일치는 미세한 위치 오류나 길게 누적되는 cache 오류까지 배제하지 않는다. 현재 범위를 “두 token 단일-image cache smoke”로 유지해야 한다. 미래의 장문/beam/multi-image 지원을 이 검사로 주장하면 안 된다.

선택적 보완: 원래 native RGB 경로 자체의 cache-versus-full 오차도 같은 tolerance로 먼저 기록하면 허용 오차가 EO 슬롯 변경 때문에 커진 것인지 분리하기 쉽다. 허용 오차를 실패 후 조용히 넓히는 대신 새 설정/receipt로 구분한다. 현재 한 단계 smoke를 위해 장문 검사를 추가할 필요는 없다.

## P2 — provider와 checkpoint 정체성은 별도 검증이 필요하다

위치: `vlm_interface_smoke.py:134–138, 219–227, 373–426`.

- script는 provider의 model이 autograd 경로에 있고 파라미터가 실제 변했는지 확인한다. 그러나 그것이 올바른 OlmoEarth checkpoint·native bands·같은 지역/시간의 RGB인지까지는 provider metadata만으로 증명할 수 없다. root의 실제 provider를 별도로 읽어야 한다.
- VLM은 config hash와 optional commit hint만 기록한다. 이미 보관된 model acquisition receipt/weight shard hashes를 연결하면 정확한 재현본을 식별할 수 있다. 매 smoke마다 8B weights 전체를 다시 hashing할 필요는 없지만 config hash만으로 weights까지 고정됐다고 하면 안 된다.
- encoder LR=1e-6일 때 BF16 encoder parameter를 직접 업데이트하면 변화가 dtype 해상도 아래로 사라질 수 있다. provider는 encoder/optimizer parameter를 FP32로 유지하고 VLM dtype 표시는 VLM용으로만 취급하는 것이 맞다. 현재 update assertion은 이런 문제를 false PASS로 숨기지 않고 실패시킨다.
- 제공된 mask/token source를 `encode()`에서 detach하거나 no-grad로 감싸면 안 된다. 별도 tensor를 requires_grad로 바꿔도 실제 encoder gradient가 없으면 현재 script는 실패한다. 실제 provider에서도 임의 precomputed embedding을 encoder 학습의 증거로 대체하면 안 된다.

## 코드상 적절한 부분

1. EO는 일반 text modality 슬롯에 들어가며 image token ID, native RGB processor, Qwen vision, DeepStack 자체를 대체하지 않는다. image 뒤 삽입점과 image token 수를 검증한다. EO에 native spatial RoPE를 얻었다는 주장을 하지 않는다.
2. frozen embedding output에 `clone` 후 EO tensor를 대입하는 경로는 PyTorch의 CopySlices autograd를 보존한다. VLM weight를 freeze해도 입력에 대한 미분은 살아 있다. 실제 training block은 no-grad 밖에 있으며 EO feature, EO projected tensor, adapter, live encoder의 finite nonzero gradient와 실파라미터 변화를 각각 확인한다.
3. teacher-forced target을 prefix 뒤에 붙이고 마지막 `target_len+1` logits 중 마지막을 제외하는 CE 정렬은 다음-token 학습의 index와 일치한다.
4. 원 RGB prompt의 hook 전·inactive hook·EO optimizer 이후 logits를 비교한다. EO 슬롯이 추가된 다른 sequence를 원 RGB logits와 같아야 한다고 요구하지 않는 구분도 맞다.
5. actual generation 경로에서 injection은 prefill 1회, one-token cached decode는 bypass 1회인지 검사한다. cache 길이와 동일 prefix full recomputation도 비교한다. 일반 `inputs_embeds` 우회로보다 native placeholder/RoPE 경로를 보존하기에 적절하다.
6. precomputed features 사용 시 최종 상태가 `ENCODER_NOT_TESTED`로 구분된다. generic CE target은 gold가 아니라고 명시하며, 1-step loss나 gradient만으로 EO의 의미 이해·정확도 향상·방법 신규성을 주장하지 않는다.
7. frozen VLM에 gradient가 생기면 실패한다. optimizer는 adapter와 provider encoder의 trainable parameter만 받으며 원 RGB parity도 다시 검사한다. 다만 receipt의 optimizer membership 숫자는 실제 set-intersection을 세는 대신 구성을 전제한 기록이므로, shared-module provider를 허용하는 범위로 확장할 때는 explicit identity intersection을 추가하는 편이 좋다.

## 공식 구현 대조

[Transformers v5.15.0 Qwen3-VL 구현](https://raw.githubusercontent.com/huggingface/transformers/v5.15.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)을 직접 확인했다. input-ID 기반 image placeholder 치환 뒤 visual mask와 DeepStack feature를 language model에 넘기는 경로, 새 full forward 시 M-RoPE를 재계산하는 조건은 이 hook 설계와 부합한다. 실제 실행본의 version과 source hash는 receipt로 확인해야 한다.

## 최종 실행 판정 기준

P1 capture freshness를 고친 뒤 실제 provider와 실제 checkpoint를 연결하여 실행한다. PASS가 나와도 의미는 “동일한 native RGB 경로를 보존하면서 EO encoder→adapter→고정 VLM CE의 미분과 한 번의 update가 가능했다”까지다. EO 정보의 유용성, 새로운 개념 전이, sensor 관측 가능성, 독립 EO retention은 별도 실험에서 검증한다.
