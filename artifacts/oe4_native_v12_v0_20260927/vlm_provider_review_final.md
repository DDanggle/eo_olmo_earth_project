# 실제 H5 provider 및 VLM 인터페이스 최종 독립 검토

2026-09-27. 원본 코드를 수정하거나 CPU/GPU 모델을 실행하지 않은 정적 검토다. 앞선 `vlm_interface_review.md`는 그대로 보존한다.

확인한 SHA256:

- `vlm_interface_smoke.py`: `09d3bd38f111210069cf4ba180bcd8d2f563b7def184a70534b982dbd339b7ca` — 요청한 고정본과 일치.
- `real_h5_provider.py`: `79721ed854b08a7b2334fb2a34353c31af1e281d2709edcb05f0b4587dfd1803`.

**판정: 예정된 실제 자료 한 번의 gradient/update smoke를 실행해도 된다. 새로운 중대 입력·graph 결함은 찾지 못했다. 기존 stale visual capture P1은 수정됐다. 실제 실행 성공은 아직 확인하지 않았다.**

## 수정된 P1

`vlm_interface_smoke.py:57–77, 361–378`은 generation 이전 capture index를 저장하고 그 이후 새 capture가 정확히 하나인지 검사한다. 현재 sequence의 mask shape, image placeholder 위치, visual token 수, DeepStack depth 및 각 feature shape를 확인한 다음 원 RGB feature hash와 비교한다. capture가 아예 없을 때 이전 값을 가져와 통과하던 경로가 사라졌다. baseline에도 같은 freshness 검사를 적용했다.

## 실제 provider에서 확인한 사항

- `real_h5_provider.py:19–33`은 기존 manifest의 파일·crop·compact timestep·원래 timestamp index를 사용한다. H5의 presence mask와 timestamp index를 대조하며, raw S2/WorldCover/timestamp를 합친 SHA를 이전 manifest의 frozen crop SHA와 비교한다. 임의 새 이미지나 임의 embedding으로 대체하는 경로가 아니다.
- `real_h5_provider.py:44–50`은 공식 COMPUTED normalization을 한 번 적용하고 config/weights 두 파일 hash를 기대값과 비교한 뒤 공식 loader를 사용한다. full model에서 실제 encoder를 가져오며 FP32 parameter를 유지한다. LR 1e-6의 BF16 직접 업데이트 손실 우려가 해소되어 있다.
- `real_h5_provider.py:55–61`은 모든 관측 S2 pixel의 online mask를 만들고 실제 encoder forward의 native token tensor를 반환한다. official mask 계약은 pixel 해상도이므로 이 mask 크기는 맞는다. `reshape`는 autograd를 끊지 않는다. `encode()` 안에 detach, no-grad, tensor 재생성 또는 캐시된 feature 반환이 없다.
- encoder를 eval mode로 두는 것은 이번 deterministic engineering test에서 gradient를 막지 않는다. 첫 shape/generation용 no-grad 호출과 실제 backward용 새 호출도 분리되어 있다. 고정 VLM을 통과한 loss가 실제 encoder parameter에 도달하는지와 optimizer 이후 실제 변화는 상위 script가 다시 검사한다.
- `real_h5_provider.py:71–76`은 같은 raw crop의 첫 observation에서 B04/B03/B02를 선택해 RGB를 만든다. fixed DN stretch를 model normalization이나 cloud 판정으로 혼동하지 않고 기록한다. 제공된 prepare 명령에서 생성한 RGB와 provider config를 함께 넘기면 지리적 crop 일치가 성립한다.

## 남는 제한 — 이번 실행 blocker 아님

1. **RGB 파일 결합은 실행기 계약이다.** 상위 script는 임의 `--image`도 받을 수 있고 provider config에는 RGB SHA가 없다. root는 이 provider가 생성한 `rgb.png`와 같은 디렉터리의 config를 한 쌍으로 호출해야 한다. 이후 재사용 인터페이스에서는 provider metadata의 expected RGB SHA와 상위 image SHA를 비교하면 잘못된 짝을 자동 차단할 수 있다.
2. **EO는 두 시점, RGB는 첫 시점이다.** 현재 입력은 “같은 crop의 한 RGB observation + 다시간 S2 표현”이다. 양쪽 관측 정보량이 동일하다는 비교가 아니며 RGB 대비 효과를 주장할 통제가 아니다. 현재 generic CE target은 정답 라벨이 아니므로 이것이 이번 gradient 검사 자체를 무효로 만들지는 않는다.
3. **소스 identity를 receipt에 연결해야 한다.** provider가 source/deps 경로를 선두에 넣지만 실제 import된 OlmoEarth `__file__`나 전체 source tar SHA를 자체 assertion/metadata로 기록하지는 않는다. 새 process와 고정 공식 source 경로를 사용하는 현재 실행에서는 기존 launch/source receipt를 함께 보관하면 된다. 잘못 캐시된 package가 있는 장기 process용으로 확장할 때는 실제 module path assertion을 추가한다. 하드코딩한 checkpoint SHA의 공식 출처는 앞선 pinned-download receipt에 연결한다.
4. 현재 provider는 original v1.2 checkpoint SHA를 강제한다. native objective smoke에서 갱신된 checkpoint가 VLM으로 이어진 것을 증명하는 실험은 아니다. 주장 가능한 것은 원래 OlmoEarth encoder에 VLM CE gradient가 도달하여 한 번 갱신되는 경로다.
5. 단일 crop, 16 일반 text slots, flattened token 평균 pooling, 두-token generation만 검증한다. 위치·센서 정보의 보존, EO 추가 정보의 유용성, 새로운 개념 전이, 독립 EO 성능 유지, 학습된 wrapper의 저장·재로딩은 이번 범위에 없다.

## 결과를 보고 판정할 것

실제 receipt에서 새 prefill capture 1개, native RGB/DeepStack parity, cache parity, EO feature/adapter/실제 encoder의 finite nonzero gradients, encoder FP32 실파라미터 변화, frozen VLM gradient 부재를 확인해야 한다. 이 조건들이 통과하면 **실제 EO 입력에서 OlmoEarth→adapter→고정 VLM으로 이어지는 학습 graph가 동작했다**고 말할 수 있다. **VLM이 EO를 이해하게 됐다거나 연구 방법이 기존 방법보다 좋다는 의미 증거는 아니다.**
