# OE4 실제 CPU 실행 receipt 독립 확인

2026-09-27. 검토: `native_cpu_contract_v1_receipt.json/receipt.json`, `cpu_v1_step_log/train_log.jsonl`, `runtime_smoke_v1.py`, `public_loss_config_bridge.md`, `parent_audit_v0/parent_group_audit.json`. step log는 최초 리뷰 뒤 추가로 확인했다. 이 검토자가 재학습하거나 서버 checkpoint를 직접 열지는 않았다.

**실제 CPU 한 step의 공식 native objective→encoder 업데이트→저장/재로딩 경로가 통과했다는 보고는 증거에 부합한다. GPU 학습·VLM 학습·성능 개선이 완료됐다는 증거는 아니다.**

| 항목 | 확인한 증거 |
|---|---|
| 실행 | `device=cpu`, train2/dev2, batch2, 1 optimizer step, 실제 train indices `[0,1]` |
| 완료 시각 | 2026-09-27 02:37:55 KST; 전체26.95초, optimizer 구간2.12초 |
| 소스 일치 | 실제 script SHA `b737a7254c2d66518e3acc8b8314918c8529d21d33e04347b49be6029cac0b2c`가 receipt와 일치; loss/recipe hash도 독립 source snapshot과 일치 |
| objective 활동 | train의 base·InfoNCE 양수 step이 각각1; dev에서도 둘 다 양수 |
| gradient 실측 | step JSONL의 `encoder_nonzero_grad_parameter_tensors=213`; clipping 전 전체 grad norm `0.50138837099` |
| encoder 실제 변화 | attention Q/K·RoPE·normalization 등 추적6개 tensor의 max absolute delta가 약 `1.00–1.0133e-6` |
| 저장본 복원 | 공식 loader 재로딩 뒤 S2 token shape `[2,8,8,2,1,768]`; token max/mean delta0; dev total loss delta0 및 batch별 base/contrastive/total allclose 통과 |
| dev objective | `0.5838051438 → 0.5838217735`, 차이 `+0.0000166297` |

dev loss는 미세하게 **증가**했다. 한 step·두 dev 파일·원 사전학습 자료의 공학 진단이므로 개선 또는 악화의 연구 결론을 내릴 수 없다. 감소를 PASS 조건으로 삼지 않은 점은 사전에 정한 계약과 일치한다.

## gradient 수치의 증거 수준

성공 경로의 소스는 nonzero encoder gradient가 없거나 target gradient가 있거나 전체 grad norm이 nonfinite면 실패하도록 되어 있다. 따라서 일치하는 source SHA와 PASS receipt는 그 guard 통과를 뒷받침한다. 추가로 전달된 `cpu_v1_step_log/train_log.jsonl`의 실제 step1에서 **nonzero gradient를 가진 encoder parameter tensor213개**를 직접 확인했다. 이는213개 개별 scalar parameter가 아니라213개 parameter tensor라는 의미다. train total `0.4185774624`, base `0.4185709953`, InfoNCE `0.0000064697`, sample indices `[0,1]`도 일치한다. 양수 loss가 각 component 단독의 encoder gradient 크기를 분해해 증명하는 것은 아니다.

## loss config bridge

원래 script와 v1의 차이를 확인했다. 공식 LossConfig의 `pop('type')` 및 train module 생성 후 config 직렬화 순서가 누락의 소스 경로를 설명한다. v1은 정확한 public config와 공식 recipe hash, 나머지 parameter 전부의 일치를 요구하고 두 registry type만 deep-copy한 메모리 설정에 복원한다. 실제 receipt에도 두 변경과 생성된 공식 loss class가 기록되어 있다. loss를 임의 대체한 수정은 아니며 최초 constructor 실패를 학습 결과로 세지 않는다.

## 지역 그룹 감사

audit의 selected records에서 직접 다시 집계했다. 전체3996 H5는 exact-coordinate key999개, 각4파일이다. 기존72파일 진단은 train64파일/63그룹, dev8파일/7그룹이며 **train–dev exact-coordinate 중복은0**이다. train 내부 중복은3979/3977, dev 내부 중복은1267/1266이다. 따라서 “train/dev 누출이 확인됐다”는 표현은 이 감사로 뒷받침되지 않는다.

exact-coordinate 그룹 분리는 지리적 이격·footprint 비중첩·사전학습 비노출을 증명하지 않는다. 또한 이 audit의72파일 manifest hash는 CPU run의4파일 manifest와 다르므로,72파일 dev 내부 중복을 CPU의 dev2에 그대로 적용해서도 안 된다.

GPU native controller 및 VLM one-shot의 대기/실행 여부는 이 CPU receipt에서 확인할 수 없다. 최종 보고에서는 최신 controller receipt를 별도 근거로 연결하고, CPU native 경로 성공과 향후 GPU/VLM 검증을 분리해야 한다.
