# Strict deterministic P4 replay

2026-08-25, physical GPU1, `CUDA_VISIBLE_DEVICES=1`, seed 1, one epoch. 두 실행은 같은
`code_sha256=e9ae32adcc5f71bc162fb160e7b81d2796def289672384b01576a84760a74dee`를 사용했다.

| 검사 | 결과 |
|---|---|
| epoch/train loss/val IoU (`seconds` 제외) | exact equal |
| val 전체 metric dict | exact equal |
| test 전체 metric dict | exact equal |
| val per-sample SHA-256 | `34bf142303bff211abd2ad285e25aad5e283776c635f15510a5b3451e31d1529` 양쪽 동일 |
| test per-sample SHA-256 | `d9f8b1a2891f0995bfebcef84d70fcf0ad7ea551454051425e129f36b08dbf6d` 양쪽 동일 |
| checkpoint SHA-256 | `731bfacf975df29edc65f853a1d4729dfb7506dbdb8b3d7b06061e49fae478c6` 양쪽 동일 |
| checkpoint state tensor | 모든 key bitwise equal, max-abs diff `0.0` |

강제한 계약은 `torch.use_deterministic_algorithms(True)`, cuDNN deterministic/benchmark-off,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off, float32 matmul `highest`다.

수정 전 동일-seed P4 full replay는 원 v2와 test IoU가 `0.122826` 대 `0.143442`(+16.8%),
precision이 `0.134929` 대 `0.168741`(+25.1%)로 갈렸다. 따라서 RNG seed만 기록한 v2는
재현 완료로 보지 않는다. 원 결과와 실패 replay 모두 같은 디렉터리에 보존한다.

## P2-tiny CUDA 예외와 복구

strict 전체 실행은 `max_pool3d_with_indices_backward_cuda`에서 중단됐고 `avg_pool3d`도 같은
deterministic 구현 부재로 실패했다. 경고 모드로 후퇴하지 않고 P2-tiny pooling을
`인접 시점 평균 + spatial max_pool2d`로 분해했다. 이는 공식 3D U-Net이 아니며 최종 strong
baseline으로 쓰지 않는다.

최종 P2-tiny smoke C/D(`code_sha256=a1545d13…`)는 아래가 양쪽에서 같았다.

| 검사 | 동일 SHA / 결과 |
|---|---|
| checkpoint | `e5b676260776ce63d88ef5ddff94c1bdbd741600b91e20eb5b22a4d543590d81` |
| val per-sample | `cbabfbffae66eac0b29aa1a7b5d8aaddcc96d902d0084eb820b7b9ded361b563` |
| test per-sample | `093523dd31b59cea58feadb87985e85c8a204ba4be078149349b5dc28e21be19` |
| metric dict / checkpoint tensor | exact equal / max-abs diff `0.0` |

## 최종 40-epoch full run ↔ P4-only replay

최종 code SHA `478c6af5…`와 동일 runtime fingerprint에서 full P1→P2→P4 실행의 P4와 P4-only를
대조했다.

| 검사 | 결과 |
|---|---|
| epoch별 train loss / val IoU (`seconds` 제외) | 40/40 exact equal |
| best epoch / best val IoU | 37 / 0.11181, exact equal |
| val/test 전체 metric dict | exact equal |
| val/test per-sample SHA | `9cbab148…` / `36559f4f…`, exact equal |
| checkpoint SHA | `74c2c67d43c0f42cdec4735bfa2c8f030622c2055ebf6de9479bef30e6a7f08d`, exact equal |
| 모든 state tensor | bitwise equal, max-abs diff `0.0` |
| cached fit+val wall time | **950.5초 vs 520.0초, 불일치** |

따라서 arm 순서와 무관한 **수치 재현성은 복구**됐다. 반면 wall time은 같은 코드·GPU에서도
system/IO/order 상태에 민감했다. 이 pilot 시간으로 end-to-end 또는 accuracy-cost 우위를 주장하지
않고, randomized arm order·isolated repetition·cold/warm cache를 분리한 G-C를 따로 실행한다.
