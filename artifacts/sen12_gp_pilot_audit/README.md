# Sen12 G-P pilot audit artifacts

이 디렉터리는 `holdout_chimanimani` 개발 fold의 결과 계보를 보존한다. 세 결과 모두 같은
봉인 split(train 5,542 / val China 159 / development holdout Chimanimani 1,133)을 쓴다.

| 파일 | SHA-256 | 상태 |
|---|---|---|
| `m23_8ep/holdout_chimanimani_pilot_8ep.json` | `a37699a4a0f5cc187928ab7a39850d184e9528a7611cf38146a43b31a9c4cc4e` | 8-epoch 최초 관측. flawed sampled AUPRC, test 노출 |
| `m23_40ep/holdout_chimanimani_pilot.json` | `b8372b99f0e0d7a7a00400c38f5ab2f9364c153bff81cc00764979f495b83bda` | protocol amendment. flawed sampled AUPRC/RNG |
| `cache/cache_audit.json` | `58fab487d112743d3f2e54f9c563c9d66a493f8339f8dcd8e1b2078c67f45457` | 6,834 cache content audit, 4/4 gate pass |
| `v2/sen12_gp_pilot_v2/holdout_chimanimani_pilot.json` | `5b87fa5efb13020e218c20875f3e3d57d86b5a739287df45613d6c7fe1de8821` | exact 지표·독립 RNG. 단 CUDA strict determinism 전이라 최종 증거 제외 |
| `v2/artifact_verification.json` | verifier 출력 참조 | checkpoint/per-sample SHA와 threshold aggregate 독립 재계산 통과 |
| `determinism/final/sen12_gp_pilot_det_final/holdout_chimanimani_pilot.json` | `038b10f677e3559e60e6061672d4ba9bc402fd25bc0c9ceb3221175ce7946cb8` | strict CUDA final, code `478c6af5…`, development-only |
| `determinism/final/artifact_verification.json` | verifier 출력 참조 | final 3 arm checkpoint/per-sample/threshold aggregate 전부 통과 |
| `determinism/final_replay/sen12_gp_p4_det_final_replay/holdout_chimanimani_pilot.json` | `33993964f693b2a85a1124dc68916c2a754adf3ef5adce4ce40365bfcfbab54c` | final P4-only; full P4와 metric/checkpoint/tensor bitwise 일치 |

8/40-epoch JSON을 고치거나 덮어쓰지 않는다. 교정 v2는 별도 `v2/`에 저장한다.
`determinism/pre_fix_replay/`는 같은 seed P4 replay가 원 v2와 test IoU **0.122826→0.143442**로
갈린 실패 계보다. `determinism/smoke_a`와 `smoke_b`는 strict CUDA 계약 적용 뒤 checkpoint SHA,
val/test per-sample SHA, 모든 지표, 모든 tensor가 bitwise 일치(max-abs diff 0)한 복구 증거다.
40-epoch strict 결과는 `determinism/final/`에 별도 보존하며 v2를 덮어쓰지 않는다.
해석 계약은 `docs/GP_PILOT_VALIDATION_AUDIT.md`가 canonical이다.
