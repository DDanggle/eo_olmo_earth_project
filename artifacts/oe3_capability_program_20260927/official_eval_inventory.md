# OE3 공식 EO 평가 준비물 — 2026-09-27

이 문서는 **독립 EO 기준선을 실행하기 위한 준비물**이다. 모델 성능 결과나 실행 완료 보고가 아니다. 공식 clone `0497dfbb6711ded4e6bf10cf089fc1e4d58c186b`를 읽고 public CLI parser를 추출해 인수만 검증했다. 설치·다운로드·서버 실행은 하지 않았다.

## 우선 두 과제

| task id | 자료/모달리티 | 평가 | 데이터 환경변수 |
|---|---|---|---|
| `sen1floods11` | 준비된 64×64 S1 홍수 마스크, 2 classes | frozen encoder + linear dense head, mIoU | `FLOODS_DIR` |
| `m_cashew_plant` | GeoBench 256×256 S2 토지피복 마스크, 7 classes | frozen encoder + linear dense head, mIoU | `GEOBENCH_DIR` |

공통 개발 설정은 patch 4, 공간 격자 유지, time/bandset mean pooling, 50 epochs, head LR 0.1, 공식 split이다. 언어는 사용하지 않는다. 이 평가는 연구 전체의 최종 범위가 아니라, 후속 큰 학습이 기존 EO 능력을 어떻게 바꾸는지 확인할 기준선이다.

`FLOODS_DIR`에는 `flood_train_data.pt`, `flood_valid_data.pt`, `flood_test_data.pt`가 필요하다. `GEOBENCH_DIR/segmentation_v1.0/m-cashew-plant`에는 GeoBench task specification과 sample/partition 파일이 필요하다. 현재 존재 여부는 이 작업에서 확인하지 않았다.

## 실행 전에 해결할 두 호환성 문제

1. **가중치 형식:** 서버 개발 모델은 기존 기록상 `config.json + weights.pth`인 HF 공개 번들이다. 공식 `full_eval_sweep --checkpoint_path`는 distributed checkpoint를 `trainer.load_path`로 전달한다. HF 폴더를 이 옵션에 그대로 넘길 수 있다고 확인되지 않았다. 공식 `model_loader.load_model_from_path`를 호출하는 작은 config bridge 또는 검증된 distributed export가 필요하다. 변환 전후 encoder tensor/hash와 고정 입력 출력 동등성을 확인해야 한다.
2. **normalization 기본값:** `m_cashew_plant` registry는 dataset 통계를 사용하지만 `full_eval_sweep --defaults_only`는 모든 과제에 pretrained 통계를 강제한다. 첨부 JSON의 command는 후자이며 이 차이를 명시했다. 비교 조건마다 통계를 바꾸지 말고 resolved config를 고정한다.

## CLI와 실행 상태

정확한 인수와 명령 템플릿은 `official_eval_inventory.json`에 있다. `--task-names`, `--defaults_only`, `--select_best_val`, `--dry_run`을 현재 source parser로 검증했다. checkpoint/data 경로는 명시적 placeholder다.

`--dry_run`은 내부에서 `dry_run_evaluate` subprocess를 호출하므로 공식 학습 의존성이 필요하다. 로컬 nano 환경에는 `olmo_core`, `geobench`, `omegaconf`, `rslearn`, `h5py`가 없어 full dry-run은 수행하지 않았다. 또한 공식 all_evals의 기본값은 test 실행이 켜져 있으므로 개발 명령에는 `run_on_test=False`를 명시했다.

## 비교 범위

v1-Tiny와 `scripts/official/v1/tiny.py`로 개발 환경을 고정한다. 최신 모델·Base 규모의 학습은 후속이다. 기존 BEN S2 데이터는 디버깅 참고로 보존한다. 원 공식 목적 추가 학습 대조군에는 정합된 파생 지도 target이 필요하므로 BEN S2만으로 대체하지 않는다.

두 dataset wrapper 모두 고정 날짜를 쓰므로 이 기준선으로 시간 변화 능력을 주장할 수 없다. Sen1Floods11의 준비된 tile filtering과 nonfinite 제거도 공식 조건에 포함되며 실제 사용 split 크기를 나중에 기록해야 한다.
