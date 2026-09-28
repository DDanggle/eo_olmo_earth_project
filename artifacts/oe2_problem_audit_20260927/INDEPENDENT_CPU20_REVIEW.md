# CPU20 독립 결과 감사

저장 영수증 및 집계 범위에서 `consistent=true`, `valid_within_scope=true`다. 5개 source snapshot이 launch pin과 로컬 원본에 모두 일치하고, 보호 대상 4개 코드의 실행 전후 size/mtime_ns/SHA가 같다. 원 OE1 train 512패치에서 hash순·MGRS당1개로 20개를 선택하는 규칙을 재구성했다. 원 metadata/질문, raw NPY/array SHA 영수증, 20개 native shape, 5개 train16/held-out4 fold와 MGRS 비중복, 장면당 1회 held-out, 100개 유한 scene metrics를 검산했다. 저장 평균·중앙값 모두 재계산과 일치한다.

| 입력 표현 | 장면평균 MAE | full 대비 MAE 증가 | full 우세 장면 |
|---|---:|---:|---:|
| full: 30×30 spatial, bandset 평균 | 0.12450431 | 기준 | — |
| 4×4 pooling: 16개, nearest 복원 | 0.15526964 | +24.71% | 16/20 |
| 8×8 pooling: 64개, nearest 복원 | 0.14578301 | +17.09% | 16/20 |
| 좌표만 | 0.27245118 | +118.83% | 20/20 |
| fold train의 목표 평균 | 0.27214136 | +118.58% | 20/20 |

16개 입력 대비 full의 상대 MAE 감소는 **19.81%**다. 분모가 다르므로 이를 위의 **16개 입력의 full 대비 증가 24.71%**와 혼동하면 안 된다. 64개 대비 full 감소는 14.60%다. full과 16개의 평균 대응 MAE 차이는 0.03076532이며, 표는 같은 20개 장면을 비교했다. full은 모든 장면에서 pooling보다 좋은 것은 아니다.

full의 평균 within-scene R²는 -36.20이지만 중앙값은 +0.627이고 14/20장면에서 양수다. `S2B_MSIL2A_20170718T115359_N9999_R023_T29UPB_08_36`의 R²=-717.914가 평균을 크게 끌어내린다. 이 장면은 목표 표준편차가 약 0.02346인데 bias/MAE가 +0.62677이다. 각 장면 자체의 목표 평균을 기준으로 하는 R²라, 장면 간 편향과 작은 장면 내 변동이 결합하면 큰 음수가 가능하다. 이는 오류를 숨길 이유도, 모든 장면에서 신호가 없다고 결론낼 이유도 아니다. 모든 장면은 그대로 유지했고, 목표 분산·평균과 R² 관계는 `train_mean`의 MSE/bias 및 train-only intercept로 독립 대수 검산했다.

**감사 범위:** 이 복사본에는 raw NPY, native feature archive, 학습된 ridge 계수, 셀별 예측이 없다. 따라서 배열 SHA/shape와 encoder 불변은 실행 영수증을 원 pin과 대조한 것이며, 원 픽셀에서 목표와 MSE를 다시 계산하거나 모델을 재실행하지 않았다. 원본 array 검산·학습 재현을 새로 했다는 표현은 부정확하다.

**과학적 범위:** 목표는 raw DN의 B08/B04 비율을 native cell 안에서 평균한 센서 신호다. calibration/건강 상태/의미 segmentation/VLM 판독 성능은 아니다. 원 native shape는 `[1,30,30,1,3,192]`지만 모든 주요 arm에서 3개 bandset을 똑같이 평균했다. full은 이 평균 뒤의 900개 공간 토큰이며 bandset 평균의 정보 손실은 조사하지 않았다. Encoder는 고정했고, ridge readout은 각 fold train 16장면으로 실제 적합했다. 따라서 실행 설명의 “no training”은 **encoder/VLM 학습 없음**으로 한정해야 한다. nearest 복원과 점별 선형 decoder에서 나타난 차이를 되돌릴 수 없는 정보 손실이나 새로운 사전학습 목적의 성공으로 해석할 수 없다. 20개 MGRS는 다르지만 지리 buffer가 없고, fold의 train 집합도 겹친다. 이것은 고정된 개발 진단의 기술적 효과 크기다.

재현 스크립트: `audit_cpu20_results.py` (production code import 없음). 전체 결과: `independent_probe_audit.json`. 기존 output이 있으면 덮어쓰지 않으므로 재실행 시 새 `--out` 경로를 지정한다.
