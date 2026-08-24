# OlmoEarth v1↔v1.2 제주 P0 실행 런북

갱신일: 2026-08-23

## 먼저 확인할 약점

이 P0는 **정확도·한국 전이효과 실험이 아니다.** 독립 인간 정답이 없고 SCL BestClear 제주 전수
입력도 아직 없으므로, 지금 허용되는 주장은 동일 입력에서 OlmoEarth 릴리스만 바꿨을 때 표현과
후보 순위가 얼마나 움직이는지에 대한 paired representation audit뿐이다.

현재 서버·인증·영구 저장소는 정상이다. 2026-08-23에 GPU0이 비고 GPU1에서만
`knee-proj` 학습이 계속되는 것을 확인한 뒤, GPU0 전용으로 8표본×2릴리스=16개 출력을 완주했다.
GPU1 프로세스는 중단·감속하지 않았다.

## 동결된 계약

| 축 | 고정값 |
|---|---|
| 모집단 | 제주 54 spatial windows × 2023–2026 = 216 site-years |
| 사건 | 같은 window의 인접연도 162 pairs |
| smoke | 라벨·행정근거를 보지 않고 2023/2026 품질 proxy 양끝에서 고른 8 site-years |
| 입력 | 12개 Sentinel-2 layer, 208 tensor/metadata files, 동일 audit view를 두 릴리스가 공유 |
| 시간 | P0 primary는 양쪽 모두 `use_legacy_timestamps: true` |
| 출력 | v1 8개 + v1.2 8개 |
| 금지 | accuracy, negative transfer 감소, 원인규명, 한국 일반화, FL privacy 주장 |

고정 식별자:

- exact-input manifest SHA-256:
  `bc149353a2fe0f1e9e0dd3b48a1445bba4c08ec4d08461c8ce9246f156039218`
- checkpoint manifest SHA-256:
  `f325e3f6d306fc26be98ec765eff23ad4b3bffa729ad4f7d0abfd7c9bf0f7ce7`
- v1 commit: `93589e2dee5b5c95a660d1e9365bc017ea7f35d6`
- v1 weights SHA-256:
  `551c1cc53337c6faaddead88071d7ebd2bd53ec271600fa6f0ee0a518c8b6e11`
- v1.2 commit: `581aa9baaa7aed4348c0903617eb92ee9f89e2ec`
- v1.2 weights SHA-256:
  `57f7b66faf206db1307670673839e639d3a19c305f6ad968c62392ad3e88deec`

## 1. 상태 확인

프로젝트 루트에서만 실행한다.

```bash
./bin/nx status
./bin/nx sh 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
```

선택할 GPU의 출력이 한 줄이라도 있으면 `--execute`를 붙이지 않는다. 다른 GPU의 작업은 실행
거부 사유가 아니다. `run_olmo_release_smoke.py`는 index→UUID를 먼저 확인해 선택한 GPU의 active
process만 검사하며, 그 GPU가 사용 중이면 실패하도록 만들었다.

## 2. 무변경 preflight

```bash
./bin/nx sh 'cd /home/work/data/olmoearth && env -u PYTHONPATH .venv-master/bin/python code/run_olmo_release_smoke.py --checkpoint-manifest release_audit_p0/checkpoints.json --exact-inputs release_audit_p0/smoke_inputs_exact.json --dataset-root release_audit_p0/smoke_dataset --config-dir config --rslearn .venv-master/bin/rslearn --output-dir release_audit_p0/results --preflight-output release_audit_p0/preflight.json'
```

`records=8`, `release_runs=2`, 두 manifest SHA 일치, `gpu_processes=[]`, `ready=true`를 모두 확인한다.

## 3. GPU가 비었을 때만 분리 실행

```bash
./bin/nx sh 'cd /home/work/data/olmoearth && mkdir -p release_audit_p0/results && setsid nohup env -u PYTHONPATH .venv-master/bin/python code/run_olmo_release_smoke.py --checkpoint-manifest release_audit_p0/checkpoints.json --exact-inputs release_audit_p0/smoke_inputs_exact.json --dataset-root release_audit_p0/smoke_dataset --config-dir config --rslearn .venv-master/bin/rslearn --output-dir release_audit_p0/results --preflight-output release_audit_p0/preflight.json --gpu 0 --execute > release_audit_p0/results/launcher.log 2>&1 < /dev/null & echo $!'
```

진행 확인:

```bash
./bin/nx sh 'tail -n 80 /home/work/data/olmoearth/release_audit_p0/results/launcher.log'
./bin/nx sh 'test -f /home/work/data/olmoearth/release_audit_p0/results/COMPLETE.json && echo COMPLETE || echo RUNNING_OR_FAILED'
```

## 4. 완료 판정

다음 조건이 모두 맞아야 완료다.

1. `results/COMPLETE.json`이 존재하고 `run_summary.json` SHA가 일치한다.
2. 각 릴리스에 8개 output GeoTIFF가 있고 모두 SHA-256 inventory에 들어 있다.
3. 두 실행이 동일한 exact-input manifest를 사용했다.
4. v1과 v1.2의 config 차이는 모델 ID/path/output layer 외에는 없다.
5. CKA·이웃 보존·거리 순위 상관만 기술하고 정확도나 전이 이득으로 부르지 않는다.

완료 뒤 지표 계산:

```bash
./bin/nx sh 'cd /home/work/data/olmoearth && env -u PYTHONPATH .venv-master/bin/python code/analyze_olmo_release_smoke.py --run-summary release_audit_p0/results/run_summary.json --complete-marker release_audit_p0/results/COMPLETE.json --exact-inputs release_audit_p0/smoke_inputs_exact.json --output-dir release_audit_p0/results/analysis'
```

분석기는 output size/SHA·mtime과 COMPLETE marker, 양쪽 8개 `sample_id`·input bundle·spatial
cluster, CRS·transform·bounds·shape·valid mask를 먼저 검사한다. 보고 지표는 spatial linear CKA와
row-normalized sensitivity, toroidal-shift null, pooled-site-year의 k=1/2 이웃 보존과 chance correction,
pairwise Euclidean-distance Spearman, 7개 spatial cluster leave-one-out 범위다. 8개 표본에는
alignment를 학습·평가할 분리가 없으므로 raw cross-version cosine·Procrustes·p-value를 계산하지
않는다.

## 5. 2026-08-23 실측 결과

- GPU0 순차 실행: v1 202.633초, v1.2 196.830초, 각 8개 output, 총 16/16
- output inventory: 각 파일 SHA-256·mtime·sample/input/spatial-cluster identity 검증 통과
- pooled site-year geometry: linear CKA 0.981, pairwise Euclidean-distance Spearman 0.889
- 이웃 보존: top-1 0.75(무작위 기대 0.143, chance-corrected 0.708), top-2 1.00
- 7-cluster leave-one-out: Spearman 0.838–0.957, top-1 0.667–0.857, top-2 0.917–1.00
- window 내부 spatial CKA: 평균 0.427, 범위 0.133–0.828; toroidal-shift null 초과분 평균 0.247

해석은 두 층으로 제한한다. 이 8개에서는 site-year 간 전역 거리·이웃 구조가 상당 부분
유지됐지만, 같은 창 내부의 국소 공간 표현 이동은 크고 표본별로 불균일했다. 정답 라벨,
BestClear 입력축, held-out gallery가 없으므로 더 나은 정확도·구름 강건성·한국 일반화·기존
cache 무재계산 가능성을 뜻하지 않는다.

재현 산출물은 `artifacts/release_audit_p0/results/`의 `run_summary.json`, 완료 marker, 모델별 로그,
`analysis/analysis_summary.json`, `per_window_metrics.csv`에 있다. 16개 대형 GeoTIFF는 서버의
`release_audit_p0/smoke_dataset/`에 두고 로컬에는 SHA inventory만 보존한다.

## 6. 증거 번들 폐쇄

로컬에 과거 `ready=false` preflight가 남아 있던 불일치를 발견해, launcher 첫 JSON과 SHA
`b63c8c60e7314fb77be579657a5a0a5c5e49bcb277c4ee69676fea249f1a2a2b`로 같은 실제 실행본으로
교체했다. 다음 명령은 서버 원본이 모두 있어야 성공한다.

```bash
./bin/nx sh 'cd /home/work/data/olmoearth && env -u PYTHONPATH .venv-master/bin/python code/verify_olmo_release_bundle.py --project-root /home/work/data/olmoearth --bundle-root /home/work/data/olmoearth --require-raw --output release_audit_p0/results/verification_v1/verification.json --complete-output release_audit_p0/results/verification_v1/VERIFICATION_COMPLETE.json'
```

실측 결과는 758/758 checks, raw 228파일·7,851,565,383 bytes, missing/failure 0,
`FULL_EVIDENCE_VERIFIED`다. raw 228개는 exact input 208, checkpoint 4, output GeoTIFF 16이다.
같은 analyzer를 새 디렉터리에서 다시 실행한 결과 summary JSON SHA
`7bfeac8d22ef71e0d5f9de3db378068e8f42221bcdf768ae00d334deca47314c`와 CSV SHA
`8a25a7cfa727f87b7e2cae8bde715171b17aa64d8a9716337a7eea1c2ddf8e36`가 기존과
byte-identical했다. 이 폐쇄는 실행·기술통계의 무결성을 증명하지만 금지 주장을 승격하지 않는다.

## 다음 승격 순서

1. 동결된 8 site-year stress set에서 BestClear 12기간·96쌍의 선택 trace·SCL/reflectance SHA·
   grid/mask·RGB·2건 replay 결정성 gate를 먼저 통과한다. 현재 1-window×4-period 코드는 이
   계약을 충족하지 않으므로 재사용하지 않는다.
2. 위 gate 뒤에만 54×4 legacy 입력의 full content hash·216×2 release audit와 SCL BestClear
   216 site-years를 검토해 `input recipe × release` 2×2를 만든다.
3. sealed 확률표본 300건과 별도 active/train 300건을 만들고, 최소 120건을 이중 판독한다.
4. 공통 입력 계약을 동결한 뒤에만 scratch·generic vision·TerraMind/Prithvi 점수표를 연다.
5. 실제 원자료 반출 불가 기관 3곳이 확보되기 전에는 federated learning을 구현하지 않는다.
