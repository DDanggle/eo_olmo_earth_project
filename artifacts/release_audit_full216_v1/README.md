# OlmoEarth v1→v1.2 제주 full-216 릴리스 감사

## 약점부터

이 결과에는 사람 정답 라벨이 **0개**이고, 표본은 제주 한 개의 결정적 9×6 grid뿐이다.
따라서 정확도 향상, negative transfer, 구름 강건성, 공공데이터 결합 효과, 제주·한국 일반화,
변화탐지 성능은 측정하지 않았다. 확인한 것은 같은 frozen legacy 입력에서 모델 릴리스만
v1→v1.2로 바꿀 때 표현 좌표와 검색 identity가 얼마나 유지되는가뿐이다.

결론은 **직접 cross-version representation-identity gate 실패**다. 216개 패널의 pooled 관계
구조는 상당히 보존되지만, 동일 공간 token의 좌표는 직접 호환되지 않고 사전 등록한 선형
bridge로도 95% gate에 미달했다. 따라서 downstream task 평가 전에는 기존 v1 cache와 v1.2
query의 운영 혼용이 검증됐다고 볼 수 없다. 이번 결과를 보고 새 bridge를 골라 같은 sealed
test에 다시 평가해서도 안 된다.

## 동결된 실험 계약

- 입력: 제주 54개 위치 × 2023–2026 = 216 site-years, 12 Sentinel-2 periods
- 입력 폐쇄성: 5,616파일, 56,684,540,847 bytes, SHA-256 전수 고정
- split: calibration 30위치/120건, embargo 6/24, sealed test 16/64,
  이전 smoke에 노출된 disclosed-audit 2/8
- bridge fit과 hyperparameter 선택: calibration만 사용
- headline 평가: sealed test만 사용; embargo와 disclosed-audit은 fit·headline에서 제외
- sealed analyzer: output raster를 읽기 전 one-time `PREANALYSIS_LOCK`을 생성하고 한 번만 실행;
  같은 sealed split의 반복 실행을 새 독립 증거로 세지 않음
- 실행: 같은 입력·timestamp track·batch 8·workers 4, GPU0 UUID
  `GPU-58459350-e802-b3ee-03be-fd3451eda731`

## 실행 결과

| 릴리스 | 출력 | wall time | 처리량 | GPU0 util p50/p90 | peak VRAM |
|---|---:|---:|---:|---:|---:|
| v1 | 216/216, 52,833,394,208 bytes | 3,756.12초 | 55.26 crops/s | 88% / 89% | 4,291 MiB |
| v1.2 | 216/216, 52,758,021,087 bytes | 2,250.12초 | 92.25 crops/s | 72% / 77% | 2,719 MiB |

두 릴리스 모두 768×256×256, float32, EPSG:32652이며 모든 값이 finite이고 65,536/65,536
token이 usable·nonzero였다. 총 432개 출력은 105,591,415,295 bytes다. v1.2는 이 실행의
end-to-end 처리량 기준 v1보다 약 1.67배 빨랐다. 실행 계약과 선택 장치는 GPU0 UUID에 고정됐고,
GPU1은 이 추론의 선택 GPU가 아니었다.

## 봉인 평가 결과

R@1은 sealed 16개 위치의 4,096 query token을 같은 릴리스 16,384-token gallery에서 먼저
native sanity check한 뒤, 반대 릴리스 gallery로 검색한 값이다.

| bridge | v1.2 query→v1 gallery R@1 | v1 query→v1.2 gallery R@1 | 판정 |
|---|---:|---:|---|
| native same-release ceiling | 1.0000 | 1.0000 | sanity 통과; exact/1e-6 near tie 0 |
| identity, no bridge | 0.0000 | 0.0000 | 실패 |
| calibration mean shift | 0.00024 | 0.0000 | 실패 |
| translated orthogonal Procrustes | 0.49097 | 0.43604 | 실패 |
| affine ridge | **0.69727** | **0.60889** | 최선이지만 0.95 gate 실패 |

affine ridge의 위치 하나씩 제외한 최소 R@1은 각각 0.67760과 0.58464였다. v1→v1.2의
1e-6 tie-tolerance pessimistic R@1은 0.60791이다. 사전 등록한 4방법×2방향 8개 gate는 모두
실패했고 `full_cache_compatibility_promoted=false`다. task 성능 gate는 라벨이 없어 `null`이다.

동시에 sealed 64건의 pooled site-year geometry는 CKA 0.97857, pairwise-distance Spearman 0.95251로
높았지만 동일 token raw cosine 평균은 −0.00860이었다. 이는 모순이 아니라 **패널 관계 구조는
남고 좌표계 identity는 깨진 structural release shift**다. 이는 세계 일반화를 뜻하지 않는다.
높은 CKA만으로 backward-compatible cache를 주장할 수 없다.

## 허용되는 주장

- 216 frozen exact inputs에서의 split별 기술적 release drift
- calibration-only bridge와 sealed representation-identity retrieval
- sealed cross-site neighbor 및 consecutive frozen manifest-window contrast의 순위 연속성

마지막 항목은 변화탐지가 아니다. 특히 2025→2026은 rolling-2026 manifest라 prospective
annual event로 해석하지 않는다.

## 금지되는 주장

- task 정확도 또는 정확도 개선, negative transfer 및 그 감소
- 구름 강건성, 공공데이터 효과, 제주·한국 모집단 일반화
- 의미적 검색 정확도, 인과·시간 변화탐지
- 모델-native·운영 cache backward compatibility
- 이번 sealed 결과를 본 뒤 고른 nonlinear bridge의 같은 split 재평가

## 증거 파일

- `analysis_strict1/analysis_summary.json`: 결과·gate·허용/금지 주장 SSOT
- `analysis_strict1/per_window_metrics.csv`: 216 site-year 기술통계
- `analysis_strict1/sealed_test_per_cluster_metrics.csv`: 16 sealed 위치×방법×방향 지표
- `paired_evidence_strict1/evidence_summary.json`: 5,616입력·432출력 폐쇄성
- `paired_evidence_strict1/paired_outputs.jsonl`: 216개 v1/v1.2 output pair
- `v1/result/`, `v1_2/result/`: 실행 summary, telemetry, 완료·사후 검증 marker
- `input_freeze/`, `spatial_split/`, `promotion_b008w04_strict1/`: exact input, split,
  batch/worker promotion 계약

핵심 SHA-256:

- paired evidence: `9931cddfc16de5cf10657f2138755166549f929ca0fa85c611cc5b72cfa1e2f1`
- paired output manifest: `33b2e8c743f113610476c42d1a897b973bdefe3c620a6106809be2a03d25d8ba`
- sealed analysis summary: `56030ea0dd95e94b76a45b45637016c01d928b2c076cc4ada941ed640f7c185d`
- preanalysis lock: `537ba9239509279f134225ccd8ecfb7ae1d135df5c0e3d0f39ed74c9ad4d3d01`

## 다음 연구 gate

이번 sealed 결과를 이미 보았으므로 nonlinear bridge·distillation은 **새 untouched geographic
split**을 먼저 hash-freeze한 뒤에만 평가한다. 실제 전이효과·구름·공공데이터 개선 주장은
독립 이중판독 라벨과 prospective 시공간 split이 생긴 다음, scratch·task-specific baseline·다른
GeoFM을 동일 입력/decoder/label budget으로 비교할 때만 연다.
