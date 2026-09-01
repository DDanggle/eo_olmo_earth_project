# CVPR 복기 감사 — 네팔 사이드카 이전 본선 상태 점검 (2026-08-31)

목적: 네팔 작업(M66–M85)을 사이드카로 닫고 본선(MountainShift: frozen OlmoEarth 전이)으로 복귀하며,
기존 결과를 하나씩 복기해 **논문에 쓸 수 있는 것 / 깨진 것 / 미실행**을 가름. 새 실행 없음 — 장부·문서 재독 기반.

## 1. 살아 있는 핵심 결과 (논문 뼈대)
| 주장 | 근거 | 상태 |
|---|---|---|
| frozen OlmoEarth reuse가 8지역 확증에서 region-macro **.2722 vs P2 .1966 (+.0756)**, per-region 6/8 승 | M65, recipe v2 clean 절차(M58) | **생존.** 단 M61 강등: "사전학습 > scratch"는 viability이지 우월성 아님 |
| 빈 타일 FP가 P4에서 더 낮은 경향 | M59·M63·M64, MS-86 | **8지역 중 7지역 생존**, 지역 중앙 P2/P4 5.02×. 단 threshold .5 기술통계이며 FP-budget matching 전에는 운영·인과 주장 금지 |
| Presto C1a common-grid | MS-87 | `.1092`로 P4/P2에 8/8 패배. generic frozen effect 한 모델 반증; off-domain/native-grid/retrospective 경계 유지 |
| 지역 이득 이질성 실재(thrissur +.127 vs hiroshima +.062; indonesia 패배, itogon all-seed 실패; M60 val↔test 해리) | M60·M63·M65 | 생존 — D·E(적응/라우팅) 동기부여로 사용 |
| 확증 절차 자체(위반 자기신고 M57 → 재설계 M58 → clean 3연승) | M57–M64 | 생존 — 논문 부록의 자산 |

### 2026-08-31 2차 claim 감사

- `P4`는 새 2단 architecture가 아니라 **frozen cache + small decoder arm**이다. confirmatory
  procedure는 provenance protocol이므로 방법 기여로 올리지 않는다.
- `P2`는 우리 S12q·LOCO 계약의 matched official-architecture baseline이지 supervised SOTA가 아니다.
- 세 초기 지역의 오경보 5~21배 관찰은 MS-86에서 전 8지역으로 다시 집계했다. P4가 7/8에서
  낮고 P2/P4 중앙값은 5.02×지만 threshold 0.5 결과이므로 FP-budget-matched 감사 전에는
  공간 문맥의 원인 증거가 아니라 가설이다.
- Korea의 polygon 면적 유사성만으로 annotation match 또는 `T-m−T-x=annotation effect`를
  주장할 수 없다. ontology·time·provenance를 통과하지 못하면 joint-shift stress test로 낮춘다.
- 상세 claim과 queue는 `PAPER_NARRATIVE_2026_08_31.md`가 최종 근거다.

## 2. 복기에서 확인한 문제·정정 필요
1. **C1(Presto) 완료 범위**: 정규화·commit·month·WGS84·6,834 cache·C1a matched decoder 3-seed가
   MS-87에서 닫혔다. 남은 것은 native 128² C1b sensitivity와 label-budget이다.
2. **C1은 이제 untouched가 아님**: 8지역 결과 개봉 후 실행이므로 matched retrospective control로만 서술 가능.
   untouched OLMo-vs-GeoFM은 한국 cohort가 맡는다(문서에 이미 명시 — 논문 서술도 이 구분 유지).
3. **네팔 사이드카의 M79가 C1을 선점하지 않음**: M79는 Δz(비지도 변화) 과제에서 Presto 하한 비교(6/7 OLMo 우위)였고,
   C1은 S12q 지도학습 분할 + 동일 decoder다. 논문에서 혼동 금지. 단 M79의 진단(픽셀 시계열 Δ가 계절 성분에 지배)은
   C1의 사전등록 예측 1("Presto < OLMo, > scratch")과 정합적 참고로만 인용.
4. **R-event(검색) 재실행 대기**: 기존 AP@100 철회, P@10 .538 > raw .432이나 사전 2×-base 게이트 실패(M17-2).
   shared-cache 두 번째 task 주장은 현재 **못 씀**. 재설계 없이는 논문에서 빼거나 음성으로만.
5. **비용 서사**: M38 — 벽시계는 오염, FLOPs 재계산으로 손익분기 생존하되 baseline 의존. 논문의 비용 표는 FLOPs 기준만.
6. **M39 정정 유지**: OLMo wrapper는 월 해상도 양자화 — "timestamp 비대칭" 문제는 존재하지 않음. 네팔 M79 v1(월 오류 폐기)도
   같은 축의 교훈 — 시간 부호화 서술 시 한 문단으로 정리 가능.
7. **개발기 잔재 주의**: M23("전 지표 우위")·M30(95% 게이트 실패)·M43→M47 정정 사슬은 확정표에서 제외된 상태 유지.
   M52는 M54·M55로 강등된 서술만 인용.
8. **한국 자산**: M9(공식 split 누수)·M10(13 cluster 동결)·M29/M35(12밴드 물질화, 24.6% 격자 밖 정정) — T-m(한국 전이) 실행 전
   AI-Hub 물질화 재검(M35의 실패 타일 제외 목록)이 선행 조건.
9. **서버 상태**: 2026-09-01 GPU0/1 memory 0 MiB. 규칙 4b에 따라 GPU0은 사용하지 않고 GPU1만 쓴다.
   nepal 관련 임시 산출물이 artifacts를 크게 만들었으니 본선 실행 전 디스크 확인 권장.

## 3. CVPR까지의 실행 순서 (2차 claim 감사 반영)
0. **완료 — MS-86 CPU audit**: P4 empty-FP 7/8·중앙 5.02×, P2/P4 tile-oracle
   region-macro +.02375·5/8 ≥.02. method 필요조건만 통과.
1. **완료 — MS-87 C1a**: Presto cache seal + common-grid 8지역×3seed.
2. **C1b native-grid**: pooling confound를 분리하되 product sensitivity로 보고.
3. **naive fusion + GeoContextGate**: source-region method를 Korea 개봉 전에 승격/폐기.
4. **label-budget 축**(1/5/10/100%) — nested region/class subset, 최소 3 subset seed.
5. **한국 sealed external transfer** (M35 정정 + ontology/time/provenance 감사 후) — test first-look는 여기서만.
6. **C2 Clay release pair**는 위 본선이 닫힌 뒤 breadth로만 검토.

## 4. 즉시 할 일 (다음 세션)
- [x] M62 정규화·commit·full cache·C1a 상태 갱신
- [x] `audit_confirmatory_mechanism.py`를 72개 test JSONL에 실행해 MS-86 FP/fusion screen 고정
- [x] Presto full-cache + C1a common-grid(MS-87)
- [ ] C1b native-grid runner snapshot·OUTROOT preflight 후 GPU1 실행
- [ ] 디스크·artifacts 용량 점검, nepal 임시 npz 정리 목록 작성(삭제는 사용자 확인 후)
