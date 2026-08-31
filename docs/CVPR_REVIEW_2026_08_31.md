# CVPR 복기 감사 — 네팔 사이드카 이전 본선 상태 점검 (2026-08-31)

목적: 네팔 작업(M66–M85)을 사이드카로 닫고 본선(MountainShift: frozen OlmoEarth 전이)으로 복귀하며,
기존 결과를 하나씩 복기해 **논문에 쓸 수 있는 것 / 깨진 것 / 미실행**을 가름. 새 실행 없음 — 장부·문서 재독 기반.

## 1. 살아 있는 핵심 결과 (논문 뼈대)
| 주장 | 근거 | 상태 |
|---|---|---|
| frozen OlmoEarth reuse가 8지역 확증에서 region-macro **.2722 vs P2 .1966 (+.0756)**, per-region 6/8 승 | M65, recipe v2 clean 절차(M58) | **생존.** 단 M61 강등: "사전학습 > scratch"는 viability이지 우월성 아님 |
| 오경보(빈 타일 FP)가 raw 대비 5~21배 낮음 — 공간 문맥 가설 | M59·M63·M64 | 생존. C1(Presto)이 이 가설의 직접 시험 |
| 지역 이득 이질성 실재(thrissur +.127 vs hiroshima +.062; indonesia 패배, itogon all-seed 실패; M60 val↔test 해리) | M60·M63·M65 | 생존 — D·E(적응/라우팅) 동기부여로 사용 |
| 확증 절차 자체(위반 자기신고 M57 → 재설계 M58 → clean 3연승) | M57–M64 | 생존 — 논문 부록의 자산 |

## 2. 복기에서 확인한 문제·정정 필요
1. **C1(Presto) 상태 서술 불일치**: M62는 "정규화 미확정"이라 했으나 EXPERIMENT_C 문서 2026-08-28 갱신에서
   공식 정규화(shift 0, /1e4)·commit 고정(11e207a…)·month 2-D 텐서까지 닫힘. **M62의 '미해결 4건' 중 1·2번은 해소됨** —
   장부에 갱신 주석 필요. 남은 것: full 128×128 cache 추출기, throughput smoke, cache seal, matched decoder 3-seed.
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
9. **서버 상태**: GPU1 0 MiB 가용, GPU0 타 프로젝트 62.6 GB 점유(규칙 4b 유지). Sen12·Presto vendoring·S1asc까지 서버에 있음.
   nepal 관련 임시 산출물이 artifacts를 크게 만들었으니 본선 실행 전 디스크 확인 권장.

## 3. CVPR까지의 실행 순서 (CRITICAL_PATH 재확인, 변경 없음)
1. **C1 full**: Presto 128×128 cache 추출기 + smoke + seal → 동일 decoder 3-seed, 8지역 matched control. (게이트 1c)
2. **label-budget 축**(1/5/10/100%) — C1과 같은 러닝에 얹기 (EXPERIMENT_C 설계 그대로, 서브셋 seed 20260827).
3. **C2(Clay v1.5/v1.0)** — RQ3 release-pair 겸용.
4. **T-m 한국 untouched transfer** (M35 정정 반영한 물질화 재검 후) — untouched OLMo-vs-GeoFM 주장은 여기서만.
5. R-event 재설계는 시간 남을 때만; 아니면 음성 결과로 한 문단.
6. 집필: 뼈대 = §1 표 + 이질성 + 절차 부록. 마감·템플릿은 CVPR 2027 공지 확인 필요(미검증 — 예년 기준 11월 중순 제출).

## 4. 즉시 할 일 (다음 세션)
- [ ] M62에 "정규화·commit 해소(2026-08-28 계약 감사)" 갱신 주석 추가
- [ ] Presto full-cache 추출기 작성 → GPU1 smoke (64타일 처리율·메모리)
- [ ] 디스크·artifacts 용량 점검, nepal 임시 npz 정리 목록 작성(삭제는 사용자 확인 후)
