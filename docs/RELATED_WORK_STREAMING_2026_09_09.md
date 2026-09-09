# 관련연구 점검 — 캐시된 지구 FM 임베딩의 증분 갱신 (2026-09-09 웹 조사, 초록·검색 스니펫 기준, 전문 미확인 항목 표시)

## 질문
"저장된 위치 임베딩을 새 관측의 단일 취득 임베딩으로 학습 갱신하고, downstream 회복률과 실측 비용으로 평가"가 이미 있는가.

## EO 쪽(전부 창 전체 재인코딩, 상태 갱신 없음)
| 논문 | 연/출처 | 하는 일 | 갱신 방식 |
|---|---|---|---|
| AlphaEarth Foundations | arXiv 2507.22291, GEE 제품 | 연 단위 임베딩 필드(2017–2024), 요청 시 5일 요약 | 제품 실행마다 창 전체 재인코딩 |
| TESSERA / v2 | CVPR 2026 / arXiv 2607.03949 | 픽셀 시계열 Barlow Twins, 연 단위 | 재인코딩, 증분 메커니즘 언급 없음(초록) |
| **Tessera 시간 민감도 분석** | arXiv 2608.27175 (2026-08) | 동결 TESSERA로 창 길이 1년→1일 재계산, 작물 −39%/1개월, 시간 커버리지를 "조절 가능한 비용"으로 봄 | **재계산**, 증분 아님 — 평가 축에서 가장 가까운 선행 |
| OlmoEarth v1 / v1.2 | CVPR 2026 / arXiv 2605.20804 | v1.2는 예제당 MACs Pareto(v1 대비 5.3배 절감) | 창 전체, 비용은 예제당 인코딩만 |
| TerraFlow | arXiv 2603.12762 | TerraMind + 시계열 생성 목표, GEO-Bench-2 시계열 과업 SOTA | 시퀀스 전체 재인코딩 |
| ALISE, HighFM, LIANet | 2024–2026 | 불규칙 시계열 질의 사영 / 정지궤도 MAE / 시공간 신경장 | 전체 창 또는 재적합 |
| Recursive Bayesian Classifier | arXiv 2301.01796 | 임의 분류기를 영상당 상수 비용의 온라인 재귀 분류기로 | **영상당 상태 갱신**이지만 클래스 사후확률 위, FM 임베딩 아님 — 기계론적으로 가장 가까운 EO 선행 |
| rs-embed, CDSE Global Embeddings(EGU26) | 2026 | 요청 시 임베딩 생성 / 1.7억 영상별 임베딩 변화탐지 | 재계산 / 영상별, 융합 상태 없음 |

미발견: 새 취득을 받아 저장 임베딩을 갱신하는 순환·TTT·SSM 상태를 가진 EO FM. SSM 서베이(arXiv 2606.25329)·SITSMamba는 전체 시퀀스 모델.

## 일반 비전·시계열(상태 재사용은 활발 — 여기가 붐빔)
Deep Feature Flow(CVPR 2017, 특징 전파) · StreamMem(2025, 고정 크기 KV 메모리) · MemStream(2026, 동적 KV) · Decouple and Cache, V-Rex(스트리밍 VLM KV) · ViT³(CVPR 2026 oral, TTT 층) · TTT3R(ICLR 2026, 프레임당 폐형식 상태 갱신·비용 보고) · TiRex-2(2026, xLSTM 시계열 FM, 패치당 상수 비용) · DeepCoT(2025).

## 벤치마크
GEO-Bench-2, PANGAEA, NeuCo-Bench, "How to Embed Matters": 관측 도착 프로토콜·갱신 비용 측정 없음. 창 길이를 바꾼 것은 Tessera 민감도 논문뿐이며 재계산 방식.

## 판단
1. 우리 정의 그대로는 **미발견**. EO 쪽은 비어 있음.
2. 가장 가까운 선행: 평가 축 = Tessera 민감도(재계산), 기계론 = Recursive Bayesian(클래스 사후확률 재귀), 일반 비전 = StreamMem·TTT3R·TiRex-2(학습 상태 갱신 + 비용, EO 아님).
3. 제품 현실(AEF·TESSERA 연 단위, 5일 요약은 요청 시)이 "갱신 = 전체 재인코딩"임을 확인. 증분 대 재계산의 비용·품질 곡선은 아무도 발표하지 않음.
4. **위험**: 순환·KV·TTT 자체는 붐빔. 노벨티는 재귀 구조가 아니라 (a) 불규칙 재방문·구름 결손·고정 차원 공개 임베딩을 상태로 쓰는 EO 특수성, (b) 벤치마크 프로토콜(GEO-Bench-Stream), (c) 대조 실험이 붙은 실측으로 세워야 함.
5. 투고 전 확인: Tessera 민감도·OlmoEarth v1.2 전문에 "update/streaming" 절이 있는지, TESSERA v2 GitHub의 NRT 로드맵(미확인).
