# OlmoEarth v1 논문 노트 (arXiv 2511.13655, 2025-11)

정독일: 2026-08-14. 카드 요약은 STUDY.md #6~#8, 여기는 상세 노트.

> **범위 경고:** 아래 모델 크기·학습비용·평가 수치는 OlmoEarth **v1** 전용이다.
> v1.1/v1.2 수치와 합치지 않는다. 후속 릴리스·전이·shift·선택적 추론 문헌은
> `PAPER_READING_LIST.md`에서 관리한다.

## 한 줄 요약
ViT encoder-decoder + **Latent MIM Lite**(동결 랜덤 프로젝션 타깃)로 잠재공간 마스크 모델링의
붕괴 문제를 풀고, 지도(맵)/자기지도(관측)를 단일 loss로 통일한 EO 파운데이션 모델.
기여의 무게는 아키텍처가 아니라 **학습 안정성**에 있다.

## 1. 아키텍처
- ViT encoder-decoder. 입력 = 정렬된 멀티모달 이미지 시계열.
- FlexiViT식 가변 패치 임베딩 — 단, pseudo-inverse 대신 "프로젝션 고정 + 입력 리사이즈"
  ("It's probably basically equivalent"라고 논문에 명시).
- 토큰 = 2D sincos 공간 + 사인 시간 + 학습형 모달리티 임베딩. full self-attention이
  공간·시간·모달리티를 모두 가로지름.
- 사이즈: Nano(4/128/8h, 1.4M), Tiny(12/192/3h, 6.2M), Base(12/768/12h, 90M),
  Large(24/1024/16h, 300M). 디코더는 공통 depth 4 (인코딩이 일을 다 하게).
- 디코더: <MASK> 토큰 + 위치/시간/모달리티 임베딩, 인코더 출력에 cross-attend, latent 예측.
- 학습 시 랜덤화: 패치 크기 1~8, 크롭 1~12토큰, 타임스텝 3~12. 총 ~1,000억 토큰.

## 2. Latent MIM Lite (§2.4)
- 문제: MAE(픽셀 복원)=안정하나 표현 얕음 / Latent MIM·I-JEPA(잠재 복원)=좋으나 붕괴.
- 해법: 타깃 인코더를 **랜덤 초기화 후 영원히 동결된 선형 프로젝션**으로 대체.
  타깃 불변 → 붕괴 원천 차단. 랜덤 프로젝션의 특징 보존은 JL 계열 근거.
- 맵(WorldCover 등 라벨)도 같은 동결 프로젝션 통과 → 지도/자기지도 단일 loss.
  맵은 **decode-only** (절대 인코딩 안 함 — 추론은 관측만 쓰므로 train-infer 정합).
  Galileo/TerraMind는 맵을 인코더 입력으로도 씀 — OlmoEarth와의 차별점.

## 3. 마스킹 (§2.3) — 밴드셋 단위
- 랜덤 마스킹은 EO에서 너무 쉬움(시공간·모달리티 이웃에 정답 존재 → 90% 마스킹 강요).
- 밴드셋(원본 해상도별 밴드 그룹; S2=3개, Landsat=2개)마다
  {미선택 / encode-only / decode-only / encode+decode} 배정.
  → "다른 밴드셋의 부분 관측으로 빠진 밴드셋 통째 복원"으로 문제 재구성.

## 4. Loss (§2.4.1–2.4.2)
- ① 모달리티 패치 판별: 예측 vs 타깃 토큰 contrastive (cosine+CE).
  **같은 밴드셋 내에서만 대조** — cross-modal 쉬운 negative 제거가 성능에 유의미.
- ② 인스턴스 contrastive: SimCLR식, augmentation 대신 같은 입력의 **다른 랜덤 마스킹 2회**를
  positive로. 풀링된 토큰에 적용, 배치 negative, 가중치 0.1. micro-batch 32에서만 대조.

## 5. 데이터 (§2.1)
- 285,288 샘플 × 2.56km² × 1년(월별 최대 12스텝), 10m/px 리샘플.
- 관측 3: Sentinel-1, Sentinel-2, Landsat-8. 맵 6: WorldCover, OSM, WorldCereal, CDL,
  SRTM, Canopy Height. NAIP(2.5m)/ERA5(160m)는 효과 없어 제외.
- 샘플링: OSM 120개 카테고리별 최대 1만 타일, 2016–2024.

## 6. 사전학습 레시피 (§3.1)
- AdamW lr 1e-4, wd 0.02, batch 512(micro 32), warmup 8k, cosine→0.1×, 667,200 스텝.
- 비용: Nano/Tiny 각 1,149 H100h, Base 2,989 H100h, Large 5,240 B200h.
  총 13,179 GPUh / 4.3MWh. → H200×2로 Nano/Tiny급 사전학습 실험은 사정권.

## 7. 평가 (§3, Table 2–3)
- vs 12개 모델 (Galileo, TerraMind, CROMA, Prithvi v2, Clay, DINOv3-Sat, AnySat, Panopticon 등).
- kNN/LP: 24태스크 중 15승. 풀 파인튜닝: 29태스크 중 19승.
- Fig 1: MACs 대비 평균 점수의 파레토 최적 주장 (13개 임베딩 태스크).
- **정직한 자인: Large가 Base보다 항상 좋지 않음** (픽셀 시계열 임베딩에선 유의미하게 나쁨).
  EO 스케일링 미해결. → "어떤 크기를 쓸 것인가"가 실질 질문이라는 공식 근거.
- 파인튜닝 레시피 = 우리 설정과 동일 (인코더 20% 에폭 동결→해동, plateau 0.2/2/cooldown 10).

## 8. Ablation (Table 4, m-so2sat/m-eurosat/PASTIS)
Full Latent MIM 32.2(붕괴) → Lite 42.2 → +모달리티 마스킹 53.6 → +모달리티 패치판별 55.3
→ +인스턴스 대조 56.8 → +맵 62.4. (140k 스텝 축소 학습 기준)

## 9. 케이스 스터디 (§5.1)
- Global Mangrove Watch: RF F1 95.3% → OlmoEarth FT 98.1%, 월 단위 갱신 가능.
- Global Ecosystem Atlas: 플랫폼에서 15,000+ 포인트 라벨링, 북아프리카 SOTA.

## 10. 우리 프로젝트와의 연결
- 벤치마크 A: 논문의 MACs 파레토에 없는 것 = 실배포 GPU-초/km², 단계별 분해
  (다운로드/조립/추론), 릴리스 간 델타. ← 우리가 얹을 것.
- 실측 보완(2026-08-15): 동결 구간 학습은 NFS IO-bound(GPU 0%), 해동 후 GPU-bound 전환.
  "MACs ≠ 실배포 비용"의 직접 증거.
- 밴드셋 구조 = H3(SAR 태스크 릴리스 드리프트)의 측정 단위.
- 재현 이슈(진행 중): docs/lfmc.md의 test MSE 580.6이 공개 ckpt+데이터로 951.9 —
  GOAL.md PR 후보 #7 참고.

## 저자 메모
1저자군(모델링팀, 성 역순 표기): Herzog, Bastani, Zhang, Tseng, Redmon(!).
시니어: Farhadi, Krishna, Beukema. 학계: Kerner(ASU), Shelhamer(UBC).
