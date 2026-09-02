# 산사태 너머 — 다과업 확장 연구안 (2026-09-02)

## 왜 지금 이 문서인가
지금까지의 증거(M65·MS-87·MS-93·MS-94)는 전부 **한 과업(산사태 분할)·한 벤치마크(Sen12Landslides)** 위에 있음.
"얼린 지구 임베딩 캐시를 재사용한다"는 주장은 과업이 하나면 "산사태 특수 현상"이라는 반론을 못 막고,
"캐시 하나를 여러 과업이 나눠 쓴다"는 비용 주장은 정의상 과업 2개 이상이어야 성립함.
이 문서는 **어떤 과업을 어떤 순서로 어떤 규칙으로 붙일지**를 결과를 보기 전에 고정함. 등록 JSON: `config/task2_extension_prereg_v0.json`.

## 0. 조사로 확인한 사실 (근거 링크)
- OlmoEarth 논문의 평가군은 GEO-Bench 7종 + BreizhCrops·CropHarvest(분류) + **PASTIS·MADOS·Sen1Floods11(분할)** + 파트너 과업
  (AWF 토지피복, LFMC 회귀, Mangrove, Nandi 작물, GEA 생태계, **Forest Loss Driver**(S2 4pre+4post, 10클래스), Marine Infra, 선박, **Solar Farm**).
  [OlmoEarth 논문](https://arxiv.org/abs/2511.13655) · [Ai2 블로그](https://allenai.org/blog/olmoearth-models)
- 파트너 과업은 **rslearn 데이터셋 형식 tar 로 공개**되어 있음(`gs://ai2-olmoearth-projects-public-data/evals/partner_tasks/{solar_farm,mangrove,lfmc,marine_infra,...}.tar`)
  이고 frozen/부분동결/미세조정 설정 파일이 함께 있음. [rslearn_projects olmoearth_evals README](https://github.com/allenai/rslearn_projects/blob/master/rslp/olmoearth_evals/README.md)
- 부모 레포 `olmoearth_run_data/` 에 12개 과업 config 존재: `satlas_solar_farm`(S2 4시점, 이진 seg, 전지구), `forest_loss_driver`(S2 4pre+4post, 분류 10, 아마존),
  `mangrove`(S2 12시점, 4클래스, 전지구), `awf/nandi/mozambique_lulc/togo_cropland`(지역 토지피복 seg), `lfmc`(S2+**S1** 12시점 회귀) 등.
- PANGAEA 벤치마크(11 데이터셋)는 GeoFM 이 supervised baseline 을 일관되게 이기지 못함을 보고 → "재사용이 항상 낫다"는 주장은 금지. [PANGAEA](https://arxiv.org/abs/2412.04204)
- 인코더 PEFT 선행: [PEFT-GFM, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Scheibenreif_Parameter_Efficient_Self-Supervised_Geospatial_Domain_Adaptation_CVPR_2024_paper.html),
  [DEFLECT, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Thoreau_Parameter-Efficient_Adaptation_of_Geospatial_Foundation_Models_through_Embedding_Deflection_ICCV_2025_paper.html).
  임베딩-제품 프레이밍: [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291). 배포 교훈: WorldCereal(PAPER_READING_LIST).
- 재난 다시점 공개 후보: [SenForFlood 2025](https://github.com/GEOHUM-PLUS/SenForFlood)(350 사건, 전/중 S1+S2+마스크, 254 GB),
  [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)(11 사건, S1+S2, 4,831 chip), SEN12-FLOOD(시계열, chip 라벨).

## 1. 과업 선정 규칙 (결과 보기 전 고정)
(a) 10 m Sentinel-2 **다시점** 입력(S1 은 선택) — 우리 캐시 계약(4×14d 또는 그에 준하는 시계열)에 매핑 가능
(b) 픽셀 마스크 또는 타일 라벨 (분할 우선, 분류는 헤드 감도로만)
(c) **자연스러운 지리 그룹**이 있어 leave-one-region-out 이 가능 (전지구 → UTM 존/대륙, 사건 → 사건 단위)
(d) rslearn 형식 또는 공개 tar 로 **계약 공학이 이미 끝난** 데이터 (새 로더 작성 최소화)
(e) 도메인 우선순위: 재난 > 토지변화 > 정적 토지피복

## 2. 후보 표

| 과업 | 입력 | 라벨 | 지리 이동 구조 | 계약 적합 | 규모/비용 | 판정 |
|---|---|---|---|---|---|---|
| **satlas_solar_farm** | S2 4시점 | 이진 마스크 | 전지구 → UTM 존/대륙 8폴드 | ★ rslearn 원본 | tar 공개; 캐시 1회 | **Task-2 primary** |
| **forest_loss_driver** | S2 4 pre + 4 post | 10클래스(타일) | 아마존 내 공간 블록 | ★ rslearn 원본, 부모 레포 config | tar 공개 | 분류 헤드 감도(전/후 구조가 산사태와 동형) |
| mangrove | S2 12시점 | 4클래스 마스크 | 전지구 | ★ | 100k 표본(크다) | 예비 |
| **Sen1Floods11** | S1+S2 (단일시점) | 홍수 마스크 | **11 사건 = 사건 LORO** | ○ OlmoEarth 평가군, 단일시점 처리 필요 | 4,831 chip | **Task-3(재난·S1)** |
| PASTIS-R | S1+S2 시계열 | 작물 마스크 | 프랑스 4폴드 | ○ 시점 수 가변 → 라벨-무관 시점 선택 규칙 필요 | 중간 | 계절 신호 대조군 |
| SenForFlood | S1+S2 전/중 | 홍수 마스크 | 350 사건 | ○ 신규(2025) | 254 GB | 대용량 후속 |
| lfmc | S2+S1 12시점 | 회귀 | 전지구 | ★ | 41k | 회귀 확장(후순위) |
| xBD / Landslide4Sense / HLS Burn Scars / DynamicEarthNet | VHR / 단일시점 / Landsat / Planet | — | — | ✗ 계약 불일치 | — | 기각 |

## 3. 왜 이 순서인가 — 도메인 축을 의도적으로 넓힘
산사태(희소·자연재해·산악) → **태양광 발전소**(전지구·인공 구조물·이진, 계절 무관) → **홍수**(재난·수체·S1 필수).
세 과업이 모두 "P4(캐시 재사용) ≥ 최고 raw"이면 재사용 주장이 도메인을 넘고, 태양광에서만 뒤집히면
"재사용 이득은 희소·자연변화 과업에 국한"으로 **정직하게 좁힘**. 어느 쪽이든 논문이 됨. 한국 3-task 는 여전히 최종 외부·비용 시험.

## 4. Task-2 프로토콜 (등록; JSON 과 동일)
- 폴드: 전지구 타일을 UTM 존(또는 대륙) 으로 그룹 → 표본 수 균형을 맞춘 **8 폴드** LORO, 폴드별 val 지역 규칙은 Sen12 와 동일(다음 폴드).
- arm: P2(UNet3D raw)·P3(U-TAE raw)·P4(frozen OlmoEarth 768×32×32 + EmbDecoder) — Sen12 recipe(40 epoch, BCE, pos_weight cap 50) **그대로**, seed 3.
- 지표: positive-tile macro IoU(primary), FP 매칭 작동점, tie-correct AP. 통계 단위 = 폴드(지역).
- **판정 규칙**: P4 가 최고 raw arm 을 ≥5/8 폴드에서 이기면 "재사용 재현"; 그렇지 않으면 "도메인 의존"으로 범위 축소. 기준 사후 완화 금지.
- 그 뒤 **PT-1 동일 프로토콜**(K=5/20, A0/A1, A2 는 MS-94 로 하차했으므로 **A1 vs A4 raw few-shot**) 을 개발 폴드 2개에서.
- 비용: 캐시 추출 1회(타일 수 × 4시점, GPU1) + 8폴드×3arm×3seed = 72 학습(Sen12 기준 ~16분/run → ~19 GPU시간, 공유 경합 시 ×1.5).

## 5. 실행 단계와 파일 (모두 새 파일; 보호 4파일 수정 없음)
1. `code/fetch_task2_solar_farm.sh` — GCS tar 다운로드·SHA 봉인 (서버 `/home/work/data/task2_solar_farm/`).
2. `code/audit_task2_contract.py` — 밴드 순서·시점 수·해상도·마스크 이진성·타일 크기 감사 → `contract_audit.json`(fail-closed).
3. `code/build_task2_geo_folds.py` — UTM 존/대륙 그룹 → 8폴드 + val 규칙, 표본 SHA, `loco_folds_task2.json`.
4. `code/extract_task2_cache.py` — `extract_sen12_fold_cache.py` 패턴을 따르는 새 추출기(emb_fp16·mask_u8·raw_u16·cache_audit.json).
5. `pilot_sen12_gp_heads.py` 를 `--cache/--folds/--contract` 인자로 실행(스냅샷 러너 `code/run_task2_confirmatory.sh`).
6. 결과 → MEASURED_FINDINGS `MS-95` 이후 번호, 판정 규칙 4항 그대로.
**Step 0(병행, CPU, 라벨 미개봉)**: 한국 preflight — `AIHUB_CUBE_V2_CONTRACT` 40-타일 파일럿 게이트, `model_view_10band`, M10 공간 split 유지.

## 6. 하지 않는 것
- Task-2 결과를 보고 폴드/지표를 바꾸는 것. 한국을 개발에 쓰는 것. "OlmoEarth 가 모든 GeoFM 보다 낫다"고 쓰는 것(Prithvi/Clay 미실행).
- MS-94 이후 A2(CacheTune) 계열을 다른 과업에서 되살리는 것 — stop rule 은 과업 무관.

## 7. 이 문서가 답하는 사용자 질문
"산사태만 보는 게 좁지 않나?" → 좁음. 그래서 **계약이 이미 풀린 과업(Ai2 파트너 과업)** 으로 도메인 축을 넓히는 경로를 등록했고,
첫 과업(태양광)은 다운로드·감사·폴드·캐시까지 **CPU/GPU 1회 작업**으로 우리 파이프라인에 그대로 붙음. GPU 실행은 승인 후 시작.
