# Ai2 upstream 기여: 실질 가치·제출 순서·EO 생태계 비교

검토일: 2026-09-11. 기준 문서: [PR 재진입 목록](PR_REENTRY_2026_09_10.md),
[명세서](../PR_DOSSIER.md), 실제 sample 브랜치와 공개 upstream.
**이 문서는 우선순위 제안이다. PR/issue 제출, push, GPU 재실험은 하지 않았다.**

## 1. 결론: 첫 제출과 대표작을 구분한다

**첫 PR은 sample schema, 대표 엔지니어링 기여는 SCL scoring, 연구 재현 역량은 LFMC 이슈다.**
그다음 확장은 새 모델을 억지로 추가하는 것이 아니라, 외부 사용자가 임베딩의 입력·품질·버전을
확인하고 안전하게 재사용하도록 기존 기능을 연결하는 작은 튜토리얼/검증 도구다.

이는 채용 결과나 머지를 보장하는 판단이 아니다. 평가 기준은 실제 사용자 영향, 직접 재현,
리뷰 범위, 유지보수 부담, 보여줄 수 있는 결과이며 **PR 가치와 논문 novelty는 별개**다.

| 실행 순서 | 후보 / 명세서 번호 | 실제 가치 | 준비도·다음 행동 | 보여줄 역량 |
|---|---|---|---|---|
| 1 | sample annotation schema / #1 | bundled 학습 준비 예제가 모델 실행 전에 실패하는 문제를 해결 | 1파일 브랜치·영문 본문 있음. 현행 upstream 재확인 완료; Linux 재실행은 과거 검증과 구분 | 작고 정확한 첫 기여. 연구 novelty는 없음 |
| 2 | SCL categorical scoring / #10a | 보간된 class ID로 구름/맑음 비율과 장면 순위가 왜곡되는 것을 막음 | BestClear **및 FirstValid** 최소 수정 + 실제 재격자 회귀 테스트 필요 | **가장 좋은 EO+시스템 대표작**. 표준 원칙을 실제 결함에 적용 |
| 3 | LFMC checkpoint/document mismatch / #2 | 공개 모델 평가를 재현할 수 있게 provenance 확인 요청 | 수치는 과거 실측. 공개 파일 식별자는 오늘 재확인. 완전한 평가 recipe/로그 묶음 보강 후 issue | 평가 재현·원인 분리·협업 태도. 잘못된 업로드로 단정 금지 |
| 4 | release/runtime preflight / #13 + #9 일부 | 긴 다운로드·GPU 실행 전에 모델/환경/입력 불일치를 설명 | 먼저 작은 호환성 표와 설정 점검. lock 일괄 업그레이드·기본값 변경 금지 | 외부 온보딩과 재현 가능한 실행 경험 |
| 5 | forest-loss 외부 실행 안내 / #3·#6 | 내부 API 권한 없는 사용자가 실행 불가 이유를 빨리 알게 함 | 문서 issue 우선; URL 설정/401 실패는 현행 runner로 별도 재현 후 fail-fast 제안 | 운영 마찰 감소. 내부 provider를 임의 교체하지 않음 |
| 6 | SCL 보조자산 의존성 / #10b | 반사도만 요청했는데 scoring에 SCL이 필요한 숨은 요구를 드러냄 | 완전한 PC 예제·친절한 에러부터. 자동 dependency API는 maintainer RFC | 데이터 획득부터 품질 평가까지 연결 |
| 7 | 품질·실제 시간·버전을 보여주는 embedding 튜토리얼 / #9 확장 | 결과를 받는 사람이 무엇을 비교할 수 있는지 이해 | 기존 제주 자산 활용, 비중복 범위 협의. 아직 구현/제출 아님 | 기술을 사용자 판단으로 연결하는 포트폴리오 |
| 8 | release별 partial-band 지원 / #12 | 지역 제공 데이터의 밴드 부족을 명시적으로 처리 | **보류**: 공개 config E2E 재현·지원 정책·학습 의미 합의가 먼저 | 전략적 가치는 크지만 첫 PR로는 범위/위험이 큼 |

2와 3은 준비를 병행할 수 있다. 이는 소요시간·수락률의 수치 예측이 아니다.
한꺼번에 8개를 내지 말고 1개의 작은 PR, 1개의 기술 PR, 1개의 재현 이슈로 시작한다.

## 2. 오늘 무엇을 직접 확인했나

공개 GitHub REST/raw 및 HF metadata를 읽었다. GitHub connector는 미연결이라 공개 읽기 경로를 사용했다.
오늘 학습/다운로드 파이프라인 전체를 재실행한 것은 아니다.

| 대상 | 9/11 재확인 |
|---|---|
| olmoearth_projects | main `23a3d7b799ba1fbb0c9138cb1444166ae1d3dd0a`; 최신 머지는 내부 forest-loss deployment #39 |
| rslearn | master `c47952f44811532b9f6e3a8561bd104fe53aa1a6`; latest release `v0.1.14`, 2026-08-25 |
| sample upstream | 6 feature 모두 legacy `es_*`; 로컬 브랜치는 같은 6 label을 `oe_labels.category`로 보존, 1 file +36/−24 |
| open queue | projects open issue 13개·PR 4개(#37/42/43/64). 목록상 schema 및 LFMC 수치 불일치 직접 중복 미발견. 제출 직전 재검색 필요 |
| SCL | `sentinel2_scl.py:115,279` 둘 다 layer resampling을 scoring read에 전달. `tile_utils.py`도 이를 raster read로 넘김 |
| SCL 의존성 | PC Sentinel2 `planetary_computer.py:347–361`은 layer band_sets에 필요한 assets를 선택; 별도 SCL 미요청 시 scoring 자산 누락 경로가 남음 |
| runtime | project lock은 rslearn 0.0.23 / pretrain 0.0.2 / runner 0.1.12. **오래됐다는 사실만으로 버그는 아님**; 사용하는 config/model과의 실패를 보여야 함 |
| LFMC 문서 | `docs/lfmc.md`의 test MSE 580.6 유지 |
| LFMC 파일 | 1,139,505,083 B; LFS SHA-256 `20064f6a0a7a70acc1d5e304bc8050bbe07e86891a60806ee585fffbeb86f92f` |
| HF 날짜 정정 | repo revision `92f32915e29014d50b55f43522a91e3dd2fe610a`, repo lastModified **2025-11-03**. 파일의 lastCommit은 `0c27933f49c5ec99444783845481bf7b972af812`, **2025-10-30**. 둘을 혼동하지 않음 |

HF 파일을 오늘 다시 내려받아 재평가하지 않았다. 현재 식별자와 과거 측정 파일의 로컬 full hash를
재현 묶음에 함께 넣는 것이 최종 provenance 확인이다. 공개 metadata만으로 평가 경로까지 검증되지는 않는다.

또한 외부 계정이 이슈를 작성한 이력은 확인되지만, **사용자 계정의 현재 작성 권한은 로그인 후 확인할 사항**이다.
비로그인 GitHub 페이지의 제한 문구나 `has_issues=true`만으로 전면 제한/허용을 단정하지 않는다.

공개 근거: [projects revision](https://github.com/allenai/olmoearth_projects/tree/23a3d7b799ba1fbb0c9138cb1444166ae1d3dd0a),
[rslearn release](https://github.com/allenai/rslearn/releases/tag/v0.1.14),
[LFMC file metadata](https://huggingface.co/api/models/allenai/OlmoEarth-v1-FT-LFMC-Base/tree/main?expand=true).

## 3. SCL이 가장 좋은 기술 PR인 이유와 제출 조건

반사도는 연속량이므로 bilinear가 적합할 수 있지만 SCL의 숫자는 물리량이 아니라 class ID다.
범주 사이를 보간하면 원래 없던 class가 만들어져 equality/count 기반 score를 바꿀 수 있다.
다만 **동일 격자라 보간이 발생하지 않는 실행은 영향을 받지 않을 수 있다**.

현재 결함은 [FirstValid scoring L115](https://github.com/allenai/rslearn/blob/c47952f44811532b9f6e3a8561bd104fe53aa1a6/rslearn/dataset/sentinel2_scl.py#L115)와
[BestClear scoring L279](https://github.com/allenai/rslearn/blob/c47952f44811532b9f6e3a8561bd104fe53aa1a6/rslearn/dataset/sentinel2_scl.py#L279)에 있다.
우리 [로컬 adapter](../code/scl_compositor.py)는 BestClear의 scoring만 nearest로 바꾸며 반사도 보간은 유지한다.
이 adapter는 upstream PR 완성본이나 두 클래스 회귀 검증을 대신하지 않는다.

제출 전 필요한 최소 검증:

1. 20 m SCL → 10 m target 또는 half-pixel offset처럼 **실제로 재격자되는** 범주형 fixture.
2. nearest scoring에서 class 집합/clear fraction이 기대와 일치하는지 확인.
3. FirstValid와 BestClear 각각에서 두 후보의 score/선택이 기대대로 나오는지 확인.
4. 선택된 **반사도는 기존 bilinear 유지**. 전체 layer를 nearest로 바꾸는 수정은 아님.
5. 같은 격자·nodata·동률 처리·SCL 없음 에러·기존 single-item 정책 회귀 확인.

현행 테스트는 동일 `PROJECTION`/`BOUNDS` fixture에서 bilinear를 넘기는 사례가 있다.
따라서 테스트가 bilinear 인자를 쓴다는 것과 SCL 보간 오류를 검증한다는 것은 다르다.
[기존 테스트](https://github.com/allenai/rslearn/blob/c47952f44811532b9f6e3a8561bd104fe53aa1a6/tests/unit/dataset/test_sentinel2_scl.py)를 확장해야 한다.

**제주 연결:** [v7 전후 이미지](../artifacts/figures/v7_rgb_pairs.png)와 [보고서](../artifacts/results/v7_summary.json)는
현장에서 이 문제를 만나 품질을 확인했다는 증거다. 한 window/4기간의 bad proxy 95.64% 감소는
candidate pool·SCL 선택 등 복합 개입 결과이며, nearest 단독 효과나 downstream 정확도 향상이 아니다.
이 수치를 PR의 성능 헤드라인으로 쓰지 않는다.

## 4. LFMC: 좋은 이슈지만 현재 초안은 과신을 더 줄여야 한다

문서 MSE 580.6, 공개 checkpoint 평가 951.9, 별도 fine-tuning run 558.8은 재현 질문으로 가치가 있다.
그러나 낮을수록 좋은 MSE를 “성능의 60%”로 바꾸면 안 된다. 재학습 결과가 가깝다는 사실도
공개 checkpoint의 업로드 오류를 인과적으로 입증하지 않는다.

- 버전 대조는 **시험한 두 환경의 val 평가**에서 차이가 작았다는 것. 모든 환경 효과 기각 아님.
- epoch/step 차이는 batch, accumulation, world size, sampler, resume, 마지막 epoch 처리에 영향받음.
  다른 데이터 snapshot의 확정 증거로 쓰지 않는다.
- OlmoEarth pretrained backbone을 다시 task fine-tuning한 것이지 foundation model을 scratch pretraining한 것이 아님.
- 문제 해결 요청은 “재업로드하세요”보다 **documented run의 checkpoint·config·split·평가 절차를 확인해달라**가 먼저다.
- 원 test/val window IDs, count, checkpoint SHA, 원본/실행 config diff, dependency versions,
  평가 명령과 로그가 한 묶음으로 있어야 한다. test에 맞춰 checkpoint를 선택한 것처럼 보이지 않게
  epoch 33의 선택 기준도 명시한다.

공개 [LFMC model card](https://github.com/allenai/olmoearth_projects/blob/23a3d7b799ba1fbb0c9138cb1444166ae1d3dd0a/docs/lfmc.md)와
로컬 [정정한 이슈 초안](../ISSUE_DRAFT_lfmc.md)을 별도로 읽을 수 있게 준비한다.

## 5. 다른 EO에 있는 것과 OlmoEarth의 정확한 빈틈

**모델, 라이브러리, 호스팅 플랫폼, embedding 제품을 같은 물건처럼 비교하지 않는다.**
아래 “빈틈”은 검사한 공개 경로의 한계/제안이며 Ai2 비공개 플랫폼 전체에 없다는 뜻이 아니다.

| 비교 대상: 이미 제공하는 것 | OlmoEarth/rslearn 현재 상태 | 우리가 보완할 정확한 범위 |
|---|---|---|
| TorchGeo: float에는 bilinear, int에는 nearest를 기본 선택 | SCL scoring이 반사도 resampling을 상속하는 경로가 남음 | SCL 전용 read만 categorical-safe하게 수정. 새로운 EO 알고리즘 아님 |
| Google Cloud Score+: pixel QA `cs/cs_cdf`와 source/model/software 식별자 | rslearn에 이미 SCL compositor와 cloud-mask 관련 transform이 있음 | QA를 새로 발명하지 말고 **실제 사용 장면·AOI 유효 면적·제외 이유**를 embedding 예제 옆에 전달 |
| TerraTorch/TerraMind: 명시적 modality/band subset 설정 | OlmoEarth도 modality·normalization 설정이 있지만 release별 band-group 제약 존재 | 지원/미지원 입력을 먼저 설명하고 fail-fast. 밴드 선택 API가 있다고 어느 모델이나 같은 성능을 보장하지 않음 |
| AlphaEarth 제품: MODEL_VERSION / DATASET_VERSION / PROCESSING_SOFTWARE_VERSION / 시간 범위 | OlmoEarth는 임의 기간 export가 가능하고 rslearn은 context metadata·토큰 수 기록 기반도 있음 | 기존 metadata를 이용해 **실제 소비 시간·weights revision·전처리·quantization**을 사람이 읽는 receipt로 연결 |
| STAC Processing·MLM: 처리 계보·모델 입력/출력 서술 | OlmoEarth 플랫폼은 이미 STAC를 사용 | 별도 표준을 만들지 말고 기존 schema와 연결 가능한 최소 sidecar/validator 논의 |

근거: [TorchGeo resampling](https://docs.torchgeo.org/en/stable/api/datasets.html#torchgeo.datasets.RasterDataset.resampling),
[Cloud Score+ 제품 명세](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED),
[TerraMind band subset](https://github.com/torchgeo/terratorch/blob/main/docs/guide/terramind.md#subset-of-input-bands),
[AlphaEarth 제품 명세](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL),
[STAC Processing](https://github.com/stac-extensions/processing), [STAC MLM](https://github.com/stac-extensions/mlm).

이 비교는 ecosystem API/제품 설계 사례이지 경쟁 모델의 정확도 비교가 아니다. 특히 annual AlphaEarth
제품의 연도 간 일관성 설명을 서로 다른 OlmoEarth checkpoint 간 호환성 보증과 혼동하지 않는다.

### 중복이라 새 기능으로 제안하지 않을 것

- “OlmoEarth에는 캐시가 없으니 추가”: rslearn에 이미
  [EmbeddingCache](https://github.com/allenai/rslearn/blob/c47952f44811532b9f6e3a8561bd104fe53aa1a6/rslearn/models/embedding_cache.py)가 있다.
  이것은 window/crop 기반 메모리 캐시이며 연구의 durable streaming state와는 다르다.
- “AlphaEarth 입력 지원을 처음 추가”: 이미
  [GoogleSatelliteEmbeddingV1 data source](https://github.com/allenai/rslearn/blob/c47952f44811532b9f6e3a8561bd104fe53aa1a6/rslearn/data_sources/aws_google_satellite_embedding_v1.py)가 있다.
- “구름 처리·유사 검색·COG export·다중 해상도·SFT가 없음”: 기존 기능이다.
  [4/23 embedding 설명](https://allenai.org/blog/olmoearth-embeddings)은 이를 소개하고 입력 품질 한계도 명시한다.
- “STAC 수집·증분 카탈로그·retry/fatal 분류가 없음”: 플랫폼의
  [7/28 인프라 설명](https://allenai.org/blog/olmoearth-infrastructure)에 이미 있다.
  특정 공개 runner의 미설정 URL 문제를 플랫폼 전체 미구현으로 일반화하지 않는다.
- “single-item SCL scoring 생략은 새 버그”: multi-item만 선택한다는 현행 문서/테스트가 있다.
  사용자가 quality gate로 오해하기 쉬운 **설정 안내 문제**부터 다룬다.
- “12개 materialize하면 4개 소비는 무조건 버그”: 현행 예제는 layers를 기간에 맞춰 조정하라고 명시한다.
  명시적 입력/실제 시간 확인을 개선하는 documentation 제안이지 silent crop 버그로 단정하지 않는다.

## 6. 다음 확장 1개만 고른다면: embedding 입력·품질 receipt

가칭이며 **제안 단계, 새 필수 API 아님**. “새로운 안전 표준”이나 “재해 탐지기”로 부르지 않는다.

기존 제주 notebook/CLI에서 한 window를 처리한 뒤 아래를 함께 보여주는 최소 예제를 제안한다.

- 요청한 기간 / 실제 사용한 source item ID와 시각 / timestep 수·순서.
- 모델 release와 weight 식별자 / normalization / S1 단위 / token pooling·GSD.
- RGB + SCL QA / AOI clear·valid 비율 / 무엇을 제외했는지. 미측정 값은 unknown.
- export dtype·nodata·공식 dequantization 경로 / downstream 비교 가능한 조건.
- 입력 이상은 `unsupported`, 관측 부족은 `insufficient observation`, 유효 입력은 `ready`로 구분.
  이것이 모델의 의미론적 정확도를 보증하는 PASS는 아니다.

**세 단계로 쪼갠다.**

1. **시간/품질 예제 보완:** 새 embedding 추출 예제에서 legacy timestamps 사용 여부를 명시.
   현행 wrapper는 `use_legacy_timestamps=True`를 유지하고 warning을 낸다. 신규 프로젝트와
   과거 fine-tuned head의 호환성 요구를 구분하며 전역 기본값을 무작정 바꾸지 않는다.
2. **기존 metadata로 receipt 만들기:** context/output metadata를 재사용할 필드를 먼저 확인.
   weights/input identity를 읽을 수 없으면 invented hash 대신 unknown을 기록한다.
3. **작은 CPU preflight:** 설정상 알 수 있는 band/order·SCL dependency·환경 지원 여부부터 검사.
   실제 clear fraction은 materialize 후 계산하므로 다운로드 전 게이트라고 잘못 부르지 않는다.

좋은 완료 기준은 문서 분량이나 새로운 class 수가 아니라, 예제 사용자 한 명이
“어떤 영상이 이 벡터에 들어갔는가 / 이 두 벡터를 같은 조건으로 비교하는가”를 답할 수 있는 것이다.
GPU 실험과 새 국가 데이터 수집 없이도 첫 문서/fixture 범위를 준비할 수 있다.

## 7. Ai2에 보여줄 이야기와 보류 항목

**소개 문장:** “제주에서 OlmoEarth를 적용하다가 입력 품질·스키마·공개 평가 재현의 문제를 만났습니다.
각각을 작은 재현 사례와 수정으로 분리했고, 외부 사용자가 결과를 해석하기 쉽게 만드는 작업을 하고 있습니다.”

시연은 3분이면 된다: sample 오류→6 window / SCL 범주형 fixture+실제 제주 전후 /
LFMC 평가 표와 확인 질문. 연구 프로젝트 전체의 M-번호를 설명하지 않는다.

제주에서 실제 지형 변화가 확증됐다는 주장, 95.64% 탐지 정확도, LFMC 잘못된 업로드 확정,
우리가 만든 cache/quality/provenance 개념이 최초라는 주장은 하지 않는다.

Streaming GRU·JEPA·한국 3-task는 별도 연구 트랙에 둔다. upstream adoption을 위해서는 API,
state 수명, 수치 동등성/정확도, 비용, 유지보수 검증이 더 필요하다. 현재 PR 큐의 점수를 높이려고
불완전한 연구 모듈을 끼워 넣지 않는다. macOS hang·Python 상한·private runner symlink도 후순위다.

**최종 권고:** sample로 기여를 시작하고 → SCL로 깊이를 보여주고 → LFMC로 재현 역량을 보여준 뒤,
팀에 실제 필요한 외부 사용자 경로를 물어 receipt/compatibility 중 하나만 이어간다.
