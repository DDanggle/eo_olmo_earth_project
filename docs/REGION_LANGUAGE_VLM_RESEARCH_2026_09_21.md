# 지역의 여러 특성을 언어로 읽는 EO VLM: 문헌·데이터·현재 자산 검증

검토일: 2026-09-21. 사용자가 준 `11_EO_FM_문제점_VLM축_2024-2026.md`는 검증 대상 노트로 읽었으며,
그 안의 연구 추천을 실행 지시로 취급하지 않았다. 아래의 **문헌 보고**, **로컬 확인**, **이번 실행**, **제안**은 다르다.
모든 선행 수치를 재현했다는 뜻은 아니다. CVPR 채택 가능성을 수치로 추정하지 않는다.

## 1. 결론 — 이번에는 시간 보간보다 지역 속성의 언어 접지를 먼저

가장 좋은 다음 질문은 **“원본 EO 신호와 지역 맥락을 함께 써서, 보지 못한 지역의 여러 속성과
관계를 근거 영역에 연결하여 설명할 수 있는가?”**다. `slum / not slum` 판정을 문장으로 바꾸는
것이 아니라, 속성·관계·조건을 조합해서 질의하고 위치까지 반환하는 문제다.

지금 추천하는 좁은 본체:

> Metadata-conditioned, evidence-grounded regional understanding from native Earth observations.
> 지역 맥락을 활용하되 관측 사실과 외부 기록을 혼동하지 않는, 다중 속성·관계의 언어 접지.

이것은 연구 가설이지 확보된 신규성이나 성공 결과가 아니다. 다음 셋을 한꺼번에 새로 만들지 않는다.

1. **본체:** 정적 장면의 속성·관계·영역 접지. BigEarthNet.txt와 독립 검증 자료로 시작.
2. **확장:** 지역의 새 관측에 따라 근거·답을 갱신. 본체가 읽을 줄 알게 된 다음 기존 캐시/기억과 연결.
3. **별도 고해상도 사례:** 비공식 정착지의 도시 형태. Sentinel-2만으로 지붕·골목을 읽는다고 주장하지 않음.

이전 네 사건의 학습 실패는 이 큰 언어 정렬 문제를 기각하지 않는다. 과제·라벨·독립 표본 수가 다르다.
반대로 더 큰 공개 데이터가 생겼다고 이전 scorer/시점 누출/사건 독립성 문제도 해결된 것은 아니다.

## 2. 먼저 바로잡을 선행연구 주장

| 기존 생각 | 확인한 사실 | 우리에게 남는 질문 |
|---|---|---|
| AAAI 슬럼 연구는 단순 classification | GRAM은 region-aware MoE와 target adaptation을 쓰는 **segmentation** | 서로 다른 물리적 속성·관계·근거를 언어로 조합하는 효용을 추가해야 함 |
| Native multispectral/SAR와 언어의 연결은 없음 | TerraMind, DOFA-CLIP, MS-CLIP, TimeSenCLIP이 직접 관련 | OlmoEarth라는 encoder 이름 교체만으로 새로움이 되지 않음 |
| 메타정보를 자연어로 넣는 것이 신규 | MetaSegNet은 기후 등 metadata→지리 prompt→cross-modal fusion | 잘못된/오래된 context, 관측과 충돌하는 context를 어떻게 다룰 것인가 |
| 지역의 특성을 설명하면 도시 VLM 최초 | UrbanVLP, UrbanLLaVA, CityLens가 직접 겹침 | native EO 신호의 추가 가치, 근거 위치, 지역 간 일반화, 불확실성 |
| TerraMesh를 받으면 바로 caption 학습 가능 | 공개 base release는 센서·지형·토지피복 중심. TerraMind의 학습 캡션과 공개 파일 계약은 다름 | 별도 TerraMesh-Masks의 caption+mask 공개를 확인해야 함 |

GRAM의 데이터 repo는 **영상은 라이선스 때문에 배포하지 않으며** Wayback zoom16 약1.2m/px로 재취득하라고
안내한다. 논문의 10m 언급은 공개 prediction의 다운샘플링 문맥과 구분해야 한다. 논문/웹 자료를
그대로 “공개 10m 슬럼 이미지-언어 데이터”로 바꾸어 읽으면 안 된다.
[AAAI 원문](https://ojs.aaai.org/index.php/AAAI/article/view/41227),
[GRAM 데이터 계약](https://github.com/DS4H-GIS/GRAM-Dataset).

## 3. 학습하려는 feature를 먼저 나눈다

| 층 | 예시 | 가능한 증거·라벨 | 중요한 한계 |
|---|---|---|---|
| 표면·분광 특성 | 수역, 수목, 나지, 불투수면, 수분/식생 지표 | S2 band·지수, S1, 지표지도, 사람 검수 | 지수가 오염·파괴·불법을 직접 뜻하지 않음 |
| 형태·구성 | 밀도, 면적 비율, 연결 성분, 배치의 규칙성 | native spatial token + mask, VHR building/road vectors | 작은 건물/골목은 VHR가 필요 |
| 공간 관계 | 숲 옆 경작지, 물가에 접한 건조 지역, 도로 사이 조밀한 건물 | polygon adjacency/distance/direction + grounding | 경계 오차와 GSD에 따른 허용오차를 둠 |
| 지역 맥락 | 기후대, 고도, 행정구역, 보호지역, 공공시설 유형 | DEM/기후/OSM/공공 기록 | context prior이며 현재 영상의 관측 정답이 아님 |
| 시간 변화 | 수역 확대, 산림 감소, 건설 진행 | 실제 전후 관측·시간별 mask·사람 설명 | 보간 상태는 새 관측이 아님 |
| 사회·법적 해석 | 비공식 정착지, 허가 여부, 서비스 결핍 | 공식 조사/문헌/공공기록 | 지붕 모양으로 빈곤·소유권·상하수도 상태를 확정하지 않음 |

출력 예시는 “슬럼이다”보다 아래처럼 구성한다. 이것은 설계 예시이지 실제 장소 분석 결과가 아니다.

- **관측:** 특정 영역의 건물 배치가 조밀하고 불규칙하다. 근거는 영상 A의 polygon P.
- **계산:** 건물 피복 비율은 X, 수역까지 거리는 Y. 공간 자료와 단위/오차를 명시.
- **기록:** 공공 자료 B는 해당 행정구역을 특정 주거 유형으로 분류한다. 자료 기준연도와 범위를 명시.
- **한계:** 위성영상만으로 개별 가구의 소득·소유권·위생시설을 판단할 수 없다.

`language description`을 목표로 잡아도 문장 길이·유창함이 아니라 **atomic claim의 사실성 + 위치 + 출처**를 평가한다.

## 4. 지금 사용할 수 있는 데이터 지도

S = 공식 schema/sample/repo까지 확인. P = 논문 본문/저자 자료. A = 초록/프로젝트 수준.
라이선스는 배포 페이지의 표시를 기록한 것으로, 혼합 배포의 법적 적합성을 보증하지 않는다.

| 자료 | 실제 텍스트의 출처 | 공간·시간 / 센서 | 지금의 쓰임과 주의 |
|---|---|---|---|
| **BigEarthNet.txt** [S/P] | 라벨맵·메타→템플릿→LLM 정제, 별도 human-verified bench | S1/S2 patch ID·촬영시각·lat/lon, bbox/VQA/caption | 최우선 언어/공간 정렬. 유럽 10개국, broad LULC이지 슬럼 설명 데이터 아님 |
| **TerraMesh-Masks** [S] | Overture/OSM 기반 mask+caption, 검수 eval | TerraMesh S2/S1/DEM, 좌표와 modality별 시각 | native EO↔phrase↔mask의 직접 후보. 아직 repo에 preprint coming soon, 배포 완결성 실측 필요 |
| **SkyScript** [S/P] | OSM tag 조합; 일부 ChatGPT 재서술 | multi-source RGB, pickle bbox/time, source/year/OSM ID | 지역 기능·관계 어휘. 이미지와 OSM의 시간 불일치, 원천별 권리 확인 |
| **Major-TOM index** [S] | 텍스트 없음; 구조화 지형/기후/토양/인구/행정 값 | geospatial index, S2 index 시각·footprint | 기존 pooled cache의 지역 context 후보. enriched index와의 join은 아직 미실행 |
| **ChatEarthNet** [S/P] | WorldCover 속성+LLM, 일부 검수 | S2 RGB 및 두 false-color 3-band 묶음 | caption-style 학습. 확인한 공개 schema에서는 정확한 lat/lon/time join 불명확 |
| **LHRS-Align** [S/P] | OSM/VGI→LLM | 고해상도 RGB·좌표 filename; 취득시각 불명확 | 지역 의미 학습. 이미지 자체보다 좌표 기반 재수집 계약 |
| **RS5M** [S/P] | 웹 문장 + BLIP2 + 원 데이터 라벨 템플릿 혼합 | 다양한 RGB, 일부 fMoW/BEN 지리·시각 | 넓은 어휘/검색 사전학습. schema와 권리 혼합; 원천별 처리 필요 |
| **VRSBench** [S/P] | GPT-4V 초기문장 + 사람 확인 | RGB caption/grounding/VQA, geo/time 부족 | 공간 언어 평가. native S2/기억 데이터와 직접 조인 불가 |
| **RSITMD / RSICap** [P/S] | 작은 사람 캡션 corpus | RGB, geo/time 부족 | 문장·검색 외부검증. 지역 grounding/시간 갱신을 보증하지 않음 |
| **GeoText-1652** [S/P] | VLM 생성·referee·부분 사람 검수 | satellite/drone/ground, bbox-language | 로봇의 cross-view 연결 참고. acquisition-time 계약은 별도 |
| **Sentinel2Cap** [S/A] | human S2 benchmark + Qwen3-VL 생성 부분 | S2, 공개 repo/parquet | 같은 센서의 독립 caption 평가 후보. geo/time·split은 추가 감사 |
| **DLR poverty-area building footprints** [S/P] | 자연어 없음; 사람이 그린 건물·block polygon | 44 AOI/35도시, 2002–2017 영상 기반 | 형태 속성의 정량 gold 후보. VHR 원본 비공개, 과거 날짜와 S2 현재를 섞지 않음 |
| **Atlas of Informality** [S] | 지역별 설명/조사·mapping | 정착지의 경계·역사적 변화 | 사회·도시 맥락 후보. tile-level 독립 캡션으로 취급 금지 |
| **OSM / Wikidata / Wikipedia** [S] | crowd tags / factual graph / 사람이 쓴 문서 | geometry/coordinates/revision | 지역 근거 RAG. 지도 편집시각과 실제 사건시각을 구분 |

### 4.1 최우선 소스의 직접 링크와 계약

**BigEarthNet.txt.** 464,044 image pairs와 약9.55M annotation rows는 다른 숫자다. 같은 patch의
여러 질문·문장을 서로 다른 독립 이미지로 세면 안 된다. text parquet만 약467MB; 원본S1/S2 전체는
약110GiB다. 필드는 `ID,s1_name,patch_id,input,output,type,category,split,latitude,longitude,country,season,climate_zone`.
`bench`는 1,082 image pairs/15,029 annotations로 저자가 보고한다. 문서 판독과 이번 실제 파일 집계를 구분한다.
parquet의 lat/lon은 scalar 위치이며 bbox target은 정규화된 이미지 좌표다. 지리적 footprint는
`patch_id`와 원본 raster/metadata를 연결해서 검증해야 하며 text parquet만으로 확보됐다고 하지 않는다.
[데이터](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt),
[논문](https://arxiv.org/abs/2603.29630), [원본 영상](https://bigearth.net/).

**TerraMesh-Masks.** phrase에 맞는 binary mask를 제공하므로 “문장은 잘 쓰는데 어디를 말하는지 모른다”는
문제에 더 직접적이다. base TerraMesh의 공개 caption 유무와 혼동하지 않는다. code Apache2,
mask ODbL, image CC-BY-SA4.0로 서로 다르다. eval loader는 S2L2A/S2L1C/S1/DEM을 열거한다.
[코드·계약](https://github.com/IBM/TerraMesh-Masks),
[학습 mask 배포](https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh-Masks),
[검증 배포](https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh-Masks-Eval).
학습 입력 영상은 별도 [base TerraMesh](https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh)에서 받는다.
공식 README의 전체 데이터 용량은 약18TB이므로 작은 shard로 검증한 뒤 필요한 부분만 취득한다.

**SkyScript.** 공개 전체 unfiltered pool은 5.2M, 논문의 2.6M은 CLIP top-50% filtered subset이다.
논문 이후 이미지가 두 배로 늘었다는 뜻은 아니다. 2024/12 language-polished caption CSV는 별도다. 파일명은 OSM ID,
영상 source, 연도를 갖는다. 상세 pickle의 `box,time,center_tags,surrounding_tags`가 정확한 join에 더 중요하다.
GSD가 0.1–30m로 넓어 “위성 RGB”를 하나의 해상도로 묶지 않는다.
[논문·데이터](https://github.com/wangzhecheng/SkyScript),
[작은 5K CSV](https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/dataframe/SkyScript_val_5K_filtered_by_CLIP_openai.csv).

**Major-TOM.** 기존 로컬 감사에서 OlmoEarth/Clay 두 embedding 파일의 **248,719행**이
`grid_cell+product_id`로 1:1 대응함을 확인했다. `unique_id`는 두 파일 간 교집합이 0이었다.
이는 **새 enriched index와의 1:1 join을 검증한 결과는 아니다**. 새 index는 grid 규격·footprint·관측시각을
다시 맞춰야 한다. 특히 384px crop의 의미를 10km cell 전체 설명으로 잘못 붙이지 않는다.
[Core 249k](https://huggingface.co/datasets/Major-TOM/Core-S2L2A-249k),
[enriched index](https://huggingface.co/datasets/Major-TOM/index).

### 4.2 사람 언어와 synthetic supervision을 혼동하지 않는다

지도 라벨에서 문장을 생성하여 **train에 쓰는 것은 정상적인 supervision**이다. 그 자체가 누수는 아니다.
문제는 (a) test 정답을 만든 지도/필드를 test input에도 넣거나, (b) 같은 patch의 다른 QA가 train/test에
나뉘거나, (c) 동일 지도에서 만든 silver 정답만으로 실제 영상의 독립적인 이해·변화 확인을 주장하는 경우다.

따라서 데이터는 `visual-target`, `allowed-context`, `label-generation-only`, `evaluation-gold`로 나눈다.
`country/season/climate`는 일반 context 질의에서는 입력 가능하지만, 이것을 맞히는 visual-only 과제에서는
해당 필드 및 직접 유도하는 위치/날짜를 주지 않는다. context QA 점수와 visual reasoning 점수를 별도 보고한다.

## 5. 현재 코드에서 실제로 연결 가능한 부분

로컬 확인 파일: `code/earthtalk_projector_train.py`, `config/earthtalk_projector_prereg_v0.json`,
`artifacts/results/majortom_contract_audit.json`, 이전 원장 감사 문서.

| 기존 부품 | 재사용 | 바꿔야 할 것 |
|---|---|---|
| frozen OlmoEarth single-observation spatial tokens | spectral/spatial backbone으로 사용 | 새 데이터의 band order/scale/GSD/time/missing mask를 별도로 검증 |
| 768→2048→LLM hidden projector, 출력 RMS 보정 | 안정적인 입력 정렬의 초기 baseline | pooled-only뿐 아니라 좌표 있는 region queries/grounding loss 비교 |
| 32×32→8×8 avg-pool, 64 tokens | 빠른 coarse baseline | 40m token 기준 한 pooled block=160m. 작은 객체·관계 정답에 병목 가능 |
| Olmo-3-7B frozen language backbone | 첫 adapter-only baseline 가능 | text-only도 같은 instruction/SFT 예산으로 비교; 무학습 text-only만으로 공정성 주장 금지 |
| Major-TOM pooled 768-d vectors | scene retrieval·multi-attribute probe | 풀링으로 사라진 위치를 복원했다고 주장 금지. grounding에는 재추출 필요 |
| landslide QA/기존 event memory | 추후 실제 시계열 갱신 실험의 뼈대 | old top12 선택·synthetic gold·quadrant scorer를 새 벤치마크에 복사 금지 |

현재 projector는 개별 날짜 type embedding은 있지만 별도 2D 위치 embedding은 없다. LLM의 sequence
position과 encoder 내부 공간 정보가 있으므로 “위치가 완전히 없다”는 뜻은 아니다. 다만 픽셀/실거리와
명시적인 출력 영역을 연결하는 계약과 supervision이 약하다.

좌표가 다른 공개 imagery의 nearest-neighbor embedding을 caption과 짝지어 학습하지 않는다.
같은 지역 이름도 exact footprint·acquisition date·crop의 일치를 대체하지 않는다.

## 6. 최소 아키텍처 — 처음에는 조립보다 한 가지 가설을 검증

### 6.1 관측 채널과 context 채널

관측 채널: `S2/S1 → frozen OlmoEarth spatial tokens → coordinate-aware region adapter → LLM`.
context 채널: `metadata/document → provenance-tagged typed records → 별도 context tokens → LLM`.

출력은 `(attribute, value, region, evidence_id, evidence_kind, uncertainty)`의 구조화된 claim과 짧은 문장이다.
`evidence_kind`는 `observed`, `derived_measurement`, `external_record`, `predicted`를 구분한다.
연구 목적은 이 schema를 예쁘게 출력하는 것이 아니라 필요한 관측/문서를 정확히 선택하게 학습하는 것이다.

처음에는 단순 MLP + 고정 grid를 기준선으로 둔다. 제안 adapter는 query-conditioned region pooling과
좌표/센서/해상도 embedding을 갖고, 메타정보는 region reading을 조건화하되 직접 관측 evidence를 생성하지
못하게 한다. Q-Former/Perceiver류 resampler나 gate를 쓴 사실만으로 신규성은 없다.

고해상도 도시 형태가 목적이면 **별도 VHR RGB branch**가 필요하다. S2는 토지피복·수분·계절적 배경,
VHR는 건물/도로 형태를 맡긴다. baseline도 동일 VHR를 받아야 하며 성능 차이를 OlmoEarth로 잘못 귀속하지 않는다.
그 branch는 이번 S2 중심 본체가 통과한 뒤 추가한다.

### 6.2 학습 신호

1. **Attribute / relation supervision:** presence만 아니라 면적, count, 인접, 상대 위치.
2. **Grounding supervision:** 같은 phrase를 실제 bbox/mask에 연결. language CE만으로 끝내지 않음.
3. **Matched hard negatives:** 같은 국가·계절·대략적인 토지피복인데 관계/영역이 다른 장면.
4. **Source supervision:** 영상에서 확인한 사실과 context-only claim, 관측 불가를 구분.
5. **Counterfactual context:** 맞는/누락/무관/틀린 metadata. 목표는 모든 답의 불변성이 아니다.
   타당한 context-dependent 해석은 바뀔 수 있지만 관측되지 않은 객체를 생겼다고 주장하면 안 된다.

학습 초기에는 frozen backbone+adapter, 그다음 작은 LLM LoRA를 비교한다. 마지막으로 backbone 일부 LoRA.
관계 접지에 encoder 정보가 부족한데 decoder만 계속 키우지 않는다. 자연어만 감독하면 RGB teacher가
볼 수 없는 분광 속성은 자동으로 학습되지 않는다. 해당 속성은 독립 측정/라벨 또는 적절한 sensor task가 필요하다.

### 6.3 시간 보간/로봇 기억을 붙일 위치

언어로 읽힌 공간 record가 안정된 뒤 `region_id`별 observation/claim을 저장하고 새 관측으로 갱신한다.
가중치를 매 관측마다 온라인 재학습할 필요는 없다. 로봇의 semantic map/scene graph처럼 상태를 바꾸되,
EO는 재방문·구름·다중센서·해상도·관측 지연을 별도로 처리한다.

보간은 `expected state / anomaly proposal`에 둘 수 있다. 보간 임베딩은 observed evidence로 승격하지 않는다.
정적 BEN 캡션 실험으로 short-gap interpolation이나 실시간 update를 검증했다고 쓰지 않는다.

## 7. CVPR용 검증과 중단 기준

### 7.1 필수 비교군

- metadata-only LLM, question-only LLM: 동일 튜닝 예산.
- source-table + deterministic template: 수치/출처 문장을 LLM 없이도 생성하는 기준선.
- 잘 튜닝한 RGB VLM: zero-shot만이 아니라 matched SFT를 포함.
- naive OlmoEarth projector, RGB+OlmoEarth fusion, 제안 region/source-aware adapter.
- native multispectral를 받는 DOFA-CLIP/MS-CLIP의 retrieval/open-vocabulary head; 생성 task에서는 가능한 matched decoder.
- EarthDial/GeoChat/LHRS/GeoGround 중 과제와 공개 실행 계약이 맞는 강한 모델.

native vs RGB에는 같은 AOI/date/crop/GSD·질문·LLM·token budget·튜닝예산을 맞춘다.
추가로 **동일 OlmoEarth의 RGB-band-only arm**, 3-channel band-group VLM,
multispectral teacher를 RGB에 distill한 SATtxt를 비교해야 encoder 차이와 센서 정보 차이를 분리할 수 있다.
RGB를 고정하고 non-RGB를 matched pair에서 교체/누락하는 검증도 넣되, 물리적으로 불가능한 조합으로
생긴 단순 OOD 붕괴를 “분광 근거 사용”이라고 해석하지 않는다. in-distribution band ablation과 독립 sensor gold를 함께 본다.

### 7.2 평가 split과 측정

QA 행이 아니라 **patch→지역 block→도시/생태권** 단위로 분리한다. 같은 imagery/OSM polygon/문서의 파생
문장이 split을 가로지르지 않게 한다. pretrained encoder/teacher의 benchmark exposure도 공개한다.

지표는 attribute macro-F1, relation accuracy, box/mask IoU, factual claim precision/recall,
unsupported-claim rate, evidence citation correctness, abstention risk-coverage와 비용.
자연어 유사도/LLM-as-judge 단독으로 결론내리지 않는다. CI는 독립 지역 수준으로 집계한다.

세 가지 일반화를 구분한다: 새로운 지역, 새로운 속성 조합/관계, 새로운 촬영 조건.
country-heldout 하나로 세 가지를 모두 검증했다고 쓰지 않는다. BEN의 Europe 밖 일반화는 별도 자료가 필요하다.

### 7.3 제안하는 순서 — 아래 학습은 아직 실행하지 않은 계획

| 단계 | 최소 실험 | 통과해야 다음 단계로 감 |
|---|---|---|
| R0 | 공개 schema·band·시각·footprint·exact join + encoder forward | 조인/입력 계약을 실제 산출물로 확인 |
| R1 | 2k–10k unique train patches의 frozen-adapter 작은 학습, 별도 지역 검증 | visual arm이 question/meta-only보다 visual task에서 유리한지 |
| R2 | 20k–50k patch, relation/grounding, 3 seeds, source-isolation ablation | naive projector와 matched RGB-SFT보다 독립 지역에서 이득 |
| R3 | 사람 검수 외부지역 및 metadata conflict/missing test | 정확도 이득이 환각/출처 오류 증가의 대가가 아닌지 |
| R4 | 실제 시계열과 단일 응용(도시 형태 또는 환경 변화) | 새로운 관측의 정보에만 반응하고 근거를 올바르게 갱신 |

H100/H200 한 장으로 시작할 수 있는 것은 작은 frozen-adapter/LoRA 실험이다. 전체 원본 다운로드,
encoder 추출과 학습 시간은 throughput 실측 전 확정하지 않는다. 첫 학습은 예컨대 **4 GPU-hour 상한의
10k-image pilot**로 예산을 정하고 throughput/loss/data-bound 여부부터 본다. 9.55M QA 전체를 먼저 돌리지 않는다.

중단/방향전환 기준:

1. 영상 제거/섞기에도 visual task가 유지되면 학습 문제를 먼저 고친다.
2. native-band 이득이 RGB+metadata와 동일하면 “EO FM만의 장점” 주장은 철회한다.
3. 제안 adapter가 naive projector와 독립 지역에서 동률이면 복잡한 모듈 주장을 철회한다.
4. 관측만으로 판별 불가능한 사회적 라벨은 그 사실을 인정하고 공공 기록 task로 분리한다.
5. 외부지역·사람 검증 없이 내부 synthetic benchmark만 좋아지면 CVPR 제출 근거로는 부족하다.

## 8. 이번 직접 실행 검증

사전 계약: `config/region_language_contract_prereg_2026_09_21.json`.
실행 코드: `code/audit_region_language_contract_20260921.py`.
서버 outroot: `/home/work/data/olmoearth/region_language_contract_20260921`.

범위는 text parquet ≤650MB, compressed image prefix ≤32MiB/8 complete patches,
GPU1 frozen encoder forward만이다. VLM 학습·벤치마크 성능 측정이 아니다.
실제 서버는 H100이 아니라 NVIDIA H200 두 장이었다. 기존 네 확증 실행 코드의 size/mtime/SHA256은
새 파일 전송 전후 불변으로 확인했다. shared venv는 수정하지 않고 데이터 리더만 별도 deps 경로에 둔다.

### 8.1 실제 파일 감사 완료

| 항목 | 실측 |
|---|---:|
| 다운로드한 text parquet | 466,819,745 bytes |
| 전체 annotation rows | 9,553,962 |
| 고유 영상 patch ID | 464,044 |
| train / validation / test / bench 고유 patch | 229,114 / 118,095 / 115,753 / 1,082 |
| bench 문항 | 15,029 |
| 6개 split 쌍의 exact patch-ID 교집합 | 모두 0 |
| caption / binary / MCQ / bbox rows | 463,932 / 3,625,160 / 3,259,184 / 2,205,686 |
| 이미지 archive 실제 읽은 압축 byte | 1,048,576 |
| 12개 band가 완전한 원본 영상 / exact text join | 8 / 8 |
| 8개 영상에 정확히 연결된 언어 문항 | 212 |

split ID disjoint는 위치적으로 떨어졌다는 증거가 아니다. spatial buffer·지역 holdout은 별도다.
bench는 저자 설명상 test에서 선별됐고 실제 공개 파일에서는 별도 split으로 재배정돼 ID가 겹치지 않는다.
표본은 archive 앞부분의 같은 scene 인근 patch들이다. 대표성·일반화를 전혀 검증하지 않는다.

**메타정보 지름길 실측:** 국가 463,274 + 계절 463,662 + 기후 463,637 = **1,390,573 MCQ**.
각 행의 대응 metadata field와 선택지를 단순 문자열 비교하는 알고리즘이 전 문항을 파싱하고 전부 맞혔다.
이는 전체 MCQ의 42.7%에 해당한다. 100%의 **metadata lookup**이지 VLM 성능이 아니다. 해당 필드를 입력으로 제공하는 arm의 합산 점수에
이 문항을 섞으면 visual understanding을 크게 과장할 수 있다는 직접 증거다.
데이터셋 자체가 잘못됐다는 결론도 아니다. context QA라면 정당한 과제이고 visual QA와 분리하면 된다.

마찰 기록: shared venv에는 pyarrow/zstandard가 없었다. 서버의 pyarrow=19.0.1 제약 때문에 최초21.0.0
설치는 실패했다. 제약을 존중해 19.0.1과 zstandard0.25.0을 실험 폴더 `deps/`에만 설치했다.
초기 실패 로그는 `audit.log`, 성공한 재실행은 `audit_retry1.log`로 보존했다. 기존 venv는 변경하지 않았다.

GPU 첫 시도에서 `.venv`의 wrapper가 `normalize` 인자를 받지 않는 것을 확인했다. `.venv-master`의
현재 rslearn wrapper는 이 인자를 지원하므로 그 환경으로 전환했다. 과거 extraction 스크립트가 있어도
실제 실행 venv 계약이 자동으로 같아지는 것은 아니다. 최초 오류는 `gpu_smoke.log`에 보존했다.

### 8.2 H200 frozen encoder smoke 완료

- 모델: **OlmoEarth-v1-Base**, `.venv-master`, 물리 GPU1=NVIDIA H200.
- 원본 8개×12 band. band 이름으로 모델 순서에 맞추고 10/20/60m band를 120×120/10m로 resample.
  12개 band의 공간 범위 일치를 확인하고 CRS/acquisition timestamp를 기록했다. 실제 저해상도 band의 물리적 분해능이
  resample로 10m가 되는 것은 아니다.
- 각 영상에서 **30×30×768 spatial features**를 얻었고 **8/8 모두 finite**였다.
- model load를 포함한 코드의 측정 구간 **5.17초**, peak PyTorch allocated GPU memory **0.607GiB**.
  서버 예약 비용·전체 wall time·GPU reserved memory·LLM/학습 메모리 수치는 아니다.
- 원본 이미지와 언어 ID를 정확히 연결하고 native input을 현재 encoder로 읽는 경로가 실행됐다.
  **언어 alignment 학습, 출력 문장의 정확도, 일반화, SAR 추가 가치, CVPR 방법의 성능은 미검증**이다.
- 전체 59GiB S2 archive를 받지 않고 앞부분 1MiB만 읽었다. 8개 인근 영상이라 성능 표본으로 쓸 수 없다.
- raw chip의 RGB contact sheet를 생성했다. scaling은 각 chip의 RGB 통합2–98 percentile stretch이며
  정량 비교용 반사도 그림이 아니라 방향·내용 sanity check용이다.
  실제 그림을 열어 농경지·수목·하천이 보이는 인접 장면임을 확인했다. 이는 caption/수종/정량면적
  gold 검수를 대신하지 않으며 도시·슬럼 대표 표본도 아니다.

로컬 증거:

- [파일 감사 결과](/Users/dgyi/dong/ai_projects/olmoearth_projects/_work/artifacts/region_language_contract_20260921/evidence/audit_summary.json)
- [GPU 입력 검증](/Users/dgyi/dong/ai_projects/olmoearth_projects/_work/artifacts/region_language_contract_20260921/evidence/gpu_smoke.json)
- [원본 RGB 확인 그림](/Users/dgyi/dong/ai_projects/olmoearth_projects/_work/artifacts/region_language_contract_20260921/evidence/rgb_contact_sheet.png)
- 실패/성공 로그와 exact image-text sample은 같은 evidence 폴더에 보존.

재현용 SHA256:

```text
BigEarthNet.txt.parquet  d3b97f999456016bb13c2a8e94b8f47825654f07a0394a6b266a38b750ca1554
CPU audit code          acef8a148262114e63a5db1e0b03711ef1fdde3ce8c79790eb0ca4f377a1068a
GPU + RGB preview code  c77b5664a4c06d0e47f2b275d940211de11d43e66aa35918fe84dedb1e15aa52
```

CPU code 사본은 실행 후, preview 기능을 추가한 버전으로 교체하기 전에 보존했다. pre-run snapshot이라고
부르지 않는다. 사전 계약 JSON은 데이터 감사 전에 서버에 전송했다. 학습·기존 gate 재판정은 없었다.

## 9. 관련 문헌을 읽는 우선순위

단순 나열이 아니라 우리 주장을 깨는 순서다. 아래는 저자·학회·공식 배포의 1차 자료이며,
일부는 초록/repo 계약 수준 확인이다. 공개했다는 주장과 지금 내려받아 재현됐다는 결과를 혼동하지 않는다.

### A. “native EO와 언어 연결이 처음”이라는 주장을 깨는 문헌

1. [TerraMind, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Jakubik_TerraMind_Large-Scale_Generative_Multimodality_for_Earth_Observation_ICCV_2025_paper.html): native 다중센서와 caption/좌표의 any-to-any 학습. [코드](https://github.com/IBM/terramind). 학습과 공개 caption inference 지원 버전을 구분.
2. [DOFA-CLIP / GeoLangBind, 2025](https://arxiv.org/abs/2503.06312): 가변 spectral modality를 text space에 정렬. [코드·weights](https://github.com/xiong-zhitong/DOFA-CLIP). 생성형 대화 모델은 아님.
3. [Beyond the Visible / MS-CLIP, 2025](https://arxiv.org/abs/2503.15969): 12-band S2 contrastive VLM. [코드](https://github.com/IBM/MS-CLIP). 공개 weights와 caption training corpus의 공개 상태를 구분.
4. [TimeSenCLIP, ISPRS 2026](https://arxiv.org/abs/2508.11919): S2 time series와 co-located ground-photo CLIP 정렬. 시간축 언어 연결도 빈 분야가 아님.
5. [EarthDial, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Soni_EarthDial_Turning_Multi-sensory_Earth_Observations_to_Interactive_Dialogues_CVPR_2025_paper.html): multispectral/multitemporal/multiresolution dialogue. native GeoFM backbone과 band-composite vision encoder를 구분하되 직접 비교를 피하지 않음.

### B. “지역 특성+metadata+언어가 처음”이라는 주장을 깨는 문헌

6. [GRAM, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/41227): 지역 특성 MoE + unseen-city segmentation adaptation.
7. [MetaSegNet, TGRS 2024](https://arxiv.org/abs/2312.12735): metadata를 geographic text prompt로 바꾸어 segmentation과 결합.
8. [UrbanVLP, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/35024): satellite+streetview와 calibrated generated text. caption hallucination/homogenization이 이미 문제로 제시됨.
9. [UrbanLLaVA, ICCV 2025](https://github.com/tsinghua-fib-lab/UrbanLLaVA): 도시의 다중모달 spatial reasoning/instruction learning.
10. [CityLens](https://arxiv.org/abs/2506.00530): 도시 socioeconomic sensing benchmark. [공식 코드](https://github.com/tsinghua-fib-lab/CityLens)에는 ICLR 2026로 표시.
11. [SkyScript, AAAI 2024](https://arxiv.org/abs/2312.12856): georeferenced OSM semantics와 overhead imagery 정렬.
12. [RemoteCLIP 대조와 별도로, Ground Remote Alignment / GRAFT](https://arxiv.org/abs/2312.06960): ground imagery를 language bridge로 사용. 직접 caption이 없다고 학습이 불가능한 것은 아님.

### C. 분류를 넘어선 언어·공간 감독과 검증

13. [BigEarthNet.txt, 2026](https://arxiv.org/abs/2603.29630): S1/S2 instruction·caption·spatial referring-expression 데이터.
14. [TerraMesh, CVPR EarthVision workshop 2025](https://arxiv.org/abs/2504.11172): multimodal EO corpus. 언어 source로 쓸 때는 별도 caption 계약 확인.
15. [TerraMesh-Masks, 공개 repo](https://github.com/IBM/TerraMesh-Masks): phrase-mask supervision; 아직 repo에 preprint coming soon으로 표시.
16. [ChatEarthNet, ESSD 2025](https://doi.org/10.5194/essd-17-1245-2025): globally distributed satellite captions. 지도 기반 supervision의 이득과 한계.
17. [LHRS-Bot / LHRS-Align, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9511_ECCV_2024_paper.php): VGI/OSM 기반 remote-sensing understanding.
18. [RS5M / GeoRSCLIP](https://github.com/om-ai-lab/RS5M): 큰 혼합 image-text corpus의 직접 비교군.
19. [VRSBench](https://github.com/lx709/VRSBench): caption+grounding+VQA. caption 문자열만 아닌 영역을 평가.
20. [GeoGround](https://arxiv.org/abs/2411.11904): HBB/OBB/mask 언어 grounding. bbox 출력 자체는 신규가 아님.
21. [Sentinel2Cap](https://github.com/LucreziaT/Sentinel2Cap): human S2 caption 평가 후보. 현 캐시와 exact join은 미확인.
22. [GeoText-1652](https://github.com/MultimodalGeo/GeoText-1652): cross-view language-grounding 참고.
23. [Qwen3-VL RGB-only SFT, WACV workshop 2026](https://openaccess.thecvf.com/content/WACV2026W/CV4EO/html/Wang_Steering_Instruction-Tuned_Vision-Language_Models_for_Multi-Label_Landcover_Classification_WACVW_2026_paper.html): 작게 튜닝한 RGB baseline을 무시하면 native multispectral 기여를 과장할 수 있음.
24. [DLR 건물 footprint 데이터, Scientific Data 2026](https://www.nature.com/articles/s41597-026-07945-2): 도시 형태의 정량 supervision. [실제 파일](https://download.geoservice.dlr.de/BUILDING_FOOTPRINTS/files/).

### D. 언어가 붙은 지역 기록

25. [OSM data/copyright](https://www.openstreetmap.org/copyright): ODbL, 사람이 작성한 tag가 중심. [history](https://wiki.openstreetmap.org/wiki/Planet_History)에서 평가 cutoff 이전 기록을 분리.
26. [Wikidata access](https://www.wikidata.org/wiki/Wikidata:Data_access/en): QID/좌표/시간 qualifier/출처를 구조화 evidence로 사용. 이미지 관측과 동일시하지 않음.
27. [Wikipedia Geosearch](https://www.mediawiki.org/wiki/API:Geosearch): 좌표 기반 human prose 후보. revision과 유효 시각·출처·재사용 조건 보존.
28. [Atlas of Informality](https://www.atlasofinformality.com/): 정착지 단위 지역 맥락. 동시점 tile caption으로 재해석하지 않음.

### E. 2026년 직접 경쟁 — 위의 설계를 더 좁혀야 하는 이유

29. [SATtxt, CVPR 2026](https://github.com/ikhado/sattxt): multispectral teacher를 RGB representation에 distill하고 언어와 정렬. “RGB가 spectral semantics를 못 쓴다”는 단정에 대한 직접 반례.
30. [TerraScope, CVPR 2026](https://arxiv.org/abs/2603.19039): pixel-grounded EO reasoning, text+mask 출력. 근거 mask를 낸다는 것만으로 신규성 없음.
31. [From Coordinates to Candidate Regions, 2026-09](https://arxiv.org/abs/2609.08391): RS temporal change를 region selection으로 풀며 per-frame ROI·geometry·time token을 사용. temporal region token 자체도 선행.
32. [T-REN, 2026-04](https://arxiv.org/abs/2604.18573): frozen visual features를 language-aligned adaptive region tokens로 압축하며 streaming extension도 제시. 제안 regionizer의 직접 baseline이지 이름만 빌려 새 방법으로 제시할 수 없음.
33. [Beyond Zooming, 2026-07](https://arxiv.org/abs/2607.25993): UHR optical EO의 multi-tool evidence search. active robot/agent를 붙이는 broad novelty와 충돌.
34. [Earth-OneVision, 2026](https://arxiv.org/abs/2606.10819): 여러 센서·시계열/영상의 생성형 EO 이해. broad capability를 경쟁점으로 삼으면 규모 싸움. 논문의 센서 입력 변환과 공개 실행 상태를 구분.
35. [EarthMind, 2025](https://arxiv.org/abs/2506.01667): optical/MS/SAR cross-sensor fusion과 expert QA. [코드](https://github.com/ahmad-naghavi-ozu/EarthMind).
36. [SkySense-O, CVPR 2025](https://github.com/zqcrafts/SkySense-O): EO encoder 기반 region/mask-language와 open-world 이해. 정적 RGB지만 “GeoFM→language 자체가 처음”의 반례.
37. [SkyNative, 2026](https://arxiv.org/abs/2605.17949): encoder-free RS language 모델. 여기서 native는 native multispectral이 아니라 raw RGB patch의 LLM 입력이라는 의미.
38. [GeoLink, NeurIPS 2025](https://arxiv.org/abs/2509.26016): OSM과 imagery의 multi-granularity pretraining. 자연어 생성 모델은 아니지만 metadata fusion 기준선.
39. [GLACIA, WACV workshop 2026](https://github.com/lalitmaurya47/GLACIA): native six-band Prithvi branch와 RGB MLLM branch의 late fusion. “Prithvi를 VLM과 쓴 적 없다”는 주장도 피한다.
40. [GAIA](https://arxiv.org/abs/2502.09598): 여러 mission의 web-rendered EO image, caption, metadata. raw multispectral와 rendered picture를 혼동하지 않는 조건에서 보조 어휘 데이터 후보.

이 자료들 때문에 **region adapter + source tag + memory를 합쳤으니 신규**라는 결론도 성급하다.
벤치마크에서 무엇이 아직 실패하는지 먼저 보여야 한다. 특히 BigEarthNet.txt의 RS-InternVL은
S1/S2 encoder와 LLM을 이미 연결한다. 따라서 우리 첫 실험은 새 시스템 전체를 주장하기보다,
강한 이웃 모델 대비 native sensor 증거·지역 맥락·언어 grounding 사이의 구체적 결함 하나를 찾는 단계다.

## 10. 최종 우선순위

**계속 팔 것은 지역 특성의 compositional grounding이다.** 먼저 `무엇·얼마나·어디·무엇과 이웃`을
native EO + 안전한 context로 읽게 한다. 다음은 실제 VHR가 있는 도시 형태, 마지막이 새 관측에 따른
claim-memory update다. PDE·시간 보간·로봇형 기억·대규모 RAG를 모두 첫 논문에 묶지 않는다.

CVPR를 노리는 근거는 “VLM을 썼다”가 아니라, 기존 모델의 구체적 실패를 독립 지역에서 재현하고
그 실패를 겨냥한 방법이 강한 RGB/native/metadata baseline을 이기는 것이다. 지금은 그 실험을 할 수
있는 데이터와 부품이 확인된 상태이지, CVPR급 결과가 확보된 상태는 아니다.
