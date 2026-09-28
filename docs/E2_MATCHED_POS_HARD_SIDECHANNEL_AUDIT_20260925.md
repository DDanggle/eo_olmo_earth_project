# E2 S1 positive–hard-negative exact-prompt matching 감사

2026-09-25. 코드와 메타데이터만 읽었다. 예측 파일을 열거나 점수를 계산하지 않았고, 모델·새 실험을 실행하지 않았다.

## 판단

**같은 사건, 같은 literal 날짜, 같은 슬롯, 같은 prompt를 가진 pos/hard_neg 비교는 텍스트·시간·센서·입력 형식에 의한 직접 구별을 상당히 통제한다.** 같은 checkpoint/seed 내에서는 두 입력의 가변 부분이 EO token 값으로 좁혀진다. 따라서 성공 결과가 나올 경우 “동일한 비영상 입력 조건에서 EO 임베딩 값이 source label 구별에 유용하다”는 해석은 가능하다.

하지만 비교 대상은 서로 다른 tile이고, 영상 속 지형·토지피복·관측 품질·노이즈·zero-fill·기존 수역·촬영 기하가 남는다. **홍수라는 특정 현상 의미, 두 시점의 변화, 실제 발생 시점 또는 순수 픽셀 정보의 인과적 기여를 입증하는 비교는 아니다.** 일반 E2의 pre/pre negative보다 시간/slot 우회를 강하게 통제하는 보완 분석이다.

## 1. 날짜·slot·prompt가 같아지는 근거

- [E1 dates_for](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/e1_flood_qa_v0.py:49): 날짜는 각각 metadata flood_date−24/−12/0일로 생성한다.
- [pos 생성](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/e1_flood_qa_v0.py:89), [hard_neg 생성](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/e1_flood_qa_v0.py:94): 두 kind 모두 `dates=[pre_2,post]`, `slots=[pre_2,post]`다. within-tile neg만 `[pre_1,pre_2]`다.
- 로컬 `artifacts/streaming_review_20260909/kurosiwo_s1_cache/meta.jsonl`의 event→flood_date 집합만 읽어 확인한 결과, 43개 event 모두 flood_date가 하나였다. 이는 날짜 메타데이터 확인이며 모델 성능 계산이 아니다.
- 다만 builder 자체는 event 내 날짜 동일성을 assert하지 않는다. 후속 분석에서는 **저장된 실제 QA row의 event + literal dates tuple + slots tuple + 최종 user prompt**를 검증해야 한다. event ID만으로 같다고 대체하지 않는다.
- [hard 선택](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/e1_flood_qa_v0.py:82): hard candidates는 split 전체에서 tile ID의 hash 순으로 선택한다. 처음부터 event별 matching된 자료가 아니다. 양쪽 kind가 존재하는 exact-match stratum만 분석 가능하고, 제외 event/문항 수와 이유를 별도 기록해야 한다.

## 2. Encoder에 들어가는 것과 들어가지 않는 것

[추출 입력 구성](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/extract_kurosiwo_s1_cache.py:23)은 `RasterImage(image=VV/VH crop, timestamps=...)`와 `ModelContext(inputs=..., metadatas=[])`를 만든다.

| 항목 | 코드상 직접 전달 여부 | 남는 의미 |
|---|---|---|
| S1 VV/VH 값 | 전달 | 관측 내용과 센서/품질 흔적을 포함 |
| 사건일 기반 synthetic timestamp | 전달 | exact 날짜 match 시 양쪽 동일; 실제 취득일은 미검증 |
| 센서 modality / patch 크기 / crop grid | 전달·고정 | S1끼리 같은 조건, 상대 공간 위치 encoding은 남음 |
| 위경도·event ID·tile ID·AOI 이름 | 전달 안 함 | tile ID는 파일 선택에만 사용; 지리적 특성은 영상에서 간접 인식될 수 있음 |
| 참조 flood mask / valid_u8 / pflood / label | encoder에 전달 안 함 | source label 구성·표본 선택에는 사용됨 |
| 명시적 quality 점수 | 전달 안 함 | 결측·산란 패턴 같은 영상 내 품질 단서는 남음 |

[raw/label 읽기 및 인코딩](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/extract_kurosiwo_s1_cache.py:42)에서 mask와 valid를 읽지만 `window`에는 cube와 timestamps만 전달한다. [raw zero 처리](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/extract_kurosiwo_s1_cache.py:43)는 원 backscatter 0을 −30dB로 채워 영상 입력에 남긴다.

서버의 현재 `.venv-master/lib/python3.11/site-packages/rslearn/models/olmoearth_pretrain/model.py:595–692`도 읽었다. sample kwargs는 modality image tensor, modality availability mask, timestamps로 구성한다. 657–667행의 관측 있는 시점 mask는 전 영역 `ONLINE_ENCODER` 값이며 Kuro 참조 flood/valid mask가 아니다. 즉 변수명 `sentinel1_mask`를 정답 mask가 주입되는 것으로 해석하면 안 된다. 현재 wrapper에도 per-tile 좌표나 event/label metadata를 끼워 넣는 경로는 확인되지 않았다.

## 3. Projector/LLM 입력

- [eo_tokens](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/e2_multi_reader_v0.py:208): tile는 `.npy` 선택 및 메모리 cache key 용도다. 실제 값은 선택한 두 slot의 spatial embedding, 각각 64 token, 그리고 B−A 64 token이다.
- [token type](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/e2_multi_reader_v0.py:224): 두 kind 모두 type `[0]*64+[1]*64+[3]*64`, 총 192 token이다. pos/hard의 kind나 tile ID를 type embedding에 넣지 않는다.
- [projector](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/e2_multi_reader_v0.py:193): 입력은 token 값과 type뿐이다. 같은 seed의 같은 projector를 사용해야 한다.
- [prompt](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/e2_multi_reader_v0.py:228): sensor 이름, phenomenon 이름, 두 날짜와 고정 질문만 넣는다. 좌표·event·tile·flood_frac·kind·gold는 포함하지 않는다.
- [평가](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/e2_multi_reader_v0.py:279): `build(..., answer=None)` 상태로 호출하고 greedy generation한다. gold 답을 embedding에 붙이는 243–246행은 학습에만 적용된다. 평가 결과에 저장되는 `text_gold`가 질문 입력에 들어가는 것은 아니다.

따라서 literal prompt, 같은 model/projector seed, 동일 192 token 형식에서 양쪽을 비교하면 명시적 비영상 입력은 같다. blind 또는 zero-EO에서는 같은 prompt/shape/type이면 동일 입력이 되므로 별도 새로운 실행 없이도 통제의 원리를 명확히 설명할 수 있다. 실제 저장된 예측의 일치 여부는 이번 감사에서 확인하지 않았다.

## 4. source label과 hard=0의 제한

[E1 tile_stats](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/e1_flood_qa_v0.py:35): `labelled = (valid_u8==1) & (mask_u8>0)`, flood는 그 영역의 class3 픽셀이다. 원 224×224 전체가 아니라 cache의 가운데 192×192 crop 기준이다.

- pos: `flood_frac >= .02`. 최소 valid_frac 조건은 없다.
- hard_neg: `flood_px == 0` AND `valid_frac >= .90`.
- 따라서 “hard=0”은 **crop 안에서 유효·라벨된 영역의 홍수 라벨이 0**이라는 뜻이다. 최대 10% 미확인 영역, crop 밖, annotation 오류까지 실제 홍수 부재가 보장되지 않는다.
- pos와 hard의 유효 영역 조건이 비대칭이다. pos는 적은 유효 픽셀에서도 양성이 될 수 있고, encoder는 valid mask로 입력 픽셀을 제거하지 않는다. invalid 영역/zero-fill 패턴의 차이가 class와 연관될 수 있다. 실제 차이의 크기는 이번에 계산하지 않았다.
- hard=0은 permanent-water=0이라는 뜻이 아니다. 영구 수역 비율과 지형·토지피복은 일치시키지 않았다.
- 0<홍수비율<.02의 중간 사례는 양성에도 hard에도 들어가지 않는다. matched 성능의 대상은 선택된 두 극단의 source-label 구분이며 전체 홍수 탐지 난이도와 다르다.
- pos/hard 라벨은 post의 홍수 참조 mask에 기반한다. pre_2에 실제로 홍수가 없었는지, 두 관측 사이에 새로 홍수가 발생했는지, 다른 변화가 없었는지는 독립적으로 검증하지 않았다. same-date matching도 이 라벨 의미를 강화하지 않는다.

## 가정·미확인 항목

확인한 생성 코드와 현재 wrapper가 실제 현재 cache를 만들었다는 개별 과거 lineage는 앞선 cache 감사와 같이 완전 검증되지 않았다. 이번의 matching은 현 캐시·QA 계약을 전제로 한다. 정확한 취득시각/순서, 상대 관측기하·SAR 보정/정합, source mask 정확도, 장소·지표면 배경·quality 분포의 균형, 모델 사전학습 노출은 새로 확인하지 않았다.

새 분석 명세의 가장 좁은 결론 문구는 다음과 같다: **“동일 사건·날짜 문구·관측 슬롯의 서로 다른 tile에서, EO 임베딩 값이 post 홍수 source label 유무를 구별하는 데 기여하는지 평가한다.”** 모집단과 event support를 명시하고, 대조군 선택은 모델 출력과 무관하게 고정해야 한다. 특정 현상 이해·변화 이해·인과성은 별도 검증 대상이다.
