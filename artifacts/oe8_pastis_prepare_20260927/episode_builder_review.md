# OE8 episode builder 최종 독립 정적 검토

2026-09-27. 검토 파일: `/private/tmp/oe8_pastis_prepare_20260927/episode_builder.py`, 실제 `download/review_bundle_v0/episodes/episode_contract.json`. 코드 SHA256 `64c6df1fe10b40da8cc076286a6184be05ec0db261a4462af0548aaba8eff6d6`가 실제 실행 contract와 일치함을 별도 확인했다. 원본 수정·GPU 실행·모델 학습 없음.

**판정: 실제 v0 산출물을 재생성해야 할 critical leakage/semantics 오류는 이 정적 검토에서 발견하지 못했다.** 실행된 원 배열과 원 라벨의 수치 검산은 별도 독립 감사 결과이며, 이 검토는 코드 경로·계약의 의미를 점검했다.

## 확인한 사항

1. **query gold 접근 분리:** `build_catalog`는 query 원 정답을 읽지 않는다. class 선택은 train_pool48의 eligible object patch coverage에서만 수행한다. dev query마다 선택된 클래스의 모든 directed pair를 생성하므로 query의 target 존재 여부로 public task를 고르지 않는다. public train/dev JSONL hash를 동결한 뒤 `write_scoring`에서 dev 정답을 읽고, 이후 public hash 불변도 검사한다.

2. **source bank의 감독 학습 제외:** `assign_roles`가48train/16bank/16dev를 frozen hash 규칙으로 다시 계산하고 manifest의 `training_partition` 및 `supervised_training_allowed`와 대조한다. train catalog의 query와 support는 train_pool만, development support는 source_bank만 사용한다. bank label은 공개 support mask를 만드는 데 쓰지만 현재 코드에서 train episode 감독으로 흘러가지 않는다. bank raw와 감독 정답이 미래 trainer/native replay에 들어가지 않도록 하는 실제 loader 검사는 아직 별도 작업이다.

3. **양성·혼동·void 의미:** support는 원 instance의 우세 crop class가95% 이상이며 해당 class·label_valid에 속한 pixel만 mask로 저장한다. mask가64pixel 미만이거나 원 instance의8관측 중 결측이 있으면 제외한다. target와 counter class가 다르고, 반복 object_key 및 query와 같은 patch를 막는다. query 정답은 `semantic==target_class AND label_valid`이며 void19는 빠진다. target-absent 사례는 제거하지 않고 scoring에 별도로 표시한다. 연간 crop label을 건강·변화시점 정답으로 바꾸는 코드가 없다.

4. **K와 cohort:** 같은 query/class의 support sequence에서1/2/4/8 prefix를 사용한다. development는 고정 shared source-bank sequence를 재사용한다. class4의 bank capacity7로 K8이 불가능한 directed pairs가 있으나, 부족 여부는 dev query gold나 모델 성능을 보지 않고 source support availability로 계산한다. `k8_auc_cohort`/`k8_auc_eligible`를 실제 scoring에서 적용해96개 공통 query-pair의 모든 K 곡선만 동일 AUC에 사용하고,192개 lower-K 집계는 별도 표로 유지해야 한다. 현재 기록된 이 구분은 적절하다.

5. **pair ID·class mapping shortcut:** `pair_id`는 class pair로부터 만든 deterministic hash이므로 모델에 입력하면 학습 중 class를 암기하는 shortcut이 된다. 현재 contract가 pair/base/episode ID·patch/object ID·file path·hash·class mapping을 model prompt에서 금지하고, `pair_catalog`·`source_objects`·scoring·manifest class counts를 analysis-only로 지정한다. public episode 자체에 이 정보가 보관되는 것은 인덱싱 목적상 가능하지만, 미래 loader가 entire episode JSON을 prompt에 넣지 않아야 한다. 이 실행에서는 모델을 실행하지 않았으므로 실제 leakage-free inference가 검증된 것은 아니다.

## 유지해야 할 한계

- 같은 patch는 분리했지만 exact footprint·cross-patch parcel identity는 미검증이다. bank/train 최소 centroid 거리의 추가 검사는 유용한 근거지만 정밀한 geometry 비중첩 검증과 동일하지 않다.
- 입력 NPZ에는 query8관측이 모두 있다. 아직 취득하지 않은 관측·전체 관측 quality flag·다른 split의 정답을 실제 모델 경로에서 차단하는 acquisition/loader guard는 아직 미구현이다. 이것은 입력 catalog의 결함을 새로 찾았다는 뜻이 아니라 학습 시작 전 남은 gate다.
- source support는8관측 전체를 사용하고 그 비용을 장부에 포함하도록 계약되어 있다. query만2→3→4관측이라고 기록하며 support 비용을 누락하면 안 된다.
- 소수4개 알려진 class의 exemplar 조건 문제다. 교정 쌍은 공개 mask 합성이며 실제 사람 교정이나 새 class 발견을 입증하지 않는다. 부정 예시를 실제로 활용하는지는 향후 역할 교환·교정 제거 등의 성능 통제로 검증할 부분이다.

**다음 판정:** 입력과 episode 준비 PASS는 유지 가능하다. 실제 학습 이전에 model-visible whitelist, scoring 경로 접근 차단, split/permission 검사, acquired-observation gate를 실행 검증해야 한다. 이 준비 단계의 검토를 과학적 성능·일반화·CVPR 신규성 PASS로 해석하지 않는다.
