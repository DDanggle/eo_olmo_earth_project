# OE8 prepare_inputs.py 독립 검토

2026-09-27. 대상 `/private/tmp/oe8_pastis_prepare_20260927/prepare_inputs.py`의 업로드 전 정적 검토. source·GPU 변경 없음. 배열 수치의 독립 검산은 다른 검수자의 실행 결과와 구분한다.

## 즉시 반영할 사항

1. **contract의 source stat 연속성 값이 실패 때도 True다.** 검토본192행에서 `source_shards_unchanged=unchanged`인데 `verified_source_stat_continuity=True`가 상수다. 원 파일이 처리 중 변경되어 status가 `incomplete_or_source_changed`여도 연속성 검증은 통과한 것으로 남는다. `verified_source_stat_continuity=unchanged`로 연결하고 before/after의 size/mtime_ns를 보존한다. `fresh_whole_shard_hash_this_run=False`는 현재대로 유지한다. size/mtime 연속성은 원 sha를 새로 계산한 것과 같지 않다.

2. **결측 없음과 학습 허가가 같은 필드로 읽힌다.** 검토본168~175행은 source_bank/dev_query도 `clean_training_eligible=True`가 될 수 있고, `dict(c,...)` 때문에 source_bank의 원 `role='train'`도 남는다. 이 필드/legacy role만 쓰는 downstream loader는 bank 정답을 학습에 넣을 수 있다. 권장: `strict_no_missing_eligible` 등 입력 품질 필드로 이름을 좁히고, `supervised_training_allowed=(training_partition=='train_pool')`를 별도 기록한다. 원 role은 `original_candidate_role`로 옮기거나 계보 전용임을 명시한다. trainer/episode builder는 authoritative `training_partition`만 사용하고 bank/dev ID가 어떤 train query/support/replay 감독에도 들어가지 않는지 교차 검사한다. CPU 추출 자체를 막는 배열 오류는 아니다.

## 추출은 진행 가능하되 계약에 남길 사항

3. **48/16/16은 patch 분리이며 필지 독립성 검증 완료가 아니다.** bank를 학습지역의 원32개 중 hash순8개로 고른 것은 label을 보지 않는 적절한 방법이다. 그러나 같은 parent에서 인접 patch/경계 필지가 겹칠 가능성을 검사하지 않았다. raw instance ID가 patch별 local ID라면 다른 patch의 같은 숫자를 전역 객체 ID처럼 비교해서도 안 된다. `parcel_disjoint_verified=False`, `bank_train_footprint_overlap_audit_pending=True`를 명시하고, 비교 학습 전에 확인 가능한 공간/원객체 검사를 하거나 주장 범위를 patch-disjoint 개발로 제한한다. 이 이유만으로 원 입력 추출을 중단할 필요는 없다.

4. **서로 다른 mask 축 순서를 기계 계약으로 명시하면 downstream 오정렬을 막을 수 있다.** 실제 배열은 `normalized_s2:H,W,T,B12`, `raw_selected_s2:T,B10,H,W`, `observation_valid:T,H,W`, `nodata_observed:T,B10,H,W`, `band_observed:B12`, `timestamps:T,3`이다. 현재 normalized_shape만으로 sidecar 축을 판정하면 H/W/T를 혼동할 수 있다. contract에 tensor_layouts와 timestamps의 `[day,zero_based_month,year]`를 명시하고 verifier/loader에서 shape를 검사한다. `band_observed`는 원 밴드 취득 여부이며 pixel/date별 nodata mask와 다른 뜻이다.

## 검토에서 문제가 없었던 부분

- 원 int16 S2를 직접 읽고 INPUT→NATIVE 재배열 후 공식 computed mean/std를1회 적용한다. B01←B02와B09←B8A, native순서의10실측/2보완 표시는 서로 맞는다. 주어진 코드에서 double normalization·clip은 보이지 않는다.
- `raw==-10000`을 운영상 보수적 sentinel로 취급하고 다른 음수는 보존한다. 한 실측 밴드라도 sentinel이면 해당 pixel/date의 normalized12밴드를0으로 만들고 원값·실측별 nodata·pixel/date validity를 별도로 남긴다. 이 zerofill은 물리적 반사도0이라는 주장도, 올바른 missing-aware encoder 경로가 구현됐다는 주장도 아니다. 현재 정책 문서가 이 제한을 밝혀 두었다.
- `label_valid=semantic!=19`와 관측 validity를 분리한 것은 적절하다. 연간 mask가 관측마다 읽힌다는 보장은 없으므로 학습/평가가 선택한 temporal window에서 invalid/void를 어떻게 처리할지는 별도 loader 정책이 필요하다.
- 구름 마스크를 만들었다고 하지 않고 `cloud_quality_certified=False`를 유지한다. no-sentinel 사례를 맑은 영상으로 해석하지 않아야 한다.
- 날짜8개는 고정된 원 날짜 인덱스로 영상과 함께 선택하며 눈으로 맑은 날짜를 고르거나 label/prediction으로 선택하지 않는다.

**판정:** 원 배열 추출을 막을 변환 산술 오류는 이 정적 검토에서 찾지 못했다. 1·2의 기계 계약을 고치고, 3·4의 해석/축 계약을 명시한 뒤 CPU 추출·독립 검증으로 진행할 수 있다. 이 준비 성공은 source bank의 객체 독립성·관측 판독 가능성·새 방법의 성능을 입증하지 않는다.
