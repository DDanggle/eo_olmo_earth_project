# AI-Hub Sentinel-2 cube v2 계약

동결일: 2026-08-26. M35 결과를 본 뒤 작성한 **재물질화 사전 계약**이다. 이 문서를 바꾸면
새 schema/version으로 남기고 v2 실행 중 임계값을 바꾸지 않는다.

## v1 철회 범위

- v1의 2,539는 파일 생성 성공 수이지 유효 cube 수가 아니다.
- 624개(24.6%)가 all-band zero 10% 이상이고 일부는 100% zero였다.
- all-band zero <1%인 1,912개도 사후 후보일 뿐 usable 판정이 아니다.
- v1 arrays·manifest·audit는 오류 증거로 보존하고 downstream 학습·평가에서 사용하지 않는다.

## v2 target contract

| 항목 | 동결값 |
|---|---|
| target CRS | inventory의 EPSG:32652 |
| target grid | inventory bbox, 1024×1024, 정확히 10 m (rounding tolerance 0.05 m/span) |
| 날짜 | AI-Hub 관측일 UTC 하루 |
| 플랫폼 | AI-Hub `S2A/S2B`와 STAC `sentinel-2a/2b` 일치 필수 |
| collection | Planetary Computer `sentinel-2-l2a` |
| bands | B02 B03 B04 B08 B05 B06 B07 B8A B11 B12 B01 B09 |
| candidate order | item ID 오름차순; overlap은 먼저 온 valid pixel 우선 |
| resampling | nearest primary; bilinear은 후속 민감도 arm만 |
| dtype | uint16 |
| cloud filter | item-level `eo:cloud_cover ≤ 60`; 값 없음은 제외 |
| coverage | 12밴드 공통 valid mask가 target의 **99.9% 이상** |
| missing asset | 한 밴드라도 없으면 해당 item의 그 밴드는 기여 불가; 최종 공통 coverage로 판정 |
| write policy | v2 별도 디렉터리, array+validity mask 원자적 저장, v1 덮어쓰기 금지 |

유한하지 않거나 10,240 m 정사각형이 아닌 inventory bbox는 STAC 검색 전에
`invalid_target_grid`로 제외한다. 네트워크/reader `error`는 과학적 제외가 아니므로 재실행 때
재시도하고, manifest의 완료 항목이나 결정론적 제외와 합치지 않는다.

`coverage`는 픽셀값 0 여부가 아니라 raster source mask/nodata를 target grid로 warping한 유효성으로
계산한다. 실제 0 반사도와 nodata를 혼동하지 않는다. 99.9% 미만은 학습 때 0으로 채우지 않고
`excluded.jsonl`에 candidate IDs, band별 coverage, 공통 coverage와 함께 fail-closed한다.

## 재실행 전·후 gate

1. 40개 층화 pilot: v1 severe 20 + v1 clean 20, 서로 다른 날짜·플랫폼·경계 유형을 포함한다.
2. pilot 40/40에서 platform match, 12 asset, shape/dtype, common coverage ≥99.9%를 만족해야 전수 실행한다.
3. pilot마다 source item bbox/geometry와 target UTM/WGS84 bbox, band별 validity, 공통 validity를
   보존해 footprint 교집합을 독립 감사할 수 있게 한다.
4. 전수 결과는 예외 수가 아니라 coverage 분포·band별 nodata·all-band-zero 분포를 다시 감사한다.
5. 전수 중 임계값을 낮추지 않는다. 탈락이 많으면 v2 결과로 기록하고 v3 계약을 새로 쓴다.
6. downstream split별 제외율과 희소 class별 제외율을 보고해 selection bias를 검사한다.
7. 공개 논문용 시각 표본은 AI-Hub 이용정책 범위의 최소 예시만 사용하고 원본·cube를 재배포하지 않는다.

## 판정

- `materialized`: 파일이 생성됨.
- `coverage_valid`: 12-band common validity ≥99.9%.
- `health_audited`: 전수 내용 감사와 층화 시각 감사 통과.
- `experiment_eligible`: 위 세 조건 + split/class selection-bias 감사 통과.

이 네 상태를 합쳐 단순히 `success`라고 쓰지 않는다. RQ2 학습에는
`experiment_eligible`만 들어간다.

## M65/C1 외부 전이에서 쓰는 model view

12밴드 물질화는 원천을 보존하기 위한 **data contract**이지 모든 arm에 12밴드를 주겠다는 뜻이
아니다. Sen12 S12q와 Presto C1의 공통 관측은 B02/B03/B04/B08/B05/B06/B07/B8A/B11/B12
10밴드다.

- external-transfer primary는 위 10밴드만 쓰고, OLMo v1의 B01/B09 band group은 Sen12와 Korea
  양쪽에서 동일하게 missing 처리한다.
- Korea B01/B09까지 쓰는 12밴드 arm은 OLMo-only sensitivity다. Presto/P2/P3와 같은 표의
  primary comparison으로 올리지 않는다.
- 따라서 v2 cube는 12밴드 validity를 계속 봉인하되, downstream manifest에 `model_view_10band`
  hash와 `materialized_12band` hash를 둘 다 기록한다.

이 분리를 하지 않으면 Korea의 성능 변화가 country transfer인지 추가 밴드 정보인지 구분되지 않는다.
