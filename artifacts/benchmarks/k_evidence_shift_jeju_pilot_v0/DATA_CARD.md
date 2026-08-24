# K-EvidenceShift Jeju pilot v0 data card

## 판정

이 산출물은 **벤치마크가 아니라 누수 검사를 통과한 audit pilot**이다. 현재 14개 후보에는 독립 인간 정답이 0개이므로 정확도·전이효과·원인 규명 성능을 보고하면 안 된다.

## 현재 범위

- 후보 레코드: 14
- 500 m 공간 그룹: 13
- 공유 materialized-window 그룹: 8
- assistant 시각 pre-annotation: 14 (ground truth 사용 금지)
- 독립 인간 정답: 0
- 시점 정렬 공식 사건 보강 근거: 0 (인과 주장 불가)
- 보류: 14/14
- PNU 출처 충돌: 1

## 누수 계약

- 같은 500 m 공간 그룹은 향후 서로 다른 split에 둘 수 없다.
- 같은 rslearn materialized window를 공유하는 후보도 서로 다른 split에 둘 수 없다.
- 현재 scene graph가 연결되어 cloud/quality task의 scene-disjoint split은 만들 수 없다.
- API snapshot은 마지막 EO 관측 뒤에 수집됐고 공개시점이 동결되지 않았으므로 prospective model input에서 제외한다.
- 행정자료 no-match는 음성 원인 라벨이 아니다.
- 현재 GK2A 2 km grid는 수집됐지만 과거 6시점 조회는 API 제한으로 실패했다. Sentinel-2 B02 임계값은 cloud proxy일 뿐 한국형 구름 계측값이 아니다.
- sealed probability test와 이중 판독 세트는 아직 만들지 않았다.
- 다중 모델용 common Sentinel-2 입력 계약도 아직 동결되지 않았다. P0의 OlmoEarth 릴리스 감사는 별도의 exact-input 계약을 사용하므로 전이 성능표가 아니다.

누수 검사 결과: `pass_for_audit_pool_not_a_train_test_split`

## 승격 조건

CVPR 실험 준비 상태: `False`. 최소 3개 독립 지역, 300개 독립 블라인드 판독, 그중 120개 이중 판독, sealed probability test, 동결된 paired-input 계약, 4개 paired-input baseline 완료가 필요하다. 연합학습은 실제 원자료 반출 불가 기관 사일로가 3곳 이상일 때만 승격한다.
