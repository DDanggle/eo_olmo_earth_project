# K-Earth 공공데이터 연결 카탈로그

최종 확인: 2026-08-22  
대상: 제주 공식 오름 368개 evidence ledger와 OlmoEarth 변화 후보  
범위: 한국 전체 공공데이터의 전수 목록이 아니라, **변화 검증·원인 후보·보류 판정에 연결 가능한
공식 데이터의 연구용 shortlist**다.

사용자가 지금 신청할 서비스, 현재 실제 보유본, 시계열 custody, 사업·연구 판정은
`K_EARTH_PROGRAM_STATUS.md`의 다섯 축에서 관리한다. 이 문서는 source-level 세부 join 계약을 맡는다.

## 결론부터

공공데이터를 더 붙인다고 원인 규명이 자동으로 가능해지지 않는다. 가장 재현 가능한 연결축은
`PNU(필지고유번호) + 공식 polygon + 유효시점`이며, 각 데이터는 다음처럼 역할을 분리해야 한다.

1. **기하 연결축**: 연속지적도에서 PNU와 필지 polygon을 얻는다.
2. **원인 후보 사건**: 환경영향평가 사업구역, 건축인허가, 개발행위허가, 산림사업을 공간·시간으로
   교차한다.
3. **독립 상태 관측**: 팜맵, 토지피복지도, 임상도, 국토지리정보원 항공사진으로 실제 지표 상태를
   확인한다.
4. **영향·규제 문맥**: 생태자연도, 국토환경성평가지도, 보호지역, 국가유산은 변화의 중요도와
   검토 우선순위를 정하되 원인으로 쓰지 않는다.
5. **입력품질·교란 감사**: 천리안 2A 구름탐지와 기상관측으로 위성 입력의 공통오류를 독립 점검한다.

따라서 가장 강한 주장은 `변화 footprint ∩ 공식 사건 polygon/PNU`가 존재하고 사건일이
관측 전후 구간과 맞으며, 별도의 항공사진·상태지도에서도 변화가 확인될 때만 허용한다. 단순 주소
일치, 같은 리, 최근접 시설, 현재 지도는 D급 문맥이다. 조회 결과가 0이어도 해당 출처의 시공간·
행위유형 coverage가 완전하다는 증명이 없으면 U(unknown)다.

2026-08-22 첫 구현은 이 원칙을 실제 원본에 적용했다. 2025 제주 FarmMap 289,379 polygon 중
유효 PNU는 289,367건이었고, 변화 후보 4좌표 중 `oreum_v6_r08` 1건만 polygon 안에 들었다.
다만 FarmMap 항공 관측일 2022-12-30은 후보의 변화 전 영상 2024-05-16보다 503일 빠르므로,
이는 B급 **변화 전 농지 상태**이지 원인 사건이 아니다. OSM으로 위치화한 오름점 243개 중
7점도 농지 polygon 안에 들었지만 입력 위치가 공식 오름 경계가 아니므로 C급에 머문다.
따라서 A/B급 원인 근거는 여전히 0/368이고 선택적 보류 정책은 유지된다.

## 접근성 표기

| 표기 | 의미 |
|---|---|
| 파일 | 포털의 파일 다운로드 절차로 획득; 스냅샷 날짜를 반드시 보존 |
| 자동승인 API | 활용신청·서비스키가 필요하지만 자동승인으로 안내됨 |
| 로그인/신청 | 별도 로그인, 도엽 신청 또는 제공기관 승인 절차 필요 |
| 협의 | 과거판·법정 원본 등은 담당기관 확인이 필요 |
| 탐색 후보 | 공식 항목은 확인했지만 현재 join에 필요한 필드/coverage를 아직 검증하지 못함 |

## 공식 데이터 카탈로그와 연결 방법

| 우선 | source_id | 공식 데이터·제공처 | 접근/형식 | 공간·시간 연결 | evidence 역할 | 현재 한계 |
|---|---|---|---|---|---|---|
| P0 | `vworld_cadastral` | [VWorld 연속지적도 2.0](https://www.vworld.kr/dev/v4dv_2ddataguide2_s003.do?svcIde=cadastral) | 별도 dataset 신청 없음; 재승인 뒤 대표점 gate 및 257점 bounded snapshot 성공 | 후보/OSM 대표점으로 PNU·cadastral polygon 획득; 공식 주소 정규화는 별도 미완료 | 모든 PNU 기반 사건의 기하 backbone; 정확 중첩을 재현하는 join 재료 | 256 feature·1 `NOT_FOUND`, 고유 PNU 235. 대표필지≠오름경계이며 `r08` FarmMap/VWorld PNU 충돌을 해소하지 않음 |
| P0 | `me_eia_area` | [환경부 환경영향평가 사업구역 WFS](https://www.data.go.kr/data/15142907/openapi.do) | 실제 제주 bbox WFS 성공·13 polygon | `the_geom`, `MGTNO`, `BSNS_OD`, `BSNS_NM`, 등록·수정일과 변화 footprint/기간 교차 | 공식 사업구역과 시점을 잇는 핵심 원인 후보 | 14후보 직접 중첩 0. 평가대상 사업만 포함하며 실제 착공·완공을 뜻하지 않음 |
| P0 | `buildinghub_permit` | [국토교통부 BuildingHUB 건축인허가](https://www.data.go.kr/data/15136267/openapi.do) | 제주 45 법정동·2023–2026·111 page·8,794행 수집; [월별 원시자료](https://www.hub.go.kr/portal/opn/lps/idx-lgcpt-pvsn-srvc-list.do?pageIndex=10) 병행 | 대지 PNU와 허가·착공·사용승인일을 관측구간과 교차 | 건축 사건 후보 | 14후보 중 FarmMap PNU가 있는 `r08`은 같은 법정동 77행이나 exact PNU 0. 주소/법정동 일치는 D/U 문맥 |
| P0 | `forest_private_ops` | 산림청 사유림업무지원 [사업개별소재지](https://www.data.go.kr/data/15120886/fileData.do) + [사업별 작업종](https://www.data.go.kr/data/15134337/fileData.do) + [수종계획](https://www.data.go.kr/data/15120694/fileData.do) | 파일 CSV | `사유림사업번호 + PNU`로 소재지·설계/시공/감리·작업종·수종계획을 연결하고 사업연도/일정을 추가 결합 | 벌채처럼 보이는 변화를 합법적 숲가꾸기·조림과 구분할 강한 원인 후보 | 개별소재지 표만으로 정확 실행일을 확정할 수 없음. 관련 사업 기본/실적표와 날짜 coverage audit 필요 |
| P0 | `molit_dev_permit` | [국토교통부 개발행위허가](https://www.data.go.kr/data/15021109/fileData.do) | 파일 CSV | PNU, 지목, 면적, 용도지역, 신청·허가일, 개발유형·목적을 parcel/change 기간과 결합 | 개발 사건 후보 | 현재 보유 snapshot은 제주 240행이나 2023·2024 행이 0이고 2025도 4행뿐. no-match는 음성 증거가 아님 |
| P1 | `mafra_farmmap_jeju` | [농림축산식품부 2025 제주 팜맵 SHP](https://www.data.go.kr/data/15104491/fileData.do), [전국 팜맵 API](https://www.data.go.kr/data/15057368/openapi.do) | 파일 SHP / 자동승인 API; **파일 ingest 완료** | 필지 경계·19자리 PNU·영상연도·변경/판독 코드와 변화 footprint 교차 | 농경지 경계·변경 상태를 독립 확인하는 B급 상태 근거 | 289,379 polygon 중 PNU placeholder 12건. `r08` 관측은 변화 전 503일이라 baseline일 뿐이며, 법적 효력·원인 시점은 보장하지 않음 |
| P1 | `ngii_aerial` | [국토지리정보원 항공사진](https://www.data.go.kr/data/15059918/fileData.do) | 로그인/검색·신청, TIFF | 주소/좌표/촬영연도로 전후 고해상도 영상을 고르고 같은 footprint를 사람이 판독 | Sentinel 변화의 존재·시기를 검증하는 독립 B급 관측 | 다운로드·승인 절차와 촬영일/구름/도엽 coverage가 불균일; 사진만으로 행정 원인은 확정 못함 |
| P1 | `kma_gk2a_cloud` | [기상청 천리안 2A 구름탐지](https://www.data.go.kr/data/15077314/openapi.do) | 최신 2 km grid 127,040값 성공; 과거 OlmoEarth 6일은 최근 2일 제한으로 실패 | 관측시각·격자를 Sentinel scene 시간/footprint와 결합 | 현재/향후 scene 품질 보조 | 이 endpoint로 역사 common-mode failure를 소급 감사할 수 없음; 2 km라 SCL 대체도 불가 |
| P1 | `me_landcover` | [환경공간정보서비스 토지피복지도 WMS](https://aid.mcee.go.kr/api/land.do) | 공개 WMS 성공·14후보×2023–2025 42 PNG | 동일 footprint에 대해 제작연도별 class를 비교 | 상태 전이를 독립 확인 | 42장은 모두 연도별 hash가 다르지만 분류·제작 변화일 수 있음. 실제 사건/원인으로 해석 금지 |
| P1 | `forest_type_5k` | [산림청 1:5,000 임상도](https://www.data.go.kr/data/15093362/fileData.do) | 파일 SHP | 수종·영급·경급·수관밀도와 footprint 교차, 제작연도 보존 | 산림 baseline·층화와 산림 상태 확인 | 현장과 다를 수 있다는 공식 주의가 있고 갱신시차 존재. 제3유형 라이선스(출처표시+변경금지) 검토 필요 |
| P1 | `molit_zoning` | [국토교통부 용도지역지구도](https://www.data.go.kr/data/15058773/openapi.do) | VWorld WMS/WFS, 키 필요 | 변화 parcel과 용도지역·지구·구역 polygon 교차 | 허용행위·규제 문맥과 층화 | 현행도는 과거 원인이 아님. 과거 고시일·변경이력 없이는 D급 |
| P1 | `molit_urban_facility` | [국토교통부 도시계획시설도](https://www.data.go.kr/data/15057507/openapi.do) | VWorld WMS/WFS, 키 필요 | 도로·공원·교통·공급·환경·방재 등 계획시설 polygon과 교차 | 인프라 계획 문맥·후속 사건 조회 seed | 계획 지정이 실제 공사를 의미하지 않으며 현행 snapshot만으로 시점 불명 |
| P2 | `molit_land_character` | [국토교통부 토지특성정보](https://www.data.go.kr/data/15123549/openapi.do) | WMS/WFS/속성 API | PNU별 이용상황·고저·형상·도로접면을 parcel에 부착 | 표본 층화·대안 설명·지가모형 공변량 | 원인 사건이나 변화 정답이 아니라 시점이 있는 상태/공변량 |
| P2 | `molit_building_use` | [국토교통부 건물용도 공간정보](https://www.data.go.kr/data/15123458/openapi.do) | WMS/WFS | 공식 건물 geometry·용도와 footprint 교차 | 현재 건축물 존재·용도 문맥 | 현행 건물의 준공시점을 보장하지 않아 BuildingHUB 이력이 필요 |
| P2 | `me_ecological_map` | [환경부 생태자연도·환경공간정보서비스](https://egis.me.go.kr/) | 최신 SHP는 로그인; API 제공; 과거자료는 협의 | footprint와 등급·고시 version을 교차 | 생태가치·영향심각도와 검토 우선순위 | 최신 고시도만으로 변화 원인·시점 판단 불가; 과거판은 국립생태원 협의 필요 |
| P2 | `me_ecvam` | [국토환경성평가지도 Open API](https://ecvam.neins.go.kr/api/apiGuide.do) | 이메일 서비스키, WMS; TIFF 다운로드 가능 | 종합·법제·환경생태 평가 레이어와 footprint 교차, 지도 version 보존 | 보전위험 층화·영향 severity | 최소지표법으로 한 고가치 항목이 최종등급을 결정할 수 있음. 원인/결과변수로 오용 금지 |
| P2 | `kdpa_protected` | [한국보호지역 통합DB관리시스템](https://www.kdpa.kr/) | SHP/Excel | 보호지역 polygon, 지정연도/WDPA version과 footprint 교차 | 보호상태·법적 검토 우선순위 | 사이트 자체가 법정 고시와 차이가 날 수 있음을 고지. A급 법적 판단은 원 고시 재확인 필요 |
| P2 | `heritage_spatial` | [국가유산청 지정유산 공간정보 SHP](https://www.data.go.kr/data/15148507/fileData.do) | 파일 SHP | 지정유산·보호/규제 공간과 footprint 교차 | 문화유산 영향·규제 위험 flag | 지정·규제 유효시점과 원 고시 확인 전에는 원인 근거가 아님 |
| P2 | `kma_surface_weather` | [기상청 ASOS 일자료](https://www.data.go.kr/data/15059093/openapi.do) + [관측지점정보](https://www.data.go.kr/data/15139439/openapi.do) | 자동승인 API | 가장 가까운 지점의 좌표·이전 이력과 날짜별 강수·습도·기온을 scene에 연결 | 계절·수분·대기 교란 공변량 | 점 관측의 공간대표성이 낮고 지점 이전 가능. 토지변화 원인이 아님 |
| P2 | `forest_fire_event` | [산림청 산불발생정보 API](https://www.data.go.kr/data/3070842/openapi.do) | 탐색 후보, API | 발생일·위치 정밀도·피해면적 필드를 먼저 확인한 뒤 footprint/기간 교차 | 화재 원인 후보 | 현재 공식 설명만으로 모든 사건의 좌표 정밀도와 completeness 미확인; schema audit 전 D/U급 |
| P2 | `jeju_forest_use_aggregate` | [제주시 산지이용지정현황](https://www.data.go.kr/data/15056266/fileData.do) | 파일 CSV, 연간 시계열 | 연도별 건수·면적을 제주 전체 denominator로 연결 | parcel no-match의 누락·제도차이를 감사하는 coverage 근거 | 개별 필지/오름 join 불가. 개발행위허가와 행정 개념도 달라 직접 대조 불가 |
| P3 | `forest_ops_legacy` | 산림청 디지털산림경영 [숲가꾸기](https://www.data.go.kr/data/15110958/fileData.do) / [조림](https://www.data.go.kr/data/15110957/fileData.do) | 파일 | PNU·사업연도·착수/종료일·작업종을 과거 변화와 연결 | 과거 control/방법 검증 | 2015–2020 중심이며 국유림 제외 등 모집단 제한. 2023–2026 원인판정용 아님 |
| P3 | `seogwipo_building_fallback` | [서귀포시 건축착공허가 현황](https://www.data.go.kr/dataset/15024938/openapi.do) | 지역 API | 지번주소·건축용도·착공처리일을 국가 BuildingHUB 결과와 대조 | 국가 API의 지역 coverage audit·fallback | 서귀포시/분기자료로 범위 제한; 국가자료와 중복 제거 필요 |

## 368개 레지스트리에 넣을 join contract

### 1. 객체를 분리한다

```text
oreum_registry
  oreum_id, official_name, official_address, official_list_date,
  representative_pnu?, official_boundary_status

observation_event
  change_id, oreum_id?, change_geom, t_before, t_after,
  sentinel_scene_ids, scl_quality, gk2a_quality, model_version

parcel_anchor
  pnu, parcel_geom, cadastral_snapshot, address_normalization_method,
  representative_only

administrative_event
  source_id, record_id, event_type, pnu?, event_geom?,
  event_date_start?, event_date_end?, permit_date?, status

context_state
  source_id, record_id, state_class, state_geom,
  valid_from?, valid_to?, snapshot_date
```

현재 오름 레지스트리의 OSM peak point는 C급 위치 seed다. 공식 주소에서 얻은 한 PNU도
`representative_only=true`로 시작한다. 실제 오름 경계 또는 변화 segmentation polygon이 없으면
점 buffer는 후보 API 조회 범위를 만드는 데만 쓰며 B급 경계 중첩으로 승격하지 않는다.

### 2. 공간과 시간을 동시에 맞춘다

```text
spatial_match = ST_Intersects(change_geom, official_event_or_parcel_geom)
temporal_match = official_event_interval overlaps
                 [t_before - tolerance, t_after + tolerance]
```

- `tolerance`는 결과를 보기 전에 고정한다. 첫 실험은 90일을 후보로 하되, 허가→착공 지연을
  고려해 30/90/180일 민감도 분석을 별도 보고한다.
- exact PNU match와 polygon intersection은 분리 저장한다. 같은 행정리 또는 최근접 거리는
  `candidate_discovery`이며 원인 match가 아니다.
- 면적 겹침률(`intersection_area/change_area`, `intersection_area/event_area`)과 경계까지 거리도
  원값으로 보존해 임계값을 사후 변경해도 재현 가능하게 한다.

### 3. evidence edge를 명시적으로 저장한다

```text
evidence_edge(
  edge_id, oreum_id, change_id, source_id, source_record_id,
  relation, spatial_method, intersection_area_m2, distance_m,
  temporal_method, day_gap, evidence_grade,
  no_match_interpretable, reviewer_decision, created_at
)
```

| 출력 등급 | 최소 조건 | 허용 주장 |
|---|---|---|
| A | 공식 원 레코드 자체와 식별자·version 확보 | 해당 행정기록/지정이 존재함 |
| B | 공식 polygon/PNU가 변화 footprint와 교차하고 시점도 부합하거나, 독립된 날짜별 상태 관측이 부합 | 공간·시간상 일치하는 공식 근거가 있음 |
| D | 주소/행정리/거리·현행 레이어만 부합 | 후속 조회 단서가 있음 |
| U | 키·과거판·coverage·schema 부족 또는 no-match 해석 불가 | 판단 보류 |
| M | OlmoEarth score/합의 | 조사 우선순위만 제시 |

**원인 확정**은 A/B edge 하나만으로 자동 생성하지 않는다. `공식 사건 B + 독립 상태관측 B +
사람 검수`가 모이고, 대안 사건도 검토됐을 때 `cause_supported`로 보낸다. 환경·문화유산·보호지역
중첩은 `impact_priority`를 바꿀 뿐 원인 label을 바꾸지 않는다.

### 4. 데이터 누락 자체를 first-class record로 만든다

각 수집마다 다음 manifest를 남긴다.

```text
source_manifest(
  source_id, retrieval_uri, fetched_at, snapshot_date,
  valid_from, valid_to, spatial_coverage, population_definition,
  crs, geometry_type, license, request_hash, schema_hash,
  row_count, jeju_row_count, coverage_audit_status
)
```

`no_match_interpretable=true`가 되려면 다음 네 조건을 모두 만족해야 한다.

1. 후보의 위치와 관측기간이 출처의 공식 coverage 안에 있다.
2. 찾는 행위유형이 그 데이터의 모집단에 포함된다.
3. 스냅샷·API pagination·폐기/변경 이력을 포함해 수집이 완전하다.
4. PNU/geometry와 날짜 필드가 null이 아니며 join 오류 검사를 통과했다.

한 조건이라도 실패하면 no-match는 U다. 특히 현재 국토부 개발행위허가 snapshot의 제주 행은
2023·2024가 0인 반면, 제주시 산지이용지정현황은 별도 행정 개념이지만 2023년 714건·230.6 ha,
2024년 542건·74.2 ha의 활동을 보고한다. 둘을 같은 허가로 비교할 수는 없지만, “어느 한 시스템의
0건”을 제주 행정활동 없음으로 일반화할 수 없다는 coverage 경보다.

## 먼저 구현할 세 패키지

### P0-A — Geometry spine

- 완료: 등록 domain을 고정하고 대표점 `status=OK` 뒤에만 후보 14+오름점 243으로 확장했다.
- 완료: 256 feature·고유 PNU 235를 raw/hash와 함께 저장하고 `NOT_FOUND` 1건을 coverage 누락으로 남겼다.
- 다음: 공식 주소 368건을 법정동 코드·`산` 여부·본번/부번으로 정규화하고 대표점 PNU와 대조한다.
- 다음: `r08`의 dated FarmMap/current VWorld PNU 충돌과 공유 대형필지 14개를 geometry/version으로 감사한다.
- 성공 기준: 대표필지와 오름경계를 분리하고 PNU exact/conflict/unresolved coverage를 368 분모로 보고.

### P0-B — Administrative event stack

- 환경영향평가 WFS, BuildingHUB, 사유림 사업 3표, 개발행위허가를 공통
  `administrative_event`로 정규화한다.
- 변화 footprint와 공간교차 + 30/90/180일 시간 민감도를 계산한다.
- 성공 기준: 사건별 모집단·snapshot을 manifest에 남기고, no-match 가능/불가능 비율을 별도 보고.

### P0-C — Independent validation stack

- 2025 제주 팜맵을 먼저 내려받아 변화 후보·오름 대표필지와 교차한다.
- 국토지리정보원 전후 항공사진 표본을 사람 blind review로 판독한다.
- 천리안 2A와 Sentinel SCL을 결합해 cloud common-mode failure를 품질 gate로 차단한다.
- 성공 기준: 모델 score를 보지 않고 판독한 독립 상태근거와 scene 품질 판정이 남음.

#### P0-C 1차 구현 상태 — FarmMap

- 원본 ZIP의 두 SHP를 EPSG:5179와 EUC-KR DBF로 오프라인 처리했다. 제주시 156,122건,
  서귀포시 133,257건, 합계 289,379건이며 밭 146,482·과수 74,446·비경지 34,799·시설
  33,539·논 113건이다.
- 변화 후보 4좌표와 OSM 오름점 243개를 공간격자 후보축소 뒤 polygon `covers`로 결합했다.
  변화 후보는 `r08` 1건 B, 오름점은 7건 C이며 point miss는 비농지/무변화 음성 증거가 아니다.
- 상태시점은 행정 갱신일보다 실제 관측에 가까운 `FLIGHT_YMD`를 우선한다. `r08`은 밭,
  PNU `5013025324202000000`, 항공 2022-12-30, 갱신 2023-12-08이고 변화 관측구간은
  2024-05-16→2025-05-13이다. 503일 이전 baseline이라 `official_pre_change_state_at_point`로
  기록했다.
- FarmMap과 보유 개발행위허가의 exact PNU 연결은 206행·50개 PNU·144개 FarmMap polygon이다.
  이는 필지 단위 교차출처 문맥일 뿐 사건일과 변화 footprint가 함께 맞지 않아 원인으로 쓰지 않는다.
- 모든 원본·schema·파생파일 hash를 manifest에 고정했다. 17개 단위/통합 테스트, Ruff,
  `compileall`을 통과했고 같은 입력·고정 retrieval time의 두 실행에서 8개 출력이 byte-identical했다.
- API 후속: BuildingHUB 8,794행·EIA 13 polygon·토지피복 42장은 수집됐고 기존 PNU 58개 중
  BuildingHUB exact PNU 9개가 확인됐다. 그러나 14후보의 A/B급 corroboration은 0이다.
- 미완료: 항공사진 blind review, P0-A 지적 geometry, 사유림 사건은 아직 없다. GK2A 최신 grid는
  받았지만 역사 6관측일은 조회 제한으로 실패했다. 따라서 원인 판정은 생성하지 않는다.

## 논문 실험으로 바꾸는 방법

데이터를 붙인 뒤 성능 한 숫자를 보고하지 않는다. 다음 **증거-source ablation**을 고정한다.

```text
M0  OlmoEarth + OSM point
M1  M0 + official PNU/cadastral geometry
M2  M1 + FarmMap/land-cover/forest-state observations
M3  M2 + permits/EIA/forest administrative events
M4  M3 + aerial/field independent review
```

각 단계에서 368개 고정 분모로 다음을 측정한다.

- queryable coverage, time-aligned coverage, `no_match_interpretable` rate
- investigate / decide / abstain 비율
- 사람 판정 대비 selective risk와 coverage–risk 곡선
- 출처 하나를 뺐을 때 결정이 바뀌는 수와 잘못 강해진 주장 수
- 오름 유형·행정구역·토지피복별 침묵률 차이(누락 편향)
- API·사람검수·GPU 비용과 후보 1건당 검증시간

이 설계의 박사급 질문은 “한국 지도를 많이 붙이면 좋아지는가?”가 아니라 다음이다.

> 공식 근거가 불완전하고 서로 다른 행정 시스템에 흩어져 있을 때, Earth foundation model은
> 어떤 변화만 안전하게 말해야 하며 데이터 누락은 그 선택적 보류를 공간적으로 어떻게 편향시키는가?
