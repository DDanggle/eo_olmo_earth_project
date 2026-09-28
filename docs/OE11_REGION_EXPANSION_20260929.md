# OE11: 훈련 지역·입력 확대와 위치별 교정 비교 구현

2026-09-29 수동 실행 기록. 겨울밀 복구는 선행 조건에서 제외했다. 기존 P2 결과·상한은 보존하고 예약은 중지 상태다. 이번 기록은 새 입력과 코드의 실제 실행에 관한 것이며 새 성능 결과는 아직 없다.

## 실제 확보한 입력

정상으로 검증된 기존 PASTIS shard0000의 메타데이터에서 원래80개를 모두 제외했다. 후보649개 중 서로 다른 분할 중심점 거리가2.2km 미만인2개를 제외하고, 훈련/교정 bank는 사전 고정한 네 클래스의 균형 규칙, 내부검증은 hash 규칙으로288개를 선정했다. 훈련·bank·내부검증을10km 좌표 블록으로 나누고 최종 거리 관문은 haversine으로 계산했다. 정확한 필지 경계 독립성까지 증명한 것은 아니다.

| 준비 항목 | 이전 | 이번 |
|---|---:|---:|
| 훈련 지역 | 2 | 3 |
| 훈련 query 영상 | 48 | 192 |
| 분리된 교정 bank 영상 | 16 | 48 |
| 개발 검증 영상 | 16, 한 지역 | 48, 세 source 지역의 다른 블록 |
| 패킷당 준비한 관측 | 8 | 8 |

세 부모지역 t31tfj/t32ulu/t31tfm마다 훈련64·bank16·내부검증16개다. t31tfm은 이전 개발지역을 새 source 지역으로 편입한 것이며 이후 이 지역 점수를 미관찰 지역 전이라고 부르지 않는다. 기존80개 patch ID는 새 묶음에 포함되지 않았다. t30uxv의182개 metadata행은 건너뛰고 그 지역 영상/정답 payload는 열지 않았다. 사전학습·과거 프로젝트의 완전 미노출 여부는 별도 감사 대상이다.

| 대상이 존재하는 훈련 영상 수 | 기존48개 | 새192개 |
|---|---:|---:|
| 초지 | 47 | 185 |
| 옥수수 | 23 | 107 |
| 포도밭 | 16 | 66 |
| 사료용 콩과 | 18 | 99 |

서로 같은 영상에 여러 대상이 있어 합계는192를 넘는다. 이는 class 존재 수이며 적격 support 객체 수와 다르다. 기존 겨울밀 수치와 새 네 대상 평균은 직접 연결하지 않는다.

CPU 추출은229.13초에288/288개 완료했다. 별도 코드로576개 입력/정답 파일의 해시, 정규화의 정확한 재계산, 밴드/날짜/라벨 계약, 원80개 제외와 분할 간 거리를 검산했다. 선택 패킷의 분할 간 최소 중심점 거리는2553.06m다. 검토 묶음18파일26,033,557bytes를 nx로 회수해 전부 해시 대조했다. 검증은 저장 packet 변환 검사이며 원 tortilla payload를 독립 구현으로 전량 다시 읽은 검사는 아니다.

## 학습 목록과 실제 로더까지 연결한 범위

네 대상(초지·옥수수·포도밭·사료용 콩과)의12개 방향쌍과K=1/2/4/8을 모두 구성했다. 같은 query/방향쌍/K/episode ID를 공유하는 두 목록이다. 각 목록은훈련9,216행=192영상×12쌍×4K, 내부검증2,304행=48영상×12쌍×4K다. 행 수를 독립 영상 또는 사람 교정 수로 세지 않는다.

- `catalog_pooled_v0`: query 자신을 제외한 source 훈련 영상에서 support를 선택한다.
- `catalog_cross_parent_v0`: 훈련 때 query와 다른 부모지역의 support만 선택한다.
- 두 목록의 내부검증 support와 목록 SHA는 완전히 동일하다. 내부검증은 공통 bank를 쓰며 새 지역 전이 평가가 아니다. 같은 episode ID라도 훈련 support가 다르므로 목록 SHA까지 실행 identity에 넣는다.

감사 v1은 총23,040행의 query×쌍×K 완전성, 역할 분리, cross-parent 제한과 실제 참조 mask 해시를 검사했다. 별도 실제 로더는 두 목록×train/development×K1/K8×세 부모지역×query2/4/8관측=72회 통과했다. 입력을 읽을 때 query 정답은 읽지 않았고, 이후 선택 episode24개에서만 명시적으로 훈련/평가 target 읽기를 따로 검사했다. 원 manifest를 변경하지 않았다. 로더가4/8query 관측을 읽을 수 있다는 것은 현재 모델이 이를 처리한다는 뜻이 아니다. 현재 encoder wrapper는query2관측이다.

기존 로더가 `dev_query`와 평면 mask 경로를 요구해 새 `calibration`·분할별 mask 경로와 충돌했다. 별도 adapter에서 검증용 사본에만 별칭을 적용하고, 실제 읽기·해시는 원본을 사용하게 했다. 합성8검사와 실제72회 검사를 통과했다. 감사의 초기안은 smoke 상태만 확인해 이전 receipt가 들어갈 여지가 있었고, 독립 검토 후 v1에서 현재 manifest/catalog SHA와 묶고 Cartesian 조합 전체를 대조하도록 보완했다. 감사 v0은 실행하지 않았으며 v1을 최종 근거로 쓴다. 검토20파일8,196,395bytes를 회수해 전부 해시 대조했다.

[24개 언어 입력](../code/oe11_context_v0/contexts.json)도 생성했다. 12개 방향쌍×이름만/이름+설명이며 실제 v2 입력 계약을 모두 통과하고 재생성 결과가 동일했다. 기존 농업 카드의 문장을 그대로 썼고 출처·클래스·파일 ID는 모델 문구 밖의 provenance에 둔다. 이는 공개 농업 지식이며 필지별 지역/기상 정보나 실제 전문가 교정이 아니다. 고정된 설명이 클래스 식별자로만 작동할 수 있어 재서술·틀린 설명·학습된 익명 코드 대조가 별도로 필요하다. 현재 사람 응답0이다.

## 실제 구현한 연결부

[새 코드](../code/oe11_matching_v0/README.md)는 query 위치와 역할별 언어로 개별 양성/혼동 객체를 비교한다. 기존 날짜 평균과 query2/support8 입력은 유지한다. 역할마다 고정 reader로 텍스트 표현을 얻으며 이름/설명 및 EO 고정/공동학습을 같은 head에 연결할 수 있다.

초안의 양성 점수−혼동 점수만으로는 역할을 뒤집었을 때 확률이 보완 관계가 되어 둘 다 없는 곳을 표현하지 못했다. 실행 전 검토에서 이를 발견해 query의 배경 점수를 추가했다. 현재 positive logit은 양성 점수−logsumexp(혼동 점수, 배경 점수)다. 양쪽 모두 부재인 사례의 음성 출력과 배경 기울기를 CPU 검사에 포함했다.

실제 PyTorch 모듈·대체 EO/reader를 쓴 CPU 검사13/13 통과. 텍스트와 영상에 따른 객체 선택 변화, 모든K, 역할/객체 순서, mask 손실의 EO 경로, frozen 경로, 저장복원과 입력 경계를 확인했다. 실제 OlmoEarth/Qwen 가중치의 신규 forward·optimizer 학습은 아직 수행하지 않았다. 역할별 reader prefill이2회라 실제 토큰·비용 검사가 필요하다. 이 코드 자체를 새 원리나 성능 상승으로 주장하지 않는다.

## 영상 확인으로 남은 구체적 문제

세 지역의 훈련 사례를 사전 hash 규칙으로3개씩 그렸다. 직접 확인한 page00/page02에서 구름·연무가 보였고, t31tfm 일부 사례는 초기query 위치2/5에도 구름이 심했다. 결측 sentinel이 없다는 것과 구름 없이 판독 가능하다는 것은 다르다. 영상/날짜를 사후 제외하지 않았다. 동일한 새 데이터에서2/4/8query 관측을 비교할 실질적 근거다. 관측 확대를 언어/구조 고유 효과로 혼합하지 않고 모든 대조군에 동일한 관측을 제공한다.

분포 제약도 기록한다. t32ulu 내부검증에는 포도밭 양성이0개이며, t31tfj 옥수수 양성1개와 t31tfm 포도밭 양성2개는 작다. 초지는 내부검증48개 모두에 존재해 대상 부재 오탐을 여기서 평가할 수 없다. 지역별bank의K8 지원도 일부 클래스에 부족하다. 세 지역 bank를 합친 지원과 각 지역 안의 지원을 구별해야 한다. 이 개발 묶음은 충분한 독립 지역 평가를 대체하지 않는다.

## 다음 실행 순서

원본 OlmoEarth의 같은 가중치에서 시작한다. 이전 P2 checkpoint의 추가 학습 노출을 새 내부검증으로 가져오지 않는다. 먼저 이번 네 대상에서 공통 head의 이름만/설명 포함을 비교하고, EO 고정/갱신 대조를 둔다. 학습 목록에서는 지역을 섞은 support와 query와 다른 부모지역의 support만 쓰는 조건을 별도로 만든다. 같은 query·K·마스크 정답을 사용하면서 교정의 지역이 바뀌었을 때 학습 효과를 비교한다. 지역별 정보 부족은 목록의 K 지원 가능성에 표시한다.

구체적인 순서는 다음과 같다.

1. 이번 네 대상의 고정 train 사례로 실제 가중치 연결·양성/혼동/둘 다 부재 학습과 새 프로세스 optimizer/RNG 재개를 확인한다. 충분한 기준선이 학습되는지 먼저 확인한다. 시작점은 원본 OlmoEarth이며 기존 P2를 재개하지 않는다.
2. 같은 입력·개별 객체 비교 head에서 EO 고정/갱신×이름/설명의 네 조건(E0–E3)을 비교한다. 설명의 추가 이점은 E3−E2, EO 갱신의 이점은 E3−E1이다. 평균 support head는 구조 기여를 분리하는 대조다. 충분 학습 비교의 업데이트·학습 충분성·동일 튜닝 기회·평가 가중치는 처리량 pilot 이후 결과 전에 별도 고정한다.
3. 같은 설명과 구조에서 pooled/cross_parent 목록만 바꿔 지역을 넘는 교정 예시로 훈련한 효과를 비교한다. 같은 내부검증 목록이므로 표본 차이가 효과로 섞이지 않는다. 진짜 새 지역의 주장은 이후 남겨 둔 지역 또는 외부 지역의 별도 고정 평가로 검증한다.
4. 구름 때문에 날짜2개만으로 정보가 부족한지를 query2/4/8관측 공통 대조로 확인한다. 이를 언어 효과와 합치지 않는다. 모델의 추가 날짜 경로는 아직 구현 전이다.

새 GPU 실행 전 실제 가중치·trainer·새 프로세스 optimizer/RNG 재개와 고정 학습 조건을 확인해야 한다. 이전2시간 pilot 장부는48초 사용 상태로 보존했다. 예약을 재개하지 않았다.

## 파일 위치

- 서버 전체 입력: `/home/work/data/olmoearth/oe11_source_expand_v0/prepared_v0`
- [로컬 검토 입력·검증](../artifacts/oe11_region_expansion_20260929/review_v0/independent_packet_verification.json)
- [지역·대상 가용표](../artifacts/oe11_region_expansion_20260929/review_v0/availability_summary.json)
- [새 지역 영상판](../artifacts/oe11_region_expansion_20260929/review_v0/qa/page_02.png)
- [입력 추출 코드](../code/oe11_source_expand_v0/prepare_regions.py) · [추출 전 고정 정책](../code/oe11_source_expand_v0/selection_policy.json)
- [독립 packet 검증 코드](../code/oe11_source_review_v0/verify_export.py)
- [새 matching head](../code/oe11_matching_v0/matching_head.py) · [CPU 검사 결과](../code/oe11_matching_v0/cpu_test_receipt.json)
- [외부 지역 자료 조사](../artifacts/oe11_region_expansion_20260929/external_source_review.md): TimeMatch 프랑스30TXT/31TCJ는 구체적 후보지만 pixel-set과dense patch의 차이, meadow/fodder/포도 라벨 재매핑이 있어 바로 합치지 않았다. 이 메모의 기존2지역 유지 제안과 달리 이번 실행은t31tfm source 편입을 명시하고 진행했다.

- [통합 목록 감사·실제 로더 결과](../artifacts/oe11_region_expansion_20260929/catalog_review_v1/catalog_verification.json) · [72회 실제 입력 검사](../artifacts/oe11_region_expansion_20260929/catalog_review_v1/loader_smoke_v0.json)
- [새 목록 생성기](../code/oe11_episode_catalog_v0/episode_catalog.py) · [호환 로더](../code/oe11_loader_v0/loader_adapter.py) · [최종 목록 감사 v1](../code/oe11_catalog_review_v1/verify_export.py)
- 서버 목록: `/home/work/data/olmoearth/oe11_source_expand_v0/catalog_{pooled,cross_parent}_v0`
- [언어 입력·출처 기록](../code/oe11_context_v0/README.md)

## 후속 실제 학습 실행 준비

09/29 01:30 KST: [GPU 검증 준비 기록](OE11_GPU_VALIDATION_PREPARATION_20260929.md)에 새 trainer·fresh-process 재개 검사·고정48단계 조건·실제 파일 사전 검사를 완료했다. CPU에서는 통과했고 GPU0/1 모두 점유로 실제 GPU 학습은 아직 시작하지 않았다.
