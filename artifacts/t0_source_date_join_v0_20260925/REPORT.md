# T0 원 날짜 metadata 감사

검사 행 7000개, 정확 metadata 조인 7000개. 전체 계약 통과: True.

이는 원저자 source_date와 공개 GEO-Bench 슬롯 계약의 조인이다. 현재 tortilla 생성의 exact revision, 원 영상 값 대응, Sentinel 제품 UTC 취득시각을 검증한 것은 아니다.

모든 원 행을 보존하고 event/AOI/전체사각형이 정확히 같은 유일 후보에만 날짜를 연결했다. 원 필드가 다른 후보, 중복·누락·invalid는 summary와 joined_rows에 남긴다. 최근접·중심점·비율 기반 fallback은 없다.

날짜는 day precision이며 source-minus-synthetic은 사건일−24/−12/0의 날짜와 비교한다. gap 분포는 개별 타일 가중이며 split/event별 값을 같이 제공한다. 경험적 분위수는 (n−1)p 선형 보간이다.

시간순 flag는 진단값이며 pre 영상의 무침수나 변화시점·인과성을 입증하지 않는다. 모델/embedding/E5 결과를 읽거나 기존 입력·라벨·실험을 수정하지 않았다.
