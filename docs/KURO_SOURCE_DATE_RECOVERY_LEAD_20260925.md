# 실제 취득일 복구를 위한 확인된 경로와 남은 연결

> 후속: [원저자 날짜 metadata 복구](KURO_DATE_RECOVERY_FEASIBILITY_20260925.md)에서 두 타일의 정확한 grid 조인과 공개 슬롯 매핑을 확인했다. 원 센서 픽셀 대응·현재 archive 생성 revision은 미검증이며 기존 export의 null 날짜는 보존한다. 아래는 그 이전 확인 단계의 기록이다.

2026-09-25. 두 선택 타일의 실제 취득일을 복구한 결과가 아니다. E5 입력·날짜·encoder 캐시는 변경하지 않는다.

[원저자 KuroSiwo catalogue 코드](https://github.com/Orion-AI-Lab/KuroSiwo/blob/main/catalogue/catalogue.py)의 catalogue schema에는 `grid_id`, `flood_date`와 별도로 `source_date`, `s1_ids`, `master`, `crank`가 있다. 해당 코드는 GeoPackage를 읽고 날짜·센서 ID를 파싱한다. 따라서 원 카탈로그에 정확한 grid를 연결할 수 있다면, 사건 날짜보다 구체적인 관측 출처를 추적할 후보 경로가 있다. 이것은 코드 구조에 근거한 추론이며 선택 타일과의 연결은 미검증이다.

[저자 GeoTIFF 배포 카드](https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-GeoTIFFs)는 각 표본에 `info.json` 메타데이터가 있다고 설명한다. 전체 배포는 매우 크므로 지금 전체 데이터를 받지 않는다. 현재 서버 TIFF와 sample/source rows를 먼저 복구해 grid ID·좌표·source fields를 확인한다. 그다음 정확한 원 표본 metadata만 연결하는 방법을 검토한다.

확인 순서: 현재 두 타일의 원 row와 TIFF metadata → 원 grid/GeoPackage 또는 info.json 일치 → source_date/Sentinel-1 product ID와 슬롯 순서 검증 → 검증한 경우에만 별도 후속 데이터 계약에 실제 날짜 사용. 사건일−12/−24를 관측일로 바꾸어 쓰지 않는다. 날짜가 복구돼도 pre 영상의 무침수 여부는 별도의 관측·라벨 검증이다.
