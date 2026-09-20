#!/usr/bin/env python3
"""OlmoEarth Studio 기획서(플로우 스펙) HTML 생성기.
스크린샷 선별 목록(MANIFEST)을 sips로 축소(JPEG)해 spec_assets/에 두고, 플로우별 스트립+프레임 카드로 묶은 HTML 한 장을 쓴다.
  --img-prefix  HTML에서 이미지를 참조할 경로 접두 (로컬: ../artifacts/studio_audit/spec_assets/, 아티팩트: assets/)
  --out         HTML 출력 경로
"""
import argparse, html, json, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT/"artifacts/studio_audit"; ASSETS = SRC/"spec_assets"
ap = argparse.ArgumentParser(); ap.add_argument("--img-prefix", default="../artifacts/studio_audit/spec_assets/"); ap.add_argument("--out", default=str(ROOT/"docs/OLMOEARTH_STUDIO_FLOWS_2026_09_19.html")); ap.add_argument("--width", type=int, default=1200); ap.add_argument("--skip-convert", action="store_true"); ap.add_argument("--embed", action="store_true", help="이미지를 data URI로 넣어 HTML 한 파일로 만든다"); a = ap.parse_args()

P = "20260919_124813/screenshots/"; Q = "20260919_125146/screenshots/"; R = "20260919_125651/screenshots/"; S = "20260919_130642/screenshots/"
UUID = "e0{n}_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_{rest}"
def U(pref, n, rest): return pref + UUID.format(n=n, rest=rest)

# (flow, step, file, caption, findings)
FLOWS = [
 ("F0", "로그인 → 프로젝트 목록 → 빈 프로젝트", "계정이 처음 만나는 세 화면. 빈 프로젝트가 '무엇부터 하라'고 말하지 않는다 (결함 P0-3).", [
  ("랜딩(비로그인)", "20260919_123213/screenshots/00_landing.png", "올모어스 스튜디오 공개 랜딩. 로그인 모달은 이메일+비밀번호와 Google 두 경로.", []),
  ("프로젝트 목록", P+"e001_projects_start.png", "로그인 직후 /projects. 카드/리스트 뷰 토글, 'Create new project'. Toy project 1개.", []),
  ("Create new project 대화상자", P+"e003_projects_dialog_Create_new_project.png", "이름·설명만 받는다. 데이터 유형·목표 산출물을 묻지 않아 이후 마법사에서 되묻게 된다.", []),
  ("빈 프로젝트 대시보드", P+"e004_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_Toy_project_jeju_No_description_Created_.png", "Datasets/Models/Predictions 빈 카드 3개. '라벨 파일이 있나요?' 한 질문으로 Import vs Create tasks를 갈라줘야 한다.", ["P0-3"]),
  ("Accounts", P+"e019_users_dialog_Accounts.png", "조직 = 개인 1명. 팀·공유 단위가 없다 (P2).", ["P2"]),
 ]),
 ("F1", "프로젝트 정보구조 (사이드바 11개 메뉴)", "모든 역할에 같은 11개 메뉴가 평면으로 노출된다. 어노테이터·분석가·파트너의 뷰가 갈리지 않는다 (P2).", [
  ("Datasets (빈 상태)", P+"e007_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_datasets_Datasets.png", "'Import data' 하나가 유일한 진입. 예제 파일 버튼은 다운로드가 아닌 인라인 모달 (결함 2).", ["P1-예제"]),
  ("Data Viewer (빈 상태)", P+"e008_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_data-viewer_Data_Viewer.png", "지도+레이어+필터. 우측 'Train model / Export'가 '현재 필터 = 학습셋'이라는 뜻인데 숨어 있다 (P1).", ["P1-필터"]),
  ("Models (빈 상태)", P+"e009_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_models_Models.png", "'Build model' 진입. 카드 UI(테이블 아님, 링크 없음 — 자동화 시 확인).", []),
  ("Predictions (빈 상태)", P+"e010_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_predictions_Predictions.png", "모델이 없으면 빈 화면. 용어가 'Run model'과 섞인다 (P1 용어).", ["P1-용어"]),
  ("Map Publisher (빈 상태)", P+"e011_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_map-publisher_Map_Publisher.png", "발행 목록: Title/Description/Image/Access level/State/Model run. 'Publish'는 완료된 예측이 있어야 한다.", []),
  ("Analytics (Beta)", P+"e012_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_analytics_Analytics.png", "Annotation / Workflow 두 탭. 라벨셋 리포트·메타데이터 리포트·어노테이터/리뷰어 리포트.", []),
  ("Tasks (빈 상태)", P+"e013_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_tasks_Tasks.png", "임포트가 곧 태스크가 된다. 지도+Draw filter area, My tasks, Auto assign, Add task.", []),
  ("Areas (빈 상태)", P+"e014_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_areas_Areas.png", "예측을 돌리려면 여기서 Area를 먼저 만들어야 하는데, 어디에도 그 순서가 적혀 있지 않다 (결함 P1-Area).", ["P1-Area"]),
  ("Settings ▸ General", P+"e015_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_settings_Settings.png", "Label geometry(Point/Polygon), 이미지 어노테이션 허용, Task access 3단계, 기본 위성영상.", []),
  ("Settings ▸ Basemaps", Q+"e057_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_settings_tab_basemaps_Basemaps.png", "베이스맵 선택.", []),
  ("Settings ▸ Annotation form builder", Q+"e039_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_settings_tab_form-builder_Annotation_form_builder.png", "어노테이션 필드 정의('Add new annotation field'). 임포트로 생긴 필드와의 관계가 설명되지 않는다.", []),
  ("Settings ▸ Label sets", Q+"e058_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_settings_tab_label-sets_Label_sets.png", "labelset은 데이터셋마다 따로 생긴다 — 결함 4의 뿌리.", ["P1-labelset"]),
  ("Labs", P+"e016_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_labs_Labs.png", "iframe 앱 4개: Annotation Lab, Change Annotator, Hello Labs, Imagery Preloader.", []),
  ("Labs ▸ Annotation Lab", "with_data/lab_annotation-review.png", "큐 구성(데이터셋·태스크 상태·배정·필드·Sentinel-1 SAR)→Start annotating. 사람이 정답 라벨을 만드는 곳 — 다음 루프의 입구.", []),
  ("Labs ▸ Hello Labs", "with_data/lab_hello.png", "JWT→/api/v1/users/me, 내부 개발 가이드(ui/src/labs/CLAUDE.md) 노출 (P1 개발 문구).", ["P1-개발문구"]),
  ("My queue", P+"e017_tasks_My_queue.png", "전 프로젝트 태스크 큐. 어노테이터 역할의 홈이 될 화면.", []),
 ]),
 ("F2", "Import Training Data 마법사", "Select File → Observation Date → (Merge) → Configure New Fields → Confirm. 기본값이 라벨을 버린다 (결함 1, P0).", [
  ("Import data 진입", P+"e006_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_datasets_dialog_Import_data.png", "Datasets의 'Import data'. 지원 형식 CSV/GeoJSON, 예제 파일 모달.", []),
  ("예제: About File Formats", "examples/About_File_Formats.png", "스키마 설명이 모달 안 텍스트뿐. 다운로드·문서 링크 없음 (결함 2).", ["P1-예제"]),
  ("예제: GeoJSON", "examples/GeoJSON_Example.png", "task_name / observation_time / sample_category / sample_number / sample_true_false 규약을 여기서 역추적했다.", []),
  ("S1 Select File", "jeju_import/import_20260919_131828/step01.png", "파일 업로드. 큰 파일(네팔 298 폴리곤)은 파싱 동안 Next가 비활성 — 진행 표시가 없다.", []),
  ("S2 Observation Date", "jeju_import/import_20260919_131828/step02.png", "observation_time 열을 시간으로 인식. 형식 힌트 없음.", []),
  ("S3 Merge with Existing Data", "jeju_import/import_20260919_131828/step03.png", "두 번째 데이터셋부터 등장. Yes(권장)/No — '권장'의 근거가 없다. 우리는 No(별도 데이터셋).", []),
  ("S4 Configure New Fields — 기본값", "jeju_import/import_20260919_131122/step03.png", "'No fields selected'. 그대로 Next를 누르면 라벨 없이 지오메트리만 들어간다 — 결함 1 (P0).", ["P0-1"]),
  ("S4 Configure New Fields — select all 후", "jeju_import/import_20260919_131122/step03_selected.png", "'+ select all'을 눌러야 sample_category 등이 labelset/number/boolean으로 잡힌다.", ["P0-1"]),
  ("S5 Confirm", "jeju_import/import_20260919_131122/step04.png", "Imported / Not Imported 요약. 'Not Imported'가 강조되지 않는다.", ["P0-1"]),
  ("제출 직후", "jeju_import/import_20260919_131828/after_submit.png", "pending → ingesting → completed (~1분).", []),
  ("Datasets 3행", "jeju_import/import_20260919_131828/datasets_after.png", "제주 포인트 243 · 제주 폴리곤 243 · 네팔 폴리곤 298 (모두 completed).", []),
 ]),
 ("F3", "데이터가 들어간 뒤", "임포트 하나로 Dashboard·Data Viewer·Tasks·Analytics가 동시에 채워진다 — 이 연결이 제품의 강점이자 설명되지 않는 부분.", [
  ("Dashboard (데이터 있음)", S+"e001_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_start.png", "지도에 프로젝트 bbox, Datasets 카드 채워짐, Quick Action이 'Build a model'로 바뀜.", []),
  ("Dataset 상세 (제주)", "with_data/dataset_detail_jeju.png", "필드 목록·타입·건수.", []),
  ("Data Viewer — 두 데이터셋", "with_data/data_viewer_two_datasets.png", "제주·네팔 레이어를 한 지도에. Legend 'Color by sample_category'.", []),
  ("Data Viewer — 레이어 패널", "with_data/data_viewer_layers.png", "Datasets 필터·Start/End date·Area. 우측 Train model / Export = '현재 필터로 학습/내보내기'.", ["P1-필터"]),
  ("Data Viewer — 테이블 편집기", "with_data/data_viewer_table.png", "스프레드시트: Find&replace / Remap / Partition train-val-test / Bucket / Convert / Normalize / Concatenate / Split. 강력하지만 발견 불가.", []),
  ("Tasks (임포트 → 태스크)", S+"e017_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_tasks_Tasks.png", "243행, status Reviewed, 태그(52SBB·abstain), View geometry.", []),
  ("Analytics ▸ Labelset report", R+"e038_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_analytics_Labelset_report.png", "필드별 카운트+파이: 제주 scored 153/abstain 90, 네팔 ranked 198/unobservable 100. 라벨이 '기권 플래그'임이 여기서 보인다.", []),
  ("Analytics ▸ Workflow", R+"e003_projects_7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0_analytics_tab_workflow_Workflow_analytics.png", "Annotator / Reviewer 리포트 — 협업 라벨링을 전제한 화면.", []),
 ]),
 ("F4", "Build model 마법사 (5단계)", "S1 모델 → S2 라벨·학습데이터 → S3 공간 창 → S4 시간 카드 → S5 요약·비용. 검증은 전부 제출 뒤에 (결함 3·7).", [
  ("S1 Model", "build_model_full/20260919_131035_summary/s1_model_nano.png", "Fine-tuned / Embeddings, 소스 S2·S1·Landsat, 크기 Nano(1.7M)·Tiny·Small·Base.", []),
  ("S1 Advanced options", "build_model_wizard/20260919_130348/step1_Advanced_options.png", "LR 0.0001, 'Freeze the encoder (train a probe)' — 파인튜닝 vs 프로브 선택이 여기 숨어 있다.", []),
  ("S1 Hacker mode", "build_model_wizard/20260919_130348/step1_Hacker_mode.png", "통합 config 편집기. 투명성은 좋으나 일반 사용자에겐 개발자 문구.", []),
  ("S2 Training data", "build_model_full/20260919_131035_summary/s2_training_window_cls.png", "라벨 필드 → 산출물(Pixel seg / Window cls / Object detection). Point 라벨에 Window 분류를 허용한다 (결함 3).", ["P0-3검증"]),
  ("S2 라벨 드롭다운 (동명 labelset)", "model_jeju_polys/label_options_after_filter.png", "'sample_category (243 annotations)'가 두 번. UUID로만 구분되고 데이터셋 필터가 목록을 안 좁힌다 — 결함 4.", ["P1-labelset"]),
  ("S2 Filter 열기", "build_model_launch/20260919_163242_nepal/s2_filter_open.png", "'Training on all data · 784 annotations' → 'Choose specific datasets'.", []),
  ("S2 데이터셋 자동완성", "build_model_launch/20260919_163242_nepal/s2_dataset_options.png", "MUI Autocomplete. Escape는 마법사 전체를 닫는다.", []),
  ("S2 필터 적용", "build_model_launch/20260919_163242_nepal/s2_filter_set.png", "nepal_rasuwa_studio (298 annotations)만 선택.", []),
  ("S3 Spatial", "build_model_full/20260919_131035_summary/s3_spatial_small_advanced.png", "창 XS160/S320/M640/L1280, patch, overlap.", []),
  ("S4 Temporal — 카드 3종", "build_model_launch/20260919_163242_nepal/s4_temporal.png", "A state(기간) / A condition(전후) / A sighting(한 장). 오늘 날짜를 알면서 미래 창을 경고하지 않는다 — 결함 7.", ["P0-7"]),
  ("S4 Temporal — A state 12개월", "build_model_full/20260919_131144_summary/s4_temporal_state_advanced.png", "1~12월 선택. 라벨 날짜 연도가 올해면 창이 미래로 뻗는다.", ["P0-7"]),
  ("S5 Summary + 비용", "build_model_launch/20260919_163242_nepal/s5_summary.png", "설정 요약, Spatial split 75/25, 'Estimated cost ~1 compute unit · N of 100'. 추정 중엔 Build Model 비활성.", []),
 ]),
 ("F5", "학습 결과 — 6회 시도, 1회 성공", "실패가 제품 결함의 증거다: 제출 후 검증(3·7), 이유 없는 실패(5), 동명 labelset(4), 취소 불가(6).", [
  ("1차 실패 — 검증 에러", "model_failed/model_detail.png", "Point 라벨에 Window 분류 + 미래 창 → 제출 뒤 'Annotation validation failed 243/243'. Error 행이 있다.", ["P0-3검증"]),
  ("모델 목록 (카드)", "model_nepal/models_three.png", "training/failed 상태 칩. 카드에 실패 사유·진행률 없음.", []),
  ("네팔 'A sighting' — 41분 후 failed, 사유 없음", "model_poll/audit-nano-rasuwa-status_153329.png", "Model Info에 Error 행이 없다. 툴팁·카드 메뉴·Dashboard·queue 전부 무언 — 결함 5 (P0). 취소 버튼 없음 — 결함 6.", ["P0-5", "P2-6"]),
  ("네팔 'A condition' — 즉시 failed", "dom_probe_model_detail/172942.png", "'observation window extends to 2026-09-25 (future) 298/298'. 사건 3.5주 전 → 후 1개월 창이 오늘을 넘김 — 결함 7 (P0).", ["P0-7"]),
  ("제주 v2 — ready (Overview)", "model_poll/audit-nano-oreum-polys-cat-v2_155404_tab0_Overview.png", "polys 243, sample_category(scored/abstain), A period of time 12개월, Small 320m, Nano v1.2, Fine-tuned.", []),
  ("제주 v2 — Performance Evaluations", "model_poll/audit-nano-oreum-polys-cat-v2_155404_tab2_Performance_Evaluations.png", "Acc 81.0% / F1 78.9% / val 84창. 다수 기준선 61.9%. 라벨이 기권 플래그라 '구름 판별기'를 배운 것 — 수치는 제품 검증용.", []),
  ("제주 v2 — Predictions 탭", "model_poll/audit-nano-oreum-polys-cat-v2_155404_tab1_Predictions.png", "'No predictions yet' + Run model 활성.", []),
 ]),
 ("F6", "예측 → 발행", "Area 만들기 → Run model → predicting → (완료 후) Map Publisher ▸ Publish. 발행 화면은 파트너가 실제로 손에 쥐는 것.", [
  ("Add area 대화상자", "add_area/163537/01_open.png", "이름 + 지도(Draw polygon / Delete / Upload GeoJSON).", []),
  ("Add area — GeoJSON 업로드 후", "add_area/163537/03_filled.png", "오름 밀집 구역 15개 bbox(≈68 km²)가 제주 동부에 정확히 얹힘. 육안 확인 후 Save.", []),
  ("Areas 목록", "add_area/163537/04_list.png", "jeju_oreum_core_15 1행.", []),
  ("Run model 대화상자", "run_model/172743/01_dialog.png", "Start date(월) / End date 잠김(학습 기간 12개월) / Select existing area(s). Area 없으면 버튼 비활성.", ["P1-Area", "P2-기간"]),
  ("Run model — Area 선택 후", "run_model/172743/02_area_selected.png", "Total area 68.20 km² · Estimated cost ~1 compute unit. Run name 필수 표시지만 비워도 자동 이름으로 실행됨.", []),
  ("실행 직후", "run_model/172743/03_after_run.png", "행 생성: 모델--01-01-2025--12-31-2025--영역, 상태 pending → predicting.", []),
  ("Map Publisher (발행 전)", "dom_probe_mappub/163630.png", "Publish 버튼. 완료된 model run이 있어야 한다.", []),
  ("예측 completed (44분, 1 unit)", "predictions_poll/final_completed_181231.png", "행을 눌러도 아무 일도 없다. 결과를 보는 곳은 여기가 아니라 Data Viewer ▸ Layers ▸ Predictions — 안내가 없다.", ["P1-결과위치"]),
  ("Data Viewer ▸ Predictions 레이어", "dataviewer_predictions/233002/05_zoomed.png", "'Zoom to prediction'을 눌러야 결과 위치로 온다. 68 km² Area 전체가 단색 'abstain' 한 덩어리.", ["P0-8"]),
  ("결과 파일 = feature 1개", "dataviewer_predictions/233002/04_map_settled.png", "Download prediction results → zip 안 result.geojson 915바이트: Area bbox 1개, sample_category_2='abstain', 확률 필드 없음. Window 분류를 Area에 돌리면 Area당 라벨 하나 — 어디에도 경고가 없다 (결함 8).", ["P0-8"]),
  ("Publish 대화상자 (입력 후)", "publish/232928/02_filled.png", "Runs·Legend 표시명·Global map view·Allow feedback·Analytics 패널·Title/Description·Narration(Markdown)·Access level(Public/Restricted)·'Publish to OlmoEarth Viewer' 토글(기본 꺼짐).", []),
  ("Map Publisher — 저장 후 (State preview)", "publish/233033/04_list.png", "Restricted + 토글 꺼짐으로 Save → State 'preview', 토스트 'Viewer config created successfully'. Public 전환은 사람이 결정하도록 남겼다.", []),
  ("행 Actions", "dialog_probe_mappub_actions/233131/01_open.png", "Edit / Preview / Delete. 'Publish/Unpublish' 상태 전환이 메뉴에 없고 Edit 안의 토글에 있다.", ["P1-용어"]),
  ("OlmoEarth Viewer (파트너가 보는 화면)", "dom_probe_viewer/233130.png", "제목·Layers·Basemaps·Inferred data·Legend. 결과 위치로 자동 이동. 단순하고 좋다 — 다만 보여줄 것이 단색 사각형 하나뿐이라는 게 문제.", ["P0-8"]),
 ]),
]

FINDINGS = [
 ("P0-1", "P0", "Import가 기본으로 라벨을 버린다", "Configure New Fields 기본값 'No fields selected'. '+ select all'을 모르면 지오메트리만 들어간다.", "라벨 후보 기본 선택, Confirm에서 Not Imported 강조"),
 ("P0-3", "P0", "빈 프로젝트 첫 실행 가이드 없음", "빈 카드 3개. Import vs Create tasks 분기 질문이 없다.", "'라벨 파일이 있나요?' 한 질문 분기"),
 ("P0-3검증", "P0", "학습 검증이 제출 뒤에만", "Point 라벨+Window 분류, 미래 창을 S2/S4가 통과시키고 Build 후 실패.", "S2 지오메트리↔산출물, S4 창↔오늘 즉시 검증"),
 ("P0-5", "P0", "41분 학습 후 실패에 사유 없음", "Error 행·툴팁·카드·Dashboard·queue 전부 무언. 1 unit 소비.", "실패 사유·로그 요약 표시, 이메일 알림"),
 ("P0-7", "P0", "최근 사건(수 주 전)은 어떤 시간 카드로도 학습 불가", "A condition 후 1개월이 오늘을 넘겨 298/298 무효; A sighting은 영상 없음(추정)으로 사망.", "후 맥락을 오늘까지 자동 클램프 + 가용 영상 수 미리보기"),
 ("P1-labelset", "P1", "데이터셋별 동명 labelset을 UI가 구분 못 함", "드롭다운에 같은 텍스트 둘, UUID로만 구분. 필터가 목록을 안 좁혀 'No matching annotated tasks'.", "라벨 항목에 데이터셋명, 필터↔라벨 연동"),
 ("P1-Area", "P1", "예측엔 Area가 먼저 필요한데 안내 없음", "Area 0이면 Run model 비활성, 이유 설명 없음.", "Run model 안에서 그리기/업로드 직행 + 빈 상태 문구"),
 ("P1-필터", "P1", "'필터가 곧 학습셋'이 숨어 있음", "Data Viewer 우측 Train model/Export.", "마법사 S2와 명시 연결"),
 ("P1-용어", "P1", "용어 불일치", "Create new project↔Add project, Import data↔Import Training Data, Predictions↔Run model, Map Publisher↔Publish.", "통일"),
 ("P1-개발문구", "P1", "개발자 문구 노출", "Labs의 POST /datasets, ui/src/labs/CLAUDE.md.", "사용자 언어로"),
 ("P1-예제", "P1", "예제/스키마 문서 부재", "예제가 인라인 모달, 다운로드·문서 링크 없음.", "스키마 문서 + 다운로드"),
 ("P0-8", "P0", "Window 분류 예측이 Area당 라벨 하나 — 경고·확률 없음", "68 km² Area → result.geojson feature 1개 'abstain', 확률 필드 없음. 44분·1 unit 뒤에야 단색 사각형으로 알게 된다.", "Run model에 '이 Area는 N개 창/1개 라벨' 미리보기, 창 단위 타일링 옵션, 확률 필드"),
 ("P1-결과위치", "P1", "완료된 예측을 보는 경로가 숨어 있음", "Predictions 행 클릭 무반응. 결과는 Data Viewer ▸ Layers ▸ Predictions, 켜도 지도가 이동하지 않음.", "행 클릭 → Data Viewer 해당 레이어로 이동 + 자동 확대"),
 ("P2-6", "P2", "학습 취소 불가", "Actions = Edit name / Delete model.", "Cancel training + 유닛 규칙"),
 ("P2-기간", "P2", "예측 기간이 학습 기간 길이에 잠김", "End date 비활성, 시작월만 이동.", "'최근 N개월' 프리셋"),
 ("P2", "P2", "조직·역할 모델이 얕음", "Organization=개인, 11개 메뉴 평면.", "팀·공유 단위, 역할별 뷰"),
]

def convert(src: Path, dst: Path):
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime: return
    subprocess.run(["sips", "-Z", str(a.width), "-s", "format", "jpeg", "-s", "formatOptions", "72", str(src), "--out", str(dst)], check=True, capture_output=True)

ASSETS.mkdir(exist_ok=True); items = []; missing = []
for fid, fname, fdesc, steps in FLOWS:
    for i, (step, rel, cap, tags) in enumerate(steps, 1):
        src = SRC/rel; slug = f"{fid}_{i:02d}.jpg"
        if not src.exists(): missing.append(rel); continue
        if not a.skip_convert: convert(src, ASSETS/slug)
        items.append({"flow": fid, "step": step, "file": rel, "asset": slug, "caption": cap, "tags": tags})
(SRC/"curation.json").write_text(json.dumps({"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "items": items}, ensure_ascii=False, indent=1))
if missing: print("MISSING:", *missing, sep="\n  ")

SEV = {"P0": "sev0", "P1": "sev1", "P2": "sev2"}
def chip(t): return f'<a class="chip {SEV[t.split("-")[0]]}" href="#f-{html.escape(t)}">{html.escape(t)}</a>'
def esc(s): return html.escape(s)
import base64
def img(slug):
    if a.embed: return "data:image/jpeg;base64," + base64.b64encode((ASSETS/slug).read_bytes()).decode()
    return a.img_prefix + slug

sections = []
for fid, fname, fdesc, steps in FLOWS:
    fitems = [it for it in items if it["flow"] == fid]
    nodes = "".join(f'<li><a href="#{fid}-{i}"><span class="n">{i}</span>{esc(it["step"])}</a></li>' for i, it in enumerate(fitems, 1))
    frames = "".join(f'''<figure class="frame" id="{fid}-{i}">
  <a class="shot" href="{img(it["asset"])}" data-title="{esc(it["step"])}"><img loading="lazy" src="{img(it["asset"])}" alt="{esc(it["step"])}"></a>
  <figcaption><div class="fhead"><span class="n">{i}</span><strong>{esc(it["step"])}</strong>{"".join(chip(t) for t in it["tags"])}</div><p>{esc(it["caption"])}</p><code>{esc(it["file"])}</code></figcaption>
</figure>''' for i, it in enumerate(fitems, 1))
    sections.append(f'''<section class="flow" id="{fid}">
  <header class="fl-head"><span class="fid">{fid}</span><h2>{esc(fname)}</h2><p class="fdesc">{esc(fdesc)}</p></header>
  <div class="strip-wrap"><ol class="strip">{nodes}</ol></div>
  <div class="frames">{frames}</div>
</section>''')

findings_rows = "".join(f'<tr id="f-{esc(k)}"><td><span class="chip {SEV[sev]}">{esc(k)}</span></td><td>{esc(t)}</td><td>{esc(ev)}</td><td>{esc(fix)}</td></tr>' for k, sev, t, ev, fix in FINDINGS)
toc = "".join(f'<li><a href="#{fid}"><span class="fid">{fid}</span>{esc(fname)}</a></li>' for fid, fname, _, _ in FLOWS)
n_shots = len(items)

HTML = f'''<title>OlmoEarth Studio 플로우 스펙</title>
<meta name="description" content="OlmoEarth Studio를 실데이터(제주 오름·네팔 라수와)로 관통한 2026-09-19 제품 감사 — 7개 플로우, {n_shots}개 화면, 결함 {len(FINDINGS)}건">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--ground:#eef1ec;--surface:#ffffff;--ink:#17231f;--muted:#5f6e67;--line:#d3dbd5;--accent:#0b6e5c;--accent-ink:#ffffff;--soft:#dcebe5;
 --sev0:#a8382d;--sev0-bg:#f7e3df;--sev1:#9a6212;--sev1-bg:#f6ead3;--sev2:#4c5f77;--sev2-bg:#e2e8f0;--shadow:0 1px 2px rgba(23,35,31,.08),0 8px 24px -12px rgba(23,35,31,.25)}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--ground:#0f1613;--surface:#17221d;--ink:#e6ece7;--muted:#98a79f;--line:#2a3831;--accent:#4fc3a6;--accent-ink:#0f1613;--soft:#1e3a32;
 --sev0:#f08a7c;--sev0-bg:#3a1f1b;--sev1:#e2b063;--sev1-bg:#3a2c14;--sev2:#a9bbd3;--sev2-bg:#23303f;--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.6)}}}}
:root[data-theme="dark"]{{--ground:#0f1613;--surface:#17221d;--ink:#e6ece7;--muted:#98a79f;--line:#2a3831;--accent:#4fc3a6;--accent-ink:#0f1613;--soft:#1e3a32;
 --sev0:#f08a7c;--sev0-bg:#3a1f1b;--sev1:#e2b063;--sev1-bg:#3a2c14;--sev2:#a9bbd3;--sev2-bg:#23303f;--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.6)}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}} @media (prefers-reduced-motion:reduce){{html{{scroll-behavior:auto}}}}
body{{margin:0;background:var(--ground);color:var(--ink);font:15px/1.6 "IBM Plex Sans KR","Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif;padding-block:0 64px;padding-inline:16px}}
a{{color:var(--accent)}} code,.mono{{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.82em}}
h1,h2,h3{{line-height:1.25;text-wrap:balance;margin:0}}
.wrap{{max-width:1240px;margin:0 auto}}
.top{{display:grid;grid-template-columns:1fr;gap:24px;padding-block:40px 24px;border-bottom:1px solid var(--line)}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
h1{{font-size:clamp(28px,4vw,40px);font-weight:700;margin-top:8px}}
.lede{{max-width:62ch;color:var(--muted);margin:12px 0 0}}
.stats{{display:flex;flex-wrap:wrap;gap:10px 28px;margin-top:18px;font-variant-numeric:tabular-nums}}
.stats b{{font-size:22px;font-weight:600;display:block}} .stats span{{font-size:12px;color:var(--muted)}}
.layout{{display:grid;grid-template-columns:1fr;gap:32px;margin-top:24px}} .layout>main{{min-width:0}}
@media (min-width:1000px){{.layout{{grid-template-columns:240px 1fr}} .rail{{position:sticky;top:16px;align-self:start;max-height:calc(100vh - 32px);overflow:auto}}}}
.rail ul{{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:4px}}
.rail a{{display:flex;gap:10px;align-items:baseline;padding:8px 10px;border-radius:8px;color:var(--ink);text-decoration:none;font-size:14px}}
.rail a:hover,.rail a:focus-visible{{background:var(--soft);outline:none}}
.fid{{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--accent);font-weight:500;min-width:2.2em}}
.rail h3{{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:18px 0 8px;font-family:"IBM Plex Mono",monospace;font-weight:500}}
.legend{{display:flex;flex-wrap:wrap;gap:6px}}
.chip{{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:500;padding:2px 7px;border-radius:999px;text-decoration:none;line-height:1.5;white-space:nowrap}}
.sev0{{background:var(--sev0-bg);color:var(--sev0)}} .sev1{{background:var(--sev1-bg);color:var(--sev1)}} .sev2{{background:var(--sev2-bg);color:var(--sev2)}}
/* 파이프라인 개요 */
.pipe{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin-top:16px}}
.pipe .st{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px;position:relative}}
.pipe .st b{{display:block;font-size:14px}} .pipe .st span{{font-size:12px;color:var(--muted)}}
.pipe .st i{{position:absolute;top:10px;right:10px;width:8px;height:8px;border-radius:50%;background:var(--accent)}} .pipe .st.partial i{{background:var(--sev1)}} .pipe .st.todo i{{background:var(--line)}}
.overview h2{{font-size:20px;margin-top:28px}} .overview p{{max-width:70ch}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
th,td{{text-align:left;vertical-align:top;padding:9px 12px;border-top:1px solid var(--line)}} th{{background:var(--soft);font-weight:600;border-top:0;font-size:12px;letter-spacing:.04em}}
.tbl{{overflow-x:auto;margin-top:12px}}
/* 플로우 */
.flow{{margin-top:56px;scroll-margin-top:16px}}
.fl-head{{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;align-items:baseline}} .fl-head h2{{font-size:24px;font-weight:600}} .fl-head .fdesc{{grid-column:2;margin:0;color:var(--muted);max-width:70ch}}
.strip-wrap{{overflow-x:auto;margin-top:16px;padding-bottom:6px}}
.strip{{list-style:none;margin:0;padding:0;display:flex;gap:0;min-width:max-content}}
.strip li{{position:relative;display:flex;align-items:center}}
.strip li+li::before{{content:"";width:22px;height:2px;background:var(--line);margin:0 2px}}
.strip a{{display:inline-flex;align-items:center;gap:8px;background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:6px 12px 6px 6px;text-decoration:none;color:var(--ink);font-size:13px;white-space:nowrap}}
.strip a:hover{{border-color:var(--accent)}}
.n{{display:inline-grid;place-items:center;width:22px;height:22px;border-radius:50%;background:var(--accent);color:var(--accent-ink);font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:500;flex:none}}
.frames{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px;margin-top:20px}}
.frame{{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:var(--shadow);scroll-margin-top:16px;display:flex;flex-direction:column}}
.shot{{display:block;aspect-ratio:16/10;overflow:hidden;background:var(--soft);border-bottom:1px solid var(--line)}}
.shot img{{width:100%;height:100%;object-fit:cover;object-position:top;display:block}}
figcaption{{padding:12px 14px 14px;display:flex;flex-direction:column;gap:6px}}
.fhead{{display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px}} .fhead strong{{font-weight:600;font-size:14.5px}}
figcaption p{{margin:0;font-size:13.5px}} figcaption code{{color:var(--muted);font-size:11px;word-break:break-all}}
/* 라이트박스 */
dialog{{border:0;padding:0;background:transparent;max-width:min(96vw,1400px);width:auto}} dialog::backdrop{{background:rgba(10,16,13,.82)}}
dialog figure{{margin:0;background:var(--surface);border-radius:12px;overflow:hidden}} dialog img{{display:block;max-width:min(96vw,1400px);max-height:84vh;object-fit:contain;width:auto;height:auto}}
dialog figcaption{{flex-direction:row;justify-content:space-between;align-items:center;font-size:13px}} dialog button{{font:inherit;background:var(--soft);color:var(--ink);border:0;border-radius:8px;padding:6px 12px;cursor:pointer}}
dialog button:focus-visible,.shot:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
.note{{background:var(--surface);border-left:3px solid var(--sev1);padding:12px 16px;border-radius:0 10px 10px 0;margin-top:16px;font-size:14px}}
footer{{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}}
</style>
<div class="wrap">
<header class="top">
 <div>
  <div class="eyebrow">Product spec · 2026-09-19 · Toy project - jeju · 계정 1개(Admin)</div>
  <h1>OlmoEarth Studio 플로우 스펙</h1>
  <p class="lede">라벨 넣기 → 학습 → 예측 → 발행. 제주 오름 243곳·네팔 라수와 298창의 실데이터로 이 파이프라인을 끝까지 눌러보고, 화면 하나하나를 플로우 순서로 묶었다. 결함은 화면에 칩으로 붙어 있고, 근거 스크린샷 파일명이 각 프레임 아래에 있다.</p>
  <div class="stats"><div><b>7</b><span>플로우</span></div><div><b>{n_shots}</b><span>화면(선별) / 502 촬영</span></div><div><b>6 → 1</b><span>학습 시도 → ready</span></div><div><b>{len(FINDINGS)}</b><span>결함 (P0 5)</span></div><div><b>3 / 100</b><span>compute units 사용</span></div></div>
 </div>
</header>
<div class="layout">
 <nav class="rail" aria-label="플로우 목차">
  <h3>플로우</h3><ul><li><a href="#overview"><span class="fid">00</span>파이프라인 개요</a></li>{toc}<li><a href="#findings"><span class="fid">08</span>결함 표</a></li><li><a href="#repro"><span class="fid">09</span>재현·한계</a></li></ul>
  <h3>심각도</h3><div class="legend"><span class="chip sev0">P0 막힘·손실</span><span class="chip sev1">P1 헤맴</span><span class="chip sev2">P2 다듬기</span></div>
 </nav>
 <main>
  <section class="overview" id="overview">
   <h2>00 · 파이프라인 개요 — 어디까지 확인했나</h2>
   <div class="pipe">
    <div class="st"><i></i><b>계정·프로젝트</b><span>F0 · 확인함</span></div>
    <div class="st"><i></i><b>Import</b><span>F2 · 3개 데이터셋 완료</span></div>
    <div class="st"><i></i><b>Data Viewer·Tasks·Analytics</b><span>F3 · 확인함</span></div>
    <div class="st"><i></i><b>Build model</b><span>F4 · 5단계 전부</span></div>
    <div class="st partial"><i></i><b>학습</b><span>F5 · 6회 중 1회 ready</span></div>
    <div class="st"><i></i><b>예측</b><span>F6 · completed (결과 = 라벨 1개)</span></div>
    <div class="st partial"><i></i><b>발행</b><span>F6 · Restricted · preview (Public은 미전환)</span></div>
   </div>
   <h2>먼저 — 한계와 못 한 것</h2>
   <p>완료된 유일한 모델(제주 v2)의 수치는 <strong>과학적으로 의미가 없다.</strong> 라벨 <code>scored/abstain</code>은 우리 파이프라인의 기권 플래그(구름·유효 픽셀 대리변수)이지 사람이 확인한 정답이 아니다. 이 문서의 학습·예측은 <em>제품 경로가 끝까지 도는지</em>를 보는 연기 테스트다. 어노테이션 편집기(태스크를 열어 그리는 화면), 협업자 역할별 UI, 발행된 지도 화면은 미확인.</p>
   <h2>학습 시도 6회 — 무엇을 배웠나</h2>
   <div class="tbl"><table><thead><tr><th>#</th><th>모델</th><th>데이터·라벨</th><th>시간 카드</th><th>결과</th><th>드러난 것</th></tr></thead><tbody>
    <tr><td>1</td><td>audit-nano-oreum-window-cls</td><td>제주 Point 243 + 네팔 298 · sample_category</td><td>A state 12개월</td><td><span class="chip sev0">failed 즉시</span></td><td>Point에 Window 분류·미래 창을 제출 뒤에야 검증 (결함 3)</td></tr>
    <tr><td>2</td><td>audit-nano-rasuwa-status</td><td>네팔 폴리곤 298 · status</td><td>A sighting ±12h</td><td><span class="chip sev0">failed 41분 후</span></td><td>사유 표시 없음 (결함 5) · 취소 불가 (결함 6) · 1 unit 소비</td></tr>
    <tr><td>3</td><td>audit-nano-oreum-polys-cat</td><td>제주 폴리곤 243 · sample_category(포인트판)</td><td>A state</td><td><span class="chip sev0">failed 즉시</span></td><td>동명 labelset 오선택 → "No matching annotated tasks" (결함 4)</td></tr>
    <tr><td>4</td><td>audit-nano-oreum-polys-cat-v2</td><td>제주 폴리곤 243 · sample_category(폴리곤판)</td><td>A state 2025-01~12</td><td><span class="chip sev2">ready 48분</span></td><td>Acc 81.0 / F1 78.9 / val 84창 (기준선 61.9)</td></tr>
    <tr><td>5</td><td>(dry-run) rasuwa-status-state</td><td>네팔 298 · status</td><td>A state 2026-01~12</td><td>미제출</td><td>창이 미래로 뻗을 것이 명백 → 제출 안 함</td></tr>
    <tr><td>6</td><td>audit-nano-rasuwa-status-cond</td><td>네팔 298 · status</td><td>A condition ±1개월</td><td><span class="chip sev0">failed 즉시</span></td><td>"window extends to 2026-09-25 (future)" 298/298 — 최근 사건 학습 불가 (결함 7)</td></tr>
   </tbody></table></div>
   <div class="note"><strong>파트너 관점의 핵심 두 줄:</strong> (1) 네팔처럼 <em>사건 후 몇 주</em>가 가장 절박한 사용 사례인데, 세 시간 카드 중 어느 것으로도 성공하지 못했다. (2) 유일하게 끝까지 간 제주 경로도 결과가 <em>68 km²에 라벨 하나</em>였다 — Window 분류 모델을 Area에 돌리면 그렇게 된다는 것을 제품이 미리 말해주지 않는다. 둘 다 "돌아가긴 하는데 파트너가 원하는 답이 안 나온다"는 같은 종류의 문제다.</div>
  </section>
  {"".join(sections)}
  <section class="flow" id="findings">
   <header class="fl-head"><span class="fid">08</span><h2>결함 표 — 우선순위</h2><p class="fdesc">각 행의 칩이 위 프레임의 칩과 같은 ID다. 근거는 전부 이 계정에서 실제로 일어난 것.</p></header>
   <div class="tbl"><table><thead><tr><th>ID</th><th>문제</th><th>근거(확인함)</th><th>제안</th></tr></thead><tbody>{findings_rows}</tbody></table></div>
   <p style="margin-top:14px;color:var(--muted);font-size:13.5px"><strong>긍정적으로 확인:</strong> 빈 상태마다 안내 · Project ID 복사 · Task access 3단계 · Auto assign · 예제 데이터 · Map Publisher Access level · Embeddings를 산출물로 제공 · Hacker mode의 config 투명성 · 비용을 미리 추정 · Spatial split 기본(누수 방지) · 검증 실패 메시지는 있을 때 구체적.</p>
  </section>
  <section class="flow" id="repro">
   <header class="fl-head"><span class="fid">09</span><h2>재현·한계</h2><p class="fdesc">모든 화면은 Playwright(읽기 전용 크롤러 + 흐름별 스크립트)로 찍었고, 계정을 건드리는 스크립트는 DOM 덤프 → dry-run → 스크린샷 검증 → 실행 순서를 지켰다.</p></header>
   <div class="tbl"><table><thead><tr><th>스크립트</th><th>역할</th></tr></thead><tbody>
    <tr><td><code>code/studio_audit.py</code></td><td>로그인·전수 탐색(읽기 전용), explore_map.json</td></tr>
    <tr><td><code>code/studio_import.py</code></td><td>Import 마법사 자동화(+select all, merge No)</td></tr>
    <tr><td><code>code/studio_build_model_launch2.py</code></td><td>Build model 마법사(라벨 index, 데이터셋 필터, 시간 카드, dry-run)</td></tr>
    <tr><td><code>code/studio_models_poll.py</code> · <code>studio_predictions_poll.py</code></td><td>학습·예측 상태 폴링, 완료 시 상세 캡처</td></tr>
    <tr><td><code>code/studio_dom_probe.py</code> · <code>studio_dialog_probe.py</code></td><td>셀렉터를 쓰기 전 실물 DOM·대화상자 덤프</td></tr>
    <tr><td><code>code/make_area_geojson.py</code> · <code>studio_add_area.py</code> · <code>studio_run_model.py</code></td><td>Area 생성·예측 실행</td></tr>
    <tr><td><code>code/build_studio_spec.py</code></td><td>이 문서 생성(선별 목록 → JPEG → HTML)</td></tr>
   </tbody></table></div>
   <p style="color:var(--muted);font-size:13.5px;margin-top:14px">원본 PNG 502장은 <code>artifacts/studio_audit/</code>(git 미추적). 상세 서술은 <code>docs/OLMOEARTH_STUDIO_PRODUCT_AUDIT_2026_09_19.md</code>.</p>
  </section>
 </main>
</div>
<footer>OlmoEarth Studio 제품 감사 · 2026-09-19 · 계정 소유자의 Toy project에서만 수행 · 크레덴셜·개인정보 없음</footer>
</div>
<dialog id="lb"><figure><img id="lb-img" alt=""><figcaption><span id="lb-cap"></span><button type="button" id="lb-close">닫기</button></figcaption></figure></dialog>
<script>
(function(){{const d=document.getElementById('lb'),im=document.getElementById('lb-img'),cap=document.getElementById('lb-cap');
document.querySelectorAll('a.shot').forEach(a=>a.addEventListener('click',e=>{{e.preventDefault();im.src=a.getAttribute('href');im.alt=a.dataset.title;cap.textContent=a.dataset.title;d.showModal();}}));
document.getElementById('lb-close').addEventListener('click',()=>d.close());d.addEventListener('click',e=>{{if(e.target===d)d.close();}});}})();
</script>
'''
Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(HTML)
size = sum((ASSETS/it["asset"]).stat().st_size for it in items)
print(f"HTML → {a.out} ({len(HTML)//1024} KB) · assets {len(items)}장 {size/1e6:.1f} MB · missing {len(missing)}")
