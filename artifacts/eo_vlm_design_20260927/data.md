# EO → VLM 데이터 설계 검토 (2026-09-27)

읽기 전용 조사. 다운로드·GPU 작업 없음. 아래 수량은 공개 데이터셋 규모이며 우리가 확보·검증한 수량이 아니다. 논문 문제는 **새 환경의 제한된 교정을 받아 공간 위치와 관측 근거를 함께 갱신하는가**로 잡는다. 센서 연결 자체 또는 공개 라벨을 문장으로 변환한 것만으로 신규성을 주장하지 않는다.

## 우선순위: 코어 2개 + 스트레스 도메인 1개

### 1. PASTIS / PASTIS-R — 현재 연결을 실제 의미 과제로 전환하는 첫 코어

- 공식 공개 규모: 프랑스 2,433개 128×128 패치, 10m, 18작물 종류, 124,422필지, S2 10밴드·38–61관측. R 버전에는 S1 상승/하강 궤도 시계열이 각각 약70관측 추가된다. 픽셀 의미와 필지 instance 정답, 공식5fold가 있다. 영상 출처 THEIA, 정답은 프랑스 IGN LPIS. [공식 저장소](https://github.com/VSainteuf/pastis-benchmark)
- 원 취득일은 metadata에서 읽는다. 공식 loader는 기준일 2018-09-01 대비 일수와 영상/정답을 반환하며, 채널별 표준화를 선택한다. 기준일을 취득일로 오인하지 않는다. [공식 loader](https://raw.githubusercontent.com/VSainteuf/pastis-benchmark/main/code/dataloader.py)
- 공개 다운로드: 원 PASTIS 약29GB, dense PASTIS-R 53.7GB zip. pixel-set 전용27GB 버전은 dense grounding 대체품이 아니다. [PASTIS-R 원자료](https://zenodo.org/records/5735646)
- 현재 로컬 계획에서 서버 GEO-Bench shard 두 개의 존재만 확인됐고, shard0001 과거 hash 불일치 이력이 있다. raw10밴드/날짜/부모지역 읽기, B01/B09 결측 처리, OlmoEarth 정규화 parity 모두 미완료다. 먼저2사례 계약검사 유지. 코드 MIT와 원자료 라이선스를 혼동하지 말 것: 이번 Zenodo 텍스트 추출에서 원자료 라이선스 값은 확인되지 않았다.
- **직접 가능한 정답:** 주어진 작물 개념/양성·혼동 필지 예시로 다른 필지 찾기, mask/위치, 해당 라벨 면적·개수, 정해진 관측 묶음 안에서 근거 위치 연결.
- **새 검수가 필요한 정답:** 어느 시점·분광 특징이 구별에 유효했는지, 교정 문장의 정확성, 가려져 판독 불가능한지. 연간 작물 라벨은 월별 건강·고사·수확일·변화 시작일 정답이 아니다.
- 분할: 기존 계획대로 부모tile4개 한계를 유지한다. 개발의3개 비봉인tile와 최종1개tile을 독립국가전이로 부르지 않는다. 같은필지/인접패치를 support/query 사이에 흩뜨리지 않는다.

### 2. Kuro Siwo — 실제 변화 + 전문가 판단을 시험할 두 번째 코어

- 공개 논문: 2015–2022년 43홍수사건, 6대륙·3기후대, 67,490개의 라벨된 시계열. 전2회·후1회 S1 VV/VH, 224×224,10m와 DEM; 전문가5명이 교차검수한 영구수역/홍수/비수역 정답. 공식 평가에서10사건을 holdout. CEMS가 있는 곳은 그 정답을 수정했고 없는 곳은 새로 판독했다. [저자 논문](https://arxiv.org/html/2311.12056v3)
- 공식 HF는 GRD/SLC, 날짜/AOI metadata, valid mask를 제공하며 현재 데이터 라이선스는 CC-BY-4.0이다. [공식 GeoTIFF](https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-GeoTIFFs), [공식 WebDataset](https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset). 원 논문 MIT와 현재 데이터카드 CC-BY가 다르므로 사용할 버전의 권리를 manifest에 고정한다.
- native 경로는 **GRD부터** 시작. 공식 전처리 XML은 선형 sigma0, Lee Sigma speckle filtering, terrain correction, EPSG:3857 격자를 사용한다. log/dB 변환과 OlmoEarth S1 정규화는 원 배열 통계·공식 모델 코드로 맞춰야 한다. SLC phase4채널을 기존2채널S1처럼 넣으면 안 된다. [공식 전처리](https://raw.githubusercontent.com/Orion-AI-Lab/KuroSiwo/main/configs/grd_preprocessing.xml)
- **직접 가능한 정답:** 새 홍수 영역 찾기, 영구수역과 구분, 유효픽셀에서 mask/위치/면적. 면적은 격자·좌표계에 맞게 계산하고 투영10m를 모든 위도에서 지상100㎡로 고정하지 않는다.
- **새 검수가 필요한 정답:** 경계·혼동을 바로잡는 짧은 양성/음성교정, 어떤 전후 관측이 근거인지, 판독불가 근거. 기존 전문가 mask는 전문가가 쓴 자연어 설명이나 실제 상호작용 로그가 아니다. 원인·피해액·기후변화 귀속 정답도 없다.
- 제안 장점: 같은 물처럼 보이는 공간에서 전후 관계가 달라져 정답이 갈린다. 시간정보가 VLM 연결 중 사라지는지와 교정의 새사건전이를 동시에 시험할 수 있다. baseline에 동일한DEM·시점·기후맥락을 준다. 사건보고서에 정답 홍수 범위가 담겼으면 입력 맥락으로 쓰지 않는다.

### 3. MADOS — 해양 쓰레기/유사물질 혼동 스트레스

- 저자 공개자료: S2 174scene, 2,803개의240×240crop,15개 클래스, sparse pixel labels,3단계 신뢰도, marine debris 보고 근접성 레이어. native10/20/60m 밴드, Rayleigh-corrected reflectance, train/val/test 목록을 포함하며 압축4GB/풀린5.35GB다. [저자 Zenodo](https://zenodo.org/records/10664073)
- class0은 미주석이며 깨끗한 바다/음성 정답이 아니다. 보고근접성은 자연어 전문가 보고서 본문이 아니다. 같은scene의crop은 분할단위로 묶고 지역·날짜 재등장도 검사한다.
- **직접 가능한 정답:** 라벨된 영역에서 marine debris와 해조류·거품·자연물·oil spill 혼동 평가, confidence별 성능. **추가검수:** 자연어 구별교정, 불확실성 근거, 미주석 영역의 음성/범위 정답. 누적쓰레기량·질량·개별플라스틱 판독이나 시간변화는 이 자료만으로 주장할 수 없다.
- OlmoEarth S2 L2A와 Rayleigh-corrected 해양 반사도는 동일하지 않다. 원scene·취득일 복원이 가능한지, L2A 정합/별도입력stem이 필요한지 먼저 확인한다. 단순 밴드수 맞추기로 호환판정 금지.
- 이번1차자료 HTML에서 라이선스 값은 노출되지 않아 재배포 전 원 record metadata를 확인해야 한다. [공식 코드](https://github.com/gkakogeorgiou/mados)는 다운로드 안내와 학습 코드를 제공한다. 현재 미확보.

## 조건부 후보 2개 (즉시 핵심실험으로 추가하지 않음)

| 자료 | 쓸 이유 | 주요 제약/현재 판정 |
|---|---|---|
| Sen4AgriNet PAD | France2019↔Catalonia2019/2020 국가·연도전이,13개 S2밴드와 dense labels/parcelIDs | L1C TOA이므로 nativeL2A와 정합 필요. 공식 HF card MIT, 원LPIS ODC Attribution. 공개HF281GB; 원개념전기간과 현재배포기간을 구분. 현재 미확보 |
| Sen1Floods11 | 소규모 대체홍수 코어,446수작업chip과 S1/S2 정합 영상 | S2는13밴드 L1C×10000, S1 VV/VH dB. 전후시계열이 기본제공되지 않아 Kuro의 변화대체품 아님. 공개약14GB, 현재 미확보 |

Sen4AgriNet의 공개subset은 FR2019/Catalonia2019–2020, LPIS의168세부클래스/instance이다. [저자 HF 데이터카드](https://huggingface.co/datasets/orion-ai-lab/S4A). 저자 모델 비교는5000patch·11빈도상위클래스로3전이시나리오를 구성했다. PASTIS→S4A는 국가+자료셋+방사처리 동시변화이므로 순수지역효과 대신 동일S4A FR→ES를 우선한다. [공식 실험](https://github.com/Orion-AI-Lab/S4A-Models)

Sen1Floods11은512×512,10m이며 S1/S2취득일과 사건ID metadata·기존split을 제공한다. [공식 README](https://github.com/cloudtostreet/Sen1Floods11), [저자 CVPRW 논문](https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html). 이번 조사에서 데이터권리 상세값은 미확인. Kuro 논문의 관련연구가 말하는 Sen1Floods11 수작업label의 사건수 대신 원자료metadata를 신뢰해야 한다.

## 100인시 안에서 실제 교정 데이터를 만드는 제안

**수량은 계획이며 아직 수집되지 않았다.** 먼저30사례에 걸리는 실제 시간을 측정한다. 첫200episode는 PASTIS80/Kuro80/MADOS40을 상한으로 잡되, MADOS native변환이 미완료면 core2개100씩만 진행한다. 이후 최대600episode(240/240/120)로 확장한다. 한 episode는 input관측+질문+query영역+support교정+target정답+출처묶음이며 600개이미지와 동의어가 아니다.

100인시 예시: 절차/연습10h +600건1차검수×4분40h +120건독립재검수×5분10h +어려운건합의20h +품질/출처검사10h +예비10h. 4분가정은30사례파일럿에서만 결정하며 SAR/해양 전문가가100시간 확보된 것으로 간주하지 않는다. 모든분야의 판독자격을 자동가정하지 않는다.

- 공개mask에서 위치·면적·라벨 답을 프로그램으로 생성한다. AI는 짧은 질문/설명후보와 혼동사례만 제안한다. AI다수결은 독립정답이 아니며, 동일계열teacher/evaluator 편향을 별도기록한다.
- 사람은 근거날짜·영역, false positive를 구분하는 대조교정, 실제로 판단불가한 이유에 시간을 쓴다. sparse라벨 공백을 음성으로 채우는 작업은 자동생성하지 않는다.
- K=1/5/20 선택교정수와 전체후보검수비용을 따로 집계한다. 공개라벨로 자동추출한support는 simulated correction이고 실제전문가 교정실험과 분리한다.
- support/query의 필지/사건/scene을 분리하고, 최종holdout을 알고리즘선정·교정작성·AIprompt수정에서 봉인한다. 지역context는 예측시점이전의 일반기후·계절정보만; test사건피해문서·정답지도 금지.

## 데이터가 요구하는 최소 아키텍처와 결정 실험 (제안)

1. native OlmoEarth 토큰에 센서·실제시점·위치·validity를 보존하여 VLM으로 전달한다. 평균토큰16개와 token수·학습예산을 맞춘 날짜/영역선택 resampler를 비교한다. 토큰수증가 자체의 효과를 분리한다.
2. 같은 query의 위치/mask와 요약문이 **동일한 근거토큰**을 참조하게 한다. 이것이 spatialgrounding·정량오류를 줄이는지 본다. 별도 segmentationhead가 다 해결하고 VLM이 문장화만 한 효과를 strongbaseline으로 둔다.
3. 양성/혼동교정과 지역context로 근거선택을 바꾸되, context를제거/섞기/모순되게했을 때 영상근거를 넘어 답이 따라가는지 평가한다. VLM답변문체보다 falsepositive·maskIoU·면적오차·지원교정예산대비 새지역성능을 우선한다.
4. 실제변화(Kuro), 긴시계열개념구별(PASTIS), 유사분광물질(MADOS)이 각각 필요한정보가 달라 서로를검증한다. 모두돌아간다는사실만으로 한방법이일반화했다고 주장하지 않으며 동일정보generic공동학습보다 이득이 있어야 한다.

이번 자료선정은새기여의 근거가 아니라 좋은실패를관찰할 수 있는 실험조건이다. 첫2사례계약검증 → core에서같은정보baseline의오류관찰 → 그오류에대응하는토큰/교정방법학습 → 봉인사건/지역검증 순서가 합리적이다.
