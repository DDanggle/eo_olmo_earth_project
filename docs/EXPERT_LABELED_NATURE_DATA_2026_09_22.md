# 공개 전문가 라벨 자연환경 EO 데이터: 무엇을 재사용할 수 있는가

확인일: 2026-09-22. 공식 논문·저자 배포처 중심의 후보 조사이며, 데이터 다운로드·학습·성능 검증은 하지 않았다.
접근 가능은 공개 페이지/파일 배포 확인을 뜻한다. 실제 파일 무결성·현지 캐시 정합성까지 확인했다는 뜻은 아니다.

## 결론과 추천

**전문가가 만든 정량 라벨은 있다. 전문가의 문장형 판단 근거·시간별 믿음 갱신 정답은 별개다.**
아래 네 후보에서 샘플별 전문가 rationale, 최초 판독 가능 관측, prefix별 불확실성 gold는 확인하지 못했다.
자연환경 확장의 **우선 검토는 KuroSiwo**, 홍수 밖의 **의미/어휘 보조 후보는 ForestNet**으로 권한다.
단, 현재 SN7 v0.5의 질문·gold 계약을 대체하지 않는다. KuroSiwo도 세 시점뿐이라 장기 기억의 주 벤치가 아니다.
ForestNet은 산림손실 의미 학습 후보이지, 시간 갱신 효과를 증명하는 외부 벤치가 아니다.
기존 전문가 판독을 재사용하면 주석 비용은 줄지만, 새 질문의 정답까지 자동으로 얻는 것은 아니다.

## 후보 비교

| 데이터 | 실제 제공되는 감독 | 센서/시간 구조 | 현재 과제에 적합한 역할 |
|---|---|---|---|
| KuroSiwo | SAR 전문가 수작업 물/홍수 마스크 | Sentinel-1 GRD/SLC, 사건 전 2 + 후 1 | 자연환경·센서 확장의 짧은 근거 비교 |
| ForestNet | 전문가 산림손실 원인 분류 | Landsat 8 사후 합성/개별 장면 | 산림 의미/어휘 보조, 정적 영역 접지 |
| Landslide4Sense | 자동 후보를 전수 수동 교정한 산사태 마스크 | Sentinel-2 12밴드 + 지형 2밴드, 공개 입력은 단일 시점형 | native S2 공간 접지 보조 |
| BRIGHT | EO 전문가 건물 polygon·피해 등급 | VHR 0.3–1m, 주 설정은 사건 전 광학 + 후 SAR | 재난 종류·센서 변화에 대한 외부 reader 감사 |

## 1. KuroSiwo — 전문가 라벨의 출처가 가장 명확한 우선 후보

- 43개 홍수 사건; Sentinel-1 전 2장/후 1장과 DEM. 10m 격자의 GRD/SLC이며 RGB 광학 데이터가 아니다.
- SAR 전문가 5명이 분담 판독하고 다른 전문가가 교차 검수했으며 senior remote-sensing scientist가 감독했다.
- 정답은 no-water/permanent-water/flood. 위성 판독 정답이지 전 영역 현장 실측 정답이라고 할 수 없다.
- 근거: [NeurIPS 논문 §3·부록 Annotation process](https://arxiv.org/html/2311.12056).
- [저자 공식 HF](https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset)는 CC BY 4.0, 영상·mask·valid_mask·info.json과 사건 단위 test 구성을 명시한다. [원본 GeoTIFF 배포](https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-GeoTIFFs)도 있다.
- **라이선스 불일치:** 위 원 논문 본문은 데이터 MIT를 명시하지만 현재 HF는 CC BY 4.0이다. 데이터 전체에 단일 조건을 단정하지 않는다. 재배포 전 실제 내려받을 릴리스/버전의 동봉 라이선스·배포 조건을 확인·기록하고, 불일치가 남으면 권리자 확인 전 재배포하지 않는다.
- HF viewer 오류는 배포 파일 부재와 같지 않다. 실행 전 소량 파일·메타데이터·NaN/valid-mask 검사가 필요하다.
- 촬영 시각과 사건 시각은 있으나 픽셀별 발생일/최초 판독 가능일 gold는 없다. 후영상 마스크를 전영상 prefix 정답으로 복사하면 미래정보 누수다.
- 비전문가 20시간으로 SAR 전문가 판독을 대체하지 않는다. 기존 전문가 mask에 연결한 제한된 질문부터 검토한다.
- 기존 실험에서 사용한 사건·타일은 새 untouched test가 아니다. 로컬 사용 이력과 기존 NaN 문제를 먼저 감사한다.

## 2. ForestNet — 홍수 밖의 자연환경 의미를 위한 보조 후보

- 인도네시아 산림손실 2,756개 사례. GFC의 손실 polygon/연도를 바탕으로 전문가 판독자가 Google Earth 고해상도 영상에서 원인을 분류했다.
- Plantation, Smallholder Agriculture, Grassland/shrubland, Other로 묶은 범주다. 모든 사례의 현장조사·원인 확정이라고 읽지 않는다.
- 입력 Landsat 8은 손실 발생 뒤 최대 5년 범위에서 수집한 합성영상과 구름이 적은 개별 장면이다. native S2나 짧은 간격 시계열이 아니다.
- [Stanford 저자 페이지](https://stanfordmlgroup.github.io/projects/forestnet/)에서 CC BY 4.0과 공식 다운로드 링크를 확인했다. 이번에는 아카이브를 받지 않았다.
- 손실 연도는 있어도 정확한 월·첫 시각 증거 gold가 아니다. 사후 장면/원인 라벨을 당시 prefix 모델 입력에 넣으면 누수다.
- “어떤 토지이용 형태가 보이는가?”의 보조 감독은 가능하나 “왜 파괴했는가?”의 전문가 설명 정답이나 불법성 판단으로 확장하지 않는다.

## 3. Landslide4Sense — S2와 직접 연결되지만 전문가 추론 데이터는 아님

- [원 논문 §III-C](https://arxiv.org/html/2206.00515)는 OBIA/영상 차이 기반 후보 생성 후 Google Earth·기존 inventory로 모든 polygon을 수동 검증·수정했다고 명시한다.
- 완전 자동 라벨은 아니다. 다만 판독자의 자격·다중 검수 체계를 KuroSiwo처럼 구체적으로 확인하지 못했으므로 “전문가 100% 독립 이중주석”이라고 부르지 않는다.
- S2 12밴드+DEM/경사, 128×128, 모두 약 10m로 재표본화했다. 모든 밴드의 원래 해상도가 10m라는 뜻이 아니다.
- 공개 학습 입력은 14밴드 단일 시점형 patch다. 주석 제작에 전후 영상을 썼다는 사실이 공개 시계열 제공을 뜻하지 않는다.
- [저자 Zenodo](https://zenodo.org/records/10463239)는 공개 ZIP을 제공한다. [IBM-NASA 배포본](https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense/blob/main/README.md)은 train/val/test mask 구조를 명시한다.
- 논문 분할과 대회 분할 수가 다르므로 버전·split을 고정해야 한다. 원 GitHub 코드 MIT를 데이터 라이선스로 자동 간주하지 않는다. 이번 확인에서 데이터 권리 문구는 충분히 해소하지 못했으므로 재배포 전 추가 확인한다.
- 라벨은 경계/위치뿐이며 물질·유형·체적이 없다고 원 논문이 명시한다. 최초 확인 월·언어 rationale도 제공 정답으로 확인되지 않았다.
- Hokkaido 등 기존 사용 사건과 겹칠 수 있다. 지역명이 다르다고 사건 독립을 가정하지 않는다.

## 4. BRIGHT — 전문가 판독은 강하지만 현재 RGB reader에 즉시 꽂는 자료는 아님

- [ESSD 원 논문 §2.3](https://essd.copernicus.org/articles/17/6217/2025/)은 전문가 polygon 작성/별도 EO 검수, CEMS·UNOSAT·FEMA 피해자료, EO 전문가 재검증을 설명한다. 원자료 일부만 현장 확인을 포함한다.
- 14개 사건/23개 지역, 여러 자연·인위 재난. 주 입력은 사건 전 광학+후 SAR이며, 일부 사건의 추가 후광학을 논문에서 사용했다. 전체를 RGB 전후쌍으로 간주하지 않는다.
- 모호한 “Possibly Damaged” 일부를 명백한 피해가 보이지 않을 때 Intact로 합쳤다. Intact를 검증된 무피해나 우리 `no_visible_change`와 동일시하지 않는다.
- [공식 저장소](https://github.com/ChenHongruixuan/BRIGHT)와 [저자 연결 HF](https://huggingface.co/datasets/Kullervo/BRIGHT)에 배포가 있다. Ukraine/Myanmar/Mexico 광학은 재배포되지 않아 별도 원자료 절차가 필요하다.
- 라이선스는 사건/모달리티별이다. HF 상세 조건은 주로 광학·라벨 CC BY-NC 4.0, SAR CC BY 4.0 및 예외를 명시한다. 상단 CC BY-SA 배지와 불일치하므로 일괄 라이선스로 적지 않고 사용 사건별 조건을 확인한다.
- 사건 날짜·피해 등급은 최초 판독 가능 시점 gold가 아니며, 기본 2시점으로 장기 기억 효과를 주장할 수 없다. 전문가의 샘플별 문장 reasoning도 확인되지 않았다.

## 적용 계약: 기존 전문 주석에서 언어 감독을 만들 때

1. mask/class → 영역·면적·비교관계 → 짧은 문장은 **자동 파생 silver 언어**로 표기한다. 전문가가 쓴 설명으로 포장하지 않는다.
2. 전문가 원래 과제의 정답과 우리의 prefix-visible 정답을 따로 저장한다. 미래 영상·최종 피해·사건명을 현재 시각 증거로 대체하지 않는다.
3. 입력에 보이는 대상만 말하는 검수는 기존 20인·시간 예산 안에서 소량 수행한다. 전문가 진단·SAR 재주석을 비전문가에게 요구하지 않는다.
4. 문항/타일이 아니라 사건/AOI를 봉인하고, answer-only·evidence-supervised를 같은 데이터/비용으로 비교한다. 텍스트-only·영상 교체 대조를 유지한다.
5. 먼저 계약 정합성과 읽기 가능성을 통과시킨다. 전문가 라벨의 존재, 데이터 추가, H100 사용 자체가 VLM 기억 방법의 신규성이나 CVPR 성공을 보장하지 않는다.
