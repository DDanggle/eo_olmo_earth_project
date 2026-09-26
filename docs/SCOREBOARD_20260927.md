# 점수판 — "가장 강한 기준선 대비 +10%p" 규칙으로 본 결과 (2026-09-27)

규칙: `config/meaningfulness_rule_20260927.json` (사용자 결정). 보류 지역·사건에서 균형정확도가 **가장 강한 기준선보다 ≥ .10**,
차이의 95% CI 하한 > 0, seed 2/3 이상. 교체 통제 통과는 별도 필수. 이전 등록 판정은 바꾸지 않고 여기서만 재분류한다.

| 결과 | 비교 | 차이 | 가장 강한 기준선이었나 | 10%p 규칙 |
|---|---|---|---|---|
| MS-157 홍수 reader | vs 같은 예산 blind | +.29–.30 | 아니오 (blind는 약함) | — |
| MS-158 홍수 reader (E5 full) | vs E6 attention head (367k) | −.064 / +.032 / +.108, 구간 모두 0 포함 | **예** | **불통과** |
| MS-157 산사태 reader | vs blind | +.28 / +.20 / −.06 | 아니오 | — |
| MS-131/156 EarthTalk 산사태 | vs TEOChat zero-shot | .607 vs .581 (+.026) | 부분 | **불통과** |
| MS-159 EarthDial 산사태 | vs 우리 reader | .677 vs .789/.682/.526 | — | 비교 대상 아님 |
| E7 태양광 reader | vs E7 mean-pool MLP head | +.187 / +.156 / +.202 (reader .947/.944/.924 vs head .760/.788/.722) | **아니오** (공간 정보를 평균으로 잃는 약한 head) | **판정 보류 → E7s** |
| E7 홍수 reader | vs E7 MLP head | .792/.723/.777 vs .682/.678/.659 → +.11/+.05/+.12 | 아니오 | 판정 보류 → E7s |
| E7 산사태 reader | vs E7 MLP head | .734/.625/.620 vs .599/.622/.841 | 아니오 | 판정 보류 → E7s |

**현재 10%p 규칙을 통과한 결과: 0개.** 통제로 확인된 "내용을 읽는다"는 사실들은 유효하지만, "더 잘한다"는 아직 보여주지 못했다.

## 규칙을 통과할 수 있는 후보 (진행 중)

| 후보 | 왜 가능성이 있나 | 실험 |
|---|---|---|
| 태양광(작은 물체) 판독 | 공간 평균 head보다 +15–20%p. 공간 attention head도 이기면 LLM reader의 실질 이득 | **E7s** (GPU0) |
| 새 지역 일반화 | 지역이 바뀌면 작은 head가 더 무너지고 LLM reader가 덜 무너질 수 있음 | **X1** italy (E2/E7 reader·head) |
| 인코더 선택 | 어떤 기반 모델은 LLM 판독에서 크게 앞설 수 있음(벤치마크 축) | **R-ZOO** (reader vs mean head vs attention head) |
| 공개 EO-VLM 대비 | 같은 문항에서 EarthDial/TEOChat보다 +10%p | X1 italy에서 EarthDial과 같은 문항 비교 |
