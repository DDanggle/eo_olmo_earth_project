#!/usr/bin/env python3
"""1시간 발표 덱 빌더 — 이미지를 축소해 data URI 로 인라인한 단일 HTML 을 만든다.
수치는 MEASURED_FINDINGS 봉인값만 사용(창작 금지). 실행: python3 docs/talk/build_deck.py → docs/talk/olmoearth_session_2026_09.html"""
import base64, io, json
from pathlib import Path
from PIL import Image
HERE=Path(__file__).resolve().parent; WORK=HERE.parents[1]; NLT=Path("/Users/dgyi/dong/ai_projects/nepal-live-twin/web/public/data")
def img(path, w=880, q=78):
    im=Image.open(path).convert("RGB")
    if im.width>w: im=im.resize((w, int(im.height*w/im.width)), Image.LANCZOS)
    b=io.BytesIO(); im.save(b,"JPEG",quality=q,optimize=True); return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()
I={
 "pre":img(NLT/"scenes/s2-2026-08-12.png",720,82), "post":img(NLT/"scenes/s2-2026-08-27.png",720,82),
 "planet":img(NLT/"story/planet/ps_rasuwagadhi_0828.png",900),
 "swir":img(NLT/"story/spec_swir_post0827.png",520), "ndwi":img(NLT/"story/spec_ndwi_post0827.png",520), "true":img(NLT/"story/spec_true_post0827.png",520),
 "grid_pre":img(NLT/"story/corridor_pre_grid.png",900), "grid_post":img(NLT/"story/corridor_post_grid.png",900),
 "v003_pre":img(NLT/"candidates/v003_pre.png",420), "v003_post":img(NLT/"candidates/v003_post.png",420), "v003_delta":img(NLT/"candidates/v003_delta.png",420),
 "x001_pre":img(NLT/"candidates/x001_pre.png",420), "x001_post":img(NLT/"candidates/x001_post.png",420), "x001_delta":img(NLT/"candidates/x001_delta.png",420),
 "v064_delta":img(NLT/"candidates/v064_delta.png",420),
 "verify":img(WORK/"artifacts/figures/verify_candidates.png",1000), "jeju_v2":img(WORK/"artifacts/figures/jeju_change_v2.png",900), "jeju_v6":img(WORK/"artifacts/figures/jeju_change_v6_4vs12.png",900),
 "search":img(WORK/"artifacts/figures/first_search.png",1000),
}
# ---------- 슬라이드 정의: (kind, fields) ----------
S=[]
def slide(html, notes="", cls=""): S.append((html, notes, cls))
def sec(k, title, sub=""): slide(f'<p class="kicker">{k}</p><h1 class="sec">{title}</h1><p class="sub">{sub}</p>', "", "section")

sec("0 · 시작", "산이 무너진 날,<br>위성은 100곳을 훑었다", "2026년 8월 26일 · 네팔 라수와 · 그리고 한 대의 '얼린' AI")
slide(f'''<div class="two"><figure><img src="{I['pre']}"><figcaption>8월 12일 · 사건 전</figcaption></figure><figure><img src="{I['post']}"><figcaption>8월 27일 · 사건 다음 날</figcaption></figure></div>
<h2>같은 곳, 15일 차이. 뭐가 달라졌나요?</h2><p class="lede">힌트: 강바닥의 회색 띠 폭. 이 계곡은 폭이 100 m라서 이 사진에서는 10픽셀입니다.</p>''',
"먼저 학생들에게 30초 주고 차이를 찾게 한다. 대부분 못 찾는다 — 그게 포인트. 라수와가디 국경 합류부, 센티넬-2 10 m, 2.56 km 창.")
slide(f'''<figure class="hero"><img src="{I['planet']}"><figcaption>PlanetScope 3.8 m · 8월 28일 · © Planet Labs PBC (CC-BY-NC-4.0) · 참고용, AI 입력 아님</figcaption></figure>
<h2>확대하면 보입니다 — 토사 판, 그 안의 물길, 끊긴 도로.</h2>''',
"상업 위성이 재난 때 공개한 영상. 우리는 이걸 AI에 넣지 않았다(밴드·해상도 계약이 다름). 사람 눈 확인용.")
slide('''<h2>문제를 숫자로 바꿔봅시다</h2><div class="big"><span>강 70 km</span><span>2.56 km 창 <b>100</b>개</span><span>사람이 하루에 볼 수 있는 창 <b>?</b></span></div>
<p class="lede">사망자 수백 명, 도로 끊김, 구름. 어디부터 봐야 할까요? 오늘 이야기는 <em>"어디를 먼저 볼지 정하는 AI"</em>이고, <em>"피해를 판정하는 AI"</em>가 아닙니다.</p>''',
"오늘의 주장 경계를 첫 5분에 못 박는다: candidate(후보)까지, damage(피해)·cause(원인)·probability(확률)는 말하지 않는다.")

sec("1 · 위성이 보는 법", "사람 눈은 3개, 위성 눈은 12개", "센티넬-2 광학 · 센티넬-1 레이더 · 그리고 구름")
slide(f'''<div class="three"><figure><img src="{I['true']}"><figcaption>트루컬러 — 사람 눈(B04·B03·B02)</figcaption></figure><figure><img src="{I['swir']}"><figcaption>SWIR 합성 — 식생 초록, 젖은 흙 분홍, 물 짙은 파랑</figcaption></figure><figure><img src="{I['ndwi']}"><figcaption>NDWI 물 지수 — 물만 뽑아냄</figcaption></figure></div>
<h2>사람 눈에 같아 보이는 픽셀이 스펙트럼에서는 다릅니다.</h2>''',
"12개 밴드 중 단파적외(SWIR)가 젖은 토사를 가른다. 그래서 AI에 12밴드를 전부 넣는다. 질문: 왜 사진 한 장으론 부족할까?")
slide('''<h2>광학 vs 레이더 — 각자의 약점</h2><table class="t"><tr><th></th><th>광학 (센티넬-2)</th><th>레이더 (센티넬-1)</th></tr>
<tr><td>구름</td><td class="bad">막힘</td><td class="good">뚫음</td></tr><tr><td>밤</td><td class="bad">못 봄</td><td class="good">봄</td></tr><tr><td>사람이 읽기</td><td class="good">쉬움</td><td class="bad">어려움 (밝기=거칠기·기울기)</td></tr><tr><td>단위 함정</td><td>반사율</td><td class="bad">선형 vs dB — 우리가 한 번 틀렸음</td></tr></table>
<p class="note">몬순 히말라야에서 8월 27일 대조 지역 4곳의 구름 없는 비율: <b>84% · 35% · 15% · 0%</b>. 광학은 확률 싸움입니다.</p>''',
"레이더 단위 오류(M75)는 나중에 '틀렸던 것' 절에서 다시 나온다 — 예고만.")

sec("2 · 지구를 미리 외운 AI", "임베딩이란 무엇인가", "OlmoEarth v1 Base · Ai2 · 우리는 한 번도 재학습하지 않았습니다")
slide('''<h2>비유: 40 m 격자마다 <em>768개 숫자로 된 명함</em></h2>
<div class="cards"><div><b>입력</b><span>같은 자리의 위성 시계열(4시점, 12밴드)</span></div><div class="arrow">→</div><div><b>OlmoEarth (얼림)</b><span>지구 전체로 미리 학습된 인코더</span></div><div class="arrow">→</div><div><b>출력</b><span>768차원 벡터 × 64×64 격자 = "이 땅이 어떤 상태인지"</span></div></div>
<p class="lede">명함이 비슷하면 땅이 비슷하고, 같은 자리의 명함이 갑자기 바뀌면 땅이 바뀐 것. 이게 오늘 쓰는 유일한 AI 연산입니다.</p>''',
"'얼림(frozen)'을 강조: 재난마다 학습하지 않는다. 캐시 개념 예고 — 명함을 한 번 만들어 보관하고 재사용.")
slide(f'''<figure class="hero"><img src="{I['search']}"><figcaption>초기 실험: 명함이 비슷한 곳을 검색 — 제주 농경지 질의 → 비슷한 농경지들이 나온다</figcaption></figure>
<h2>첫 번째 놀라움: 라벨 없이도 "비슷한 땅"이 찾아집니다.</h2>''',
"검색 데모. WorldCover로 정량화하면 제주 농경지 정밀도 .816(무작위의 14.8배). 재미 포인트: '구글 이미지 검색인데 땅 버전'.")
slide(f'''<div class="two"><figure><img src="{I['jeju_v2']}"><figcaption>변화탐지 v2 — 그럴싸했다</figcaption></figure><figure><img src="{I['verify']}"><figcaption>눈으로 확인 — 상위 후보가 <b>전부 구름</b>이었다</figcaption></figure></div>
<h2>두 번째 교훈: 숫자가 좋아도 <em>원본 사진을 사람이 본다.</em></h2>''',
"실패 계보 v2→v6. 이 실패가 이후 모든 실험에 '육안 검증' 규칙을 남겼다. 학생들에게: 여러분 프로젝트에서 top-k를 직접 열어본 적 있나요?")
slide(f'''<figure class="hero"><img src="{I['jeju_v6']}"><figcaption>v6: 시점 4개 vs 12개 — 변화 신호가 어떻게 달라지는지</figcaption></figure>
<h2>여섯 번을 고쳐서야 "평소 변화"와 "사건 변화"를 나눌 수 있게 됐습니다.</h2><p class="lede">핵심 장치: <em>플라시보</em>. 같은 자리의 평소 2주 변화량을 재서, 그보다 크게 바뀐 곳만 후보로 남깁니다.</p>''',
"플라시보 = 아무 일 없던 기간의 변화. 임계 = 그 분포의 99퍼센타일. 이게 뒤에 나오는 '13%'의 정의다.")

sec("3 · 실험 ①", "새 지역에서도 통하나?", "8개 나라의 산사태 · 한 지역을 빼고 학습, 그 지역에서 시험")
slide('''<h2>Leave-One-Region-Out: 시험 지역의 정답은 한 장도 안 보고 학습</h2>
<div class="loro"><span>Hiroshima</span><span>Hokkaido</span><span>Indonesia</span><span>Itogon</span><span>Kyrgyzstan 1</span><span>Kyrgyzstan 2</span><span>New Zealand</span><span class="hold">Thrissur ← 시험</span></div>
<p class="lede">이걸 8번 반복. 통계 단위는 <em>지역 8개</em>이지 타일 6,834개가 아닙니다. (seed 3개는 반복 측정)</p>''',
"Sen12Landslides 벤치마크. 왜 지역 단위인가: 같은 지역 타일은 서로 닮아서 독립 표본이 아니다. 대학생들이 가장 자주 하는 실수.")
slide('''<h2>세 선수</h2><div class="cards"><div><b>P2 · UNet3D</b><span>위성 픽셀로 처음부터 학습 (raw)</span></div><div><b>P3 · U-TAE</b><span>위성 픽셀로 처음부터 학습 (raw, 시계열 특화)</span></div><div class="acc"><b>P4 · OlmoEarth 명함 + 작은 판독기</b><span>얼린 임베딩 위에 23.7만 파라미터 디코더만 학습</span></div></div>
<p class="lede">모두 같은 데이터, 같은 40 epoch, 같은 손실함수, 같은 채점(positive-tile macro IoU).</p>''',
"'같은 조건'을 반복해서 말한다. 비교 실험의 공정성은 여기서 결정된다.")
slide('''<h2>결과 (M65 · 봉인값)</h2><table class="t num"><tr><th>시험 지역</th><th>P4 명함 재사용</th><th>P2 raw</th><th>P3 raw</th><th>P4 − 최고 raw</th></tr>
<tr><td>Hokkaido</td><td class="acc">.386</td><td>.215</td><td>.221</td><td class="good">+.165</td></tr><tr><td>Thrissur</td><td class="acc">.359</td><td>.232</td><td>.232</td><td class="good">+.127</td></tr>
<tr><td>Kyrgyzstan 1</td><td class="acc">.281</td><td>.192</td><td>.173</td><td class="good">+.089</td></tr><tr><td>Hiroshima</td><td class="acc">.278</td><td>.216</td><td>.178</td><td class="good">+.062</td></tr>
<tr><td>Indonesia</td><td>.272</td><td class="acc">.284</td><td>.265</td><td class="bad">−.012</td></tr><tr><td>New Zealand</td><td class="acc">.242</td><td>.179</td><td>.188</td><td class="good">+.054</td></tr>
<tr><td>Kyrgyzstan 2</td><td class="acc">.208</td><td>.107</td><td>.104</td><td class="good">+.101</td></tr><tr><td>Itogon</td><td class="acc">.152</td><td>.148</td><td>.105</td><td>+.004</td></tr>
<tr class="sum"><td>8지역 평균</td><td class="acc">.272</td><td>.197</td><td>.183</td><td class="good">+.076</td></tr></table>''',
"7승 1패. 인도네시아에서 졌다는 걸 숨기지 않는다 — 다음 슬라이드의 '역할'로 이어진다. 절대값이 낮은 이유(어려운 spatial holdout, DEM 없음)도 한 줄.")
slide('''<h2>지역은 표본이 아니라 <em>역할</em>입니다</h2><div class="roles">
<div><b>Indonesia</b><span>반증 지역 — raw가 이긴 유일한 곳. 새 방법은 여기서 나빠지면 안 됨</span></div><div><b>Itogon</b><span>바닥 지역 — 모두 .15. 모델 탓인지 라벨 탓인지 가르는 곳</span></div>
<div><b>Hokkaido</b><span>재사용 이득 최대(+.165)</span></div><div><b>Kyrgyzstan 2</b><span>라벨 희소 — raw가 .107로 붕괴</span></div><div><b>New Zealand</b><span>아키텍처 민감 — P3가 P2를 이긴 유일한 곳</span></div><div><b>Hiroshima</b><span>시드 분산 감시</span></div></div>
<p class="note">개발용 지역(china·chimanimani)은 어떤 성능 표에도 안 들어갑니다. 한국은 봉인된 최종 시험. 네팔은 운영 데모.</p>''',
"학생 질문 유도: 왜 모든 지역을 다 학습에 쓰지 않나? → 개발과 판정을 섞으면 자기 채점이 된다.")

sec("4 · 실험 ②", "아무 AI나 되는 거 아냐?", "두 번째 얼린 모델 Presto를 같은 자리에 세움")
slide('''<h2>같은 디코더 · 같은 split · 같은 seed에서</h2><table class="t num"><tr><th></th><th>P4 OlmoEarth</th><th>P2 raw</th><th>Presto (풀링)</th><th>Presto (원격자)</th></tr>
<tr><td>8지역 평균</td><td class="acc">.272</td><td>.197</td><td>.109</td><td>.126</td></tr><tr><td>P4에 진 지역</td><td>—</td><td>7/8</td><td class="bad">8/8</td><td class="bad">8/8</td></tr><tr><td>P2에 진 지역</td><td>—</td><td>—</td><td class="bad">8/8</td><td class="bad">8/8</td></tr></table>
<p class="lede">"풀링 때문에 진 거 아니냐"는 반론을 원격자(128×128)로 재실험 → +.017 오르지만 <em>순위는 하나도 안 바뀜</em>.</p>
<p class="note">공정성 경고: Presto는 원래 12개월 시계열용이라 우리 4시점 계약에서 불리합니다 → 이 값은 Presto의 <em>하한</em>. Prithvi·Clay는 아직 안 돌렸으니 "모든 모델보다 낫다"는 못 씁니다.</p>''',
"MS-87/93. 여기서 정직성의 톤을 잡는다: 우리에게 유리한 결과에도 경고문을 붙인다.")

sec("5 · 실험 ③", "두 모델을 합치면 더 좋아질까?", "실패담 — 그리고 사전 등록이 우리를 구한 이야기")
slide('''<h2>직관: raw는 raw대로, 명함은 명함대로 잘 보는 곳이 다를 것</h2>
<p class="lede">라벨을 <em>보고</em> 타일마다 더 나은 쪽을 고르면(oracle) 평균이 +0.024 오르는 것처럼 보였습니다(M86). "그럼 어느 쪽을 믿을지 예측하는 게이트를 학습하자!"</p>
<div class="cards"><div><b>naive 4종</b><span>평균·AND·OR·logit 평균</span></div><div><b>게이트 v1</b><span>소스 지역만으로 학습한 51특징 로지스틱</span></div><div><b>게이트 v2</b><span>+ 작동점 정합</span></div></div>''',
"학생들에게 예측시키기: 될 것 같은가? 손 들어보기.")
slide('''<h2>0.5 임계의 함정</h2><table class="t num"><tr><th>게이트 v1</th><th>임계 0.5</th><th>FP를 맞춘 작동점</th></tr>
<tr><td>P4 단독</td><td>.272</td><td>.272</td></tr><tr><td>naive 평균</td><td>.279</td><td>.281</td></tr><tr><td>게이트(soft)</td><td class="good">.286 ← 이겼다!</td><td class="bad">.255 ← 졌다</td></tr></table>
<p class="lede">두 모델을 섞으면 빈 타일의 오경보가 늘어납니다. 오경보를 같은 수준으로 맞추면 이득이 <em>뒤집혔습니다</em>. 사전에 등록해 둔 "FP 매칭 규칙"이 없었다면 +0.014를 성공으로 발표했을 겁니다.</p>''',
"MS-91. 가장 교육적인 슬라이드. '작동점'을 비유로: 화재경보 민감도를 올려서 화재를 더 잡았다고 자랑하면 안 된다 — 오경보도 같이 늘었으니.")
slide('''<h2>진짜 답: 합칠 게 없었다</h2>
<div class="big"><span>작동점 맞춘 뒤<br>oracle 이득</span><span class="acc">+0.008</span><span>(FP 매칭에선 <b>−0.004</b>)</span></div>
<p class="lede">라벨을 훔쳐보는 완벽한 선택기조차 거의 못 버는데, 학습 게이트가 벌 수 있을 리 없죠. 처음의 +0.024는 두 모델을 <em>다른 작동점</em>에서 비교해 생긴 인공물이었습니다. 등록해 둔 중단 규칙대로 v3는 만들지 않았습니다.</p>''',
"MS-92. 음성 결과가 어떻게 '해명'이 되는지. 이건 융합 논문들이 흔히 빠지는 함정이라 리뷰어 방어에 오히려 쓸모 있다.")

sec("6 · 실사건", "네팔로 돌아가서", "라벨 0장으로 만든 후보 목록은 맞았을까?")
slide(f'''<div class="two"><figure><img src="{I['grid_pre']}"><figcaption>회랑 창들 · 사건 전</figcaption></figure><figure><img src="{I['grid_post']}"><figcaption>회랑 창들 · 사건 후</figcaption></figure></div>
<h2>100창 스캔 → 구름으로 판정 불가 53 → <em>47창</em> 판정 → 상위 6곳 검토 리드</h2>''',
"각 창마다 명함 변화량과 평소 변화량(플라시보 3쌍)을 비교. 숫자 '13%'는 '이 창의 40 m 격자 중 평소 범위를 넘은 비율'이다 — 피해 면적이 아니다.")
slide(f'''<div class="three"><figure><img src="{I['v003_pre']}"><figcaption>1위 Dalphedi · 전</figcaption></figure><figure><img src="{I['v003_post']}"><figcaption>후</figcaption></figure><figure><img src="{I['v003_delta']}"><figcaption>AI Δ — 주황 = 평소 범위 초과</figcaption></figure></div>
<div class="three"><figure><img src="{I['x001_pre']}"><figcaption>대조: Tadi Khola(사건 없음) · 전</figcaption></figure><figure><img src="{I['x001_post']}"><figcaption>후</figcaption></figure><figure><img src="{I['x001_delta']}"><figcaption>AI Δ — 거의 없음 (2.3%, 장면 밖 35% 제외)</figcaption></figure></div>
<h2>사건 지역 13.3% vs 대조 지역 2.3%. "아무 일 없음"이 어떻게 보이는지가 중요합니다.</h2><p class="note">정정(오늘 아침): 이 대조 창의 35%가 위성 장면 밖(검정)이었음 — 처음 발표값 1.3%는 빈 땅으로 희석된 값. 이 슬라이드를 만들다가 발견했습니다. 사전 등록 대조 창 p009(장면 안 100%, 관측 97%)는 1.0%.</p>''',
"대조 창을 관측성으로 사후 선택했다는 한계(M77) + 오늘 발견한 0-채움 정정을 그대로 말한다 — '눈으로 확인' 규칙이 또 한 번 일한 사례. 사전 등록 대조 10곳 중 판정 가능은 1곳(p009)뿐 — 구름의 실태(M81).", "rows2")
slide('''<h2>그래서 맞았나? 외부 기관 3곳의 홍수 지도로 채점</h2><table class="t num"><tr><th>규모</th><th>결과</th></tr>
<tr><td>2.56 km 창 단위 (M86)</td><td class="bad">판정 불가 — 리드 6/6 적중이지만 비리드도 88%가 홍수 안</td></tr>
<tr><td><b>40 m 토큰 단위 (M88)</b></td><td class="good">AUROC <b>.846</b> (12만 토큰, 기저율 5.5%)</td></tr><tr><td>같은 토큰, 고전 |ΔNDVI|</td><td>.750</td></tr><tr><td>같은 토큰, 밴드 차</td><td>.694</td></tr>
<tr><td>강에서 300 m 밖 토큰만 (M89)</td><td class="good">.846 → "강줄기 그리기"가 아님</td></tr></table>
<p class="note">말할 수 없는 것: 라벨은 홍수 <em>대리</em>(피해 확정 아님), 사건 1건, AUPRC에선 post-event NDWI가 더 높음. 그래서 "탐지"가 아니라 "검토 후보".</p>''',
"학생들에게 규모의 중요성: 같은 데이터가 창 규모에선 무판별, 토큰 규모에선 .846. 어느 해상도로 질문하느냐가 답을 바꾼다.")

sec("7 · 정직성 기계", "우리가 틀렸던 것들", "사전 등록 · 봉인 · 공개 철회")
slide('''<h2>세 번의 철회</h2><div class="roles">
<div><b>M75 · 레이더 단위</b><span>선형 강도를 dB 계약에 넣음 → 라수와가디 9.8% "탐지" 철회, 재계산 후 미검출</span></div>
<div><b>M89 · 과장 하향</b><span>"강 근접성 때문이 아님" → "OSM 중심선 300/600 m 밖에서도 유지"로 정확히</span></div>
<div><b>M92 · M86 정정</b><span>+0.024 headroom은 작동점 인공물 — 융합 방향 종료</span></div><div><b>M77 정정 · 오늘</b><span>대조 창 35%가 장면 밖 0-채움 → 1.3%를 2.3%로 상향. 발표 이미지 확인 중 발견</span></div></div>
<p class="lede">규칙: 판정 기준을 <em>실험 전에</em> 파일로 쓰고 커밋한다. 결과가 나온 뒤 기준을 바꾸면 자기기만. 오늘 아침 CacheTune도 그렇게 죽었습니다(MS-94: 어댑터가 디코더 적응보다 −0.05~−0.08).</p>''',
"과학의 실제 모습. 취업·논문 관점에서 이 장부가 자산인 이유: '무엇이 안 되는지'를 증거로 아는 팀은 드물다.")

sec("8 · 다음", "캐시를 버리지 말고, 조금만 고쳐 쓰기", "Reuse or Retrain? — 라벨이 N장일 때 무엇을 업데이트해야 하는가")
slide('''<h2>행동 사다리</h2><div class="ladder">
<div><b>A0 재사용</b><span>그대로 씀 · 라벨 0</span></div><div class="acc"><b>A1 디코더 적응</b><span>라벨 5장으로 +.07~.08 (MS-94에서 확인) · 캐시 유효</span></div>
<div class="dead"><b>A2 CacheTune</b><span>저랭크 어댑터 — 게이트 불통과, 하차</span></div><div><b>A3 인코더 PEFT</b><span>캐시 무효화 · 재임베딩 비용</span></div><div><b>A4 raw 재학습</b><span>가장 비쌈</span></div></div>
<p class="lede">다음 실험은 같은 라벨 N장에서 <em>A1 vs A4 vs A3</em>. 그리고 과업을 넓힙니다: 산사태 → <em>태양광 발전소</em>(전지구) → <em>홍수</em>(레이더 필수). 한국은 3-task를 한 캐시 위에서 — "task 몇 개부터 캐시가 싸지는가".</p>''',
"MS-94: A2 어댑터는 두 지역·두 K 에서 A1 보다 −0.05~−0.08 → 등록 중단 규칙으로 하차. 양성: A1 이 라벨 5장으로 +.07~.08. MULTITASK_EXTENSION 문서 요약 — Ai2 파트너 과업(rslearn 형식)이라 계약 공학이 끝나 있음.")

sec("9 · 마무리", "다섯 줄로", "그리고 퀴즈 세 개")
slide('''<ol class="five"><li>얼린 지구 임베딩을 <em>재사용</em>하면 새 지역 산사태를 raw 학습보다 잘 찾는다 (7/8 지역).</li><li>아무 임베딩이나 되는 건 아니다 (Presto 8/8 패).</li><li>두 모델을 합치면 좋을 거란 직관은 <em>작동점 인공물</em>이었다.</li><li>라벨 0장의 실사건 후보 목록이 외부 라벨과 40 m 규모로 정합했다 (.846) — 그러나 "피해 탐지"는 아니다.</li><li>기준은 실험 전에 쓰고, 틀리면 장부에 남긴다.</li></ol>''',
"각 줄에 근거 번호를 붙여 말한다: M65, MS-87/93, MS-91/92, M88/89, M75/89/92/94.")
slide('''<h2>퀴즈</h2><ol class="quiz"><li>왜 통계 단위가 타일 6,834개가 아니라 지역 8개인가?</li><li>게이트 v1이 0.5 임계에서 이기고 FP 매칭에서 진 이유를 한 문장으로.</li><li>네팔 결과에서 "탐지했다"고 쓰면 안 되는 이유 두 가지.</li></ol>
<p class="note">더 읽기: RESTART_HERE.md → RESEARCH_TOPIC_ALIGNMENT → MEASURED_FINDINGS(M65, MS-87, MS-91/92, M88) → REGION_ROLES → MULTITASK_EXTENSION</p>''',
"답: (1) 같은 지역 타일은 독립이 아님 (2) 섞으면 빈 타일 오경보가 늘어 임계를 올려야 하고 그때 재현율이 더 깎임 (3) 대리 라벨·사건 1건·AUPRC 역전.")

# ---------- HTML ----------
css = """
:root{--bg:#101418;--bg2:#171c22;--ink:#E9E4D8;--mute:#8C948F;--acc:#D9743A;--teal:#3FB8A8;--good:#7FB77E;--bad:#D46A6A;--line:#2a3138}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--ink);font-family:'IBM Plex Sans KR',system-ui,sans-serif;height:100%;overflow:hidden}
.deck{position:relative;width:100vw;height:100vh}
.slide{position:absolute;inset:0;padding:6vh 7vw;display:none;flex-direction:column;justify-content:center;gap:2.2vh;background:var(--bg)}
.slide.on{display:flex}.slide.section{background:var(--bg2);justify-content:center}
.kicker{font-family:'IBM Plex Mono',monospace;color:var(--acc);letter-spacing:.14em;font-size:1.1rem;margin:0}
h1.sec{font-family:'Gowun Batang',serif;font-weight:700;font-size:clamp(2.4rem,6vw,5rem);line-height:1.12;margin:0;text-wrap:balance}
.sub{color:var(--mute);font-size:1.3rem;margin:0}
h2{font-family:'Gowun Batang',serif;font-weight:700;font-size:clamp(1.5rem,3.2vw,2.6rem);line-height:1.25;margin:0;text-wrap:balance}
h2 em,.lede em,li em{color:var(--acc);font-style:normal}
.lede{font-size:1.25rem;line-height:1.6;max-width:70ch;margin:0}.note{color:var(--mute);font-size:1rem;line-height:1.55;max-width:80ch;margin:0}
.two{display:grid;grid-template-columns:1fr 1fr;gap:2vw}.three{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.6vw}.two img{max-height:46vh;object-fit:cover}.three img{max-height:38vh;object-fit:cover}.slide.rows2 .three img{max-height:24vh}
figure{margin:0}figure img{width:100%;display:block;border:1px solid var(--line)}figcaption{font-family:'IBM Plex Mono',monospace;color:var(--mute);font-size:.8rem;margin-top:.5em}
.hero img{max-height:62vh;width:auto;max-width:100%;margin:0 auto}
.big{display:flex;gap:4vw;align-items:baseline;flex-wrap:wrap}.big span{font-family:'Gowun Batang',serif;font-size:clamp(1.6rem,3.6vw,3rem);line-height:1.2}.big span.acc,.big b{color:var(--acc)}
.cards{display:flex;gap:1.2vw;align-items:stretch;flex-wrap:wrap}.cards div{flex:1 1 200px;border:1px solid var(--line);padding:1.1em 1.2em;background:var(--bg2)}
.cards div.acc{border-color:var(--acc)}.cards .arrow{flex:0 0 auto;border:0;background:none;font-size:2rem;color:var(--mute);align-self:center}
.cards b{display:block;font-size:1.05rem}.cards span{display:block;color:var(--mute);font-size:.95rem;margin-top:.4em;line-height:1.45}
table.t{border-collapse:collapse;width:100%;font-size:1.05rem}table.t th,table.t td{text-align:left;padding:.55em .8em;border-bottom:1px solid var(--line)}
table.t th{color:var(--mute);font-family:'IBM Plex Mono',monospace;font-size:.85rem;letter-spacing:.06em;font-weight:500}table.num td{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums}
td.acc{color:var(--acc);font-weight:700}td.good{color:var(--good)}td.bad{color:var(--bad)}tr.sum td{border-top:2px solid var(--ink);font-weight:700}
.loro{display:grid;grid-template-columns:repeat(4,1fr);gap:.8vw}.loro span{border:1px solid var(--line);padding:1em;font-family:'IBM Plex Mono',monospace;text-align:center}.loro .hold{border-color:var(--acc);color:var(--acc)}
.roles{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1vw}.roles div{border-left:3px solid var(--teal);padding:.2em 1em}.roles b{display:block}.roles span{color:var(--mute);font-size:.95rem;line-height:1.45}
.ladder{display:grid;grid-template-columns:repeat(5,1fr);gap:.8vw}.ladder div{border:1px solid var(--line);padding:1em;min-height:9em}.ladder .acc{border-color:var(--acc)}.ladder .dead{opacity:.45;text-decoration:line-through}.ladder b{display:block}.ladder span{color:var(--mute);font-size:.9rem;line-height:1.4;display:block;margin-top:.4em}
ol.five{font-size:1.35rem;line-height:1.7;max-width:75ch;padding-left:1.4em}ol.quiz{font-size:1.4rem;line-height:1.8;padding-left:1.4em}
.bar{position:fixed;left:0;top:0;height:3px;background:var(--acc);width:0;z-index:5}
.hud{position:fixed;right:1.2vw;bottom:1vh;font-family:'IBM Plex Mono',monospace;color:var(--mute);font-size:.8rem}
.notes{position:fixed;left:0;right:0;bottom:0;background:#000c;color:var(--ink);padding:1em 7vw;font-size:1rem;line-height:1.5;display:none;border-top:1px solid var(--acc)}.notes.on{display:block}
.help{position:fixed;left:1.2vw;bottom:1vh;font-family:'IBM Plex Mono',monospace;color:var(--mute);font-size:.75rem}
@media (max-width:820px){.slide{padding:4vh 5vw}.two,.three,.roles,.ladder{grid-template-columns:1fr}.loro{grid-template-columns:1fr 1fr}}
@media (prefers-reduced-motion:no-preference){.slide.on{animation:fade .25s ease}}@keyframes fade{from{opacity:0}to{opacity:1}}
"""
slides_html="\n".join(f'<section class="slide {c}" data-n="{i}">{h}</section>' for i,(h,n,c) in enumerate(S))
notes_json=json.dumps([n for _,n,_ in S],ensure_ascii=False)
html=f"""<title>산이 무너진 날, 위성은 100곳을 훑었다</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700&family=IBM+Plex+Sans+KR:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}</style>
<div class="bar" id="bar"></div>
<main class="deck" id="deck">{slides_html}</main>
<div class="notes" id="notes"></div>
<div class="help">← → 이동 · N 발표자 노트 · Home/End</div><div class="hud" id="hud"></div>
<script>
const NOTES={notes_json};const S=[...document.querySelectorAll('.slide')];let i=Math.max(0,Math.min(S.length-1,parseInt(location.hash.slice(1))||0));let showNotes=false;
function go(n){{i=Math.max(0,Math.min(S.length-1,n));S.forEach((s,k)=>s.classList.toggle('on',k===i));document.getElementById('bar').style.width=((i+1)/S.length*100)+'%';document.getElementById('hud').textContent=(i+1)+' / '+S.length;const nb=document.getElementById('notes');nb.textContent=NOTES[i]||'';nb.classList.toggle('on',showNotes&&!!NOTES[i]);history.replaceState(null,'','#'+i);}}
addEventListener('keydown',e=>{{if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown')go(i+1);else if(e.key==='ArrowLeft'||e.key==='PageUp')go(i-1);else if(e.key==='Home')go(0);else if(e.key==='End')go(S.length-1);else if(e.key.toLowerCase()==='n'){{showNotes=!showNotes;go(i);}}}});
addEventListener('hashchange',()=>{{const n=parseInt(location.hash.slice(1));if(!isNaN(n)&&n!==i)go(n);}});
document.getElementById('deck').addEventListener('click',e=>{{if(e.target.tagName==='IMG')return;go(e.clientX>innerWidth/2?i+1:i-1);}});go(i);
</script>"""
out=HERE/"olmoearth_session_2026_09.html"; out.write_text(html,encoding="utf-8"); print("slides",len(S),"bytes",out.stat().st_size)
