# E8 입력 계약: EarthDial (CVPR 2025) — 2026-09-26

E8 = 우리 content-use 통제 프로토콜(실제 / 같은 타일 영상 교환 / 빈 영상 / 단일 영상)을 공개 EO-VLM인
EarthDial에 적용하는 실험. 이 문서는 **모델을 한 번도 돌리지 않고** 1차 출처(repo 코드, HF config, 논문)만
읽어 정리한 입력 계약이다. 실행 코드: `code/e8_earthdial_protocol_v0.py`.

출처 고정점
- repo: `hiyamdebary/EarthDial` @ `92c1260703ae8b3ddcac17fce6e8234de717308b` (2025-06-20). 아래 링크 전부 이 커밋 기준.
  `R = https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b`
- HF: `akshaydudhane/EarthDial_4B_RGB` @ `ae752c47`, `EarthDial_4B_MS` @ `835030e1`, `EarthDial_4B_Methane_UHI` @ `b97a203a`
  (config.json·added_tokens.json·inference.py만 받음, 가중치는 받지 않음)
- 논문: arXiv 2412.15190 (HTML 판) §3.1–3.2, Table 2

## 0. 먼저: 약점과 미검증 (요약)

1. **실행으로 검증한 것은 없다.** 아래 전부 코드·config를 읽은 결과다. selftest는 순수 로직(arm 구성, 채점)만 확인한다.
2. **HF 체크포인트에 모델 코드가 없다.** config의 `auto_map`은 `modeling_internvl_chat.py`를 가리키지만 파일 목록에 없다.
   → `trust_remote_code`로 `AutoModel` 로딩은 실패할 것으로 예상. repo의 `src/`를 import해야 한다(그들의 `inference.py`도 그렇게 함).
3. **transformers 버전.** repo는 `transformers==4.37.2` 고정([`pyproject.toml`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/pyproject.toml)). 번들된 Phi-3 코드가
   최신 transformers(서버 `.venv-master`는 Olmo-3 때문에 훨씬 최신일 것)에서 `generate(inputs_embeds=…)`·캐시 API로 깨질 수 있다.
   probe 전에 별도 venv(4.37.2 + peft + timm 0.9.12 + einops + sentencepiece 0.1.99)를 준비하는 편이 안전하다. 미검증.
4. **우리 과제용 템플릿은 없다.** Sentinel-2 bi-temporal용 센서 태그가 없다(아래 §3). 가장 가까운 공식 템플릿을 빌렸고,
   날짜를 넣은 것은 우리 변경이다.
5. **_MS / SAR 부분은 거의 전부 추론**(§5). 특히 SAR이 _MS에 들어 있다는 것은 config의 `_name_or_path` 계보에서 추론한 것이다.

## 1. 추론 API (RGB)

| 항목 | 값 | 근거 |
|---|---|---|
| 클래스 | `earthdial.model.internvl_chat.InternVLChatModel.from_pretrained(dir, torch_dtype=bf16, low_cpu_mem_usage=True).eval().cuda()` | [`rs_change_detection_test.py:345-350`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/eval/rs_change_detection/rs_change_detection_test.py#L345-L350), HF `inference.py` |
| 토크나이저 | `AutoTokenizer.from_pretrained(dir, trust_remote_code=True, use_fast=False)` (sentencepiece `tokenizer.model`) | 같은 곳 |
| 호출 | `model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=…)` | [`modeling_internvl_chat.py:447-502`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/model/internvl_chat/modeling_internvl_chat.py#L447-L502) |
| 대화 템플릿 | `phi3-chat`: `<\|system\|>\n{InternVL 중국어 system msg}<\|end\|><\|user\|>\n{q}<\|end\|><\|assistant\|>\n`, stop = `<\|end\|>`(32007) | [`conversation.py:388-404`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/conversation.py#L388-L404), config `template` |
| 영상 토큰 | 각 `<image>` → `<img>` + `<IMG_CONTEXT>`×(256×타일 수) + `</img>`. 448/14=32 → pixel shuffle 0.5 → 16×16=256 | `modeling_internvl_chat.py:57, 477-479` |
| 전처리 | RGB 변환 → 448×448 BICUBIC resize → ToTensor → ImageNet mean/std | [`dataset.py:330-336`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/train/dataset.py#L330-L336) |
| 동적 타일 | `dynamic_preprocess`(1–6 타일 + 썸네일, config `max_dynamic_patch=6`, `use_thumbnail=true`) | [`dataset.py:770-823`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/train/dataset.py#L770-L823) |
| 디코딩 | README: "greedy decoding"으로 평가. 단 CD eval 스크립트 기본 `num_beams=5`(`--dynamic`이면 1로 강제) | README, `rs_change_detection_test.py:314, 355-357` |
| 크기 | safetensors 합계 8.29 GB (bf16 ≈ 4.15B 파라미터). 추론 VRAM ≈ 9–10 GB 예상(미측정) | `model.safetensors.index.json` |
| flash-attn | 필수 아님. ViT는 import 실패 시 naive attention으로 fallback([`modeling_intern_vit.py:22-27,118-120`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/model/internvl_chat/modeling_intern_vit.py#L22-L27)); Phi-3도 import 실패 시 경고만. 설치 권장 버전 `flash-attn==2.3.6` | README, `modeling_phi3.py:48-62` |
| 기타 의존성 | `modeling_internvl_chat.py`가 `peft`를 import(17행). `train/dataset.py`는 `decord`·`cv2`를 import하므로 우리 스크립트는 `dynamic_preprocess`를 복사해 씀(동일성 로컬 확인) | |

우리 PNG는 512×512(`code/sentinel_qa_gen.py:11-14`, 밴드 [2,1,0]/3000 clip)이므로 `dynamic_preprocess`는 max_num 3이든 6이든
**타일 1개**를 준다(썸네일은 타일이 2개 이상일 때만 붙음). 즉 영상당 256 토큰.

## 2. 여러 장(bi/multi-temporal) 입력

- 학습: 영상마다 타일을 만들고(`max_num = max(1, 6 // n_images)`), `pixel_values`를 이어 붙이고, 대화 속 `<image>`를
  **영상마다 하나씩** 순서대로 치환한다. [`dataloader.py:318-386`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/train/dataloader.py#L318-L386),
  [`dataset.py:602-604`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/train/dataset.py#L602-L604) (`preprocess_phi3`). 즉 올바른 추론은
  `num_patches_list=[영상1 타일 수, 영상2 타일 수]`.
- **공식 CD eval의 특이점:** [`rs_change_detection_test.py:260-266`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/eval/rs_change_detection/rs_change_detection_test.py#L260-L266)은
  `num_patches_list`를 넘기지 않는다 → `chat()`이 `[전체 타일 수]`로 처리해 **첫 `<image>`에 두 영상 토큰을 모두 넣고, 두 번째
  `<image>`는 문자열로 남는다**(`modeling_internvl_chat.py:454-455, 477-479`). 토큰 수가 맞아 오류는 없다. 그들의 논문 수치가
  이 경로로 나왔는지는 모름. 우리는 학습과 같은 영상별 `num_patches_list`를 쓴다(스크립트에서 `<image>` 개수 = 영상 수 assert).
- 논문 §3.1: "each image through the ViT … stack and concatenate these tokens" — 위 코드와 일치.
- `[s1_vh_temp_10]`은 `constants.py:27`에 있으나 **토크나이저 added_tokens에는 없다**(일반 텍스트로 쪼개짐).

## 3. 태그와 변화/재해 템플릿 (공식 eval 질문 원문)

태그 토큰(added_tokens.json, RGB·MS 동일): 과제 `[changedet] [identify] [grounding] [refer] [classify] [caption] [treeclassify]`,
센서 `[s2_rgb_10] [s2_ms_10] [l8_rgb_30] [hr_rgb_0.5] [hr_rgb_temp_0.5] [hr_rgbi_0.5] [s1_vh_10] [s1_vh_1]`
([`constants.py:17-47`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/train/constants.py#L17-L47)).

공식 eval 결과 파일의 질문 원문([`results/*.jsonl`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/eval/rs_change_detection/results)):
- 변화 캡션(LEVIR/Dubai/SYSU): `[changedet] [hr_rgb_temp_0.5] <image> <image> \n are there any semantic change detected in the provided images?`
- xBD 재해 유형: `[changedet] [hr_rgb_temp_0.5] <image> <image> \n Analyze the images to identify the type of disaster that occurred. Options: …`
- xBD 예/아니오(가장 가까운 형태): `[hr_rgb_temp_0.5] <image>  <image> \n Are there any buildings affected due to disaster? Please answer yes or no."Answer in single word.`
- 여기서 `\n`은 **백슬래시+n 두 글자**(JSON에 `\\n`). 검출 eval은 이를 실제 개행으로 바꾸지만(`detection_test.py` `org_prompt→req_prompt`),
  CD eval은 치환 줄이 주석 처리돼 있어 문자 그대로 들어간다(`rs_change_detection_test.py:178`).

**E8 선택(기본 `--template earthdial --sensor hr_temp`):**
```
2장: [changedet] [hr_rgb_temp_0.5] <image> <image> \n Did a landslide occur between the first image (taken on {d1}) and the second image (taken on {d2})? Please answer yes or no.
1장: [s2_rgb_10] <image> \n Is there a landslide visible in this image? Please answer yes or no.
```
이유: (a) 모델이 학습한 유일한 bi-temporal 형식이 `[changedet] [hr_rgb_temp_0.5] <image> <image> \n …`이다. (b) 해상도 태그
(0.5 m)는 틀리지만 S2용 temporal 태그가 없다. (c) 날짜는 우리 추가 — 없으면 같은 타일 두 항목의 프롬프트가 똑같아져 교환 arm이
"텍스트를 따른다"를 판별할 수 없다. 민감도 분석용으로 `--sensor s2`(2장에도 `[s2_rgb_10]`)와 `--template plain`
(`Image-1 ({d1}): <image>\nImage-2 ({d2}): <image>\n…`)을 둔다. 어느 쪽을 주 분석으로 할지는 사전등록에서 고정해야 한다.

참고로 그들의 태그 사용은 일관되지 않다: LCZ(Sentinel-2 다중분광) 테스트 질문이 `[s1_vh_10]`로 시작한다
(`rs_classification/results/rs_LCZ_test.jsonl`). 태그를 "정답 형식"으로 과신하지 말 것.

## 4. 이 과제가 EarthDial의 학습 분포 밖인 지점

- 산사태 데이터셋 없음(논문 Table A.1). 가장 가까운 것은 xBD(항공 0.5 m 재해 전후), QuakeSet(S1 SAR 지진 유무), FloodNet(항공).
- S2 10 m RGB는 Stage 1 사전학습(Satlas/SkyScript)과 BigEarthNet 분류에 있으나, **S2 bi-temporal 변화 질의는 학습에 없다**
  (Stage 2 temporal은 전부 `jpg_A,jpg_B` 고해상도 RGB: [`Stage2_RGB_Temporal_Finetunning.json`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/shell/data/Stage2_RGB_Temporal_Finetunning.json)).
- 따라서 E8의 낮은 d_real은 "모델이 영상을 안 읽는다"가 아니라 "과제가 분포 밖"일 수 있다. E0와 같은 해석 규칙
  (`d_real < 0.10` → uninformative)을 쓴다.

## 5. _MS 체크포인트: 다중분광·SAR 입력 (코드 판독, 실행 미검증)

**경로.** 3채널이 아닌 입력은 모델에 들어가기 **전에** 데이터 쪽에서 `model.sequential_vit_features(pixel_values, 'bilinear')`를
불러 ViT 특징을 미리 만든다([`dataloader.py:263-282`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/train/dataloader.py#L263-L282) 단일, `:355-363` 다중;
eval: [`classification_test.py:160-174`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/eval/rs_classification/classification_test.py#L160-L174)).
그 결과 `[N, 1024, 1024]`(토큰×ViT 차원)가 `pixel_values`로 `chat()`에 들어가고, `extract_feature`는 `shape[2]==1024 or ndim==3`이면
ViT를 건너뛰고 pixel shuffle+MLP만 적용한다([`modeling_internvl_chat.py:372-397`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/model/internvl_chat/modeling_internvl_chat.py#L372-L397)).

**3채널 묶음.** [`sequential_vit_features`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/model/internvl_chat/modeling_internvl_chat.py#L294-L370):
채널을 저장 순서대로 3개씩 끊어(`i:i+3`) ViT에 넣고, 3개가 안 되면 마지막 채널을 복제해 채운다. CLS 제거 후 묶음별
`[1, 1024 tok, 1024 dim]`을 얻는다.

**"bilinear" 병합의 실제 동작.** [`bilinear_interpolate_and_concat`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/model/internvl_chat/modeling_internvl_chat.py#L237-L292):
묶음 k개를 ⌈√k⌉×⌈k/행⌉ 격자로 놓고 각 `[1024 tok × 1024 dim]` 행렬을 **(토큰 축, 특징 차원 축) 평면에서** bilinear 축소한 뒤 이어
붙여 다시 1024×1024를 만든다. 즉 공간 격자가 아니라 "평탄화된 토큰 순서 × 은닉 차원"을 이미지처럼 보간한다. 예: 12밴드 → 4묶음 →
각 묶음이 토큰 512개 × 차원 512개를 차지. 묶음이 1개(SAR 1–2채널)면 항등 보간이다. 논문 §3.1의 "aggregation and reduction in size
using bilinear interpolation via the AnyRes block"이 이것. 코드에서 읽은 그대로이며 의도와 다를 가능성도 있다.

**정규화·밴드 순서 (Stage 3 설정 [`Stage3_MS_SAR_finetunning.json`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/shell/data/Stage3_MS_SAR_finetunning.json)).**

| 데이터 | 키 | bands | normalization | 비고 |
|---|---|---|---|---|
| BigEarthNet S2 | `tif_ms` | 12 | `s2_l2a`: SSL4EO 12밴드 mean/std([`constants.py:55-57`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/train/constants.py#L55-L57)) | 밴드 순서는 파일 저장 순서. B1..B12(B10 제외) 추정, 미확인 |
| LCZ42 (So2Sat S2) | `tif_ms` | 10 | `s2_norm`: mean 0 std 1(이미 스케일된 반사율 가정) | |
| TreeSatAI | `rgbi` | 4 | /255 | |
| Satlas S1, SAR 선박 | `png`/`jpg` | 1 | `s1`: mean −20.26, std 5.91 (dB) | |
| QuakeSet (S1 전후) | `jpg_pre,jpg_post` | config 1, 실제 배열 `[512,512,2]` float32 | `s1` | HF `dataset_info.json`. 2채널(VV,VH 추정)→3채널 패딩, 영상별 1024×1024 특징 |

입력 형식은 HF `datasets` arrow의 float 배열(C,H,W 또는 H,W,C→transpose)이며, 다중분광 경로는 **동적 타일을 쓰지 않는다**
(`dynamic_image: false`, 448로 `MultiChannelResize` [`dataset.py:239-250`](https://github.com/hiyamdebary/EarthDial/blob/92c1260703ae8b3ddcac17fce6e8234de717308b/src/earthdial/train/dataset.py#L239-L250)).

**어느 체크포인트가 SAR을 아는가.** 공개 체크포인트는 3개뿐(RGB, MS, Methane_UHI; 작성자 계정 목록). config `_name_or_path`
(학습 시작 가중치 경로로 해석):
- RGB: `…/4B_Full_6Nov_pretrain_VIT_MLP_LLM_1` → Stage 1 가중치에서 시작 = **Stage 2(RGB+temporal) 산출물**
- MS: `…/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change` → Stage 2에서 시작 = **Stage 3(MS+SAR) 산출물**
- Methane_UHI: `…_RGBFinetune_Change_MS` → MS에서 시작, `[l8_ms_30] [uhi] [hyper_rgb_3]` 토큰 추가(vocab 32038)

Stage 3 설정에 S1(Satlas_S1, 선박, QuakeSet 변화)이 포함되므로 **SAR은 _MS가 담당**하는 것으로 추론된다. 별도 SAR 체크포인트는
공개되지 않았다(`EarthDial_4B_SAR` 조회 시 없음). 이 추론은 `_name_or_path` 해석에 기대므로 미검증.

## 6. 미검증 목록 (실행 전 확인할 것)

| # | 항목 | 확인 방법 |
|---|---|---|
| U1 | repo `src/` + 서버 transformers 버전으로 로딩·`chat()`이 도는가 | probe 1회. 실패 시 4.37.2 전용 venv |
| U2 | 영상별 `num_patches_list`와 공식 eval 방식(첫 `<image>`에 몰아넣기) 중 무엇이 그들 수치의 경로인가 | 공식 xBD 샘플 몇 개로 두 방식 답 비교(선택) |
| U3 | 질문 속 `\n`이 학습 데이터에서도 문자 그대로였는가 | 학습 arrow의 `conversations` 열 확인(파일이 수 GB라 미실시) |
| U4 | `[hr_rgb_temp_0.5]` vs `[s2_rgb_10]`, 날짜 유무가 답을 바꾸는가 | `--sensor`/`--template` 민감도 arm |
| U5 | _MS가 SAR을 안다(=Stage 3 산출물) | _MS로 S1 단일 영상 질의 몇 개, QuakeSet 형식 재현 |
| U6 | S2 다중분광의 밴드 순서·정규화(`s2_l2a` 12밴드 순서) | BigEarthNet_S2 test arrow 1개 확인 |
| U7 | "bilinear" 병합이 (토큰, 차원) 평면 보간이라는 판독 | _MS로 12밴드 텐서 1개 통과시켜 중간 shape 로깅 |
| U8 | 추론 VRAM·속도 | probe 로그 |
| U9 | Kuro Siwo S1 홍수(E2 데이터)를 _MS에 넣는 형식: 1채널 VH dB + `s1` 정규화 + `[s1_vh_10]`, 2장 전후 | 계획만. v0 스크립트는 S2 RGB 산사태만 다룸 |

## 7. 서버 준비 (실행 전, 이 문서 작성 시점에는 하지 않음)

```bash
# 가중치
huggingface-cli download akshaydudhane/EarthDial_4B_RGB --local-dir /home/work/data/olmoearth/models/EarthDial_4B_RGB
# 코드 (HF에 모델 코드 없음)
git clone https://github.com/hiyamdebary/EarthDial /home/work/data/olmoearth/third_party/EarthDial
git -C /home/work/data/olmoearth/third_party/EarthDial checkout 92c1260703ae8b3ddcac17fce6e8234de717308b
# 실행
CUDA_VISIBLE_DEVICES=1 env -u PYTHONPATH .venv-master/bin/python -B code/e8_earthdial_protocol_v0.py --probe
```
