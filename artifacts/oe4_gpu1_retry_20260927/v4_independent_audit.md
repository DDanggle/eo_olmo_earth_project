**GPU1 VLM v4 독립 감사 — 2026-09-27**

**PASS: 실제 H5 기반 OlmoEarth encoder → adapter → frozen Qwen3-VL의 1-step 학습 경로가 FP32에서 통과했다.** snapshot_04 원본 파일 4개의 SHA256·크기, controller와 receipt, 로컬 소스 4개의 hash 및 35개 일관성 검사를 재계산해 모두 통과했다. 추가 실험이나 서버 쓰기는 없었다. 실제 tensor를 GPU에서 다시 재생한 감사는 아니다.

| 확인 항목 | 검산 결과 |
|---|---|
| 최종 상태 | PASS_live_encoder_adapter_vlm_gradient_contract; controller completed, returncode 0 |
| 장치 | 물리 GPU 1, UUID GPU-8b485982-e8bb-004a-a10a-e37637e5e2bb; CUDA_VISIBLE_DEVICES 일치 |
| 완료·시간 | 2026-09-27 15:43:42.905685 KST; receipt 89.877104초 |
| 최대 PyTorch 할당 | 37,108,215,296 bytes = 34.559719 GiB |
| 정밀도 | FP32, float32 matmul highest, cuDNN TF32 false |
| native RGB cache max / mean delta | 2.67028808594e-05 / 4.89206104248e-06 |
| EO 포함 cache max / mean delta | 2.38418579102e-05 / 4.06409299103e-06 |
| cache 기준 | 양쪽 allclose·top1 true; v3와 동일 atol .15 / rtol .01 |
| EO 주입·cache | prefill 1회, cached decode bypass 1회; native/EO cache 길이 85/101 |
| RGB·DeepStack 보존 | 이미지 및 DeepStack 3층 feature 기록 hash 일치; 원래 RGB logits는 hook 설치 전후 및 EO update 이후 bitwise 동일 |
| 실제 gradient 행 재집계 | encoder 211/211, adapter 4/4: 전부 유한한 양수 norm, 이름 중복 0 |
| EO feature / embedding gradient norm | 0.173413202167 / 13.8959159851 |
| clipping 전 전체 gradient norm | 454.317596436; 소스에서 norm 1.0 clipping 후 step |
| 실제 추적 가중치 변경 | encoder 4개 전부 양수: 9.98377799988e-07–1.01327896118e-06; adapter 4개 전부 양수: 9.99891854008e-05–0.000100016593933 |
| frozen VLM | gradient parameter 수 0, optimizer 등록 parameter 수 0 |
| optimizer / CE | 1 step; loss 6.2336306572; target “The image shows land cover.” |

제어기·로그·receipt의 종료 상태가 일치한다. 보호 스크립트 4개, 실행 소스 및 입력 파일의 hash·mtime·크기는 실행 전후 동일하다. 실행 worker source SHA256은 `b7917b3076f1688ac8f0557377b69be419699538069538b2c12d8140a014531e`, 최종 receipt SHA256은 `69f77f5fe63e17509d4389aa16d14f19a8c832f0a5605089171ac94181bc663d`다. native 완료 상태와 준비 데이터 manifest hash도 이전 native 산출물과 일치한다.

**해석 범위.** 일반 CE 문장과 2-token 생성 “The image”는 정답 평가가 아니다. 이미지 1개·batch 1·1 step에서 학습 신호가 실제 EO encoder까지 전달되고 가중치가 바뀌는 경로를 입증했다. 전체 VLM 학습, 의미 정확도, 성능 향상 또는 일반화는 검증하지 않았다. 가중치 변화는 추적한 encoder 4개와 adapter 4개의 실제 delta 기록이며 전체 encoder tensor의 delta를 계산한 결과는 아니다. 소스에는 VLM 단계의 checkpoint 저장·재로딩 검증이 없다.

**native 결과와의 연결.** 앞서 native smoke 8 step 및 development 32 step은 실제 encoder 업데이트와 저장·재로딩을 통과했다. 각각 dev total 변화는 −0.000163003802 및 +0.000213615584로 성능 향상이 일관되게 관측된 것은 아니다. 이번 VLM provider는 원본 Base weights `57f7b66faf206db1307670673839e639d3a19c305f6ad968c62392ad3e88deec`를 강제한다. 따라서 native checkpoint_after를 이어 학습한 VLM 결과나 CPT 효과의 증거가 아니다. VLM provider 입력은 실제 Sentinel-2 두 시점이며, native 학습에서 쓴 WorldCover는 이번 VLM encoder의 입력 modality가 아니다.

**BF16 이력.** v3의 cache 실패(max .25, mean .0362606160, top1 true)는 보존한다. v4는 동일 허용오차로 FP32 cache와 native RGB control이 통과한 별도 실행이며 BF16 해결 또는 실패 원인이 오직 반올림이었다는 결론은 아니다.

근거: [v4 receipt](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/artifacts/oe4_gpu1_retry_20260927/snapshot_04/runs/vlm_interface_gpu1_v4/receipt.json), [v4 controller](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/artifacts/oe4_gpu1_retry_20260927/snapshot_04/vlm_sequence_gpu1_v4_status.json), [실행 소스](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/oe4_native_v12_v0/vlm_snapshot_v4/vlm_interface_smoke.py), [provider](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/oe4_native_v12_v0/vlm_snapshot_v4/real_h5_provider.py). 기계판독 검산값은 같은 이름의 JSON에 있다.
