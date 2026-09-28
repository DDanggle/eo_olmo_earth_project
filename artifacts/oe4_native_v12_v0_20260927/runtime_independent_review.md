# OE4 실제 H5 학습 경로 독립 검토

검토일: 2026-09-27. 담당: prior-art redteam agent. 코드를 수정하거나 서버·GPU에서 실행하지 않았다. 아래 판정은 정적 코드와 공식 소스 대조에 한정된다.

**판정: 새 출력 경로와 실제 유휴 GPU를 확보한 뒤, 예정된 8-step engineering smoke를 실행해도 된다. 현재 즉시 중단할 코드 결함은 찾지 못했다. 다만 저장본 재로딩, WorldCover loss의 유효 기여, 비활성 band dropout은 명시적으로 다뤄야 한다. 결과가 좋아도 downstream 개선·공식 CPT 재현·새 방법의 효능을 뜻하지 않는다.**

## 검토한 고정본

- `runtime_smoke.py`: SHA256 `a1633b8ce66b7cb12e466b3bc75ae219785da5714b2490c47e8a86492006f71a`
- `loader_contract.md`: SHA256 `005e49e61febcb646d6b630a277293b5a70d5ce5dd7d202bc0b4e498a678a39e`
- `official_source_0497dfb.tar.gz`: SHA256 `a984c40296ec8e1e825c77bf7d61b75f66e4c72152449f04a642498acc97d3c1`
- 공식 소스 commit: `0497dfbb6711ded4e6bf10cf089fc1e4d58c186b`; 모델 revision: `2e99a734a30e9aeb993aaa39946c6dcf554739e0`.
- 추가 검토: `acquire_assets.py`, `inspect_runtime.py`, 공식 `latent_mim.py`, `model_loader.py`, `train/loss.py`, `train/masking.py`, `train/train_module/{contrastive_latentmim,train_module}.py`, `nn/{st_model,flexi_vit}.py`, `data/dataset.py`, `dataset/{sample,convert_to_h5py}.py`, `scripts/official/v1_2/base.py`.

## 수정·확인할 사항

### 1. WorldCover loss가 실제로 기여했는지 별도로 확인해야 한다 — 결과 해석 P1, 실행 blocker 아님

`runtime_smoke.py:287–296, 329–366`은 base와 InfoNCE를 기록하지만, 최종 통과 조건은 encoder의 gradient와 weight 변화다. 이 조건은 **InfoNCE만** 작동해도 만족한다. 현재 S2만 online 입력이고 WorldCover만 decode 대상이므로, 작은 crop이 균질하면 공식 loss의 same-target negative masking이 모든 WorldCover sample의 기여를 건너뛸 수 있다. 공식 `train/loss.py:598–643`은 valid negative가 없는 sample을 건너뛴 뒤 contributing sample 수로 나눈다.

현재 로그의 `base`를 반드시 확인하고, 0이면 이를 실패 숨김 없이 `contrastive-only effective update`로 보고해야 한다. 더 좋은 기록은 view별 WorldCover decoder token 수와 same-target masking 이후 기여 sample 수다. 후속 run을 위해 이 카운트를 추가하는 것은 적절하지만, 사후에 loss가 좋은 crop만 골라 현재 run을 대체해서는 안 된다. 전체가 0이면 사전에 정한 더 큰 crop/추가 native target을 새 설정으로 실행해야 한다. “두 공식 목적의 경로를 모두 검증했다”라는 문구는 비제로 base 기여를 확인한 뒤에만 사용한다.

### 2. 공식 pretraining의 band dropout은 현재 비활성 — scope 기록 P2

`runtime_smoke.py:239, 322`의 로딩과 `model.train()`만으로는 v1.2 band dropout이 켜지지 않는다. 공식 `nn/st_model.py:812–826`은 patch embed의 활성 dropout rate를 0으로 만들고, `enable_band_dropout()`을 별도로 요구한다. 공식 trainer는 `train/train_module/train_module.py:225–232`에서 **online encoder에만** 이 메서드를 호출한다.

이번 실행은 “공식 loss component를 이용한 smoke”이므로 비활성 상태로 실행해도 된다. 다만 receipt와 계약에 `band_dropout_configured_rate`, `band_dropout_active_rate`, `disabled_for_engineering_smoke`를 기록하는 편이 정확하다. 원래 pretraining recipe에 더 가까운 run에서는 online encoder만 활성화하고 target은 계속 0인지 확인한다. 단순히 target 포함 모든 모듈에 활성화 메서드를 적용하면 안 된다.

### 3. 저장 checkpoint를 실제 다시 읽어보지는 않는다 — artifact 검증 P2

`runtime_smoke.py:359–365`는 full state_dict와 config를 저장하지만 `load_model_from_path`로 재로딩하거나 동일 입력의 loss/output을 재검증하지 않는다. state_dict 구조는 공식 loader와 맞지만, 현재 증거는 “저장했다”까지이며 “재사용 검증 완료”는 아니다. 저장된 config에는 원래 flash-attn 설정이 남아 있고 런타임 플래그만 SDPA로 바꿨으므로, portable reload 검사에도 같은 backend 설정을 적용해야 한다.

저장 후 별도 process 또는 기존 model 해제 후 재로딩하여 strict load 성공과 같은 dev seed의 loss 일치를 기록하면 충분하다. optimizer state가 없으므로 이를 trainer resume checkpoint라고 부르면 안 된다. 최초 GPU 실행 자체를 이 보완 때문에 지연할 필요는 없다.

### 4. 900초는 강제 timeout이 아니다 — 운영 P2

`runtime_smoke.py:317–319`의 wall-time 검사는 optimizer step 시작 전에만 있다. 최초 로딩·dev-before, 실행 중인 CUDA op, dev-after·checkpoint 저장은 이 guard로 중단되지 않는다. 서버 실행기가 hard wall bound가 필요하면 전체 process의 timeout을 관리해야 한다. `max-wall-seconds`를 모든 연산에 대한 강제 보장이라고 보고하지 않는다.

### 5. 준비·증거의 소규모 보완 — P3

- `runtime_smoke.py:109–113`이 loader/loss/masking/recipe와 checkpoint를 해시하지만 실제 encoder/normalizer 소스의 변경은 이 네 파일 해시만으로 검출하지 못한다. 상위 acquisition/launch receipt에 보관한 전체 source tar SHA와 git commit을 결과 receipt에 연결하면 충분하다.
- `acquire_assets.py`의 이미 다운로드된 asset 재사용 분기는 기존 receipt SHA와 현재 파일만 비교한다. 현재 요청의 URL/revision/expected byte와 옛 receipt도 일치하는지 확인하는 편이 좋다. 이번 새 경로 최초 다운로드의 blocker는 아니다.
- archive 약17.9GB 외에 추출본과 weights 공간이 필요하다. 디스크 여유와 유휴 GPU 확인은 실행기 책임이며 현재 script가 스케줄러 역할을 하지는 않는다.
- 32 train files와 batch2, 8 steps면 실제 exposure는 16 files다. script는 이를 정확히 기록한다. 사용자 보고도 “32개로 학습했다” 대신 manifest/exposure를 구분해야 한다.

## 적절하게 구현된 통제

1. 개발용 split은 loss 관측 전에 파일명 hash로 정하고, crop/timestamp hash와 탈락 사유를 기록한다. loss에 따른 예제 선택은 없다. 파일 분리는 지역 독립을 뜻하지 않으며 원래 사전학습 자료에 대한 diagnostic이라는 제한도 명시되어 있다.
2. H5의 compact S2 축과 `missing_timesteps_masks`의 True=present를 연결한다. timestamp는 공식 `[day, month-1, year]` 규약과 일치한다. 존재 mask의 길이·개수 검증도 있다. 두 번째 modality 시간 정렬, 누락 pixel, 전체 modality loader는 이번 범위가 아니므로 증명했다고 하면 안 된다.
3. raw를 공식 normalization으로 한 번만 변환하며, WorldCover static axis를 추가한다. 공간 크기가 다르면 몰래 resample하지 않는다. fully observed crop이라는 제한도 명시한다. cloud/no-data 의미를 임의로 생성하지 않는다.
4. `model.eval()` 및 같은 batch/seed를 이용한 dev-before/after 비교는 stochastic mask 교락을 크게 줄인다. 읽은 target patch embedding에는 BatchNorm이 없고, band dropout이 0이므로 `model.train()`의 target mode 전파가 현재 frozen target을 바꾸는 결함은 확인되지 않았다.
5. target의 모든 파라미터가 frozen인지 확인하고, optimizer는 requires_grad만 받으며, backward 이후 target gradient도 검증한다. 공식 고정 projection target/EMA=(1,1)/exit0 설정 외에는 실패하도록 제한한다.
6. public checkpoint strict loader를 그대로 사용하고 모델 파라미터를 임의 재초기화하지 않는다. 공식 base loss와 이미 weight=.05를 포함한 InfoNCE의 합성은 공식 trainer와 부합한다. InfoNCE weight를 두 번 적용하지 않는다.
7. loss finite, gradient finite, nonzero encoder gradients, encoder 일부 실파라미터 변화가 모두 확인되어야 한다. dev-loss 감소를 통과 기준으로 삼지 않는다. 실패해도 step JSONL을 남긴다.
8. config·package path·version·hash를 보관하고, source checkout과 master environment를 직접 변경하는 구조가 아니다. full official training과 작은 objective smoke의 차이를 문서에서 명확하게 인정한다.

## 실행 뒤 독립 판정에 필요한 최소 결과

`receipt.json`, `train_log.jsonl`, `data_manifest.json`, launcher의 GPU/device 및 source tar receipt를 확인한다. 성공 보고에서는 실제 step 수·사용 파일 수·비제로 base 기여 여부·encoder 변화·target gradient 부재·현재 저장/reload 상태를 분리한다. 본 검토는 source-level GO이며 실제 import/forward/backward/모델 재로딩의 성공을 선취해서 보증하지 않는다.
