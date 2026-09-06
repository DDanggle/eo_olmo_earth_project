# OLMoEarth 연구 재시작 지점
> ## 2026-09-06 18:40 KST — MS-109 반영 + GEO-Bench action benchmark 경계 (최신)
>
> **새로 닫힘**: Solar에서 OlmoEarth base cache `.6081/.9262`(macro IoU/AP), Galileo base
> `.4587/.7952`; paired 차이 `+.1494/+.1309`, 8/8 fold OlmoEarth 우위, one-sided Wilcoxon
> `p=.00390625`. Sen12 한 과업 의존 반론은 약해졌다. 단 Solar fold는 독립 과업 8개가 아니고
> seed 1개다.
>
> **중요한 역설**: Sen12와 Solar 모두 OlmoEarth가 top action이라, MS-109는 cache-value 주장을
> 강화하지만 selector 필요성은 아직 증명하지 않는다. 먼저 task/region/budget별 **top-action
> rank reversal 또는 cost Pareto crossover**와 best-static 대비 oracle headroom을 계산한다.
> normalized headroom `<.02`이고 독립 group 교차가 없으면 learned selector를 만들지 않는다.
>
> **프로브 MS-110**: effective rank는 family 분리에서 생긴 exploratory signal이다. family 평균
> `n=4`에서는 Spearman `.400, p=.600`; predictor 주장 금지. 절대 metric 대신 task 안의 pairwise
> action gain/regret를 예측한다.
>
> **새 SSOT**: 쉬운 novelty/실험 설계 `docs/EARTHCACHE_GEOBENCH_UPGRADE_2026_09_06.md`,
> machine-readable draft `config/geobench_cache_action_prereg_v0.json`. 기존
> `safe_cache_action_prereg_draft_v0.json`은 region/support A0/A1 안전 정책이며, 새 cross-task
> model/cache benchmark와 합쳐 표본 수를 부풀리지 않는다.
>
> **데이터 상태**: FOTW·DynamicEarthNet은 적재 검증 완료, PASTIS는 미완료. 현재 세 공개 후보가
> 모두 dense segmentation이므로 classification 1개(TreeSatAI 또는 BigEarthNet-v2)와 regression
> 1개(BioMassters)를 추가하기 전 `GEO-Bench 전반`이라고 쓰지 않는다.
>
> **다음 순서**: PASTIS 무결성 → action/metric/chip/time 계약 freeze → 기존 개발표의 G0 oracle
> headroom → 실제 cold/warm/raw-I/O/storage 비용 → Core-6 action matrix → G0 통과 시에만 simple
> selector. GPU 실행은 GPU1이 비고 보호 규약을 만족할 때만 한다.
>
> ## 2026-09-06 09:40 KST — 아침 스냅샷 (이하 이력)
>
> **한 줄**: 논문 몸통이 비어 있던 상태에서 벗어나는 중. 새 downstream 과업 1개 확보(fotw), 2개 수신 중(PASTIS·DynamicEarthNet). 라벨 없는 캐시 가치 예측기에 첫 리드(effective_rank ρ=.635, CI가 0 배제). GPU는 남의 작업으로 전면 점유.
>
> **상태 한 번에 보기**: `./bin/nx sh 'bash /home/work/data/olmoearth/code/status.sh'`
> (GPU·실행 중 체인·DONE/FAILED 마커·GEO-Bench 수신·보호 4파일 해시를 한 화면에)
>
> **방향 (2026-09-06 새벽 사용자와 합의, 다시 바꾸지 않음)**
> 1. 벤치마크는 **GEO-Bench-2 위에 얹는다** — PASTIS + Fields of the World + DynamicEarthNet + 기존 Sen12·Solar = 과업 5개. GEO-Bench-2에는 few-shot·캐시 재사용·비용 축이 없다(공식 프로토콜은 전량 라벨 fine-tuning). 우리는 그들의 **데이터·split만** 쓰고 arm끼리 비교한다. 리더보드 숫자와 직접 비교 금지(M24 교훈).
> 2. **진단이 아니라 예측기**를 만든다. "유사도 지표(CKA 등)가 기능 등가를 예측 못 한다"는 ICLR 2023 등 선행연구가 점유 → 기여 아님. 남은 빈칸 = "라벨 없이 무엇이 캐시 가치를 예측하는가".
> 3. 교차모델 선형 브리지(Clay→OlmoEarth)는 5폴드 전부 사전등록 불통과로 **사망**. 되살리지 않는다.
>
> **확보한 것**
> - fotw: sha256 통과, train 4,000 / val 1,000 / test 2,000, 샘플 `image_a/image_b (4,256,256)` + `mask`. **단 4밴드(RGB+NIR)** — 논문 표의 "S2/Multi"와 달리 다중밴드가 아니다. 계약 주의.
> - PASTIS: S2 10밴드가 우리 Sen12/한국 캐시와 **같은 집합**(B08 위치만 다름, `band_order`로 정렬). + S1 asc/desc. 20 class 시계열 분할. 수신 중.
> - DynamicEarthNet: planet 4밴드 + S2 12밴드(B01/B10 포함). 우리 10밴드 전부 보유. 수신 대기.
> - 프로브 14캐시: `artifacts/cache_probes.json`, `artifacts/probe_correlation.json`. 사전등록 ρ≥.70 불통과(NO_PREDICTOR_AT_N14). effective_rank ρ=+.635 CI[+.13,+.88], participation_ratio ρ=+.631 CI[+.04,+.93]. 물리 프로브(NDVI 복원)는 판별력 없음. 확증은 n=70(과업 5개)로 사전 지정.
>
> **GPU 필요 단계 (지금 막힘 — GPU0/1 모두 타 사용자 100%)**
> - (A) 경쟁 캐시(Clay/Galileo/Prithvi)를 **Solar**에 — 현재 "어느 표현이 캐시 가치 있나" 결론이 downstream 과업 1개(Sen12)에만 얹혀 있음. 최대 구멍. 파이프라인 있음.
> - (B) PASTIS/fotw/DEN에 OlmoEarth·경쟁 캐시 추출 → 8폴드 디코더 → 프로브 확증 n=70.
> - GPU가 비면 (A) 먼저. 규약 4b: `nvidia-smi`로 GPU1이 비었는지 확인, 남의 프로세스 있으면 멈춤.
>
> **재발 방지 장치 (2026-09-06 신설)**
> - `code/preflight.py geobench` — 비싼 단계 앞 5초 예비검사(클래스·url·sha·band_order·필수인자). 통과해야 진행.
> - `code/test_chain_rc_pattern.py` — `echo "$(date) rc=$?"` 오용 린터. 현역 0건. 봉인 결과 생산 스크립트 15개는 HISTORICAL(계보 보존).
> - **설계 규칙**: 다운로드와 검증을 한 사슬에 묶지 않는다. 검증 실패가 독립 다운로드를 막았던 것이 7시간 손실의 원인(2026-09-06 02:31 fotw verify 실패 → pastis/DEN 미시작).
> - 밴드 계약: `raw_u16 = (10밴드, 12시점, 128, 128)`, **B08 = idx 3**. `test_cache_probes.py`에 손계산 검증.
>
> **환경**: `.venv-geobench`(격리, `--system-site-packages`, `PIP_CONSTRAINT=`, `numpy<2`). `.venv-master`는 건드리지 않음. 데이터 `geobench2/<dataset>/`.
>
> **정정 이력(이 사이클)**: "용량 필요조건" 철회(차이 .003) · "독립 설정 4개" 철회(상관된 반복) · A1>A4h Sen12 7/8 · 한국 큐브 error 6건 재시도 전 experiment_eligible 아님 · 브리지 "두 공간 이미 같다" 1폴드 과잉진술 철회.
>
> ## 2026-09-06 인수인계 (다른 컴퓨터에서 이어가기)
>
> **중심 질문(확정)**: 하나의 Earth 표현 캐시를 새 task·새 지역에서 언제 REUSE/ADAPT/RE-EMBED/
> REQUEST할 것인가. 버전 마이그레이션(A)은 model-shift 부록이다.
> 최신 상세 감사 `docs/CVPR_BIG_PICTURE_AUDIT_2026_09_05.md`, 쉬운 큰그림 재판정
> `docs/PAPER_STATE_2026_09_06.md` + 이 절이 SSOT다.
>
> **확정된 결과(장부 MEASURED_FINDINGS.md)**: M65·MS-98 재사용 우위(산사태 6/8, 태양광 8/8) ·
> MS-96/97/99 few-shot A1>A4w 16/16, A1>A4h는 산사태 7/8·태양광 8/8(단 A1>A0 K5는
> 두 task 모두 5/8) · C0-dev(안전정책 개발화면) · MS-100(브리지: identity AP .02→.90 복구,
> 단 저장 gate summary는 IoU field 수정 필요, R5 무익, v1.1 R6 무효) · MS-101/102(두 번째 FM은
> single-seed diagnostic이며 보편 우월 근거 아님) · MS-105(group-concat `.168`, readout-only 설명
> 기각) · MS-108(architecture 7캐시×8폴드 완결, scale/family별 cache/raw 경계) ·
> M104(한국 큐브 v2 파일 감사를 통과했으나 experiment_eligible 아님).
>
> **서버(ainexus h200-dev, `./bin/nx`만 사용, GPU1만) 최신 상태(9/6 01:05 KST)**
> - `BV1_CHAIN4_DONE`: Galileo group-concat `.168`, raw보다 우위 1/8 — readout만 바꿔 rescue되지 않음.
> - `AIHUB_V2_FULL_DONE`: inventory 2,699 중 manifest 2,536, excluded 163. 그러나 excluded에 재시도
>   대상 `error` 6건이 섞였고 audit가 이를 통과시킨 결함이 있어 **experiment_eligible 아님**.
> - `arch_axes_chain.sh` **완료**: `ARCH_AXES_DONE` 2026-09-05T16:05:15Z. 추출 7/7,
>   decoder 56/56, schema/fold/seed/metric/SHA 검사 missing·invalid 0.
>   `galileo_base_half` 최종 `.138`, raw 우위 0/8. 검증/요약은
>   `artifacts/arch_axes_verify.json`(`ca1d7730…`)과
>   `artifacts/arch_axes_summary.json`(`405cb800…`). 마지막 fold는 다른 GPU 1 프로세스와
>   겹쳤으므로 wall-clock을 비용 근거로 쓰지 않는다. 현재 우리 GPU job은 없다.
> - direct SSH는 RSA host-key changed(`zBP7cfCx…`, known_hosts 96행) 경고를 낸다. 관리 터널로
>   접속은 됐지만 키는 임의 삭제하지 않았다. 다음 운영 전에 플랫폼에서 fingerprint를 확인한다.
> - 로컬 맥 launchd(09:30/21:30) + GitHub Actions(`DDanggle/gk2a-archive`, 09:40/21:40 KST): GK2A 일일 수집. 상태 `gh run list -R DDanggle/gk2a-archive`.
>
> **다른 컴퓨터에서 시작하기**
> 1. `git clone git@github.com:DDanggle/eo_olmo_earth_project.git _work` (이 저장소). 서버 접속 CLI는 별도 저장소 `h100-setup`(nexus)이며 `bin/nx`가 그것을 부름. `.env`(API 키)는 저장소에 없음 — 수동 복사.
> 2. `./bin/nx tunnel up` → `./bin/nx sh 'nvidia-smi'`. 터널이 자주 끊김: `pkill -f "backend.ai app h200-dev sshd"; ./bin/nx tunnel up`.
> 3. 서버 코드 위치 `/home/work/data/olmoearth/code/` (로컬 `code/`를 `./bin/nx push $PWD/code/<f> olmoearth/code/`로 동기화). 보호 4파일(pilot_sen12_gp_heads.py, sen12_official_baselines.py, extract_sen12_fold_cache.py, audit_sen12_fold_cache.py)은 확증 실행 중 푸시 금지.
> 4. 원격 명령 안에서 `pkill -f "<스크립트명>"`은 자기 셸까지 죽임 → `pgrep -f "^bash code/<name>"` 또는 `^\./\.venv-master/bin/python code/<name>` 패턴으로.
>
> **다음 순서(9/5 감사로 수정)**
> 1. P0 무결성: AI-Hub error 6건 재시도·selection-bias gate, Solar cross-CRS 누수 감사,
>    release gate의 `iou_frozen_thr` 재생성, v1.1 R6 무효 표기.
> 2. Korea 희소 class의 eligible-cluster metric과 128-chip unit를 label 개봉 전에 dated amendment로 동결.
> 3. 공개 Task-3(Sen1Floods11 우선)의 CACHE/RAW baseline + 단순 support-only action rule 동결.
> 4. Task-3 one-shot action-regret와 cold/warm/re-embed/storage 실측 비용을 먼저 닫는다.
> 5. 그 뒤 Korea 3-task를 외부 사례로 한 번 개봉한다. 10/09에 CVPR go/no-go.
>
> **2026-09-04 갱신**: 큰 그림은 `docs/BIG_PICTURE_2026_09_04.md`(버전 전환 연속성 벤치마크 + A 브리지 + OlmoEarth-KR + FoldRefresh + C). 실행 중: Clay v0(탐색), A 체인(`logs/release_chain.log`).

갱신: 2026-09-06
활성 과학 기준점: **MS-96/97 + MS-98/99 + Earth Embedding Continuity A/B/C 설계**
감사 시작 기준 HEAD: `95ed699`. A/B/C config는 `efce8b8`, Korea 3-task config는 `6f47156`에
커밋됐으나 파일명/status의 `draft` 표시는 과거 상태라 dated amendment로만 정정한다.

이 파일은 새 세션의 첫 진입점이다. Nepal 대응 데모는 현재 CVPR/transfer 임계경로가 아니며,
전용 코드·문서·원본·중간 산출물은 sibling 저장소
`/Users/dgyi/dong/ai_projects/nepal-live-twin`이 소유한다.

## 한 문장 연구 질문

> **세계·과업·모델 release가 바뀔 때, 저장된 Earth embedding과 downstream head를 그대로 쓰고,
> head만 적응하고, 표현을 migration하고, 재임베딩하거나 라벨을 더 요청할 시점을 정확도와
> label·GPU·raw-I/O·cache invalidation 비용으로 결정할 수 있는가?**

세 축을 별도 논문으로 벌리지 않는다.

- **A — release migration**: OlmoEarth v1→v1.2에서 old cache/index/head를 살리는가.
- **B — product validity**: Clay·AlphaEarth 등 다른 embedding product에서도 같은 결정 문제가
  성립하는가.
- **C — safe action**: support label과 contract만 보고 A0 reuse/A1 head-adapt/A3 re-embed/
  REQUEST를 고르는가.

설계 SSOT는 `docs/ABC_EMBEDDING_CONTINUITY_2026_09_04.md`다.

## 지금까지 닫힌 양성 결과

### Task 1 — Sen12Landslides, 8개 geographic holdout

- source-only frozen OlmoEarth cache + decoder(P4): region-macro `.2722`.
- raw UNet3D(P2) `.1966`, raw U-TAE(P3) `.1834`; 최고 raw 대비 7/8 지역 우위.
- target tile K=5/20에서 A1 cache-head adaptation은 raw full A4w를 8/8, parameter-matched
  A4h를 방향 기준 **7/8** 이겼다(MS-96/97). fixed-exposure에서도 A1>A4w 8/8이다.
- 그러나 K=5에서 A1>A0는 5/8뿐이다. `라벨 5장이면 항상 적응`은 금지한다.

### Task 2 — Solar Farm, 8개 UTM-zone fold group

- A0 cache no-adapt `.591`, stratified A1 K=5 `.582`, K=20 `.609`.
- raw A4w K=5/20 `.240/.245`, A4h `.257/.291`; cache pathway가 raw adaptation을 8/8
  이겼다(MS-98/99).
- random K=5에서 support 12/24가 positive tile 0장이었고 A1이 `.426`으로 붕괴했다.
- tie-correct AP는 A0 `.9252`가 A1 K=5/20 `.8651/.9044`보다 높다. 따라서 action은 task뿐
  아니라 support 구성과 배포 utility/threshold 계약에 의존한다.
- Solar fold는 독립 지역 8개가 아니라 UTM-zone 기반 group이다. Sen12의 `8지역`과 같은
  일반화 단위라고 부르지 않는다.

### Model shift

- M1: 같은 scene의 OlmoEarth v1/v1.2 token identity R@1은 양방향 0. Procrustes·affine ridge도
  등록된 retrieval compatibility gate를 실패했다. pooled CKA `.979`는 task continuity 증거가 아니다.
- M85는 v1/v1.2 radar/optical 성능 비교이지 cache migration 실험이 아니다.

## 닫힌 음성 방향 — 이름을 바꿔 되살리지 않는다

- CacheTune A2 low-rank spatial residual: MS-94에서 A1보다 `.05–.08` 낮아 stop rule 발동.
- prediction fusion·GeoContextGate: MS-90B/91/92 종료. FP-matched oracle headroom도 거의 0.
- label-free winner router와 block routing: 등록 gate 실패.
- MoE: action complementarity와 untouched selector 성공 전에는 열지 않는다.

## 2026-09-04 Clay v0 실행 경계

서버에서 확인된 사실:

- `clay_cache`: 6,834/6,834, 실패 0, Clay v1.5, 1024-d.
- 그러나 native `16×16`을 bilinear로 `32×32`에 확대한 cache다. audit의
  `all_gates_pass=true`는 파일 완결성만 보증하며 비교 공정성을 보증하지 않는다.
- source decoder chain과 few-shot wait chain이 실행 중이었다. 확증 실행 중 서버 코드 push 금지
  규칙에 따라 이 검토에서는 서버 코드를 건드리지 않았다.
- 이 v0 결과는 **interpolated deployment-adapter exploratory baseline**으로만 보존한다.
  B1 common-physical 또는 B2 native confirmatory 결과로 쓰지 않는다.
- current Clay few-shot FP-matched IoU는 Clay A0의 FP budget을 쓰고, 비교하려는 historical raw
  A4는 OlmoEarth A0 budget을 썼다. 서로 다른 작동점이므로 report 간 primary IoU 비교는 금지한다.
  threshold-free tie-correct AP만 탐색 비교할 수 있다.

## 아직 주장할 수 없는 것

- A1이 A0보다 항상 낫다.
- Clay 하나로 product-agnostic 원리가 증명됐다.
- v1→v1.2 bridge가 old task head/index를 보존한다. 아직 downstream migration은 0회다.
- AlphaEarth 연간 embedding으로 사건 전후/실시간 변화를 측정한다.
- support label 없이 self-training하면 정확도가 개선된다.
- Korea sealed target 또는 독립 Task-3에서 safe action policy가 통과했다.
- OLMoEarth가 모든 GeoFM보다 보편적으로 우월하다.

## 다음 실행 순서

1. **P0 증거 복구(CPU)**: Solar random support ID/양성 수/SHA, WGS84 cross-CRS distance,
   exact-query A4w0, 서버 48-run 원시 report/snapshot을 로컬 봉인한다.
2. **A release migration**: full-12-band Solar의 exposed 2fold에서 v1/v1.2 exact-scene bridge가
   old-head AP·fixed-threshold IoU를 보존하는지 screen한다. 새 query raw read는 정상 비용이고,
   피하는 것은 과거 archive 전체의 raw backfill이다.
3. **B-v1 Clay**: v0 종료 뒤 native 16×16 smoke를 새로 봉인한다. B1은 80 m·16×16·256-d
   compact cache, B2는 native product로 분리하고 같은 report/threshold에서 raw를 재평가한다.
4. **C A3 ceiling + deterministic guardrail**: exposed 4 unit, K=20에서 공식
   LayerDecayAdamW와 q/v LoRA sensitivity를 측정한다. positive 0이면 A0/REQUEST, 그 외에도
   leave-one-tile-out 하한이 0보다 클 때만 A1을 허용한다.
5. **untouched first-look**: policy와 threshold를 동결한 뒤 독립 Task-3 또는 Korea sealed target을
   한 번만 연다.

AlphaEarth는 Solar/static mapping B2 뒤에만 둔다. 연간 64-d product이므로 event Sen12와 같은
시간 gate에 넣지 않는다. Prithvi는 Clay 뒤 contract-shift sensitivity다.

## 읽는 순서

1. 이 파일
2. `docs/ABC_EMBEDDING_CONTINUITY_2026_09_04.md`
3. `docs/ASSET_INVENTORY.md`
4. `docs/CRITICAL_PATH.md`
5. `MEASURED_FINDINGS.md`
6. `docs/PAPER_NARRATIVE_2026_08_31.md` — 이전 narrative, 현재 A/B/C 문서가 실행 방향을 대체
7. `GOAL.md` 마지막 Worklog

기계 판독 사전등록 파일(파일명에는 `draft`가 남았으나 `efce8b8`에 실험 전 커밋됨):

- `config/release_migration_prereg_draft_v0.json`
- `config/second_fm_cache_prereg_v1_draft.json`
- `config/safe_cache_action_prereg_draft_v0.json`

결과 뒤 발견된 gate/metric 결함은 원 파일을 조용히 고치지 않고 dated amendment와 재생성 report로
남긴다. 결과가 나온 뒤 문구·gate를 바꾸면 사전등록이 아니다.

공식 Ai2 checkout `..`에는 사용자 수정이 남아 있다. 연구 재시작 작업에서
`olmoearth_run_data/forest_loss_driver/{dataset.json,model.yaml}`과 `.pnpm-store/`를 건드리지 않는다.

> **2026-09-05 18:40**: `nepal-live-twin`(DDanggle/eo-rasuwa)은 이 작업공간 관리 대상에서 제외함. 여기서 푸시·동기화하지 않음.
>
> **2026-09-05 18:30 추가**: 서버 체인 `code/arch_axes_chain.sh`(`logs/arch_axes.log`, 끝 표시 `ARCH_AXES_DONE`) — 아키텍처 축 7캐시(olmo_nano/tiny/base_half, galileo_nano/tiny/base_half, clay_in256_half) → 8폴드 디코더. 표: `code/bv1_summary.py`에 캐시 이름 추가해 실행. 가설·판독 규칙: `config/second_fm_cache_prereg_v1_draft.json` addendum_v1b.
