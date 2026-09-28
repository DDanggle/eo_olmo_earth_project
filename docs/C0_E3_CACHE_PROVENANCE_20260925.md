# C0/E3 캐시의 단일 관측 생성 경로·시간 provenance 감사

2026-09-25, 로컬 코드·git 이력과 서버의 기존 코드/audit/로그만 읽었다. 캐시 재생성, 모델 실행, 서버 파일 수정은 하지 않았다.

## 결론과 확정 범위

| 구분 | S2 `olmo_streaming_dev/single_fp16` | S1 `kurosiwo_s1_cache/single_fp16` |
|---|---|---|
| 생성 코드 원리 | 한 시점만 encoder 입력에 전달: 독립 인코딩 | 한 시점만 encoder 입력에 전달: 독립 인코딩 |
| 실제 추출 실행의 근거 | 서버 최초/확장 로그와 audit 확인 | 서버 7,000개 완료 로그와 chain rc=0 확인 |
| 현재 모든 파일의 당시 생성 이력 완전 검증 | 미확정: per-file raw/model/code/output 연결 manifest 부재, 기존 파일 재사용 가능 | 미확정: per-file raw/model/code/output 연결 manifest 부재, 기존 파일 재사용 가능 |

현재 확인한 경로에는 여러 시점을 joint attention으로 인코딩한 뒤 시간축을 잘라 `single_fp16`으로 저장하는 동작이 없다. 따라서 그러한 누출을 발견했다는 이유로 C0/E3를 invalid 처리할 근거는 이번 점검에서 나오지 않았다. 논문 표현은 **“단일 관측 입력으로 생성하도록 구현된 캐시; 추출 실행 근거 확인; 개별 과거 생성 lineage는 완전 재검증하지 않음”**이 정확하다.

## 코드 근거

코드 기준점: [S2 시간축 slice](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/extract_olmo_streaming.py:40), [S2 single 생성](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/extract_olmo_streaming.py:46), [S1 시간축 slice](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/extract_kurosiwo_s1_cache.py:32), [S1 synthetic timestamp](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/extract_kurosiwo_s1_cache.py:44), [S1 single 생성](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/extract_kurosiwo_s1_cache.py:45).

- `code/extract_olmo_streaming.py:37–46`: `window(cube,ts,t,t+1)`을 12번 호출하며, 내부에서 `cube[:,t0:t1,...]`와 `ts[t0:t1]`만 `pooled`에 넘긴다. `pooled`가 해당 1시점으로 `RasterImage`/`ModelContext`를 새로 구성하고 `w.model`을 호출한다. `teacher_fp16`의 prefix `[0:c]`는 별도 호출이며 single의 원천으로 사용하지 않는다.
- `code/extract_kurosiwo_s1_cache.py:23–45`: `window(cube,ts,[k])`를 시점마다 호출한다. `cube[:,idx,...]`의 temporal length가 1이고, batch는 동일 시점의 공간 crop 9개다. `[0,1,2]` teacher3와 `[0,1]` stale2는 별도 결과다. pixelwise dB 변환은 시점 간 평균/표준화를 하지 않는다.
- `code/extract_sen12_fold_cache.py:152–181`: S2 raw cache는 선택한 날짜의 원 band 값을 쌓아 저장한다. multitemporal encoder 출력은 별도의 `emb_fp16`에 저장되며 raw cube를 대체하지 않는다.
- `code/kurosiwo_export_npy.py:11–14`: 원 tortilla의 세 관측을 raw 시간축으로 쌓는다. 이 단계에서 세 영상의 feature를 joint encoding하지 않는다.
- 서버 설치 `.venv-master/.../olmoearth_pretrain_v1/data/normalize.py`는 저장된 band normalization 설정의 mean/std로 정규화한다. 다른 시점 영상으로 새 mean/std를 계산하는 경로가 아니다.

관련 git 이력도 일치한다. `b1438a9`가 teacher와 single을 별도로 추출하는 T1 경로를 추가했고, `4a18746`은 full-window의 per-time token이 이미 contextualized된다는 점을 명시하고 single-acquisition 경로 재사용을 추가했다. 즉 두 표현의 차이가 당시에 인식되어 있었다.

## 실행 provenance 확인

로컬과 서버 추출 코드 SHA256이 동일했다.

- S2 `398cb0add9e84b3905d091fe72a4487b57c944d15302bd883c322f8eef3a7ea0`
- S1 `317288d1c3db1ef57dccaa57d38b0cf6717f901032596b20bdb893b67d9c72e4`

서버 `/home/work/data/olmoearth/` 아래:

- `logs/x_olmo_streaming.log`: 3,372개 완료, n_skipped=0, teacher c12와 sealed cache의 audit_max=0.
- `logs/x_olmo_streaming_ext.log` 및 `olmo_streaming_dev/olmo_streaming_audit.json`: 확장 실행 3,704개 완료, n_skipped=0, audit_max=0, timestep_units_per_tile=52. audit SHA256 `c2da8a9ef1cecfb22e957dc9fa52a2752a4a2c838d86482c757e83c6fca2ee8a`.
- `logs/t1_chain.log`: 2026-09-07T12:03:13Z extraction rc=0. 직후 audit 실패 문구는 audit_max=0.0을 false로 해석한 gate bug였다는 후속 로그가 있다.
- `logs/t1_extend.log`: 2026-09-07T18:54:51Z extension extraction rc=0.
- `logs/x_kurosiwo_s1.log`: done=7000, elapsed_s=8572.03017042391, `KUROSIWO S1 CACHE DONE`.
- `logs/kurosiwo_chain.log`: 2026-09-08T04:26:07Z extraction rc=0. `code/kurosiwo_chain.sh`는 `.venv-master/bin/python code/extract_kurosiwo_s1_cache.py`를 호출한다.

단, S2 audit의 sealed 비교는 **teacher c12** 비교다. 각 single의 독립성을 수치로 검증한 것이 아니다. S2는 shape/dtype이 맞는 기존 파일을, S1은 existing single/teacher 파일을 재사용할 수 있다. 추출 당시 encoder checkpoint/library/code/raw/output의 개별 해시 연결이 없어 로그만으로 현재 모든 cache의 생성 경로를 암호학적으로 확정할 수 없다. C0/E3가 이번에 기록한 cache SHA는 현재 소비한 입력의 동일성을 보장하며, 과거 생성 provenance까지 복원하지 않는다.

## 별도로 유지할 제한

1. S2의 12-of-15 clear-date 선택은 전체 관측 기간의 SCL 품질을 참조한다. 픽셀 feature가 시점별 독립이어도 표본 선택은 retrospective다. 온라인 도착 시점에서 이용 가능했던 선택으로 주장하지 않는다.
2. S1 single에는 사건일과 사건일−12/−24일 synthetic timestamp가 들어간다. 이 시각은 단순히 QA prompt에 붙는 문자열이 아니라 `RasterImage(..., timestamps=...)`를 통해 encoder에 전달되어 임베딩의 시간 encoding에도 포함된다. 따라서 C0/E3를 “순수 영상 정보만” 또는 “시각 메타데이터를 제거한 이미지 단독” 실험으로 부르지 않는다. 다른 시점의 영상 내용이 attention으로 들어오는 것은 아니지만, 사건 기준 시간 메타데이터가 알려진 상태다. 실제 취득일/간격·순서 검증은 여전히 미완료다.
3. S2 현재 코드의 실제 timestamp 경로와 일부 후속 문서의 'synthetic-month encoded' 설명이 충돌한다. `4b5f716`은 real-date 수정이며 현재 코드는 `real_timestamps`를 사용한다. 그러나 기존 cache 재사용 때문에 개별 파일의 과거 timestamp encoding까지 현재 코드만으로 일괄 확정하지 않는다. 문서에서 synthetic을 확정 사실로 반복하지 말고 provenance 미확정으로 적는 편이 안전하다.
4. 독립 입력임을 확인해도 later-only 성능은 source label 연관성을 뜻한다. 특정 물리적 변화/시간 관계/원인을 이해했다는 증거로 확대하지 않는다.

이번 감사는 여기서 종료한다. 재추출·모델 실행·추가 실험은 수행하지 않았다. C0/E3의 해석 범위는 “이 시간 메타데이터를 포함한 frozen 단일 관측 임베딩에 대한 의존성”이며, 순수 영상 정보의 인과적 기여나 온라인 시간 독립성을 주장하지 않는다.
