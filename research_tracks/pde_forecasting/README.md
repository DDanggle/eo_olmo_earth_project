# 별도 후속 트랙 — PDE / Poseidon / 기상 조건부 EO 예측

상태: **보관된 연구 구상, 현재 VLM 본체와 분리**. 정리일: 2026-09-22.

## 이 폴더에 모은 것

[Poseidon–EO–기상 연결 감사와 실험 구상](POSEIDON_EO_WEATHER_BRIDGE_AUDIT_2026_09_21.md)

- 이전 Poseidon 실험의 실제 시간 전개·pseudo-time·보간 구분.
- EO 지역 맥락 + DEM/강우/경계조건 → 물리 상태 예측 → 새 관측으로 교정하는 설계.
- FloodCastBench, EarthNet/GreenEarthNet 등 데이터 계약과 선행연구.
- scratch/Poseidon × EO 맥락 유무 대조, 관측 연산자와 독립 실제 관측 평가 계획.
- 보간된 상태와 관측 증거를 구분하는 규칙 및 실패 시 주장 축소 기준.

**이미 완성된 예측 모델이나 검증된 EO–PDE 연결 결과가 있다는 뜻은 아니다.**
이 폴더의 문서는 기존 결과 감사와 후속 실험 계획이다. 물리 예측이 현재 SN7 VLM의
시점 판독 실패를 해결했다는 증거도 없다.

## 그대로 둔 원본과 경계

- 원 PDE 저장소: `/Users/dgyi/dong/ai_projects/nips2026-1/v1-0426`.
  그 안의 `earth_robotics_bridge/design.md`, `transfer_learning/`, `multipde_transfer/`는
  이 폴더로 복사·이동하거나 수정하지 않았다. 대용량 데이터/체크포인트도 건드리지 않았다.
- [다중 현상 장기 로드맵](../../docs/MULTI_REGIME_EARTH_MEMORY_BLUEPRINT_2026_09_22.md)은
  물리뿐 아니라 EO 기억/언어 전체를 다루므로 원래 위치에 남겼다.
- [현재 관측 근거 기억 방향](../../docs/CVPR_SINGLE_CLAIM_DECISION_2026_09_22.md) 및
  MS-155 이후 reader 진단은 별도다. 본 트랙의 실험을 선행 조건으로 추가하지 않는다.
- 이전 문서 경로에는 안내 파일을 남겨 기존 링크를 보존했다.

## 다시 시작할 때

먼저 연결 감사 문서 §6의 데이터/단위/시각/forcing/관측 연산자 계약을 확인한다.
그다음 물리 초기화와 EO 맥락의 기여를 분리한 작은 대조부터 한다.
현재 새 실행·자료 취득·학습은 승인되거나 시작된 상태가 아니다.
