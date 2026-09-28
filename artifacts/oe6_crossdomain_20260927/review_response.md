# 준비 코드 독립 검토 대응

로봇/loop 조사 에이전트가 입력 준비 코드를 별도로 검토했다. raw NPZ를 행 번호만으로 연결하지 않도록 patch ID·parent·split·원 날짜·packed reference와 hash를 검사했다. 설치 로더의 영상 선택 함수와 날짜 분기를 모두 fixture에서 실행했다. computed normalization 상수 hash와 multiplier를 기록하고 별도 산술로 검산했다.20개 class 목록·crop pixel 존재·instance 정수/shape도 확인했다. 감사v0의 전체 파일 안정성 판정식은 새v1 코드에서 보강했고 원 실행v0는 보존했다.

원영상 확인에서 균등 선택에 구름이 보여 v1을 보존하고,8날짜 contact sheet를 본 뒤 v2의 두 날짜를 기록했다. 이 assistant QA를 전문가 cloud label이나 자동 선택 정책의 성능으로 세지 않는다. per-band availability는 sidecar이며 OlmoEarth의 band mask 구조를 바꾼 것은 아니다.

캐시 모듈은 root가 코드를 검토하고 같은10개 테스트를 다시 실행해 통과를 확인했다. 모델/cache 수치 실험이 아니다. 독립 검토는 내부 에이전트 검토이며 외부 동료 심사·EO 성능 재현을 뜻하지 않는다.
