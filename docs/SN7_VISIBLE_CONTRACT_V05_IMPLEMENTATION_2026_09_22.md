# v0.5: 전문가 반복 주석 없이 정답 계약을 바로잡는 준비물

현재 단계는 **개발용 과제·검수 도구 검증**이다. 새 사람 정답, VLM 개선, 기억 병목, CVPR 방법 우위는 아직 없다.
MS-155의 실패 판정·원본 runner/prereg/결과는 그대로 보존한다. 새 정의로 옛 결과를 통과 처리하지 않는다.

## 이번에 실제로 확보한 것

- 기존 서버 영상에서 6 AOI/12 episode/172 PNG, 약66MB. 각 episode는13장 또는21장이다.
  원 PNG 크기는511×512 또는512×512로 유지했다. 새 데이터 구매·취득이나 GPU 학습은 없다.
- 로컬 패키지: `labeling_pack/visible_contract_v05_20260922/annotator_pack/index.html`.
  현재 로컬 미리보기는 `http://127.0.0.1:8766/`이다. 서버가 종료되면 아래 명령으로 다시 연다.
- [준비 무결성 감사](../artifacts/sn7_visible_v05/preparation_audit.json):
  pack ID `sn7v05-198e4ff03e26f004`, 172개 영상 해시와 생성 코드3개 해시 일치.
- 계약·통합 테스트23개 통과. 합성 PNG fixture 테스트와 실제 영상의 해시 검사를 구분했다.
  실제 UI에서 초기 화면/영상 로딩/확대/미완료 제출 차단 확인. 주석 저장·정답 판독은 하지 않았다.
  브라우저 QA는 `ui_smoke_not_a_rater` 코드로만 열었고 사람 검수로 세지 않았다.
- 주석 JSON 다운로드·브라우저 재개 기능은 구현됐으나 실제 두 사람의 end-to-end 검수는 아직 없다.

```bash
python3 -m http.server 8766 --bind 127.0.0.1 \
  --directory labeling_pack/visible_contract_v05_20260922/annotator_pack
python3 -B code/audit_sn7_visible_pack_v05.py labeling_pack/visible_contract_v05_20260922
```

## 달라진 정답 계약

- 질문: 첫 기준영상 대비 새 구조물·공사 형태가 분명히 보이는 **첫 제공 관측 ID**.
  누적 신축 8동 도달월이나 실제 발생일을 답하지 않는다. 월 합성영상의 표지는 정확한 발생일이 아니다.
- 사람이 각 관측을 `no_visible_change / visible_change / unreadable / ambiguous`로 판독한다.
  명확한 변화가 있으면 `change_supported`, 전부 판독 가능하고 변화가 없으면 `no_visible_change`,
  나머지는 `insufficient_evidence`로 파생한다. 기준영상이 불명확하면 후영상이 선명해도 비교는 보류한다.
- 과거에 변화가 확인됐으면 이후 구름 때문에 그 기록을 지우지 않는다. `current_state`는 별도다.
  기준 불명확 시 보수적으로 교정한 상태와 검수자 원입력(`raw_current_state`)을 구분해 보존한다.
- 시간 부족은 `timeout` 작업 상태이며 정답에서 제외한다. 구름/해상도에 의한 판독 불가와 다르다.
- 두 독립 검수자의 답·첫 관측·마지막 무변화·현재/기준 상태·근거 ID가 일치해야 agreed로 만든다.
  불일치는 미해결로 남긴다. 합의가 현장 진실의 보장을 뜻하지 않는다.
- 근거 ID는 뒷받침하는 관측이다. 기준+변화 2장만으로 ‘가장 처음’ 또는 모든 과거의 무변화를
  증명했다고 주장하지 않는다. memory 실험의 충분 근거 정답은 별도 검증이 필요하다.

## 실제 구현 범위

`code/sn7_visible_contract_v05.py`: episode/주석 검증, tri-state 정답 파생, 합의, 날짜가 같은 영상 교체 쌍.

`code/sn7_visible_pack_v05.py`: 기존 v0.4 items를 **읽기만** 하여 검수 패키지 생성·무결성 검사·주석 병합.
최대 6 AOI × 서로 반대인 2개 사분면 = 12 episode. AOI당 cutoff 하나만 사용한다.
AOI·사분면 선택은 고정 hash로 하고, 같은 AOI의 두 영역은 검수 순서에서 떨어뜨린다.
선택은 기존 gold/privileged frame에 의존하지 않지만, 원천 후보 풀 자체는 기존에 선별되고 노출된 개발 자료다.
이는 무작위 모집단 표본·미사용 테스트셋·12개 독립 사건이 아니다.

`code/sn7_visible_review_v05.html`: 원본 crop 확대, 관측별 판정, 브라우저 로컬 저장, JSON 내려받기.
모델 답·기존 polygon 정답은 숨긴다. ROI는 고정 사분면이며 세부 영역 그리기/자유 설명 입력은 아직 없다.
기록은 외부로 전송하지 않는다. 파일 전체의 재배포 권한을 별도로 확인하기 전 공개 호스팅하지 않는다.

`finalize`는 agreed 정답에 대해서만 진단 입력 명세를 만든다.
영상 교체는 대체 영역의 **자체 정답**으로 채점한다. 두 영역의 정답이 다른지 먼저 확인한다.
텍스트-only/빈 영상의 정답은 정보 부족이며, 실제 영상의 정답과 일치하는지는 별도 leakage 분석용으로 남긴다.
현재는 명세 생성까지만 구현했다. 모델 inference runner/SFT/학습 결과는 아니다.

## 사용

검수자는 배포된 `annotator_pack/index.html`을 열고 서로 다른 익명 코드로 독립 작업한다.
가능하면 동일한 로컬 HTTP 주소/브라우저로 재개한다. 브라우저별 저장소는 다르고 파일 URL 저장 동작도 다를 수 있다.
중간·종료 시 JSON을 내려받아 별도 보관한다. `seconds`는 마지막 재개 이후의 벽시계 시간으로,
이전 중단 시간·교육·조정 시간을 포함한 총 인건비 기록을 대체하지 않는다.

```bash
python3 -B -m unittest discover -s tests -p 'test_sn7_visible*.py' -v
python3 -B code/sn7_visible_pack_v05.py finalize \
  --pack labeling_pack/visible_contract_v05_20260922/annotator_pack/pack.json \
  --annotations /absolute/path/rater_a.json /absolute/path/rater_b.json \
  --output artifacts/sn7_visible_v05_review_01
```

새 출력 폴더만 허용하며 기존 산출물 덮어쓰기는 거부한다. `diagnostic_manifest_ready`는 클래스/쌍의
준비 여부이지 과학적 gate 통과가 아니다. 양성/음성/정보부족 또는 유효한 교체 쌍이 없으면 그 부족을 보고한다.
어떤 결과가 나올 때까지 사람 주석을 무제한 늘리지 않는다. 전체 교육·판독·조정은 최대20인시 안에서 계획한다.

## 다음 연구 단계

전문가 인터뷰는 용어·과제 범위 확인만 선택적으로 받는다. 반복 라벨링·장문의 사고과정 작성은 요구하지 않는다.
공개 전문 주석은 [자연환경 데이터 조사](EXPERT_LABELED_NATURE_DATA_2026_09_22.md)의 원래 과제에 맞춰 재사용한다.
mask/class에서 만드는 문장은 자동 파생 silver이지 전문가가 쓴 해설이 아니다.
reader 학습을 한다면 같은 데이터/비용의 답-only 대 답+공간·관측 근거 감독을 먼저 비교한다.
새 실행 전 split·평가 지표·효과 기준·GPU 상한을 별도로 고정하고, 충분한 reader 뒤에만 memory를 검사한다.
