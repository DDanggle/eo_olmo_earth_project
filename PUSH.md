# GitHub 상태 — 푸시 완료 (2026-08-21)

- 원격: `git@github.com-dong:DDanggle/eo_olmo_earth_project.git` (브랜치 `main`)
- **저장소 공개 상태: public** (푸시 시점 확인). LFMC 재현성 발견(PR_DOSSIER #2)이
  Ai2에 통보되기 전 공개 상태이므로, 필요하면 GitHub 설정에서 private으로 전환할 수 있다.
- 이후 갱신: `git push` (upstream 설정됨)

아래는 최초 푸시 전에 점검한 항목들 — 향후 다른 저장소를 올릴 때 재사용한다.

## 올리기 전에 확인할 것

1. **자격증명 없음 확인** — 이 저장소에는 `.env`나 토큰이 없어야 합니다.
   ```bash
   git grep -iE "password|secret|api[_-]?key|token" -- . | grep -v "STUDIO_API_KEY\|API_TOKEN_ENV\|OEDATASETS" || echo "OK: 자격증명 없음"
   ```
2. **좌표 공개 판단** — `artifacts/results/*.json`에 제주·완도의 변화 후보 좌표가 있습니다.
   공개 위성데이터 기반이고 민감시설이 아니지만, 공개 저장소로 올릴지는 판단이 필요합니다.
3. **용량** — `artifacts/`가 약 19MB. GitHub 기준 문제없습니다.
4. **비공개 레포 내용 없음 확인** — `olmoearth_run`은 비공개 레포입니다. 우리 문서에는
   함수명·증상만 적혀 있고 소스 코드를 인용하지 않았습니다 (PR_DOSSIER #6·#7 확인).

## 푸시 절차 (실행은 사람이)

```bash
cd ~/dong/ai_projects/olmoearth_projects/_work

# 방법 A) gh CLI (brew install gh && gh auth login 후)
gh repo create olmoearth-research --private --source=. --remote=origin --push

# 방법 B) 웹에서 빈 저장소를 만든 뒤
git remote add origin git@github.com:<계정>/olmoearth-research.git
git branch -M main
git push -u origin main
```

**권장: 처음엔 `--private`.** LFMC 재현성 발견(PR_DOSSIER #2)을 Ai2에 먼저 알린 뒤
공개하는 것이 예의입니다. 공개로 바꾸는 것은 나중에 한 번 클릭이면 됩니다.

## 함께 올릴지 결정할 다른 저장소

| 저장소 | 상태 | 비고 |
|---|---|---|
| `~/dong/ai_projects/h100-setup` | 로컬 git, 원격 없음 | **자격증명은 `.gitignore`로 제외되어 있으나 반드시 재확인.** 비공개 권장 |
| `~/dong/ai_projects/olmoearth_projects` | Ai2 원본 클론 | 여기는 fork에 push (PR_DOSSIER §1 절차 참고) |

## Ai2 기여 제출 (별개 트랙)

PR·이슈 제출은 GitHub 푸시와 다른 작업입니다. `PR_DOSSIER.md`의 제출 순서를 따르세요:
① sample 스키마 PR → ② rslearn NotImplementedError 이슈 → ③ LFMC 재현성 리포트.
