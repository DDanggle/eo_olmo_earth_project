#!/bin/bash
# 오름 추적 지도를 GitHub Pages 에 올린다 (정적 export). 사람이 직접 실행한다 — 공개 표면을 만드는 단계라서.
#   bash apps/oreum-web/scripts/deploy-pages.sh
# 결과: https://ddanggle.github.io/oreum-tracker/
set -euo pipefail
export PATH=/Users/dgyi/.nvm/versions/node/v22.16.0/bin:$PATH
REPO=DDanggle/oreum-tracker
HERE=$(cd "$(dirname "$0")/.." && pwd)     # apps/oreum-web
WORK=$(cd "$HERE/../.." && pwd)            # _work

echo "▸ 1/4 연구 저장소 push (eo_olmo_earth_project main)"
git -C "$WORK" push origin main

echo "▸ 2/4 공개 저장소 준비 ($REPO)"
gh repo view "$REPO" >/dev/null 2>&1 || gh repo create "$REPO" --public \
  --description "제주 오름 변화 추적 지도 — frozen OlmoEarth 임베딩, 사전등록 계약, abstain. Independent project; not affiliated with Ai2."

echo "▸ 3/4 정적 빌드 (basePath /oreum-tracker)"
cd "$HERE" && rm -rf out .next && NEXT_PUBLIC_BASE_PATH=/oreum-tracker CI=true pnpm build >/dev/null
touch out/.nojekyll                          # _next/ 디렉터리를 Jekyll 이 무시하지 않게

echo "▸ 4/4 gh-pages 브랜치로 push"
TMP=$(mktemp -d); cp -r out/. "$TMP"/; cd "$TMP"
git init -q -b gh-pages && git -c user.name=DDanggle add -A && git -c user.name=DDanggle commit -q -m "Publish oreum tracker $(date +%F)"
git remote add origin "git@github.com-dong:$REPO.git" 2>/dev/null || git remote add origin "https://github.com/$REPO.git"
git push -f -q origin gh-pages
gh api -X POST "repos/$REPO/pages" -f 'source[branch]=gh-pages' -f 'source[path]=/' >/dev/null 2>&1 || true   # 이미 켜져 있으면 무시
echo "✔ 완료 → https://ddanggle.github.io/oreum-tracker/  (첫 배포는 1~2분 뒤 열립니다)"
echo "  항공사진 토글은 VWorld 에 ddanggle.github.io 도메인을 등록하고 NEXT_PUBLIC_VWORLD_KEY 로 다시 빌드해야 켜집니다."
