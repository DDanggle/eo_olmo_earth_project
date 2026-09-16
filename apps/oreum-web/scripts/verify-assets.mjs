// 발표 숫자를 봉인 JSON 에 대조하고, 금지 문구를 사용자 대면 복사에서 grep 한다.
// 네팔 web/scripts/verify-assets.mjs 의 규율을 그대로 옮겼다: 지도가 읽는 숫자와
// 연구 산출물이 어긋난 채 배포되는 일을 빌드 단계에서 막는다.
import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const root = new URL('..', import.meta.url).pathname;
const data = join(root, 'public/data');
const fail = (m) => { console.error('VERIFY FAIL —', m); process.exit(1); };

for (const f of ['summary.json', 'oreum.geojson']) if (!existsSync(join(data, f))) fail(`${f} 없음 — oreum_export_web.py 를 먼저 돌려라`);
const s = JSON.parse(readFileSync(join(data, 'summary.json'), 'utf8'));
const g = JSON.parse(readFileSync(join(data, 'oreum.geojson'), 'utf8'));

// 1. 분모 · 합계
if (g.features.length !== s.frame_a_total) fail(`geojson ${g.features.length}점 ≠ frame_a_total ${s.frame_a_total}`);
const scored = g.features.filter((f) => f.properties.verdict === 'scored');
const abstain = g.features.filter((f) => f.properties.verdict === 'abstain');
if (scored.length !== s.scored) fail(`채점 ${scored.length} ≠ summary.scored ${s.scored}`);
if (abstain.length !== s.abstain) fail(`보류 ${abstain.length} ≠ summary.abstain ${s.abstain}`);
if (scored.length + abstain.length !== s.frame_a_total) fail('채점+보류 ≠ 분모 — abstain 을 분모에서 뺐다');

// 2. abstain 은 점수가 null 이어야 한다. 0 은 "변화 없음" 으로 읽히므로 금지.
for (const f of abstain) if (f.properties.event_flag_frac !== null || f.properties.rank !== null) fail(`${f.properties.oreum_id}: abstain 인데 점수/순위가 있다`);
for (const f of scored) if (typeof f.properties.event_flag_frac !== 'number') fail(`${f.properties.oreum_id}: scored 인데 점수 없음`);

// 3. 순위는 1..scored 의 순열이어야 한다
const ranks = scored.map((f) => f.properties.rank).sort((a, b) => a - b);
if (ranks.some((r, i) => r !== i + 1)) fail('순위가 1..N 순열이 아니다');

// 4. 귀무 깃발율 — 구성상 ≈1%. 0.7~1.3% 밖이면 파이프라인이 깨진 것.
const nr = s.flag_rate_pooled.null_temporal_primary;
if (!(nr > 0.007 && nr < 0.013)) fail(`주 귀무 깃발율 ${(nr * 100).toFixed(2)}% — 1% 구성이 깨졌다`);

// 5. 색계급 경계는 단조증가 5색
if (s.class_breaks.length !== 4 || s.class_colors.length !== 5) fail('색계급 배열 길이');
if (s.class_breaks.some((b, i) => i && b < s.class_breaks[i - 1])) fail('색계급 경계 비단조');

// 6. 프레임: 채점된 오름은 전후 두 프레임이 반드시 있어야 한다 ("색이 있으면 누르면 보인다")
const frames = new Set(existsSync(join(data, 'frames')) ? readdirSync(join(data, 'frames')) : []);
const [y0, y1] = s.pairs.event;
const missing = scored.filter((f) => !frames.has(`${f.properties.oreum_id}_${y0}.jpg`) || !frames.has(`${f.properties.oreum_id}_${y1}.jpg`));
if (missing.length) fail(`채점된 오름 ${missing.length}곳에 전후 프레임이 없다: ${missing.slice(0, 3).map((f) => f.properties.oreum_id).join(', ')}`);

// 7. 금지 문구 — 사용자 대면 복사
const copy = readFileSync(join(root, 'app/page.tsx'), 'utf8') + readFileSync(join(root, 'app/layout.tsx'), 'utf8');
for (const bad of ['훼손을 탐지', '탐지했습니다', 'AI가 탐지', '95.64', 'Ai2 와 함께', 'Ai2 공식', '제휴하여']) {
  if (copy.includes(bad)) fail(`금지 문구: "${bad}"`);
}
if (!copy.includes('제휴·보증 관계가 없습니다')) fail('비제휴 고지 누락');

console.log(`VERIFY OK — ${s.frame_a_total}곳 · 채점 ${s.scored} · 보류 ${s.abstain} · 귀무 ${(nr * 100).toFixed(2)}% · 프레임 ${frames.size}`);
