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

// 6b. 월별 시계열 (상위 12): 색인에 있는 오름마다 JSON 이 있고, 참조된 프레임 파일이 실제로 존재해야 한다.
//     빈 달(null / frame:null)은 정상이다 — 볼 수 없었던 달을 지우지 않는다.
const seriesDir = join(data, 'series');
let seriesNote = '시계열 없음';
if (existsSync(join(seriesDir, 'index.json'))) {
  const idx = JSON.parse(readFileSync(join(seriesDir, 'index.json'), 'utf8'));
  let nFrames = 0, nDelta = 0, nEmpty = 0; const missingDelta = [];
  for (const oid of idx.oreum_ids) {
    const jp = join(seriesDir, `${oid}.json`);
    if (!existsSync(jp)) fail(`시계열 JSON 없음: ${oid}`);
    const sj = JSON.parse(readFileSync(jp, 'utf8'));
    if (Object.keys(sj.frames).length !== idx.months.length) fail(`${oid}: 달 수 ${Object.keys(sj.frames).length} ≠ 색인 ${idx.months.length}`);
    for (const [k, fr] of Object.entries(sj.frames)) {
      if (!fr || !fr.frame) { nEmpty++; continue; }
      if (!existsSync(join(seriesDir, oid, fr.frame))) fail(`${oid} ${k}: 프레임 파일 없음`);
      nFrames++;
      // Δz 파일이 JSON 에 적혀 있는데 없는 경우(2026-09-22 실측: 143 의 2022-01 이 다른 기기에서 커밋 누락) — 프레임은 막되 Δz 는 경고로.
      if (fr.delta) { if (!existsSync(join(seriesDir, oid, fr.delta.frame))) { missingDelta.push(`${oid} ${k}`); continue; } nDelta++; }
    }
  }
  if (missingDelta.length) console.warn(`VERIFY WARN — Δz 파일 누락 ${missingDelta.length}건 (JSON 이 참조하나 파일 없음; 시계열 재생성 필요): ${missingDelta.slice(0, 3).join(', ')}`);
  seriesNote = `시계열 ${idx.oreum_ids.length}곳 · 프레임 ${nFrames} · Δz ${nDelta} · 빈 달 ${nEmpty}${missingDelta.length ? ` · Δz 누락 ${missingDelta.length}` : ''}`;
}

// 6c. 라벨 대상 명단(label_targets.json): 상위 30 + 대조 30 + 재판독 5. 계약 sha 가 같아야 하고, 상위는 실제 순위 1..30 이어야 한다.
const byId = new Map(g.features.map((f) => [f.properties.oreum_id, f.properties]));
let targets = null;
if (existsSync(join(data, 'label_targets.json'))) {
  targets = JSON.parse(readFileSync(join(data, 'label_targets.json'), 'utf8'));
  if (targets.schema !== 'oreum-label-targets/1') fail('label_targets.json schema');
  if (targets.contract_sha256 !== s.contract_sha256) fail('label_targets.json 이 다른 계약에서 만들어졌다 — code/make_label_targets.py 재실행');
  const topExpected = scored.slice().sort((a, b) => a.properties.rank - b.properties.rank).slice(0, targets.top.length).map((f) => f.properties.oreum_id);
  if (JSON.stringify(topExpected) !== JSON.stringify(targets.top)) fail('label_targets.top 이 현재 순위 1..N 과 다르다');
  for (const oid of [...targets.control, ...targets.recheck]) { const p = byId.get(oid); if (!p || p.verdict !== 'scored') fail(`label_targets: ${oid} 는 채점 대상이 아니다`); }
  if (targets.control.some((id) => targets.top.includes(id))) fail('label_targets: 대조군이 상위군과 겹친다');
  if (targets.recheck.some((id) => !targets.top.includes(id) && !targets.control.includes(id))) fail('label_targets: 재판독 대상이 명단 밖');
}

// 6d. 사람 판독 라벨(labels.json, 있을 때만): 오름 id 가 명단 안에 있고 코드는 a–d, a 에는 태그, 재판독은 명단의 5곳에만.
let labelNote = '라벨 없음';
if (existsSync(join(data, 'labels.json'))) {
  const lf = JSON.parse(readFileSync(join(data, 'labels.json'), 'utf8'));
  if (lf.schema !== 'oreum-labels/2') fail('labels.json schema (v2 필요 — 예전 v1 파일이면 다시 내보내기)');
  if (lf.contract_sha256 && lf.contract_sha256 !== s.contract_sha256) fail('labels.json 이 다른 계약에서 찍혔다 — 순위가 바뀌었으니 다시 판독');
  if (!targets) fail('labels.json 은 있는데 label_targets.json 이 없다');
  if (lf.targets_sha256 && lf.targets_sha256 !== targets.targets_sha256) fail('labels.json 이 다른 대상 명단에서 찍혔다');
  const allowed = new Set([...targets.top, ...targets.control]); const tags = new Set(targets.tags.map((t) => t.tag));
  for (const [oid, l] of Object.entries(lf.labels)) {
    if (!byId.get(oid)) fail(`labels.json: 없는 오름 ${oid}`);
    if (!allowed.has(oid)) fail(`labels.json: ${oid} 는 라벨 대상(상위 30 + 대조 30)이 아니다`);
    if (!['a', 'b', 'c', 'd'].includes(l.code)) fail(`labels.json: ${oid} 코드 ${l.code}`);
    if (l.code === 'a' && !tags.has(l.tag)) fail(`labels.json: ${oid} 는 a 인데 태그가 없거나 모름 (${l.tag})`);
    if (l.code !== 'a' && l.tag) fail(`labels.json: ${oid} 는 ${l.code} 인데 태그가 붙어 있다`);
    if (l.code === 'd' && !(l.note ?? '').trim()) fail(`labels.json: ${oid} 는 판독 불가(d)인데 이유 메모가 없다`);
    if (l.recheck && !targets.recheck.includes(oid)) fail(`labels.json: ${oid} 에 재판독이 있는데 재판독 대상이 아니다`);
  }
  const n = Object.keys(lf.labels).length; const nd = Object.values(lf.labels).filter((l) => l.code === 'd').length;
  if (n >= 30 && nd / n > 0.3) fail(`판독 불가(d) 가 ${nd}/${n} — 30% 초과. 근거 자료(항공사진 연도·SAR)를 먼저 보강`);
  labelNote = `라벨 ${n}/${allowed.size} · d ${nd} · 재판독 ${lf.agreement ? `${lf.agreement.same_code}/${lf.agreement.n} 일치` : '없음'}`;
}

// 7. 금지 문구 — 사용자 대면 복사
const copy = readFileSync(join(root, 'app/page.tsx'), 'utf8') + readFileSync(join(root, 'app/layout.tsx'), 'utf8');
for (const bad of ['훼손을 탐지', '탐지했습니다', 'AI가 탐지', '95.64', 'Ai2 와 함께', 'Ai2 공식', '제휴하여']) {
  if (copy.includes(bad)) fail(`금지 문구: "${bad}"`);
}
if (!copy.includes('제휴·보증 관계가 없습니다')) fail('비제휴 고지 누락');

console.log(`VERIFY OK — ${s.frame_a_total}곳 · 채점 ${s.scored} · 보류 ${s.abstain} · 귀무 ${(nr * 100).toFixed(2)}% · 프레임 ${frames.size} · ${seriesNote} · ${labelNote}`);
