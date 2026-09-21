// 사람 판독 라벨 v2 — 상위 30 + 대조 30 = 60곳의 "정답 데이터" 를 만드는 가장 짧은 경로 (WP1, docs/LABEL_SPEC_OREUM_v1.md).
// 정적 export 라 서버가 없다: 클릭 → localStorage 즉시 저장 → 내보내기(JSON) → public/data/labels.json 으로 커밋.
// 앱은 커밋된 labels.json 을 기본값으로 읽고, 이 브라우저의 localStorage 가 그 위에 덮인다.
// 라벨은 임베딩 점수를 *설명* 하지 않는다. 사람이 항공사진·전후 프레임에서 본 것을 4갈래로만 기록하고,
// a(실질 변화)에만 무엇을 봤는지 태그 하나를 더 붙인다. 대상 명단은 label_targets.json(code/make_label_targets.py)에 봉인된다.

export type LabelCode = 'a' | 'b' | 'c' | 'd';
export type LabelTag = 'trail_erosion' | 'veg_loss' | 'facility' | 'grazing' | 'coastal' | 'other';
export type Recheck = { code: LabelCode; tag?: LabelTag; at: string };   // 같은 사람이 나중에 다시 본 결과 (자기 일치율)
export type Label = { code: LabelCode; tag?: LabelTag; note: string; at: string; recheck?: Recheck };
export type LabelFile = { schema: 'oreum-labels/2'; contract_sha256: string | null; targets_sha256: string | null; exported_at: string;
  agreement: { n: number; same_code: number } | null; labels: Record<string, Label> };
export type Targets = { schema: 'oreum-label-targets/1'; contract_sha256: string | null; targets_sha256: string; seed: number; control_rule: string;
  top: string[]; control: string[]; recheck: string[]; tags: { tag: LabelTag; short: string; long: string }[] };

export const LABEL_CODES: { code: LabelCode; short: string; long: string }[] = [
  { code: 'a', short: '실질 변화', long: '전후 프레임·항공사진에서 지표 변화(나지·시설·식생 손실)가 보인다 → 태그 하나를 더 고른다' },
  { code: 'b', short: '지속 인공물', long: '두 해 모두 같은 시설·도로가 있다 — 연간 변화가 아니다' },
  { code: 'c', short: '대기·구름', long: '구름 가장자리·해무·계절 차이로 깃발이 섰다 — 지표 변화 아님' },
  { code: 'd', short: '판독 불가', long: '프레임·항공사진으로는 판단할 수 없다 → 메모에 이유(해상도·구름·연도 불명)를 적는다' },
];
export const TAG_KEYS: LabelTag[] = ['trail_erosion', 'veg_loss', 'facility', 'grazing', 'coastal', 'other'];   // 숫자키 1–6
const KEY = 'oreum-labels-v9';

export function loadLocal(): Record<string, Label> {
  try { return JSON.parse(localStorage.getItem(KEY) ?? '{}'); } catch { return {}; }
}
export function saveLocal(labels: Record<string, Label>) {
  try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch { /* 사생활 모드 등: 화면 상태만 유지 */ }
}
export function agreement(labels: Record<string, Label>, recheckIds: string[]) {
  const done = recheckIds.filter(id => labels[id]?.recheck);
  return done.length ? { n: done.length, same_code: done.filter(id => labels[id].recheck!.code === labels[id].code).length } : null;
}
export function toFile(labels: Record<string, Label>, contract: string | null, targets: Targets | null): LabelFile {
  return { schema: 'oreum-labels/2', contract_sha256: contract, targets_sha256: targets?.targets_sha256 ?? null, exported_at: new Date().toISOString(),
    agreement: targets ? agreement(labels, targets.recheck) : null, labels };
}
export function download(file: LabelFile) {
  const blob = new Blob([JSON.stringify(file, null, 1)], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'labels.json'; a.click();
  URL.revokeObjectURL(a.href);
}
