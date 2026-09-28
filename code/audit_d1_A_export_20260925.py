"""Read-only descriptive audit. Does not finalize H or build L/R targets."""
import argparse
import collections
import datetime
import hashlib
import json
import statistics
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--repo', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
ROOT = args.repo.resolve()
sys.path.insert(0, str(ROOT / 'code'))
from sn7_visible_pack_v05 import load_pack
from sn7_visible_contract_v05 import derive_target

PACK_PATH = ROOT / 'labeling_pack/visible_contract_v05_20260922/annotator_pack/pack.json'
EXPORT_PATH = ROOT / 'labeling_pack/visible_contract_v05_20260922/exports/sn7v05-198e4ff03e26f004_dongdong.json'
pack = load_pack(PACK_PATH)  # verifies image bytes by hash; does not display images
export = json.loads(EXPORT_PATH.read_text())
assert pack['pack_id'] == export['pack_id']
episodes = {ep['id']: ep for ep in pack['episodes']}
annotations = export['annotations']
assert len(annotations) == len({a['episode_id'] for a in annotations})
assert set(episodes) == {a['episode_id'] for a in annotations}
assert all(a['annotator_id'] == export['annotator_id'] for a in annotations)
assert all(a['status'] == 'complete' for a in annotations)
rows = []
all_states = collections.Counter()
for ann in annotations:
    ep = episodes[ann['episode_id']]
    target = derive_target(ep, ann)
    counts = collections.Counter(ann['states'].values())
    all_states.update(counts)
    frame_by_id = {f['id']: f for f in ep['frames']}
    visible_frames = [f for f in ep['frames'] if ann['states'][f['id']] == 'visible_change']
    nochange_after_visible = []
    if visible_frames:
        first_visible_index = ep['frames'].index(visible_frames[0])
        nochange_after_visible = [f['id'] for f in ep['frames'][first_visible_index+1:] if ann['states'][f['id']] == 'no_visible_change']
    rows.append({
        'id': ep['id'], 'aoi': ep['aoi'], 'quadrant': ep['region'], 'start': ep['frames'][0]['date'],
        'cutoff': ep['cutoff'], 'frames': len(ep['frames']), 'seconds': ann['seconds'], 'timestamp': ann['timestamp'],
        'frame_state_counts': dict(counts), 'target': target,
        'uncertain_frames': [{'id': f['id'], 'date': f['date'], 'state': ann['states'][f['id']]} for f in ep['frames'] if ann['states'][f['id']] in {'unreadable', 'ambiguous'}],
        'visible_frame_ids': [f['id'] for f in visible_frames],
        'nochange_after_visible_ids': nochange_after_visible,
        'first_raw_visible_date': visible_frames[0]['date'] if visible_frames else None,
    })
grouped = collections.defaultdict(list)
for row in rows:
    grouped[row['aoi']].append(row)
pairs = []
for aoi, group in grouped.items():
    assert len(group) == 2
    a, b = sorted(group, key=lambda x: x['id'])
    signature_a = (a['target']['answer'], a['target']['first_change_date'])
    signature_b = (b['target']['answer'], b['target']['first_change_date'])
    pairs.append({'aoi': aoi, 'episodes': [a['id'], b['id']], 'signatures': [signature_a, signature_b],
                  'candidate_type_A_only': 'content' if signature_a[0] != signature_b[0] else ('temporal' if signature_a != signature_b else 'none')})
seconds = [r['seconds'] for r in rows]
timestamps = [datetime.datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00')) for r in rows]
first_start_inferred = timestamps[0] - datetime.timedelta(seconds=seconds[0])
result = {
    'scope': 'single-rater descriptive audit; not H consensus, L, R, or model evaluation',
    'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [PACK_PATH, EXPORT_PATH, ROOT/'code/sn7_visible_contract_v05.py', ROOT/'code/sn7_visible_pack_v05.py']},
    'pack_id': pack['pack_id'], 'split': pack['split'], 'pack_and_image_hash_integrity': 'pass',
    'annotator_id': export['annotator_id'], 'episodes': len(rows), 'aois': len(grouped), 'frames': sum(r['frames'] for r in rows),
    'timeouts': 0, 'duplicate_or_missing_episode_ids': 0,
    'total_seconds': sum(seconds), 'total_minutes': sum(seconds)/60, 'median_seconds': statistics.median(seconds),
    'min_seconds': min(seconds), 'max_seconds': max(seconds), 'seconds_per_frame_aggregate': sum(seconds)/sum(r['frames'] for r in rows),
    'inferred_start_utc': first_start_inferred.isoformat(), 'last_save_utc': timestamps[-1].isoformat(),
    'inferred_wall_seconds': (timestamps[-1]-first_start_inferred).total_seconds(),
    'class_counts': dict(collections.Counter(r['target']['answer'] for r in rows)), 'frame_state_counts': dict(all_states),
    'raw_current_state_counts': dict(collections.Counter(r['target']['raw_current_state'] for r in rows)),
    'derived_current_state_counts': dict(collections.Counter(r['target']['current_state'] for r in rows)),
    'candidate_pair_counts_A_only': dict(collections.Counter(p['candidate_type_A_only'] for p in pairs)),
    'rows': sorted(rows, key=lambda r: r['id']), 'pairs': pairs,
}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in {'rows', 'pairs'}}, ensure_ascii=False, indent=2))
for row in result['rows']:
    t = row['target']
    print(row['id'], row['frames'], row['seconds'], t['answer'], t['first_change_date'], t['last_clear_no_change_date'], t['current_state'], row['frame_state_counts'], 'evidence='+str(t['evidence_ids']), 'reversion='+str(row['nochange_after_visible_ids']))
print(json.dumps(pairs, ensure_ascii=False, indent=2))
