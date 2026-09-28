"""Descriptive T0 date audit on frozen E5 input memberships; no model outputs."""
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys


def main():
    repo = Path(sys.argv[1]).resolve()
    dest = Path(sys.argv[2]).resolve()
    if dest.exists():
        raise FileExistsError(dest)
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def read(path): return json.loads(path.read_text())
    def rows(path): return [json.loads(line) for line in path.read_text().splitlines()]
    base = repo / 'code/e5_bundle_v0/frozen_inputs'
    plan = read(repo / 'config/e5_equal_budget_prereg_v0.json')
    for name in ('items.jsonl', 'c1_selected_ids.json', 'c1_results.json'):
        if sha(base / name) != plan['parents'][name]: raise ValueError('Frozen input differs')
    items = rows(base / 'items.jsonl')
    eligible = set(read(base / 'c1_selected_ids.json')['quality_symmetric'])
    by_id = {i['id']: i for i in items}
    strata = defaultdict(list)
    for key in eligible:
        i = by_id[key]
        strata[(str(i['event']), tuple(i['dates']), tuple(i['slots']))].append(i)
    primary = {i['id'] for group in strata.values() if {i['kind'] for i in group} == {'pos', 'hard_neg'} for i in group}
    stored = read(base / 'c1_results.json')['subsets']['quality_symmetric']['per_seed']['1']['arms']['reader']
    recorded = {key for event in stored.values() for s in event['strata'] for name in ('pos_ids','hard_neg_ids') for key in s[name]}
    if primary != recorded or len(primary) != 902: raise ValueError('Primary membership differs')
    t0 = repo / 'artifacts/t0_source_date_join_v0_20260925'
    manifest = read(t0 / 'manifest.json')
    if not manifest['valid'] or sha(t0 / 'joined_rows.jsonl') != manifest['files_sha256']['joined_rows.jsonl']:
        raise ValueError('T0 source invalid')
    joined = {r['id']: r for r in rows(t0 / 'joined_rows.jsonl')}
    diagnostics = []
    for i in items:
        if i['phen'] != 'flood': continue
        r = joined[i['tile']]
        if r['status'] != 'matched_metadata' or r['partition'] != i['partition'] or str(r['event_id']) != str(i['event']):
            raise ValueError('E5/T0 identity mismatch')
        temporal = r['temporal']
        slots = i['slots']
        if i['dates'] != [temporal['old_synthetic_date_only'][s] for s in slots]:
            raise ValueError('E5 synthetic date contract differs')
        observed = [date.fromisoformat(temporal['published_slot_dates'][s]) for s in slots]
        imputed = [date.fromisoformat(d) for d in i['dates']]
        diagnostics.append({'id':i['id'], 'tile':i['tile'], 'partition':i['partition'], 'kind':i['kind'],
            'event':str(i['event']), 'primary':i['id'] in primary, 'documented_dates':[d.isoformat() for d in observed],
            'documented_gap_days':(observed[1]-observed[0]).days, 'imputed_gap_days':(imputed[1]-imputed[0]).days,
            'both_dates_equal':observed==imputed, 'gap_equal':(observed[1]-observed[0])==(imputed[1]-imputed[0])})
    groups = {'C1_primary_902':[r for r in diagnostics if r['primary']]}
    for part in ('train','test'):
        for kind in ('pos','neg','hard_neg'):
            groups[part+'/'+kind] = [r for r in diagnostics if r['partition']==part and r['kind']==kind]
    summary = {}
    for name, group in groups.items():
        gaps = [r['documented_gap_days'] for r in group]
        per_event = {e: sorted({r['documented_gap_days'] for r in group if r['event']==e}) for e in sorted({r['event'] for r in group})}
        summary[name] = {'n_items':len(group), 'n_tiles':len({r['tile'] for r in group}), 'n_events':len(per_event),
            'documented_gap_days':{'min':min(gaps),'median':statistics.median(gaps),'max':max(gaps)},
            'both_dates_equal_to_current':sum(r['both_dates_equal'] for r in group),
            'gap_equal_to_current':sum(r['gap_equal'] for r in group), 'event_gap_sets_days':per_event}
    report = {'schema':'t0-e5-input-population-date-diagnostic-v0','checked_at':datetime.now(timezone.utc).isoformat(),
        'scope':'Post-T0 descriptive membership audit, before E5 final results. No score association, exclusions or E5 changes. Dates are original metadata + published mapping; historical pixel lineage unverified.',
        'summary':summary, 'rows':diagnostics, 'source_sha256':{str(base / n):sha(base / n) for n in ('items.jsonl','c1_selected_ids.json','c1_results.json')},
        't0_manifest_sha256':sha(t0/'manifest.json'),'script_sha256':sha(Path(__file__))}
    with dest.open('x') as f: json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
