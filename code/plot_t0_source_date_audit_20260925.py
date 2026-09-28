"""Plot metadata-derived observation gaps, never model performance or sensor UTC."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    repo = args.repo.resolve()
    root = repo/'artifacts/t0_source_date_join_v0_20260925'
    independent_path = repo/'artifacts/T0_INDEPENDENT_RESULT_REVIEW_20260925.json'
    audit = read(independent_path)
    assert audit['consistent'] is True and audit['n_rows'] == 7000
    for name in ('manifest.json', 'joined_rows.jsonl', 'summary.json'):
        assert sha(root/name) == audit['source_hashes_verified'][str(root/name)]
    manifest = read(root/'manifest.json')
    assert manifest['valid'] is True
    rows = [json.loads(line) for line in (root/'joined_rows.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == 7000
    assert all(r['status'] == 'matched_metadata' for r in rows)
    diagnostic_path = repo/'artifacts/T0_E5_POPULATION_DATE_DIAGNOSTIC_20260925.json'
    diagnostic = read(diagnostic_path)
    assert diagnostic['t0_manifest_sha256'] == sha(root/'manifest.json')
    primary = [r for r in diagnostic['rows'] if r['primary']]
    assert len(primary) == len({r['id'] for r in primary}) == 902
    gap_sets = defaultdict(set)
    for row in primary:
        gap_sets[row['event']].add(row['documented_gap_days'])
    assert len(gap_sets) == 8 and all(len(v) == 1 for v in gap_sets.values())
    by_event = {key: next(iter(gaps)) for key, gaps in gap_sets.items()}
    event_counts = Counter(r['event'] for r in primary)
    args.out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.8, 5.8), gridspec_kw={'width_ratios':[1.25, 1]})
    split_stats = {}
    for part, color, title in [('train','#3977a4','Train'), ('validation','#bc7335','Validation'), ('test','#508a67','Test')]:
        gaps = np.sort([r['temporal']['intervals_days']['pre2_to_post_days'] for r in rows if r['partition']==part])
        median = float(np.median(gaps))
        split_stats[part] = {'n':len(gaps), 'median':median, 'max':int(gaps[-1])}
        x = np.r_[0, gaps, 700]
        y = np.r_[0, np.arange(1,len(gaps)+1)/len(gaps), 1]
        left.step(x, y, where='post', lw=2, color=color, label=f'{title}: n={len(gaps):,}, median={median:g} d')
    left.axvline(12, ls='--', color='#555', lw=1.2, label='Current imputed gap: 12 d')
    left.set(xlim=(0,700), ylim=(0,1.04), xlabel='Documented pre₂ → post interval (days)', ylabel='Cumulative fraction of tiles', title='All 7,000 catalog tiles, by split')
    left.legend(loc='lower right', fontsize=9, frameon=False)
    order = sorted(by_event, key=lambda event:(by_event[event],int(event)))
    y = np.arange(len(order))
    colors = ['#508a67' if by_event[event]==12 else '#a35e48' for event in order]
    right.barh(y, [by_event[e] for e in order], color=colors, height=.62)
    right.set_yticks(y, [f'{e}  (n={event_counts[e]})' for e in order])
    for yy, event in zip(y,order):
        right.text(by_event[event]+5, yy, str(by_event[event])+' d', va='center', fontsize=9)
    right.axvline(12, ls='--', color='#555', lw=1.2)
    right.invert_yaxis()
    right.set(xlim=(0,335), xlabel='Documented pre₂ → post interval (days)', ylabel='Event ID (primary items)', title='Frozen E5 primary: 902 items / 8 events')
    for ax in (left,right):
        ax.grid(axis='x', color='#ededed', lw=.7)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('A 12-day approximation hides the variation in observation intervals', fontsize=14, x=.52, y=.975)
    fig.subplots_adjust(left=.06,right=.98,bottom=.265,top=.86,wspace=.43)
    caption = ('Original metadata + published GEO-Bench slot mapping; 7,000 unique exact geometry/metadata joins.\n'
        'Six of the eight E5 primary events have intervals of 36–288 days. Its item-weighted median is still 12 days.\n'
        'Day-level source dates; historical raster correspondence and acquisition UTC remain unverified.\n'
        'Descriptive input audit only. E5 inputs are unchanged; no model-output association or causal claim.')
    fig.text(.06,.055,caption,fontsize=9,linespacing=1.55,color='#454545')
    fig.savefig(args.out/'source_date_intervals.png',dpi=180)
    fig.savefig(args.out/'source_date_intervals.pdf')
    plt.close(fig)
    report = {'created_at':datetime.now(timezone.utc).isoformat(), 'script_sha256':sha(Path(__file__)),
        'input_sha256':{str(f):sha(f) for f in (independent_path, root/'manifest.json', root/'joined_rows.jsonl',diagnostic_path)},
        'split_statistics':split_stats, 'primary_event_gap_days':by_event, 'primary_event_counts':dict(event_counts),
        'caption':caption, 'matplotlib':matplotlib.__version__, 'numpy':np.__version__,
        'outputs_sha256':{name:sha(args.out/name) for name in ('source_date_intervals.png','source_date_intervals.pdf')}}
    (args.out/'plot_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','caption')},indent=2))


if __name__ == '__main__':
    main()
