#!/usr/bin/env python3
"""Plot completed, independently audited E5 results. No models or predictions read.

Requires NumPy and Matplotlib. Produces PNG, PDF, caption and provenance in a new
directory only. There is no demo/fallback-data mode and no seed aggregation.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np

SEEDS = ('1', '2', '3')
EVALUATIONS = ('full/native', 'pair/native', 'later/native', 'delta/native', 'full/full_no_delta')
SCIENCE_KEYS = ('population', 'inputs', 'arms', 'training', 'evaluation', 'primary_analysis',
                'secondary_descriptive', 'validity', 'parents', 'pool_reproduction')
SCIENCE_SHA = 'bff66ca45d0fac6f64f276aa0a31506d3a6c56fea799c413481ccaa5882bbbad'
CAPTION = (
    'E5: equal-budget input comparison on the frozen C1 quality-controlled, same-prompt '
    'subset (445 source-positive and 457 hard-negative items; 8 events). Left: all three '
    'seeds are shown separately for balanced agreement with source labels, averaging '
    'same-prompt strata within each event and then weighting events equally. Full, pair, '
    'later and delta are separately trained from shared initialization within each seed; '
    'full -> [A,B,0] is an inference-only intervention on the trained full model, not a fifth '
    'trained model. A and B are earlier/later observation embeddings and D=B-A. All '
    'conditions retain 192 token positions; zero input blocks still pass through the '
    'projector and type embeddings. Right: the only primary contrast, trained pair minus '
    'trained full, with seed-specific 95% paired-event percentile bootstrap intervals '
    '(5,000 draws; 8 events resampled together across conditions). The dashed -0.05 line '
    'is the preregistered operational tolerance; it is not the entire decision rule, '
    'which also requires same-seed full/pair BA >= 0.60 in at least two seeds. Seeds '
    'are repeated fits, not additional independent events. Later, delta and full_no_delta '
    'comparisons are descriptive. These are exposed development data; the results alone '
    'do not establish temporal reasoning, memory benefit or independently verified change.'
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def pairs(entries):
        value = {}
        for key, item in entries:
            require(key not in value, 'Duplicate JSON key')
            value[key] = item
        return value
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def finite(value, low, high):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
            'Nonfinite/out-of-range plotted value')
    return float(value)


def same(actual, expected):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'Audit metric keys differ')
        for key in expected:
            same(actual[key], expected[key])
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'Audit metric list differs')
        for left, right in zip(actual, expected):
            same(left, right)
    elif type(expected) in (int, float):
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10), 'Audit numeric metric differs')
    else:
        require(actual == expected, 'Audit metric differs')


def load_verified(artifact, audit_path):
    artifact, audit_path = Path(artifact).resolve(), Path(audit_path).resolve()
    require(artifact.is_dir() and not (artifact / 'failure.json').exists(), 'Missing/failed run')
    audit = read(audit_path)
    require(audit.get('schema') == 'e5-independent-result-audit-v0'
            and audit.get('consistent') is True, 'Independent audit did not pass')
    original_root = audit.get('artifact')
    require(isinstance(original_root, str) and original_root, 'Missing original audited artifact root')
    remote_root = PurePosixPath(original_root)
    require(remote_root.is_absolute() and '..' not in remote_root.parts
            and str(remote_root) == original_root, 'Original audited root must be a canonical absolute POSIX path')
    require((audit['n_models'], audit['n_steps'], audit['n_training_exposures'], audit['n_answers'],
             audit['primary_n'], audit['primary_events']) == (12, 19080, 152424, 26325, 902, 8),
            'Audit completion/population counts differ')
    pins = {str(audit_path): sha(audit_path)}
    lookup_hashes = {}
    for name in ('status.json', 'scores.json', 'prereg.json'):
        path = artifact / name
        digest = sha(path)
        original_key = str(remote_root / name)
        require(audit['hashes_verified'].get(original_key) == digest, 'Independent audit hash mismatch: ' + name)
        lookup_hashes[original_key] = digest
        pins[str(path)] = digest
    status, scores, plan = [read(artifact / name) for name in ('status.json', 'scores.json', 'prereg.json')]
    require(status.get('status') == 'completed' and status.get('n_models') == 12
            and status.get('n_rows') == 26325, 'Run is not completed')
    require(scores.get('schema') == 'e5-equal-budget-scores-v0' and scores.get('valid') is True,
            'Scores are invalid')
    require(scores['verdict'] == status['verdict'] == audit['verdict'] and scores['verdict'] != 'invalid',
            'Verdict/completion records disagree')
    science = json.dumps({k: plan[k] for k in SCIENCE_KEYS}, sort_keys=True,
                         separators=(',', ':'), ensure_ascii=False).encode()
    require(hashlib.sha256(science).hexdigest() == SCIENCE_SHA, 'Frozen scientific plan differs')
    require(scores['coverage'] == {'expected': 26325, 'received': 26325, 'n_items': 5989,
            'n_train': 4234, 'n_test': 1755, 'n_primary': 902}, 'Evaluation coverage differs')
    require(set(scores['metrics']) == set(SEEDS) and set(audit['primary_metrics']) == set(SEEDS), 'Seed support differs')
    series, contrast, event_support = {}, {}, None
    for seed in SEEDS:
        primary = scores['metrics'][seed]['primary_same_prompt']
        same(primary, audit['primary_metrics'][seed])
        evaluations = primary['evaluations']
        require(set(evaluations) == set(EVALUATIONS), 'Plot condition support differs')
        series[seed] = {}
        for key in EVALUATIONS:
            metric = evaluations[key]
            events = metric['events']
            require(metric['n_events'] == len(events) == 8 and metric['n_items'] == 902, 'Primary support differs')
            if event_support is None:
                event_support = set(events)
            require(set(events) == event_support, 'Paired event support differs')
            require(sum(e['n_pos'] for e in events.values()) == 445
                    and sum(e['n_hard_neg'] for e in events.values()) == 457, 'Primary class counts differ')
            mean = float(np.mean([finite(events[e]['ba'], 0, 1) for e in sorted(events)]))
            require(math.isclose(finite(metric['macro_ba'], 0, 1), mean, abs_tol=1e-10), 'Equal-event mean differs')
            series[seed][key] = mean
        c = primary['contrasts']['pair_minus_full']
        require(c['left'] == 'pair/native' and c['right'] == 'full/native', 'Wrong primary contrast')
        differences = {e: evaluations['pair/native']['events'][e]['ba'] - evaluations['full/native']['events'][e]['ba']
                       for e in sorted(event_support)}
        same(c['event_deltas'], differences)
        values = np.array(list(differences.values()))
        draws = np.random.default_rng(20260925).integers(0, 8, size=(5000, 8))
        ci = np.quantile(values[draws].mean(axis=1), [.025, .975], method='linear').tolist()
        require(math.isclose(finite(c['delta'], -1, 1), float(values.mean()), abs_tol=1e-10), 'Contrast mean differs')
        require(isinstance(c['ci95_delta'], list) and len(c['ci95_delta']) == 2, 'CI absent')
        same(c['ci95_delta'], ci)
        contrast[seed] = {'delta': float(values.mean()), 'ci95_delta': ci}
    return {'schema': 'e5-verified-plot-data-v0', 'series': series, 'primary_contrast': contrast,
            'seeds': list(SEEDS), 'events': sorted(event_support), 'n_items': 902, 'n_events': 8,
            'verdict': scores['verdict'], 'source_hashes': pins, 'caption': CAPTION,
            'audit_original_artifact_root': original_root, 'local_render_artifact_root': str(artifact),
            'audited_input_lookup_hashes': lookup_hashes,
            'verification_scope': 'The independent audit ran at its recorded original artifact root. '
                'This plotting step verifies the local scores/status/prereg bytes against the exact '
                'original-root keys and checks plotted metrics; it does not rerun the full audit, '
                'load predictions, or recheck remote pairs/checkpoints.',
            'checkpoint_tensors_loaded_and_checked_on_cpu': audit.get('checkpoint_tensors_loaded_and_checked_on_cpu')}


def render(artifact, audit_path, out):
    data = load_verified(artifact, audit_path)
    # No backend import or output directory until the completion/audit gates pass.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out = Path(out).resolve()
    require(not out.exists(), 'Destination exists; no overwrite')
    require(not out.is_relative_to(Path(artifact).resolve()), 'Write figure outside the research run directory')
    out.mkdir(parents=True, exist_ok=False)
    try:
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                             'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
        fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 6.8), gridspec_kw={'width_ratios': [1.28, 1]})
        fig.subplots_adjust(left=.17, right=.97, bottom=.29, top=.74, wspace=.30)
        colors, markers, offsets = ['#0072B2', '#D55E00', '#009E73'], ['o', 's', '^'], [-.17, 0, .17]
        labels = ['Full  [A,B,D]\ntrained', 'Pair  [A,B,0]\ntrained',
                  'Later  [0,B,0]\ntrained · descriptive', 'Delta  [0,0,D]\ntrained · descriptive',
                  'Full → [A,B,0]\nfrozen intervention\n(descriptive)']
        for seed, color, marker, offset in zip(SEEDS, colors, markers, offsets):
            left.scatter([data['series'][seed][key] for key in EVALUATIONS], np.arange(5)+offset,
                         color=color, marker=marker, s=54, zorder=3, edgecolors='white', linewidths=.6)
        left.axvline(.5, color='#9AA0A6', linestyle=':', linewidth=1)
        left.set(xlim=(0, 1), ylim=(4.55, -.55), yticks=np.arange(5), yticklabels=labels,
                 xlabel='Balanced agreement with source labels')
        left.set_title('(a) Primary subset: all seeds', loc='left', fontsize=12, pad=14)
        left.grid(axis='x', alpha=.18)
        endpoints = []
        for j, (seed, color, marker) in enumerate(zip(SEEDS, colors, markers)):
            c = data['primary_contrast'][seed]
            lo, hi = c['ci95_delta']; point = c['delta']
            endpoints.extend([lo, hi, point])
            right.hlines(j, lo, hi, color=color, linewidth=2)
            right.vlines([lo, hi], j-.08, j+.08, color=color, linewidth=1.4)
            right.scatter([point], [j], color=color, marker=marker, s=62, zorder=3)
        right.axvline(0, color='#9AA0A6', linewidth=.9)
        right.axvline(-.05, color='#A02939', linestyle='--', linewidth=1.3,
                      label='Preregistered tolerance: −0.05')
        xmin, xmax = min(-.10, min(endpoints)-.035), max(.05, max(endpoints)+.035)
        right.set(xlim=(max(-1.02, xmin), min(1.02, xmax)), ylim=(2.5, -.5), yticks=[0, 1, 2],
                  yticklabels=['Seed 1', 'Seed 2', 'Seed 3'], xlabel='Δ balanced agreement: trained pair − trained full')
        right.set_title('(b) Only primary contrast\n95% paired-event intervals', loc='left', fontsize=12, pad=14)
        right.grid(axis='x', alpha=.18)
        right.legend(loc='upper center', bbox_to_anchor=(.5, -.18), frameon=False, fontsize=9)
        fig.suptitle('E5 | Equal-budget source-label discrimination', x=.02, ha='left', y=.975, fontsize=17, weight='bold')
        fig.text(.02, .919, 'C1 quality-controlled same-prompt subset: 902 items (445 positive, 457 hard-negative) · 8 events', fontsize=11)
        fig.legend([Line2D([], [], marker=m, linestyle='', color=c, markersize=7) for c,m in zip(colors,markers)],
                   ['Seed 1', 'Seed 2', 'Seed 3'], loc='upper right', bbox_to_anchor=(.98, .886), ncol=3, frameon=False)
        fig.text(.02, .165, 'A = earlier; B = later; D = B − A. Every condition retains 192 token positions. Zero blocks retain projector/type effects.', fontsize=9)
        fig.text(.02, .123, 'Intervals resample 8 paired events 5,000 times within each seed. Seeds are repeated fits, not additional independent events.', fontsize=9)
        fig.text(.02, .081, 'Later, delta and full_no_delta comparisons are descriptive. The −0.05 line alone is not the full preregistered decision rule.', fontsize=9)
        fig.text(.02, .035, 'Exposed development data · Source-label agreement is not proof of temporal reasoning or memory benefit.', fontsize=10, weight='bold')
        fig.savefig(out/'e5_equal_budget.png', dpi=300, facecolor='white')
        fig.savefig(out/'e5_equal_budget.pdf', facecolor='white', metadata={'Title': 'E5 equal-budget source-label discrimination', 'Subject': CAPTION})
        plt.close(fig)
        require(all(sha(Path(path)) == digest for path, digest in data['source_hashes'].items()), 'Source changed while plotting')
        data.update(plot_code_sha256=sha(__file__), created_at=datetime.now(timezone.utc).isoformat(),
                    matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
                    output_hashes={name:sha(out/name) for name in ('e5_equal_budget.png', 'e5_equal_budget.pdf')})
        (out/'caption.txt').write_text(CAPTION+'\n')
        (out/'plot_provenance.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
        return data
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps({'error':str(exc),'partial_outputs_not_valid':True}, indent=2)+'\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = render(args.artifact, args.audit, args.out)
    print(json.dumps({'out':str(args.out),'verdict':result['verdict'],'n_events':8,'n_items':902}))
