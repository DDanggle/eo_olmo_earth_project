#!/usr/bin/env python3
"""Post-hoc exploratory C0 flood ranking diagnostics; no fitting or tuning.

Reads independently audited frozen artifacts. Computes exact tie-aware AUROC,
non-interpolated tie-group average precision (AP), equal-event macro and pooled
scores. The original logit>=0 decision threshold is never modified.
Requires only Python and NumPy; no C0 implementation imports or sklearn.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ARMS = ('earlier', 'later', 'pair')
COMPARISONS = {'pos_vs_hard_neg': ('hard_neg',),
               'pos_vs_prepre_neg': ('neg',),
               'pos_vs_all_no': ('neg', 'hard_neg')}
REPO = Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def ranking_metrics(labels, scores):
    """AUROC: exact win/tie pair credit. AP: grouped descending score thresholds.

    If either class is absent, both metrics are null by the declared comparison
    support policy (including AP for all-positive events, which is trivially 1).
    Equal floating-point values are ties; no rounding, binning, or jitter.
    """
    y, z = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    require(y.ndim == 1 and z.shape == y.shape and np.isin(y, (0, 1)).all(), 'Invalid labels/scores shape or class')
    require(np.isfinite(z).all(), 'Nonfinite logit')
    n, positives = len(y), int(np.sum(y))
    negatives = n - positives
    out = {'n': n, 'n_pos': positives, 'n_neg': negatives,
           'positive_prevalence': positives / n if n else None,
           'estimable': bool(positives and negatives), 'auroc': None, 'ap': None}
    if not out['estimable']:
        out['reason'] = 'no_items' if n == 0 else ('missing_positive_class' if positives == 0 else 'missing_negative_class')
        return out
    order = np.argsort(z, kind='stable')
    y, z = y[order], z[order]
    starts = np.r_[0, np.flatnonzero(z[1:] != z[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    groups = [(int(np.sum(y[a:b])), int(b - a)) for a, b in zip(starts, ends)]
    negative_below, wins = 0, 0.0
    for group_pos, size in groups:
        group_neg = size - group_pos
        wins += group_pos * (negative_below + group_neg * 0.5)
        negative_below += group_neg
    tp, selected, increments = 0, 0, []
    for group_pos, size in reversed(groups):
        tp += group_pos
        selected += size
        increments.append((group_pos / positives) * (tp / selected))
    out.update(auroc=wins / (positives * negatives), ap=math.fsum(increments),
               distinct_scores=len(groups), tie_groups=sum(size > 1 for _, size in groups),
               tied_items=sum(size for _, size in groups if size > 1))
    return out


def summarize(rows, negative_kinds, all_events):
    selected = [r for r in rows if r['kind'] == 'pos' or r['kind'] in negative_kinds]
    def score(subset):
        return ranking_metrics([r['kind'] == 'pos' for r in subset], [r['logit'] for r in subset])
    events = {event: score([r for r in selected if r['cluster'] == event]) for event in all_events}
    supported = [e for e in all_events if events[e]['estimable']]
    skipped = [{'event': e, 'reason': events[e]['reason'], 'n_pos': events[e]['n_pos'], 'n_neg': events[e]['n_neg']}
               for e in all_events if not events[e]['estimable']]
    return {'pooled': score(selected),
            'pooled_supported_events_only': score([r for r in selected if r['cluster'] in supported]),
            'events': events,
            'event_macro': {'n_supported_events': len(supported), 'n_total_events': len(all_events),
                            'n_supported_pos': sum(events[e]['n_pos'] for e in supported),
                            'n_supported_neg': sum(events[e]['n_neg'] for e in supported),
                            'supported_events': supported, 'excluded_events': skipped,
                            'auroc': math.fsum(events[e]['auroc'] for e in supported) / len(supported) if supported else None,
                            'ap': math.fsum(events[e]['ap'] for e in supported) / len(supported) if supported else None,
                            'mean_event_positive_prevalence': math.fsum(events[e]['positive_prevalence'] for e in supported) / len(supported) if supported else None}}


def paired_event_bootstrap(pair, later):
    supported = pair['event_macro']['supported_events']
    require(supported == later['event_macro']['supported_events'], 'Paired bootstrap event support differs')
    require(supported, 'No supported event for AUROC contrast')
    deltas = {e: pair['events'][e]['auroc'] - later['events'][e]['auroc'] for e in supported}
    values = np.asarray(list(deltas.values()), dtype=np.float64)
    draws = np.random.default_rng(20260925).integers(0, len(values), size=(5000, len(values)))
    boot = values[draws].mean(axis=1)
    return {'metric': 'AUROC', 'contrast': 'pair_minus_later', 'aggregation': 'equal_event_macro',
            'comparison': 'pos_vs_hard_neg', 'n_events': len(supported), 'event_deltas': deltas,
            'delta': math.fsum(deltas.values()) / len(deltas),
            'percentile_ci95': np.quantile(boot, [0.025, 0.975], method='linear').tolist(),
            'bootstrap_replicates': 5000, 'rng': 'numpy.default_rng', 'seed': 20260925,
            'unit': 'paired event cluster; both arm AUROCs travel together',
            'excluded_events': pair['event_macro']['excluded_events'],
            'limits': 'Exploratory interval conditional on events with both classes. It does not include training/model-selection uncertainty or supply unseen-event confirmation.'}


def diagnose(artifact_dir, audit_path):
    artifact_dir, audit_path = Path(artifact_dir).resolve(), Path(audit_path).resolve()
    previous = load(audit_path)
    require(previous['consistent'] is True and previous['run_valid'] is True and previous['all_six_converged'] is True,
            'Requires a consistent valid independent C0 audit with all six models converged')
    expected = {'items.jsonl': previous['hashes_verified']['items.jsonl'],
                'predictions.jsonl': previous['hashes_verified']['predictions.jsonl'],
                'results.json': previous['input_results_sha256']}
    for filename, h in expected.items():
        require(sha(artifact_dir / filename) == h, 'Input changed since independent audit: ' + filename)
    source_items = [r for r in lines(artifact_dir / 'items.jsonl') if r['partition'] == 'test']
    require(len(source_items) == 1755 and len({r['id'] for r in source_items}) == 1755, 'Original test coverage changed')
    items = {r['id']: r for r in source_items}
    predictions = lines(artifact_dir / 'predictions.jsonl')
    keys = [(r['id'], r['arm']) for r in predictions]
    require(len(predictions) == 5265 and len(set(keys)) == 5265
            and set(keys) == {(i, a) for i in items for a in ARMS}, 'Prediction coverage/duplicates')
    for row in predictions:
        it = items[row['id']]
        require(all(row[k] == it[k] for k in ('tile', 'phen', 'kind', 'cluster')) and row['source_gold'] == it['answer'], 'Source metadata/label changed')
        require(math.isfinite(row['logit']) and row['prediction'] == ('yes' if row['logit'] >= 0 else 'no'), 'Frozen logit/sign invalid')
    flood = [r for r in predictions if r['phen'] == 'flood']
    events = sorted({r['cluster'] for r in flood})
    require(len(events) == 10, 'Expected ten flood events')
    output = {'schema': 'c0-flood-ranking-posthoc-v1', 'status': 'post_hoc_exploratory',
              'question': 'Does C0 pair-vs-later behavior reflect altered ranking as well as the fixed operating point?',
              'scope': 'Frozen C0 flood/S1 outputs only. Source-label discrimination, not independently verified change semantics.',
              'provenance': {'script_sha256': sha(__file__), 'artifact_dir': str(artifact_dir),
                             'independent_audit_path': str(audit_path), 'independent_audit_sha256': sha(audit_path),
                             'input_sha256': expected, 'numpy': np.__version__},
              'definitions': {'positive': 'kind=pos, source answer yes',
                              'pos_vs_hard_neg': 'post-pair positive rows vs post-pair dry hard_neg rows',
                              'pos_vs_prepre_neg': 'post-pair positive rows vs same-tile pre/pre kind=neg rows',
                              'pos_vs_all_no': 'positives vs both source-negative kinds',
                              'score': 'Original raw positive-class logit; no fitting, thresholds, probability transforms, rounding or binning.',
                              'auroc': 'P(score_positive>score_negative) + 0.5*P(tie), computed exactly from score groups.',
                              'ap': 'Non-interpolated average precision: sum over descending unique-score groups of recall increase times precision after the entire tie group.',
                              'event_macro': 'Arithmetic mean of per-event metrics, equal event weights. Missing-class events are null and excluded with explicit support.',
                              'pooled': 'All selected items pooled, including cross-event comparisons and items from events lacking one within-event class. pooled_supported_events_only additionally restricts to the event-macro support set.',
                              'ap_reference': 'Positive prevalence is reported as an imbalance reference, not a calibration score or exact finite-sample random-ranking expectation.',
                              'threshold': 'Original logit>=0 (sigmoid>=0.5) retained. No threshold sweep or recommendation.'},
              'metrics': {}, 'fixed_operating_point': {},
              'limitations': ['Requested after inspecting C0 classification results; exploratory, not preregistered confirmatory evidence.',
                             'No model, features, labels, splits, fitting settings or original decision threshold changed.',
                             'AUROC/AP are invariant under strictly increasing score transformations and cannot establish probability calibration.',
                             'Different AP values across comparisons/events can reflect class prevalence as well as ranking.',
                             'Few events and small per-event class supports limit interval stability; 10 events are already exposed development data.',
                             'Positive versus pre/pre negatives inherits assumed pre-event labels and imputed temporal-slot semantics.',
                             'Pooled AUROC compares examples across events; equal-event macro evaluates within-event ranking and answers a different weighting question.',
                             'For pos_vs_hard_neg, full pooled includes five positives from two events with no hard negatives; macro excludes these unsupported events. A support-aligned pooled result is also retained.']}
    for arm in ARMS:
        rows = [r for r in flood if r['arm'] == arm]
        require(collections.Counter(r['kind'] for r in rows) == {'pos': 457, 'neg': 457, 'hard_neg': 457}, 'Unexpected flood kind support')
        output['metrics'][arm] = {name: summarize(rows, kinds, events) for name, kinds in COMPARISONS.items()}
        output['fixed_operating_point'][arm] = {kind: {'yes': sum(r['prediction'] == 'yes' for r in rows if r['kind'] == kind), 'n': 457}
                                                for kind in ('pos', 'neg', 'hard_neg')}
    output['pos_vs_hard_neg_auroc_pair_minus_later'] = paired_event_bootstrap(
        output['metrics']['pair']['pos_vs_hard_neg'], output['metrics']['later']['pos_vs_hard_neg'])
    for filename, h in expected.items():
        require(sha(artifact_dir / filename) == h, 'Input changed during diagnostic: ' + filename)
    return output


def markdown(result):
    out = ['# C0 순위 진단 — 사후 탐색 분석', '',
           '고정된 C0 flood/S1 점수만 분석했다. 모델 재학습, 임계값 변경·선택은 없다. 원래 분류 기준은 logit ≥ 0(명목 sigmoid ≥ .5)이다.', '',
           '| 비교 | 입력 | pooled AUROC | event-macro AUROC | pooled AP | event-macro AP | 지원 사건 |',
           '|---|---|---:|---:|---:|---:|---:|']
    names = {'pos_vs_hard_neg': '양성 vs 사후 dry hard-negative', 'pos_vs_prepre_neg': '양성 vs 사전/사전 음성', 'pos_vs_all_no': '양성 vs 전체 음성'}
    for comparison in COMPARISONS:
        for arm in ARMS:
            item = result['metrics'][arm][comparison]; p, m = item['pooled'], item['event_macro']
            out.append(f"| {names[comparison]} | {arm} | {p['auroc']:.4f} | {m['auroc']:.4f} | {p['ap']:.4f} | {m['ap']:.4f} | {m['n_supported_events']}/{m['n_total_events']} |")
    contrast = result['pos_vs_hard_neg_auroc_pair_minus_later']; lo, hi = contrast['percentile_ci95']
    out.extend(['', f"양성 대 hard-negative의 사건 평균 AUROC 차이(pair − later)는 **{contrast['delta']:+.4f}**, 사건을 묶어 재표집한 5,000회 percentile 95% 구간은 **[{lo:+.4f}, {hi:+.4f}]**이다.",
                '이는 두 종류의 행이 모두 있는 사건에만 조건을 둔 사후 구간이며, 확인적 유의성 판단이나 미관측 사건 일반화 근거로 사용하지 않는다.', ''])
    if contrast['excluded_events']:
        out.append('지원이 없는 사건: ' + '; '.join(f"{x['event']} (pos={x['n_pos']}, hard_neg={x['n_neg']}; {x['reason']})" for x in contrast['excluded_events']) + '. 해당 지표는 NA이며 평균과 bootstrap에서 제외했다.')
        out.append('전체 pooled의 분모는 양성 457개·음성 457개이며, 해당 사건의 양성 5개도 포함한다. 사건 평균은 지원 사건의 양성 452개·음성 457개로 계산된다. JSON의 `pooled_supported_events_only`는 사건 평균과 같은 지원 집합으로 한정한 추가 결과다.')
    out.extend(['', '| 원래 고정 기준의 yes 수 | earlier | later | pair |', '|---|---:|---:|---:|'])
    for kind in ('pos', 'neg', 'hard_neg'):
        out.append('| ' + kind + ' | ' + ' | '.join(f"{result['fixed_operating_point'][a][kind]['yes']}/457" for a in ARMS) + ' |')
    late, pair = result['metrics']['later'], result['metrics']['pair']
    out.extend(['',
                f"현재 결과의 핵심: 양성 대 사전/사전 음성에서는 later→pair AUROC가 pooled {late['pos_vs_prepre_neg']['pooled']['auroc']:.4f}→{pair['pos_vs_prepre_neg']['pooled']['auroc']:.4f}, 사건 평균 {late['pos_vs_prepre_neg']['event_macro']['auroc']:.4f}→{pair['pos_vs_prepre_neg']['event_macro']['auroc']:.4f}로 낮아졌다. **이 비교의 성능 저하는 임계값 이동만의 문제로 설명되지 않는다.**",
                f"양성 대 dry hard-negative에서는 사건 평균 순위는 {late['pos_vs_hard_neg']['event_macro']['auroc']:.4f}→{pair['pos_vs_hard_neg']['event_macro']['auroc']:.4f}로 유지·소폭 향상되지만, 전체 pooled 순위는 {late['pos_vs_hard_neg']['pooled']['auroc']:.4f}→{pair['pos_vs_hard_neg']['pooled']['auroc']:.4f}로 낮아졌다. 같은 지원 사건만 pooled해도 later {late['pos_vs_hard_neg']['pooled_supported_events_only']['auroc']:.4f}, pair {pair['pos_vs_hard_neg']['pooled_supported_events_only']['auroc']:.4f}이다. 사건 내부 구별과 사건 사이 점수 비교·가중치의 영향을 나누어 봐야 하며, 이것만으로 원인이나 확률 보정 실패를 특정할 수 없다.",
                '두 비교에서 다른 결과가 나온 사후 진단이며, 새로운 threshold 선택이나 학습 처방으로 이어지는 검증 결과가 아니다.', '', '해석 경계:', '',
                '- AUROC/AP는 점수의 순위만 본다. 값의 단조 변환이나 임계값 이동으로는 순위 성능이 바뀌지 않는다.',
                '- 고정 기준의 recall/FPR 변화와 순위 지표를 함께 읽어야 한다. 순위 향상이 확인되어도 확률 보정(calibration)이 나쁘다는 결론까지는 나오지 않는다.',
                '- pooled는 사건 사이의 점수도 비교하고, event-macro는 사건 내부 순위에 같은 가중치를 준다. 두 값을 교환해 인용하면 안 된다.',
                '- AP는 양성 비율에 영향을 받는다. JSON에 각 비교·사건의 양성 비율, 클래스 수, tie 수를 함께 남겼다.',
                '- 이 분석은 C0 결과를 본 뒤 요청한 사후 진단이다. 이후 모델·threshold 튜닝이나 새로운 판정 관문을 정당화하는 확인 실험으로 취급하지 않는다.', '',
                f"입력 predictions SHA256: `{result['provenance']['input_sha256']['predictions.jsonl']}`", ''])
    return '\n'.join(out)


class RankingTests(unittest.TestCase):
    def test_hand_computed_ties_and_rankings(self):
        cases = [([1, 1, 0, 0], [4, 3, 2, 1], 1., 1.),
                 ([0, 0, 1, 1], [4, 3, 2, 1], 0., 5/12),
                 ([1, 0, 1, 0], [1, 1, 1, 1], .5, .5),
                 ([1, 0, 1, 0], [2, 2, 1, 0], 5/8, 7/12),
                 ([1, 0, 1, 0], [3, 2, 1, 1], 5/8, 3/4),
                 ([1, 0, 1, 0], [4, 3, 2, 1], 3/4, 5/6)]
        for labels, scores, auc, ap in cases:
            with self.subTest(labels=labels, scores=scores):
                got = ranking_metrics(labels, scores)
                self.assertAlmostEqual(got['auroc'], auc, places=14)
                self.assertAlmostEqual(got['ap'], ap, places=14)

    def test_missing_class_policy(self):
        for labels in ([1, 1], [0, 0], []):
            got = ranking_metrics(labels, np.arange(len(labels)))
            self.assertFalse(got['estimable'])
            self.assertIsNone(got['auroc']); self.assertIsNone(got['ap'])

    def test_tie_permutation_and_monotone_invariance(self):
        y, z = np.array([1, 0, 1, 0]), np.array([2., 2., 1., 0.])
        base = ranking_metrics(y, z)
        for ys, zs in [(y[[1, 0, 2, 3]], z), (y, z * 3 + 17)]:
            got = ranking_metrics(ys, zs)
            self.assertEqual(got['auroc'], base['auroc']); self.assertEqual(got['ap'], base['ap'])

    def test_auc_agrees_with_bruteforce_pair_credit(self):
        rng = np.random.default_rng(9)
        for _ in range(100):
            y = np.r_[np.ones(5), np.zeros(7)]; z = rng.integers(-2, 3, 12)
            comparisons = z[y == 1, None] - z[y == 0][None, :]
            expected = np.mean((comparisons > 0) + .5 * (comparisons == 0))
            self.assertAlmostEqual(ranking_metrics(y, z)['auroc'], expected, places=14)

    def test_nonfinite_and_bad_shapes_rejected(self):
        for y, z in [([1, 0], [np.nan, 1]), ([1, 0], [1]), ([2, 0], [1, 0])]:
            with self.assertRaises(ValueError): ranking_metrics(y, z)

    def test_event_support_exclusion_and_aligned_pool(self):
        rows = [{'cluster': 'a', 'kind': 'pos', 'logit': 2.},
                {'cluster': 'a', 'kind': 'hard_neg', 'logit': 1.},
                {'cluster': 'b', 'kind': 'pos', 'logit': -1.}]
        got = summarize(rows, ('hard_neg',), ['a', 'b'])
        self.assertEqual(got['pooled']['n_pos'], 2)
        self.assertEqual(got['pooled_supported_events_only']['n_pos'], 1)
        self.assertEqual(got['event_macro']['n_supported_pos'], 1)
        self.assertEqual(got['event_macro']['auroc'], 1.)
        self.assertIsNone(got['events']['b']['auroc'])
        self.assertEqual(got['event_macro']['excluded_events'][0]['event'], 'b')

    def test_bootstrap_keeps_supported_events_paired(self):
        pair = {'event_macro': {'supported_events': ['a', 'b'], 'excluded_events': []},
                'events': {'a': {'auroc': 1.}, 'b': {'auroc': 0.}}}
        later = {'event_macro': {'supported_events': ['a', 'b'], 'excluded_events': []},
                 'events': {'a': {'auroc': .5}, 'b': {'auroc': .5}}}
        got = paired_event_bootstrap(pair, later)
        self.assertEqual(got['delta'], 0.)
        self.assertEqual(got['percentile_ci95'], [-.5, .5])
        later['event_macro']['supported_events'] = ['a']
        with self.assertRaises(ValueError): paired_event_bootstrap(pair, later)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, default=REPO / 'artifacts/c0_linear_view_probe_v1_20260925')
    parser.add_argument('--audit', type=Path, default=REPO / 'artifacts/c0_independent_audit_20260925.json')
    parser.add_argument('--out-prefix', type=Path, default=Path('/private/tmp/c0_ranking_diagnostic_20260925'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        success = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RankingTests)).wasSuccessful()
        return 0 if success else 1
    paths = [args.out_prefix.with_suffix('.json'), args.out_prefix.with_suffix('.md')]
    require(all(not p.resolve().is_relative_to(args.artifact_dir.resolve()) for p in paths), 'Output must be outside input artifacts')
    result = diagnose(args.artifact_dir, args.audit)
    paths[0].write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    paths[1].write_text(markdown(result))
    print(json.dumps({'outputs': [str(p) for p in paths], 'contrast': result['pos_vs_hard_neg_auroc_pair_minus_later']}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
