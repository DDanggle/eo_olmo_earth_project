#!/usr/bin/env python3
"""Small exact-arithmetic audit of the executed v4 connector, not an EO eval.

No model, torch, GPU, network, or imagery loading. Synthetic scalar features are
intervened on AFTER the encoder; these are not real-image counterfactuals.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--provider', type=Path, required=True)
    p.add_argument('--worker', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    r = json.loads(a.receipt.read_text())
    assert sha(a.provider) == r['eo_source']['provider_file_sha256']
    assert sha(a.worker) == r['script_sha256']
    assert 'patch_size=4' in a.provider.read_text()
    assert 'flattened Hpatch,Wpatch,T,bandset(1)' in r['eo_source']['metadata']['token_layout']
    assert 'F.adaptive_avg_pool1d' in a.worker.read_text()
    row = r['eo_source']['metadata']['row']
    h, w = (v // 4 for v in row['crop_yxhw'][2:])
    t = len(row['compact_timesteps'])
    n, channels = r['eo_shape'][1:]
    k = r['adapter']['slots']
    assert (h, w, t, n, k) == (8, 8, 2, 128, 16), 'Only the recorded smoke geometry is audited'
    coords = [(y, x, date) for y in range(h) for x in range(w) for date in range(t)]
    width = n // k
    assert n % k == 0
    current = [list(range(i * width, (i + 1) * width)) for i in range(k)]
    # Same 16 slots: 4 spatial rows x 2 spatial columns x 2 dates. This
    # deliberately trades vertical resolution for preserving separate dates.
    temporal = [[i for i, (y, x, date) in enumerate(coords)
                 if y // 2 == yy and x // 4 == xx and date == tt]
                for yy in range(4) for xx in range(2) for tt in range(t)]
    assert len(temporal) == k and all(len(g) == width for g in temporal)
    assert sorted(i for g in temporal for i in g) == list(range(n))

    def vector(coord):
        return [Fraction(int(c == coord)) for c in coords]

    def pool(v, groups):
        return [sum((v[i] for i in group), Fraction(0)) / len(group) for group in groups]

    cases = []
    for label, before, after in [
        ('swap_date_at_same_patch', (0, 0, 0), (0, 0, 1)),
        ('move_one_patch_horizontally', (0, 0, 0), (0, 1, 0)),
        ('move_one_patch_vertically', (0, 0, 0), (1, 0, 0)),
    ]:
        v0, v1 = vector(before), vector(after)
        result = {'synthetic_post_encoder_operation': label,
                  'before_y_x_t': before, 'after_y_x_t': after,
                  'full_grid_differs': v0 != v1}
        for name, groups in [('executed_flat_pool', current), ('fixed_budget_separate_dates', temporal)]:
            x, y = pool(v0, groups), pool(v1, groups)
            result[name] = {'outputs_identical': x == y,
                            'max_abs_delta': float(max(abs(u - v) for u, v in zip(x, y)))}
        cases.append(result)
    report = {
        'status': 'completed_operator_analysis_not_model_performance',
        'receipt_sha256': sha(a.receipt), 'provider_sha256': sha(a.provider),
        'worker_sha256': sha(a.worker), 'audit_script_sha256': sha(Path(__file__)),
        'native_grid_h_w_t_bandset_channels': [h, w, t, 1, channels],
        'output_slots': k,
        'per_executed_bin': {'input_tokens': width, 'spatial_patches': 4, 'distinct_dates': 2},
        'pool_operator_rank_per_feature_channel': k,
        'pool_operator_nullity_per_feature_channel': n - k,
        'rank_interpretation': 'Disjoint nonempty averaging rows: rank16 for arbitrary fixed encoder features. Not an 87.5-percent semantic-information-loss claim.',
        'executed_bins': [{'slot': j, 'coordinates_y_x_t': [coords[i] for i in group]} for j, group in enumerate(current)],
        'synthetic_cases': cases,
        'limits': ['Synthetic post-encoder vectors only; no real-image accuracy or encoder deficiency measured.',
                   'Encoder tokens are contextual and may redundantly encode time/space elsewhere.',
                   'Date-preserving 16-slot pooling loses vertical distinctions in this example; no free information-preservation claim.',
                   'A dense branch retains more features but costs memory/compute and must be shared by fair baselines.',
                   'No evidence that this connection design or its repair is novel.'],
        'next_empirical_test': 'Compare full-grid, flat, spatiotemporal-grid, and learned resampler on the same labeled observations; match slots, targets, tuning and compute accounting.',
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({key: report[key] for key in ['status', 'native_grid_h_w_t_bandset_channels', 'per_executed_bin', 'synthetic_cases']}))


if __name__ == '__main__':
    main()
