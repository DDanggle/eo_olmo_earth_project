"""Count repeated support encodings inside a training forward, from metadata.

No raw observations, labels, model inference or held-out payload are opened.
Counts are engineering opportunities, NOT measured speedups or score gains.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    catalogs = {
        'original_train': 'artifacts/oe8_pastis_prepare_20260927/review_bundle_v0/episodes/episodes_train.jsonl',
        'expansion_train': 'artifacts/oe10_expansion_catalog_20260928/oe10_expansion_catalog_v0/runtime_v2/episodes_train.jsonl',
    }
    result = {'schema': 'oe10_support_compute_metadata_v1', 'catalogs': {},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limits': ['Conservative same-input-file/hash/date comparison, not array equality.',
                         'Only within-forward support reuse; never cache detached trainable features across updates.',
                         'No inference, FLOPs, memory or wall-time measurement.',
                         'Native replay, backward recomputation and VLM costs are excluded.',
                         'Changing query from two to eight dates is a new observation condition, not the old P2 task.']}
    for name, relative in catalogs.items():
        p = args.root / relative
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        per_k = defaultdict(list)
        input_contracts = {}
        for row in rows:
            if row['split'] != 'train':
                raise ValueError('Only train catalog is allowed')
            supports = [s for pair in row['support_pairs'] for s in [pair['positive'], pair['counterexample']]]
            if len(supports) != 2 * row['k_pairs']:
                raise ValueError('Pair count mismatch')
            keys = []
            for s in supports:
                key = (s['input_npz'], s['input_sha256'], tuple(s['dates_yyyymmdd']), tuple(s['observation_ids']))
                if len(s['dates_yyyymmdd']) != 8:
                    raise ValueError('Expected eight support dates')
                contract = (s['input_sha256'], tuple(s['dates_yyyymmdd']), tuple(s['observation_ids']))
                if s['input_npz'] in input_contracts and input_contracts[s['input_npz']] != contract:
                    raise ValueError('Conflicting input identity for same support file')
                input_contracts[s['input_npz']] = contract
                keys.append(key)
            per_k[row['k_pairs']].append((len(keys), len(set(keys))))
        report = {}
        for k, values in sorted(per_k.items()):
            n = len(values)
            before = sum(a for a, b in values)
            after = sum(b for a, b in values)
            report[str(k)] = {
                'episodes': n, 'support_calls_before': before, 'support_calls_if_shared': after,
                'support_call_reduction_fraction': (before-after)/before,
                'mean_unique_support_patches_per_episode': after/n,
                'unique_support_patch_count_histogram': dict(sorted(Counter(b for a,b in values).items())),
                'mean_patch_date_instances_two_query_current': (2*n+8*before)/n,
                'mean_patch_date_instances_two_query_shared': (2*n+8*after)/n,
                'mean_patch_date_instances_eight_query_shared': (8*n+8*after)/n,
            }
        result['catalogs'][name] = {'path': relative, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                  'episodes': len(rows), 'by_k': report}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {k: {'support_calls_removed_fraction': v['support_call_reduction_fraction'],
                                     'before_two_dates': v['mean_patch_date_instances_two_query_current'],
                                     'shared_eight_dates': v['mean_patch_date_instances_eight_query_shared']}
                              for k,v in data['by_k'].items()}
                      for name,data in result['catalogs'].items()}, indent=2))


if __name__ == '__main__':
    main()
