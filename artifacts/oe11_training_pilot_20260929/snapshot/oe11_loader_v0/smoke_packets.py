"""Bounded real-packet CPU loader smoke; no model, GPU, or implicit query gold."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from loader_adapter import make_loader_class

ACQUISITIONS = {2: [2, 5], 4: [0, 2, 5, 7], 8: list(range(8))}


def run(prepared_root, catalog_roots, base_loader, *, cases_per_cell=1,
        check_targets=False):
    if not 1 <= cases_per_cell <= 3:
        raise ValueError('Bounded smoke accepts one to three cases per mode/split/K cell')
    if not catalog_roots or set(catalog_roots) - {'pooled', 'cross_parent'}:
        raise ValueError('Expected explicit pooled and/or cross_parent catalog roots')
    Loader = make_loader_class(base_loader)
    manifest = Path(prepared_root) / 'manifest.jsonl'
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    report = dict(status='running', GPU_used=False, model_executed=False,
                  query_gold_access_requested=check_targets, cases=[], catalogs=[],
                  missing_cells=[], model_query_dates_supported=2,
                  limitation='CPU packet contracts only; 4/8-query model integration unimplemented; '
                             'no cloud/parcel certification or unseen-region evaluation')
    for mode, episodes_root in sorted(catalog_roots.items()):
        for split in ('train', 'development'):
            loader = Loader(prepared_root, episodes_root, split, support_mode=mode)
            for k in (1, 8):
                candidates = sorted((e for e in loader.episodes.values() if e['k_pairs'] == k),
                                    key=lambda e: (e['query_parent_tile'], e['query_patch_id'], e['episode_id']))
                # Prefer distinct query parents; fill from other unique queries.
                chosen, parents, queries = [], set(), set()
                for prefer_new_parent in (True, False):
                    for e in candidates:
                        if e['query_patch_id'] in queries:
                            continue
                        if prefer_new_parent and e['query_parent_tile'] in parents:
                            continue
                        chosen.append(e); queries.add(e['query_patch_id']); parents.add(e['query_parent_tile'])
                        if len(chosen) == cases_per_cell:
                            break
                    if len(chosen) == cases_per_cell:
                        break
                if not chosen:
                    report['missing_cells'].append(dict(mode=mode, split=split, k=k))
                    continue
                for episode in chosen:
                    for n, positions in ACQUISITIONS.items():
                        item = loader.load(episode, acquired_positions=positions)
                        model = item['model_input']; q = model['query']
                        if set(model) != {'prompt', 'query', 'support_pairs'}:
                            raise ValueError('Unexpected model input field')
                        if q['s2'].shape != (128, 128, n, 12) or q['s2'].base is not None:
                            raise ValueError('Query acquisition shape/copy mismatch')
                        if 'mask' in q or item['audit']['query_gold_read'] is not False:
                            raise ValueError('Query gold in inference load')
                        if any(s['s2'].shape != (128, 128, 8, 12)
                               for pair in model['support_pairs'] for s in pair.values()):
                            raise ValueError('Support acquisition mismatch')
                        report['cases'].append(dict(mode=mode, split=split,
                            episode_id=episode['episode_id'], query_parent=episode['query_parent_tile'],
                            k=k, query_dates=n, acquired_positions=item['audit']['acquired_query_positions'],
                            returned_support_observation_instances=item['audit']['returned_support_observation_instances'],
                            query_gold_read=False))
                    if check_targets:
                        target = loader.load_target(episode, purpose='training' if split == 'train' else 'evaluation',
                                                    training=split == 'train')
                        if target['target_mask'].shape != (128, 128):
                            raise ValueError('Target shape mismatch')
            report['catalogs'].append(dict(mode=mode, split=split, **loader.adapter_audit()))
    after = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if before != after:
        raise ValueError('Prepared manifest changed during read-only smoke')
    report.update(status='passed' if not report['missing_cells'] else 'passed_available_cells_with_shortages',
                  loads_checked=len(report['cases']), manifest_sha256_before=before,
                  manifest_sha256_after=after, manifest_unchanged=True,
                  actual_query_parents=dict(Counter(x['query_parent'] for x in report['cases'])))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', required=True)
    parser.add_argument('--base-loader', required=True)
    parser.add_argument('--catalog', action='append', required=True, metavar='MODE=PATH')
    parser.add_argument('--cases-per-cell', type=int, default=1)
    parser.add_argument('--check-targets', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    catalogs = {}
    for item in args.catalog:
        mode, path = item.split('=', 1)
        if mode in catalogs: raise ValueError('Duplicate catalog mode')
        catalogs[mode] = path
    report = run(args.prepared_root, catalogs, args.base_loader,
                 cases_per_cell=args.cases_per_cell, check_targets=args.check_targets)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'loads_checked', 'missing_cells', 'manifest_unchanged')}))


if __name__ == '__main__': main()
