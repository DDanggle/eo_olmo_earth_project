#!/usr/bin/env python3
"""Read-only catalog cross-check and small review export. No model or query gold."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def need(ok, message):
    if not ok:
        raise ValueError(message)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(exist_ok=False, parents=True)
    manifest = {r['patch_id']: r for r in rows(a.root/'prepared_v0/manifest.jsonl')}
    identities = {}
    report = {'status': 'running', 'modes': {}, 'GPU_used': False,
              'query_gold_or_model_results_read': False, 'source': str(a.root),
              'script_sha256': sha(Path(__file__))}
    for mode in ('pooled', 'cross_parent'):
        src = a.root / ('catalog_' + mode + '_v0')
        dst = a.out / mode
        dst.mkdir()
        contract = json.loads((src/'episode_contract.json').read_text())
        need(contract['train_support_parent_policy'] == mode, 'parent policy')
        need(contract['prepared_manifest_sha256'] == sha(a.root/'prepared_v0/manifest.jsonl'), 'manifest sha')
        for name, digest in contract['frozen_artifacts_sha256'].items():
            need(sha(src/name) == digest, 'frozen artifact sha: ' + name)
        for name in ('episode_contract.json', 'coverage.json', 'pair_catalog.json',
                     'roles.jsonl', 'source_objects.jsonl', 'public_catalog_hashes_before_scoring.json',
                     'frozen_artifacts_before_scoring.json'):
            shutil.copy2(src/name, dst/name)
        mode_report = {}
        identities[mode] = {}
        for split, expected in (('train', 9216), ('development', 2304)):
            catalog = rows(src/f'episodes_{split}.jsonl')
            need(len(catalog) == expected, 'full pair/K count')
            need(sha(src/f'episodes_{split}.jsonl') == contract['public_catalog_sha256'][split], 'public sha')
            need(len({e['episode_id'] for e in catalog}) == expected, 'unique episode ids')
            scope = Counter()
            masks = {}
            for e in catalog:
                q = manifest[e['query_patch_id']]
                need(q['training_partition'] == ('train_pool' if split == 'train' else 'calibration'), 'query role')
                need(e['k_pairs'] == len(e['support_pairs']), 'K count')
                scope[(e['query_patch_id'], e['pair_id'], e['k_pairs'])] += 1
                for pair in e['support_pairs']:
                    for s in pair.values():
                        r = manifest[s['patch_id']]
                        need(s['patch_id'] != e['query_patch_id'], 'self support')
                        need(r['training_partition'] == ('train_pool' if split == 'train' else 'source_bank'), 'support role')
                        if mode == 'cross_parent' and split == 'train':
                            need(q['parent_tile'] != r['parent_tile'], 'cross parent violation')
                        path = (src/s['mask_npz']).resolve()
                        need(path.is_relative_to((src/'support_masks').resolve()), 'mask path')
                        need(s['mask_npz'] not in masks or masks[s['mask_npz']] == s['mask_sha256'], 'inconsistent mask hash')
                        masks[s['mask_npz']] = s['mask_sha256']
            need(all(v == 1 for v in scope.values()), 'duplicate query pair K')
            for name, digest in masks.items():
                need(sha(src/name) == digest, 'support mask sha')
            identities[mode][split] = sorted((e['episode_id'], e['query_patch_id'], e['pair_id'], e['k_pairs']) for e in catalog)
            mode_report[split] = {'episodes': len(catalog), 'query_patches': len({e['query_patch_id'] for e in catalog}),
                'unique_directed_pairs': len({e['pair_id'] for e in catalog}),
                'ks': sorted({e['k_pairs'] for e in catalog}), 'verified_unique_support_masks': len(masks),
                'public_catalog_sha256': contract['public_catalog_sha256'][split]}
            (dst/f'episodes_{split}_sample.jsonl').write_text(''.join(json.dumps(e,sort_keys=True)+'\n' for e in catalog[:4]))
        report['modes'][mode] = mode_report
    need(identities['pooled'] == identities['cross_parent'], 'comparison query/pair/K identity')
    need(report['modes']['pooled']['development']['public_catalog_sha256'] ==
         report['modes']['cross_parent']['development']['public_catalog_sha256'], 'shared development catalog')
    smoke = json.loads((a.root/'loader_smoke_v0.json').read_text())
    need(smoke['status'] == 'passed' and smoke['loads_checked'] == 72 and smoke['manifest_unchanged'], 'real loader smoke')
    shutil.copy2(a.root/'loader_smoke_v0.json',a.out/'loader_smoke_v0.json')
    report.update(status='PASS', matched_query_pair_K_and_episode_ids=True,
                  development_catalog_identical=True, real_packet_loads_checked=72,
                  limitations=['Catalog combinations are not independent images or human corrections.',
                               'Loader smoke is sampled; all public catalog rows and referenced mask hashes are checked.',
                               'No novel-region performance, actual model execution or exact parcel geometry certified.'])
    (a.out/'catalog_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    files = {str(f.relative_to(a.out)): {'sha256':sha(f),'bytes':f.stat().st_size}
             for f in sorted(a.out.rglob('*')) if f.is_file()}
    (a.out/'export_manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
    print(json.dumps({'status':report['status'],'files':len(files),'bytes':sum(v['bytes'] for v in files.values()),
                      'modes':report['modes']}))

if __name__ == '__main__':
    main()
