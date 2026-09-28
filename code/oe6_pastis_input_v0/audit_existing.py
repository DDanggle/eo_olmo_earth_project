#!/usr/bin/env python3
"""Read-only audit of existing PASTIS shards; extract <=2 train records from verified shards.

Never downloads, repairs, modifies a source dataset, or launches a GPU job.
Partial records are inspection assets, not certified training inputs.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, default=str) + '\n')


def literals(source):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'GeoBenchPASTIS')
    result = {}
    for node in cls.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ['paths', 'sha256str', 'band_default_order', 'url']:
                    result[target.id] = ast.literal_eval(node.value)
    return result


def read_h5_payload(reference, allowed):
    import h5py
    import numpy as np
    m = re.fullmatch(r'.*?(\d+)_(\d+),(.+)', str(reference))
    if not m:
        raise ValueError('Unexpected packed H5 reference: ' + str(reference))
    offset, size = int(m[1]), int(m[2])
    path = Path(m[3]).resolve()
    if path not in allowed or size > (128 << 20) or offset + size > path.stat().st_size:
        raise ValueError('Payload outside verified shard/bounded size')
    with path.open('rb') as f:
        f.seek(offset)
        payload = f.read(size)
    if len(payload) != size:
        raise ValueError('Short payload read')
    with h5py.File(io.BytesIO(payload), 'r') as f:
        arrays = {k: np.asarray(f[k]) for k in f if isinstance(f[k], h5py.Dataset)}
        attrs = dict(f.attrs)
    return arrays, {'reference': str(reference), 'payload_sha256': hashlib.sha256(payload).hexdigest(), 'attrs': attrs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--loader', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    source = a.loader.read_text()
    config = literals(source)
    shutil.copyfile(a.loader, a.out/'installed_pastis.py')
    shutil.copyfile(a.loader.parent/'base.py', a.out/'installed_base.py')
    result = {'started_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(__file__),
              'loader_path': str(a.loader), 'loader_sha256': sha(a.loader), 'expected': config,
              'source_shards_unchanged': None, 'shards': [], 'prepared_native_training_inputs': 0,
              'status': 'hashing_existing_shards', 'gpu_used': False, 'downloaded': False}
    save(a.out/'audit.json', result)
    verified = []
    for name, wanted in zip(config['paths'], config['sha256str']):
        path = a.root/name
        row = {'name': name, 'expected_sha256': wanted, 'exists': path.is_file()}
        if path.is_file():
            before = path.stat()
            got = sha(path)
            after = path.stat()
            row.update(bytes=before.st_size, mtime_ns=before.st_mtime_ns, actual_sha256=got,
                       valid=got == wanted, stable_during_hash=(before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns))
            if row['valid'] and row['stable_during_hash']:
                verified.append(path.resolve())
        else:
            row['valid'] = False
        result['shards'].append(row)
        save(a.out/'audit.json', result)
        print(json.dumps(row), flush=True)
    result['complete_dataset_verified'] = all(s['valid'] for s in result['shards'])
    try:
        import numpy as np
        import tacoreader
        df = tacoreader.load([str(p) for p in verified])
        result['partial_catalog'] = {'rows': len(df), 'columns': list(df.columns),
                                     'split_counts': df['tortilla:data_split'].value_counts().to_dict()}
        train = df[df['tortilla:data_split'] == 'train'].reset_index(drop=True)
        result['inspected_train_records'] = []
        for idx in range(min(2, len(train))):
            rec = train.read(idx)
            record = {'partial_train_row': idx, 'catalog_record': train.iloc[idx].to_dict(),
                      'nested_columns': list(rec.columns), 'nested_metadata': rec.to_dict(orient='records'), 'payloads': []}
            bundle = {}
            for j, name in [(0,'s2'),(3,'semantic'),(4,'instance')]:
                arrays, provenance = read_h5_payload(rec.read(j), set(verified))
                provenance['name'] = name
                provenance['datasets'] = {}
                for key, arr in arrays.items():
                    provenance['datasets'][key] = {'shape': list(arr.shape), 'dtype': str(arr.dtype),
                        'finite': bool(np.isfinite(arr).all()), 'min': float(arr.min()), 'max': float(arr.max())}
                    if name in ['semantic','instance']:
                        vals, counts = np.unique(arr, return_counts=True)
                        provenance['datasets'][key]['value_counts'] = dict(zip(vals.tolist(), counts.tolist()))
                    bundle[name+'_'+key] = arr
                record['payloads'].append(provenance)
            np.savez_compressed(a.out/f'inspection_raw_{idx}.npz', **bundle)
            save(a.out/f'inspection_record_{idx}.json', record)
            result['inspected_train_records'].append({'record': f'inspection_record_{idx}.json', 'raw': f'inspection_raw_{idx}.npz'})
    except Exception:
        result['partial_read_error'] = traceback.format_exc()
    result['source_shards_unchanged'] = all(not s['exists'] or
        ((a.root/s['name']).stat().st_size == s['bytes'] and (a.root/s['name']).stat().st_mtime_ns == s['mtime_ns'])
        for s in result['shards'])
    result['status'] = 'completed_audit_native_input_contract_not_yet_passed'
    result['wall_seconds'] = time.monotonic()-start
    result['next_contract_checks'] = ['original dates and parent geography', 'raw spectral calibration',
        'native missing-band handling', 'annual crop labels and voids', 'RGB/label/spectral alignment']
    save(a.out/'audit.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['expected','shards']}), flush=True)


if __name__ == '__main__':
    main()
