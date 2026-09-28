#!/usr/bin/env python3
"""C0: separate, descriptive CPU global-mean linear probe; never changes E3.
prepare freezes the exact available E2 population/features; run verifies it and fits six models.
"""
import argparse, collections, hashlib, json, shutil, sys, time, traceback
from pathlib import Path
import numpy as np

ARMS = ('earlier', 'later', 'pair')
SHAPES = {'flood': (3, 768, 48, 48), 'landslide': (12, 768, 32, 32)}
SOURCES = {'landslide_train': 'sentinel_qa_train_v0/items.jsonl', 'landslide_test': 'sentinel_qa_v0_1/items.jsonl',
           'flood': 'flood_qa_v0/items.jsonl', 'contract': 'sen12_gp_contract/sample_contract.jsonl'}
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def load(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def write(path, data): Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
def digest(data): return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def unique(rows, key='id'):
    if len({x[key] for x in rows}) != len(rows): raise ValueError('Duplicate ' + key)
    return {x[key]: x for x in rows}
def indices(item, rec):
    if item['phen'] == 'flood':
        expected = {'pos': ['pre_2', 'post'], 'hard_neg': ['pre_2', 'post'], 'neg': ['pre_1', 'pre_2']}
        if item.get('kind') not in expected or item.get('slots') != expected[item['kind']]: raise ValueError('Flood kind/slots mismatch')
        result = [{'pre_1': 0, 'pre_2': 1, 'post': 2}[s] for s in item['slots']]
    else:
        r = rec[item['tile']]; q = r['scl_clear_fraction']
        if len(q) != 15 or len(r['times']) != 15 or not np.isfinite(q).all(): raise ValueError('Invalid S2 date contract')
        keep = sorted(sorted(range(15), key=lambda i: (-float(q[i]), i))[:12])
        dates = [str(r['times'][i])[:10] for i in keep]
        if len(set(dates)) != 12: raise ValueError('Ambiguous S2 dates')
        result = [dates.index(d) for d in item['dates']]
    if len(result) != 2 or result[0] >= result[1]: raise ValueError('Expected two distinct ordered E2 slots')
    return result

def spatial_mean(path, expected_shape):
    s = np.load(path, mmap_mode='r', allow_pickle=False)
    if tuple(s.shape) != tuple(expected_shape) or s.ndim != 4 or s.shape[1] != 768: raise ValueError('Cache shape mismatch: ' + str(path))
    out = []
    for frame in s:
        x = np.asarray(frame, dtype=np.float32)
        if not np.isfinite(x).all(): raise ValueError('Nonfinite cache: ' + str(path))
        out.append(x.mean(axis=(-2, -1), dtype=np.float32))
    return np.stack(out)

def standardize(train, test):
    train, test = np.asarray(train, dtype=np.float64), np.asarray(test, dtype=np.float64)
    if train.ndim != 2 or test.ndim != 2 or len(train) == 0 or train.shape[1] != test.shape[1]: raise ValueError('Feature shape mismatch')
    if not np.isfinite(train).all() or not np.isfinite(test).all(): raise ValueError('Nonfinite feature')
    mean, scale = train.mean(0), train.std(0)
    scale[scale < 1e-6] = 1.
    return (train - mean) / scale, (test - mean) / scale, mean, scale

def fit_logistic(train, y, test):
    import torch
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    train, test, y = np.asarray(train), np.asarray(test), np.asarray(y)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1] or y.shape != (len(train),): raise ValueError('Fit shape mismatch')
    if not all(np.isfinite(a).all() for a in (train, test, y)) or set(y.tolist()) != {0, 1}: raise ValueError('Fit needs finite inputs and both binary classes')
    X, Y, T = (torch.as_tensor(a, dtype=torch.float64, device='cpu') for a in (train, y, test))
    sample_weight = torch.where(Y == 1, len(Y) / (2 * Y.sum()), len(Y) / (2 * (1 - Y).sum()))
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True); b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1., max_iter=200, max_eval=250, history_size=20,
                            tolerance_grad=1e-7, tolerance_change=1e-12, line_search_fn='strong_wolfe')
    def objective(): return (torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, Y, reduction='none') * sample_weight).mean() + .005 * w.square().sum()
    def closure():
        opt.zero_grad(); loss = objective()
        if not torch.isfinite(loss): raise ValueError('Nonfinite objective')
        loss.backward(); return loss
    start = time.perf_counter(); initial = float(objective().detach()); opt.step(closure)
    final = float(closure().detach()); gradient = max(float(w.grad.abs().max()), float(b.grad.abs()))
    logits = (T @ w + b).detach().numpy(); model = {'weights': w.detach().numpy(), 'bias': b.detach().numpy()}
    finite = all(np.isfinite(a).all() for a in (logits, model['weights'], model['bias'], final, gradient))
    state = opt.state[w]
    diag = {'converged': bool(finite and gradient <= 1e-5 and final <= initial + 1e-10), 'initial_loss': initial,
            'final_loss': final, 'gradient_inf': gradient, 'iterations': state.get('n_iter', 0),
            'function_evaluations': state.get('func_evals', 0), 'train_n': len(Y), 'n_features': X.shape[1], 'parameter_count': X.shape[1] + 1, 'runtime_s': time.perf_counter() - start}
    if not finite: raise ValueError('Nonfinite fitted model/logit')
    return logits, model, diag

def event_scores(items, pred):
    pred = np.asarray(pred)
    if pred.shape != (len(items),) or not np.isin(pred, (0, 1)).all(): raise ValueError('Invalid predictions')
    unique(items); paired, hard = collections.defaultdict(list), collections.defaultdict(list)
    for it, p in zip(items, pred):
        if it['kind'] not in ('pos', 'neg', 'hard_neg') or it['answer'] != ('yes' if it['kind'] == 'pos' else 'no'): raise ValueError('Invalid source label')
        (hard if it['kind'] == 'hard_neg' else paired)[str(it['cluster'])].append((it['answer'] == 'yes', int(p)))
    events = {}
    for c, values in sorted(paired.items()):
        pos = [p for y, p in values if y]; neg = [p for y, p in values if not y]
        if not pos or not neg: raise ValueError('Paired event missing a class')
        events[c] = {'ba': .5 * (np.mean(pos) + 1 - np.mean(neg)), 'n': len(values), 'pos': len(pos), 'neg': len(neg), 'recall': float(np.mean(pos)), 'fpr': float(np.mean(neg))}
    he = {c: {'n': len(v), 'fpr': float(np.mean([p for _, p in v]))} for c, v in sorted(hard.items())}
    hp = [p for v in hard.values() for _, p in v]
    return {'event_macro_ba': float(np.mean([v['ba'] for v in events.values()])) if events else None, 'events': events,
            'hard_negative': {'events': he, 'n': len(hp), 'pooled_fpr': float(np.mean(hp)) if hp else None, 'event_macro_fpr': float(np.mean([v['fpr'] for v in he.values()])) if he else None},
            'tuple_counts': dict(sorted(collections.Counter(f"{i['cluster']}|{i['kind']}|{i['answer']}" for i in items).items()))}

def compare_later_pair(later, pair, phen):
    if phen not in SHAPES or set(later['events']) != set(pair['events']) or not later['events']: raise ValueError('Incompatible comparison events')
    events = {c: later['events'][c]['ba'] - pair['events'][c]['ba'] for c in sorted(pair['events'])}
    d = np.array(list(events.values())); ci = None
    if phen == 'flood':
        boot = d[np.random.default_rng(20260925).integers(0, len(d), size=(5000, len(d)))].mean(1)
        ci = np.quantile(boot, [.025, .975]).tolist()
    return {'delta': float(d.mean()), 'ci95_delta': ci, 'event_deltas': events, 'unit': 'event' if phen == 'flood' else 'two_regions_descriptive_only'}

def eligibility(root):
    """Reproduce E2 existence eligibility BEFORE inspecting dates or label-derived slots."""
    rec = unique(lines(root / SOURCES['contract']), 'sample_id'); candidates, audit = [], []
    for split in ('train', 'test'):
        for phen in SHAPES:
            path = SOURCES['flood'] if phen == 'flood' else SOURCES['landslide_' + split]
            for source in lines(root / path):
                if source.get('type') != 'Q1' or (phen == 'flood' and source['fold'] != split): continue
                cache = root / ('kurosiwo_s1_cache/single_fp16' if phen == 'flood' else 'olmo_streaming_dev/single_fp16') / (source['tile'] + '.npy')
                reasons = ([] if cache.exists() else ['missing_cache']) + ([] if phen == 'flood' or source['tile'] in rec else ['missing_contract'])
                audit.append(dict(id=source['id'], tile=source['tile'], partition=split, phen=phen, cache_path=str(cache), source_path=path, eligible=not reasons, reasons=reasons))
                if reasons: continue
                candidates.append(dict(source, phen=phen, partition=split, kind=source.get('kind', 'pos' if source['answer'] == 'yes' else 'neg'), cache_path=str(cache)))
    audit.sort(key=lambda x: (x['partition'], x['phen'], x['id']))
    frozen = {'schema': 'c0-e2-eligibility-audit-v1', 'rule': 'Exact E2: cache.exists(), and S2 tile present in contract; no inference or outcome-based exclusions',
              'items': audit, 'excluded': [x for x in audit if not x['eligible']],
              'candidate_counts': dict(sorted(collections.Counter(f"{x['partition']}|{x['phen']}" for x in audit).items())),
              'eligible_counts': dict(sorted(collections.Counter(f"{x['partition']}|{x['phen']}" for x in audit if x['eligible']).items()))}
    return candidates, rec, frozen

def check_eligibility(root, frozen):
    if eligibility(root)[2] != frozen: raise ValueError('Frozen E2 eligibility changed; no reselection')

def population(root, cfg, audit_path=None):
    items, rec, audit = eligibility(root)
    if audit_path is not None: write(audit_path, audit)
    for it in items:
        it['cluster'] = str(it['event']) if it['phen'] == 'flood' else it['fold']; it['indices'] = indices(it, rec)
        if not Path(it['cache_path']).is_file(): raise FileNotFoundError(it['cache_path'])
    unique(items)
    counts = dict(collections.Counter(f"{i['partition']}|{i['phen']}|{i['answer']}" for i in items))
    if counts != cfg['expected_counts']: raise ValueError('E2 population counts differ: ' + str(counts))
    test = {i['id']: i for i in items if i['partition'] == 'test'}
    for seed in (1, 2, 3):
        ref = unique(lines(root / f'e2_multi_reader_v0/reader_seed{seed}/answers_real_all.jsonl'))
        if set(ref) != set(test): raise ValueError('E2 exact test ID coverage mismatch')
        for key, it in test.items():
            if any(ref[key][k] != it[k] for k in ('tile', 'phen', 'kind', 'fold')) or ref[key]['text_gold'] != it['answer']: raise ValueError('E2 metadata mismatch: ' + key)
    for phen in SHAPES:
        train = [i for i in items if i['phen'] == phen and i['partition'] == 'train']; test_items = [i for i in items if i['phen'] == phen and i['partition'] == 'test']
        if {i['tile'] for i in train} & {i['tile'] for i in test_items} or {i['cluster'] for i in train} & {i['cluster'] for i in test_items}: raise ValueError('Train/test tile or event/region overlap')
    pairs = collections.defaultdict(list)
    for it in items:
        if it['kind'] not in ('pos', 'neg', 'hard_neg') or it['answer'] != ('yes' if it['kind'] == 'pos' else 'no'): raise ValueError('Source kind/answer mismatch')
        if it['kind'] != 'hard_neg': pairs[(it['partition'], it['phen'], it['tile'])].append(it)
    for pair in pairs.values():
        if len(pair) != 2 or {i['kind'] for i in pair} != {'pos', 'neg'} or len({i['cluster'] for i in pair}) != 1: raise ValueError('Incomplete source pair')
    return sorted(items, key=lambda i: (i['partition'], i['phen'], i['id']))

def prepare(a, out):
    cfg = load(a.config); root = Path(a.root)
    fixed = {'lambda': .01, 'max_iter': 200, 'max_eval': 250, 'history_size': 20, 'lr': 1., 'tolerance_grad': 1e-7, 'tolerance_change': 1e-12, 'cpu_threads': 2, 'device': 'cpu', 'dtype': 'torch.float64', 'line_search_fn': 'strong_wolfe'}
    if any(cfg['fit'].get(k) != v for k, v in fixed.items()): raise ValueError('Config differs from frozen solver implementation')
    write(out / 'status.json', {'status': 'preparing'})
    for rel, expected in cfg['expected_source_sha256'].items():
        if sha(root / rel) != expected: raise ValueError('Pinned source hash mismatch: ' + rel)
    paths = sorted(set(SOURCES.values()) | set(cfg['expected_source_sha256']))
    source_hashes = {rel: sha(root / rel) for rel in paths}; items = population(root, cfg, out / 'eligibility_audit.json')
    (out / 'items.jsonl').write_text(''.join(json.dumps(i, sort_keys=True) + '\n' for i in items)); shutil.copyfile(a.config, out / 'prereg.json'); shutil.copyfile(__file__, out / 'source.py')
    manifest = {'schema': 'c0-prepared-v1', 'root': str(root), 'source_sha256': source_hashes, 'code_sha256': sha(__file__), 'prereg_sha256': sha(out / 'prereg.json'), 'items_sha256': sha(out / 'items.jsonl'), 'eligibility_sha256': sha(out / 'eligibility_audit.json'),
                'population_ids_sha256': {s: digest(sorted(i['id'] for i in items if i['partition'] == s)) for s in ('train', 'test')}, 'counts': cfg['expected_counts'], 'tuple_counts': dict(sorted(collections.Counter(f"{i['partition']}|{i['phen']}|{i['cluster']}|{i['kind']}|{i['answer']}" for i in items).items())), 'cache_sha256': {}, 'cache_stat': {}}
    write(out / 'manifest.json', manifest)
    means = {}; required = {i['cache_path']: i['phen'] for i in items}
    for n, (path, phen) in enumerate(sorted(required.items())):
        before = Path(path).stat(); h = sha(path); value = spatial_mean(path, SHAPES[phen]); after = Path(path).stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns): raise ValueError('Cache changed while reading: ' + path)
        means[path] = value; manifest['cache_sha256'][path] = h; manifest['cache_stat'][path] = {'size': after.st_size, 'mtime_ns': after.st_mtime_ns}
        if n % 100 == 0: print(f'CPU means {n}/{len(required)}', flush=True)
    pairs = np.stack([means[i['cache_path']][i['indices']] for i in items])
    np.savez_compressed(out / 'global_features.npz', pairs=pairs, ids=np.array([i['id'] for i in items]))
    for rel, h in source_hashes.items():
        if sha(root / rel) != h: raise ValueError('Source changed during prepare: ' + rel)
    check_eligibility(root, load(out / 'eligibility_audit.json'))
    for path, observed in manifest['cache_stat'].items():
        current = Path(path).stat()
        if {'size': current.st_size, 'mtime_ns': current.st_mtime_ns} != observed: raise ValueError('Used cache changed during prepare: ' + path)
    manifest.update(features_sha256=sha(out / 'global_features.npz'), n_unique_caches=len(means), feature_shape=list(pairs.shape), environment={'python': sys.version, 'numpy': np.__version__})
    write(out / 'manifest.json', manifest); write(out / 'status.json', {'status': 'prepared'}); print('C0 prepared', flush=True)

def run(a, out):
    import torch
    if load(out / 'status.json')['status'] != 'prepared': raise ValueError('Only an untouched prepared run may start')
    manifest = load(out / 'manifest.json'); root = Path(manifest['root'])
    for fn, key in [('items.jsonl', 'items_sha256'), ('prereg.json', 'prereg_sha256'), ('global_features.npz', 'features_sha256'), ('eligibility_audit.json', 'eligibility_sha256')]:
        if sha(out / fn) != manifest[key]: raise ValueError('Frozen artifact changed: ' + fn)
    if sha(__file__) != manifest['code_sha256'] or sha(out / 'source.py') != manifest['code_sha256']: raise ValueError('Run the frozen source snapshot')
    for rel, h in manifest['source_sha256'].items():
        if sha(root / rel) != h: raise ValueError('Frozen source changed: ' + rel)
    check_eligibility(root, load(out / 'eligibility_audit.json'))
    write(out / 'status.json', {'status': 'running'}); items = lines(out / 'items.jsonl')
    with np.load(out / 'global_features.npz', allow_pickle=False) as f: pairs = f['pairs']; ids = f['ids'].tolist()
    if ids != [i['id'] for i in items] or pairs.shape != (len(items), 2, 768) or not np.isfinite(pairs).all(): raise ValueError('Feature archive contract mismatch')
    result = {'schema': 'c0-linear-view-probe-v1', 'valid': True, 'scope': 'descriptive exposed-development linear decodability; no E3 verdict', 'metrics': {}, 'fits': {}, 'model_sha256': {}, 'failures': [], 'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version, 'device': 'cpu', 'threads': 2}}
    predictions = []
    for phen in SHAPES:
        tr = [j for j, i in enumerate(items) if i['phen'] == phen and i['partition'] == 'train']; te = [j for j, i in enumerate(items) if i['phen'] == phen and i['partition'] == 'test']
        y = np.array([items[j]['answer'] == 'yes' for j in tr], dtype=np.float64); result['metrics'][phen] = {}
        for arm in ARMS:
            X = pairs.reshape(len(items), -1) if arm == 'pair' else pairs[:, 0 if arm == 'earlier' else 1]
            z, t, mean, scale = standardize(X[tr], X[te]); logits, model, diag = fit_logistic(z, y, t); key = phen + '_' + arm
            np.savez_compressed(out / (key + '.npz'), **model, mean=mean, scale=scale); result['model_sha256'][key] = sha(out / (key + '.npz')); result['fits'][key] = diag
            if not diag['converged']: result['valid'] = False; result['failures'].append(key + ': optimizer did not meet frozen convergence criterion')
            result['metrics'][phen][arm] = event_scores([items[j] for j in te], logits >= 0)
            predictions.extend(dict(id=items[j]['id'], tile=items[j]['tile'], phen=phen, cluster=items[j]['cluster'], kind=items[j]['kind'], source_gold=items[j]['answer'], arm=arm, logit=float(v), prediction='yes' if v >= 0 else 'no') for j, v in zip(te, logits))
            write(out / 'partial_fits.json', result['fits']); print('C0 fit ' + key + ' ' + json.dumps(diag), flush=True)
        result['metrics'][phen]['later_minus_pair'] = compare_later_pair(result['metrics'][phen]['later'], result['metrics'][phen]['pair'], phen)
    (out / 'predictions.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in predictions)); result['predictions_sha256'] = sha(out / 'predictions.jsonl')
    result['input_manifest_sha256'] = sha(out / 'manifest.json'); write(out / 'results.json', result); write(out / 'status.json', {'status': 'complete' if result['valid'] else 'invalid', 'valid': result['valid']})
    return result['valid']

def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare', 'run']); p.add_argument('--root', default='/home/work/data/olmoearth'); p.add_argument('--out', default='c0_linear_view_probe_v1'); p.add_argument('--config', default=str(Path(__file__).with_name('c0_linear_view_prereg_v1.json'))); a = p.parse_args()
    out = Path(a.root) / a.out
    if a.mode == 'prepare': out.mkdir(parents=True, exist_ok=False)
    else:
        if not out.is_dir() or load(out / 'status.json')['status'] != 'prepared': raise ValueError('Existing untouched prepared output required')
    try:
        if a.mode == 'prepare': prepare(a, out)
        elif not run(a, out): return 2
    except Exception as e:
        write(out / 'failure.json', {'phase': a.mode, 'error': str(e), 'traceback': traceback.format_exc()}); write(out / 'status.json', {'status': 'invalid'}); raise
    return 0
if __name__ == '__main__': sys.exit(main())
