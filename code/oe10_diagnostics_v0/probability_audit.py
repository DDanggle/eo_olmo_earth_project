"""Read-only post-hoc diagnosis of frozen OE10 probabilities, never threshold fitting.

Only already exposed development labels and a completed run's final predictions
are accepted. No model import, forward pass, optimization, or primary gate change.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def ranking(prob, gold):
    """Tie-aware ROC AUC and non-interpolated average precision."""
    prob = np.asarray(prob, dtype=np.float64)
    gold = np.asarray(gold, dtype=bool)
    pos = int(gold.sum()); neg = len(gold) - pos
    if not pos or not neg:
        return None, None
    order = np.argsort(-prob, kind='stable')
    s, y = prob[order], gold[order]
    ends = np.r_[np.where(s[1:] != s[:-1])[0], len(s)-1]
    tp = np.r_[0, np.cumsum(y)[ends]].astype(float)
    fp = np.r_[0, (ends+1)-np.cumsum(y)[ends]].astype(float)
    auc = float(np.sum(np.diff(fp / neg) * (tp[1:] + tp[:-1]) / (2*pos)))
    ap = float(np.sum(np.diff(tp / pos) * tp[1:] / (tp[1:] + fp[1:])))
    return auc, ap


def quantiles(x):
    return dict(zip(('min','p05','median','p95','max'), map(float, np.quantile(x, [0,.05,.5,.95,1])))) if len(x) else None


def guards(root):
    status = json.loads((root/'training_v0/status.json').read_text())
    protected = {}
    for name, expected in status['protected_before'].items():
        p = root.parent/'code'/name
        actual = {'sha256':digest(p), 'mtime_ns':p.stat().st_mtime_ns, 'bytes':p.stat().st_size}
        assert actual == expected, ('protected source changed',name)
        protected[name] = actual
    snap = root/'code_snapshot/oe10_p2_v3'
    manifest = json.loads((snap/'source_manifest.json').read_text())['files']
    assert all(digest(snap/name)==h for name,h in manifest.items()), 'training snapshot changed'
    protocol = root/'config/oe10_p2_execution_protocol_20260927.json'
    assert digest(protocol)==status['protocol_sha256'], 'protocol changed'
    return {'protected':protected, 'source_manifest_sha256':digest(snap/'source_manifest.json'),
            'protocol_sha256':digest(protocol)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepared-root',type=Path,required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    before=guards(a.root)
    assert a.run in [f'{arm}_{seed}_train' for arm in ('B0','B2') for seed in (270927,270928,270929)]
    run=a.root/'training_v0'/a.run
    receipt=json.loads((run/'receipt.json').read_text())
    assert receipt['status']=='training_completed' and receipt['completed_updates']==2304
    score_path=run/'score_step_002304.json'
    score=json.loads(score_path.read_text())
    assert digest(score_path)==[w['score_sha256'] for w in receipt['evaluation_windows'] if w['step']==2304][0]
    manifest_path=a.prepared_root/'manifest.jsonl'
    meta={r['patch_id']:r for r in read_rows(manifest_path)}
    pred_dir=run/'predictions_step_002304'
    pred_path=pred_dir/'predictions.jsonl'
    predictions={r['episode_id']:r for r in read_rows(pred_path)}
    scored=score['per_episode_scoring_only']
    assert len(scored)==len(predictions)==384
    assert {r['episode_id'] for r in scored}==set(predictions)
    cache={}; result=[]
    for row in scored:
        m=meta[row['query_patch_id']]
        assert m['training_partition']=='dev_query', 'unopened role prohibited'
        if m['patch_id'] not in cache:
            path=a.prepared_root/m['label_path']
            assert path.resolve().is_relative_to((a.prepared_root/'labels').resolve())
            assert digest(path)==m['label_sha256']==row['source_label_sha256']
            with np.load(path,allow_pickle=False) as z:
                sem=z['semantic']; valid=z['label_valid']
            assert sem.shape==valid.shape==(128,128) and np.array_equal(valid,sem!=19)
            cache[m['patch_id']]=(sem,valid)
        sem,valid=cache[m['patch_id']]
        g=(sem==row['target_class'])[valid]
        pr=predictions[row['episode_id']]
        path=pred_dir/pr['npz_path']
        assert path.resolve().is_relative_to(pred_dir.resolve())
        assert digest(path)==pr['sha256']==row['prediction_npz_sha256']
        with np.load(path,allow_pickle=False) as z:
            assert z.files==['probability']
            full=z['probability']
        assert full.shape==(128,128) and np.isfinite(full).all() and ((full>=0)&(full<=1)).all()
        prob=full[valid]; hard=prob>.5
        assert int(g.sum())==row['target_pixels'] and len(g)==row['valid_pixels']
        assert int(hard.sum())==row['predicted_valid_pixels']
        assert int((hard&g).sum())==row['intersection']
        auc,ap=ranking(prob,g)
        prevalence=float(g.mean())
        result.append({k:row[k] for k in ('episode_id','query_patch_id','target_class','counter_class','k','target_present','target_iou','absent_fp_area','absent_fp_case')} | {
            'prevalence':prevalence,'roc_auc':auc,'average_precision':ap,
            'ap_over_prevalence':ap/prevalence if ap is not None else None,
            'predicted_empty_at_original_threshold':not bool(hard.any()),
            'positive_probability':quantiles(prob[g]),'negative_probability':quantiles(prob[~g]),
            'positive_median':float(np.median(prob[g])) if g.any() else None,
            'negative_median':float(np.median(prob[~g])) if (~g).any() else None,
            'foreground_recall_at_original_threshold':float(hard[g].mean()) if g.any() else None})
    summary={}
    fields=('target_iou','prevalence','roc_auc','average_precision','ap_over_prevalence','positive_median','negative_median','foreground_recall_at_original_threshold')
    for k in (1,2,4,8):
        summary[str(k)]={}
        for c in (1,2,3):
            present=[r for r in result if r['k']==k and r['target_class']==c and r['target_present']]
            absent=[r for r in result if r['k']==k and r['target_class']==c and not r['target_present']]
            byquery=defaultdict(list)
            for r in present:byquery[r['query_patch_id']].append(r)
            means={field:statistics.mean(statistics.mean(r[field] for r in group if r[field] is not None) for group in byquery.values()) for field in fields}
            summary[str(k)][str(c)]={'present_query_count':len(byquery),'present_episode_count':len(present),
                'empty_present_episode_count':sum(r['predicted_empty_at_original_threshold'] for r in present),
                'query_macro':means,'absent_episode_count':len(absent),
                'absent_fp_case_rate':statistics.mean(r['absent_fp_case'] for r in absent) if absent else None}
    after=guards(a.root); assert after==before
    report={'schema_version':'oe10_frozen_probability_audit_v1','created_utc':datetime.now(timezone.utc).isoformat(),
            'run':a.run,'step':2304,'scope':'Post-hoc diagnosis on exposed dev; no threshold fitting, model inference, new training, or primary metric change. Pixel rankings do not establish calibration or generalization.',
            'hashes':{'code':digest(Path(__file__)),'score':digest(score_path),'predictions':digest(pred_path),'manifest':digest(manifest_path)},
            'all_prediction_hashes_and_original_hard_counts_verified':True,'unchanged_source_guards':after,
            'summary_by_k_class':summary,'per_episode':result}
    with a.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'run':a.run,'out':str(a.out),'k1':summary['1'],'k8':summary['8']}))


if __name__=='__main__':main()
