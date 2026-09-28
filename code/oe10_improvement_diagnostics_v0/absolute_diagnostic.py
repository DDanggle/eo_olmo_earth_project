"""Descriptive CPU audit of already exported P2 logs and scores; no model or tuning."""
import argparse, collections, hashlib, json, statistics
from pathlib import Path

def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def stats(items, field):
    v=[r[field] for r in items]
    return {'n':len(v),'mean':statistics.mean(v),'median':statistics.median(v)} if v else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--export',type=Path,required=True)
    ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); evidence={}
    exported={x['path']:x for x in json.loads((a.export/'export_manifest.json').read_text())['files']}
    def checked(p):
        rel=str(p.relative_to(a.export));m=exported[rel]
        assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256'];evidence[rel]=m['sha256']
        return p
    scoring=a.bundle/'episodes/scoring/scoring_train.jsonl'
    meta={r['episode_id']:r for r in rows(scoring)}
    results={}
    for folder in sorted((a.export/'training_v0').glob('*_train')):
        rp=folder/'receipt.json';rec=json.loads(checked(rp).read_text())
        if rec['status']!='training_completed':continue
        logs=rows(checked(folder/'train_log.jsonl'));assert len(logs)==2304
        assert [r['step'] for r in logs]==list(range(1,2305))
        assert len({r['episode_id'] for r in logs})==2304
        timing=collections.defaultdict(float); grouped=collections.defaultdict(list)
        for r in logs:
            m=meta[r['episode_id']]
            assert r['k']==m['k_pairs']
            r['clipped']=r['grad_norm']>1
            for key,val in r['phase_seconds'].items():timing[key]+=val
            bucket='first384' if r['step']<=384 else 'last384' if r['step']>1920 else 'middle'
            grouped[(m['target_class'],m['target_present'],bucket)].append(r)
        classes={}
        for (c,p,b),items in sorted(grouped.items()):
            classes[f'class{c}:present{p}:{b}']={f:stats(items,f) for f in ['mask_loss','language_loss','native_loss','grad_norm','clipped']}
        curve=[];classcurve={str(c):[] for c in (1,2,3)}
        for window in rec['evaluation_windows']:
            path=checked(folder/f"score_step_{window['step']:06d}.json")
            assert sha(path)==window['score_sha256'];score=json.loads(path.read_text())
            curve.append({'step':window['step'],'auc':score['target_iou_auc']})
            for c in (1,2,3):
                byk={}
                for k in (1,2,4,8):
                    rr=[x for x in score['per_episode_scoring_only'] if x['target_class']==c and x['k']==k and x['target_present']]
                    byq=collections.defaultdict(list)
                    for x in rr:byq[x['query_patch_id']].append(x['target_iou'])
                    byk[k]=statistics.mean(statistics.mean(v) for v in byq.values())
                classcurve[str(c)].append({'step':window['step'],'auc':sum((j-i)*(byk[i]+byk[j])/2 for i,j in zip([1,2,4],[2,4,8]))/7})
        total=sum(timing.values())
        results[folder.name]={'elapsed_seconds':rec.get('elapsed_seconds'),'training_seconds_sum':sum(r['seconds'] for r in logs),
            'phase_seconds':dict(timing),'phase_fraction':{k:v/total for k,v in timing.items()},
            'per_class_presence_window_training_stats':classes,'dev_curve':curve,'dev_class_auc_curve':classcurve,
            'last3_range':max(r['auc'] for r in curve[-3:])-min(r['auc'] for r in curve[-3:])}
    # Verify that class contributions recover the frozen equal-class macro score.
    for v in results.values():
        for i in range(len(v['dev_curve'])):
            recomputed=statistics.mean(v['dev_class_auc_curve'][str(c)][i]['auc'] for c in (1,2,3))
            assert abs(recomputed-v['dev_curve'][i]['auc'])<1e-12
    decomposition={}
    for c in ('1','2','3'):
        b0=statistics.mean(results[f'B0_{seed}_train']['dev_class_auc_curve'][c][-1]['auc'] for seed in (270927,270928))
        b2=statistics.mean(results[f'B2_{seed}_train']['dev_class_auc_curve'][c][-1]['auc'] for seed in (270927,270928))
        decomposition[c]={'B0_mean':b0,'B2_mean':b2,'class_delta':b2-b0,'contribution_to_macro_pp':100*(b2-b0)/3}
    report={'class_mean_decomposition':decomposition,'equal_class_macro_checked_windows':sum(len(v['dev_curve']) for v in results.values()),'scope':'Descriptive post-hoc only. Online training loss windows contain different shuffled episodes; not paired train accuracy. No model/GPU/new labels/threshold fitting.',
        'hashes':{'script':sha(Path(__file__)),'training_scoring':sha(scoring),'export_manifest':sha(a.export/'export_manifest.json'),'inputs':evidence},'runs':results}
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{f:v[f] for f in ['phase_fraction','dev_curve','dev_class_auc_curve','last3_range']} for k,v in results.items()},indent=2))

if __name__=='__main__':main()
