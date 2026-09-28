#!/usr/bin/env python3
"""Post-hoc E2 same-prompt flood analysis. No fitting or new model inference."""
import argparse, collections, hashlib, json, shutil
from pathlib import Path
import numpy as np

def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def require(ok, message):
    if not ok: raise ValueError(message)
def unique(xs):
    require(len({x['id'] for x in xs})==len(xs), 'Duplicate item ID')
    return {x['id']:x for x in xs}
def prompt(it):
    require(it['slots']==['pre_2','post'] and len(it['dates'])==2, 'Unexpected target slots/dates')
    return f"These are 2 Sentinel-1 observations of the same area in chronological order, taken on {', '.join(it['dates'])}: <EO> Did a flood occur between the two observations? Answer with yes or no."
def strata(items):
    groups=collections.defaultdict(list)
    for it in items:groups[(str(it['event']), prompt(it), tuple(it['slots']))].append(it)
    kept=[]; excluded=[]
    for (event, text, slots), its in sorted(groups.items()):
        by={k:[x for x in its if x['kind']==k] for k in ('pos','hard_neg')}
        entry={'event':event,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'reconstructed_prompt':text,
               'slots':list(slots),'n_pos':len(by['pos']),'n_hard_neg':len(by['hard_neg']),
               'pos_ids':sorted(x['id'] for x in by['pos']),'hard_neg_ids':sorted(x['id'] for x in by['hard_neg'])}
        if not by['pos'] or not by['hard_neg']:entry['reason']='missing_one_source_class';excluded.append(entry)
        else:kept.append(entry)
    return kept, excluded

def blind_consistency(items, blind):
    """Check all target prompts, including unsupported/quality-excluded rows."""
    groups=collections.defaultdict(list)
    for it in items:groups[prompt(it)].append(it['id'])
    violations=[]
    for text, ids in groups.items():
        values={blind[i]['parsed'] for i in ids}
        if len(values)!=1:violations.append({'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'ids':sorted(ids),'outputs':sorted(values)})
    return {'n_reconstructed_prompts':len(groups),'violations':violations,'consistent':not violations}

def event_metrics(groups, answers):
    events=collections.defaultdict(list)
    for group in groups:
        p=group['pos_ids'];n=group['hard_neg_ids']
        require(all(answers[i]['parsed'] in ('yes','no') for i in p+n),'Unparsed answer in stratum')
        tpr=float(np.mean([answers[i]['parsed']=='yes' for i in p]));fpr=float(np.mean([answers[i]['parsed']=='yes' for i in n]))
        events[group['event']].append(dict(group,recall=tpr,fpr=fpr,ba=(1+tpr-fpr)/2,decision_contrast=tpr-fpr))
    return {e:{'ba':float(np.mean([s['ba'] for s in ss])), 'decision_contrast':float(np.mean([s['decision_contrast'] for s in ss])),
               'recall':float(np.mean([s['recall'] for s in ss])), 'fpr':float(np.mean([s['fpr'] for s in ss])),
               'n_strata':len(ss),'n_pos':sum(s['n_pos'] for s in ss),'n_hard_neg':sum(s['n_hard_neg'] for s in ss),'strata':ss} for e,ss in sorted(events.items())}

def bootstrap(values):
    v=np.asarray(values,dtype=float)
    if len(v)<2:return None
    b=v[np.random.default_rng(20260925).integers(0,len(v),(5000,len(v)))].mean(axis=1)
    return np.quantile(b,[.025,.975]).tolist()

def evaluate_subset(items, predictions):
    groups,excluded=strata(items);events=sorted({g['event'] for g in groups});per_seed={};event_deltas=[]
    for seed in (1,2,3):
        arms={a:event_metrics(groups,predictions[(a,seed)]) for a in ('reader','blind')}
        require(all(set(x)==set(events) for x in arms.values()),'Event coverage mismatch')
        deltas=[arms['reader'][e]['ba']-arms['blind'][e]['ba'] for e in events]
        event_deltas.append(deltas)
        per_seed[str(seed)]={'arms':arms,'reader_macro_ba':float(np.mean([x['ba'] for x in arms['reader'].values()])) if events else None,
                            'blind_macro_ba':float(np.mean([x['ba'] for x in arms['blind'].values()])) if events else None,
                            'reader_minus_blind':float(np.mean(deltas)) if events else None,'ci95_delta':bootstrap(deltas)}
    mean_deltas=np.asarray(event_deltas).mean(axis=0) if events else np.array([])
    return {'candidate_count':len(items),'n_events':len(events),'events':events,'support_sufficient_for_interpretation':len(events)>=5,
            'supported_n_pos':sum(g['n_pos'] for g in groups),'supported_n_hard_neg':sum(g['n_hard_neg'] for g in groups),
            'unsupported_strata':excluded,'per_seed':per_seed,
            'seed_mean_reader_minus_blind':float(mean_deltas.mean()) if events else None,
            'seed_mean_ci95_delta':bootstrap(mean_deltas),'seed_mean_delta_by_event':dict(zip(events,mean_deltas.tolist()))}

def run(a, out):
    repo=Path(a.repo);cfg=read(a.plan);quality_dir=Path(a.quality)
    for rel,h in cfg['inputs_sha256'].items():require(sha(repo/rel)==h,'Frozen input changed: '+rel)
    require(read(quality_dir/'status.json')['status']=='complete','Quality audit incomplete')
    qm=read(quality_dir/'manifest.json');require(sha(quality_dir/'quality.jsonl')==qm['quality_sha256'],'Quality output changed')
    all_items=rows(repo/cfg['items_path']);require(sha(repo/cfg['items_path'])==qm['items_sha256'],'Quality audit used different item bytes')
    test=[x for x in all_items if x['partition']=='test'];test_by_id=unique(test)
    require(len(test)==1755,'Expected E2 test 1755')
    target=[x for x in test if x['phen']=='flood' and x['kind'] in ('pos','hard_neg')];target_by_id=unique(target)
    require(collections.Counter(x['kind'] for x in target)=={'pos':457,'hard_neg':457},'Expected target 457+457')
    quality=unique(rows(quality_dir/'quality.jsonl'));require(set(quality)==set(target_by_id),'Quality target coverage')
    for key,it in target_by_id.items():
        q=quality[key];require(all(q[k]==it[k] for k in ('tile','dates','slots','kind')) and str(q['event'])==str(it['event']) and q['source_answer']==it['answer'],'Quality metadata mismatch')
        require(q['eligible_symmetric_quality']==(q['valid_frac']>=.90),'Quality subset mismatch')
        require(0<=q['valid_frac']<=1 and 0<=q['flood_frac']<=1,'Quality statistics outside range')
        require((it['kind']=='pos' and it['answer']=='yes' and q['flood_frac']>=.02) or (it['kind']=='hard_neg' and it['answer']=='no' and q['flood_px']==0 and q['valid_frac']>=.90),'Source label inconsistent')
    predictions={};checks={}
    for arm in ('reader','blind'):
        for seed in (1,2,3):
            data=unique(rows(repo/f'artifacts/e2_multi_reader_v0/{arm}_seed{seed}/answers_real_all.jsonl'))
            require(set(data)==set(test_by_id),'Historical answer ID coverage')
            for key,it in test_by_id.items():
                d=data[key];require(all(d[k]==it[k] for k in ('tile','fold','phen','kind')) and d['emb_item']==key and d['text_gold']==d['emb_gold']==it['answer'],'Historical row metadata mismatch')
                require(d['parsed'] in ('yes','no'),'Unparsed answer: no implicit negative')
            predictions[(arm,seed)]=data
            if arm=='blind':checks[str(seed)]=blind_consistency(target,data)
    frozen_quality=[x for x in target if quality[x['id']]['eligible_symmetric_quality']]
    # These sets depend on metadata and quality only, never on model decisions.
    subsets={'quality_symmetric':frozen_quality,'all_original_targets':target}
    (out/'selected_ids.json').write_text(json.dumps({name:sorted(x['id'] for x in xs) for name,xs in subsets.items()},indent=2)+'\n')
    result={'schema':'c1-same-prompt-flood-diagnostic-v0','scope':cfg['scope'],'valid':all(c['consistent'] for c in checks.values()),
            'blind_consistency':checks,'subsets':{name:evaluate_subset(xs,predictions) for name,xs in subsets.items()},
            'quality_excluded_ids':sorted(x['id'] for x in target if not quality[x['id']]['eligible_symmetric_quality']),
            'provenance':{'plan_sha256':sha(a.plan),'code_sha256':sha(__file__),'quality_manifest_sha256':sha(quality_dir/'manifest.json'),
                          'quality_sha256':sha(quality_dir/'quality.jsonl'),'selected_ids_sha256':sha(out/'selected_ids.json'),'frozen_input_sha256':cfg['inputs_sha256']},
            'limits':cfg['limits'],'primary_verdict':None}
    # Inconsistent blind decisions invalidate the comparison; never remove that stratum.
    if not result['valid']:result['failure']='Same reconstructed blind prompt yielded multiple decisions; no visual-only interpretation'
    write(out/'results.json',result);write(out/'status.json',{'status':'complete' if result['valid'] else 'invalid','valid':result['valid']})
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True);p.add_argument('--quality',required=True);p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(a.plan,out/'analysis_plan.json');shutil.copyfile(__file__,out/'source.py')
    try:r=run(a,out)
    except Exception as e:write(out/'status.json',{'status':'invalid','error':str(e)});raise
    print(json.dumps({'valid':r['valid'],'subsets':{k:{kk:v[kk] for kk in ['n_events','supported_n_pos','supported_n_hard_neg','seed_mean_reader_minus_blind','seed_mean_ci95_delta']} for k,v in r['subsets'].items()}}))
    return 0 if r['valid'] else 2
if __name__=='__main__':raise SystemExit(main())
