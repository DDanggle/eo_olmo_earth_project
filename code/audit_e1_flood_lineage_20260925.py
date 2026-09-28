"""Verify original E1 flood items against frozen C0/E5, not imagery or gold truth."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timedelta,date,timezone
import hashlib,json
from pathlib import Path

E1_SHA='7c09a6b0336290f3eb765fde5d0483dcb9c5091d8ea4244a5840264966e5255b'
SUMMARY_SHA='ab3893e61082990bcc0fb8639c8282c770963050c10717b644de5c3bc93663ed'
E5_SHA='e8058fd5193374a1c32968b063e6b67032d8a4f2bc0bf659e490e09816a67a3a'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def check(ok,message):
    if not ok:raise ValueError(message)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    repo=a.repo.resolve();source=a.source.resolve();check(not a.out.exists(),'New report required')
    e5=repo/'artifacts/e5_equal_budget_v0_review_20260925/e5_equal_budget_v0'
    c0=read(repo/'artifacts/c0_linear_view_probe_v1_20260925/manifest.json')
    check(c0['source_sha256']['flood_qa_v0/items.jsonl']==E1_SHA,'Historic C0 source pin differs')
    check(sha(source/'items.jsonl')==E1_SHA,'Original E1 items SHA differs')
    check(sha(source/'summary.json')==SUMMARY_SHA,'Original E1 summary SHA differs')
    summary=read(source/'summary.json');code=repo/'code/e1_flood_qa_v0.py'
    check(sha(code)==summary['code_sha256'],'Current source differs from E1 build')
    check(sha(e5/'items.jsonl')==E5_SHA,'E5 item pin differs')
    audit=read(e5.parent/'e5_independent_audit_20260925.json')
    check(audit['consistent'] is True and audit['hashes_verified'][audit['artifact']+'/items.jsonl']==E5_SHA,'E5 audit linkage differs')
    items=rows(source/'items.jsonl');e5_items=rows(e5/'items.jsonl');by_id={x['id']:x for x in items}
    check(len(items)==len(by_id)==summary['n_items']==4851,'Original E1 item count/ID duplication')
    counts=Counter(x['fold']+'|'+x['kind'] for x in items)
    check(dict(counts)==summary['by_split_kind'],'Original summary counts differ')
    events=defaultdict(set);tiles=defaultdict(set);kinds={}
    for x in items:
        check(x['fold'] in ('train','validation','test') and x['type']=='Q1' and x['sensor']=='sentinel1','Invalid source metadata')
        check(x['kind'] in ('pos','neg','hard_neg') and x['answer']==('yes' if x['kind']=='pos' else 'no'),'Label contract differs')
        expected_slots=['pre_1','pre_2'] if x['kind']=='neg' else ['pre_2','post']
        check(x['slots']==expected_slots,'Source slot contract differs')
        d=[date.fromisoformat(v) for v in x['dates']];check(d[1]-d[0]==timedelta(days=12),'Synthetic date spacing differs')
        events[x['fold']].add(str(x['event']));tiles[x['fold']].add(x['tile']);kinds[(x['tile'],x['kind'])]=x
        if x['kind']=='pos':
            n=by_id.get(x['tile']+'_q1_neg');check(n is not None,'Missing within-tile negative')
            check(n['event']==x['event'] and n['fold']==x['fold'] and n['flood_frac']==x['flood_frac'],'Within-tile source mismatch')
            check(n['dates'][1]==x['dates'][0] and x['flood_frac']>=.02,'Pair date/flood-threshold contract differs')
        elif x['kind']=='hard_neg':check(x['flood_frac']==0.,'Hard-negative source fraction differs')
    overlap={}
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        overlap[left+'|'+right]={'events':sorted(events[left]&events[right]),'tiles':sorted(tiles[left]&tiles[right])}
        check(not overlap[left+'|'+right]['events'] and not overlap[left+'|'+right]['tiles'],'Source fold leakage')
    flood=[x for x in e5_items if x['phen']=='flood'];lookup={x['id']:x for x in flood}
    expected={x['id'] for x in items if x['fold'] in ('train','test')}
    check(len(flood)==len(lookup)==len(expected)==4569 and set(lookup)==expected,'E1 to E5 exact support differs')
    for key,x in lookup.items():
        original=by_id[key]
        check(all(x.get(k)==v for k,v in original.items()),'Inherited E1 field changed: '+key)
        check(x['partition']==x['fold'] and str(x['event'])==x['cluster'],'Derived E5 event/partition differs')
    result={'schema':'e1-flood-to-e5-lineage-audit-v0','consistent':True,'checked_at':datetime.now(timezone.utc).isoformat(),
        'source':str(source),'source_sha256':{'items.jsonl':E1_SHA,'summary.json':SUMMARY_SHA,'code':sha(code)},
        'e5_item_sha256':E5_SHA,'n_e1_items':4851,'n_e5_flood_items':4569,'validation_items_not_in_e5':282,
        'by_split_kind':dict(sorted(counts.items())),'events_by_split':{k:sorted(v) for k,v in events.items()},
        'fold_overlap':overlap,'all_original_fields_equal_in_e5':True,'all_source_dates_are_12_day_approximations':True,
        'code_sha256':sha(Path(__file__)),
        'limits':['Read-only source lineage; no model execution, mask/embedding arrays, original 7000-tile selection or physical labels revalidated.',
                  'Source pre/pre negatives remain assumptions, not independently verified temporal absence.',
                  'E1 event-disjoint splits do not mean events were previously unexposed in this research project.',
                  'This is September25 flood-question E1, not August26 context/decoder E1 factorial.']}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('consistent','n_e1_items','n_e5_flood_items','validation_items_not_in_e5','all_original_fields_equal_in_e5')}))
if __name__=='__main__':main()
