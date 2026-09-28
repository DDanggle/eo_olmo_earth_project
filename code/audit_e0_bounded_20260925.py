#!/usr/bin/env python3
"""Retrospective E0 saved-answer reaggregation; not a preregistration.

No production-function imports, model/pixel inference, remote access, or source
mutation. Reconstructs the prior inline review with explicit pinned inputs and
records this script's SHA. Does not recheck original MS131 answer reproduction.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re

DEFAULT_REPO = Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project')
PINS = {
    'artifacts/earthtalk_content_controls_v0/manifest.json':'fbd7a98ac385923290e26787ce4718f12dab20cc07451839caf87f863632a9bc',
    'artifacts/earthtalk_content_controls_v0/scores.json':'096f29bd4eaca3e9a8d855771e62417c9439e4753c43241d87495a22410b9e02',
    'artifacts/earthtalk_content_controls_v0/answers_real.jsonl':'f00b48f66217ac6646659fd683c80b36b69dbd0f55c8e1781a09b0705fdf1481',
    'artifacts/earthtalk_content_controls_v0/answers_zero_embedding.jsonl':'0e53192f390b5229cb6716f4dce8279dea22461f110ea7c6c1a118dd9a5c1642',
    'artifacts/earthtalk_content_controls_v0/answers_swap_within_tile.jsonl':'fffdd8e05dcf1b4acf62a89dd2e16c3dfc552799c075207dc6c4ab30f7891e60',
    'artifacts/earthtalk_content_controls_v0/answers_swap_cross_tile.jsonl':'bd37b94f6cd83a483a235d526fe410dc865e93b92c81fc72bbeffeb16c5250d8',
    'artifacts/earthtalk_content_controls_v0/answers_q3_swap_cross_tile.jsonl':'393e42dc0e53cb1b5df516b426f7e3f473d6c45678c46dd6cc62365477a428f9',
    'code/earthtalk_content_controls_v0.py':'510a6e55d976e5b6b104cc9e585a2ab8a010e4f6a71234eaf972955823e09e80',
    'code/extract_olmo_streaming.py':'398cb0add9e84b3905d091fe72a4487b57c944d15302bd883c322f8eef3a7ea0',
    'config/earthtalk_content_controls_prereg_v0.json':'d1233aa7d36da4078f2e9ee2490468cce8b8114e53558605fba4422e5140dbd3',
}
ARMS=('real','zero_embedding','swap_within_tile','swap_cross_tile','q3_swap_cross_tile')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def read_rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def agrees(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)


def difference(rows,label):
    groups={gold:[int(row['parsed']=='yes') for row in rows if row[label]==gold] for gold in ('yes','no')}
    require(groups['yes'] and groups['no'],'Missing binary class')
    return sum(groups['yes'])/len(groups['yes'])-sum(groups['no'])/len(groups['no'])


def interval(rows,label):
    groups=defaultdict(list)
    for row in rows:groups[row['tile']].append(row)
    tiles=sorted(groups);rng=random.Random(20260925);values=[]
    for _ in range(2000):
        sampled=[row for _ in tiles for row in groups[rng.choice(tiles)]]
        values.append(difference(sampled,label))
    values.sort()
    # The executed E0 used empirical order statistics, not interpolated quantiles.
    return [values[50],values[1949]]


def jaccard(pred,gold):
    predicted,truth=set(pred or []),set(gold)
    return len(predicted&truth)/len(predicted|truth) if predicted|truth else 1.0


def donor_rank(source,donor):
    return hashlib.sha256('|'.join(('20260925',source['id'],donor['id'])).encode()).hexdigest()


def audit(repo):
    repo=Path(repo).resolve();base=repo/'artifacts/earthtalk_content_controls_v0'
    own_sha=sha(__file__)
    for relative,pin in PINS.items():require(sha(repo/relative)==pin,'Input/source SHA mismatch: '+relative)
    manifest=read(base/'manifest.json');scores=read(base/'scores.json');prereg=read(repo/'config/earthtalk_content_controls_prereg_v0.json')
    require(manifest==scores['manifest'],'Saved manifest copies differ')
    require(manifest['code_sha256']==PINS['code/earthtalk_content_controls_v0.py'],'Executed source pin differs')
    require(manifest['projector_sha256']==prereg['model']['projector_sha256'],'Projector pin differs')
    allrows={arm:read_rows(base/('answers_'+arm+'.jsonl')) for arm in ARMS}
    require({arm:len(values) for arm,values in allrows.items()}==manifest['jobs']==scores['n'],'Arm coverage differs')
    require((manifest['n_q1_items'],manifest['n_pairs'])==(384,192),'Population differs')
    for arm,values in allrows.items():
        require(len(values)==len({row['id'] for row in values}),'Duplicate item ID')
        for row in values:
            raw=row['answer_raw'].strip().lower()
            if arm=='q3_swap_cross_tile':
                tokens=sorted(set(re.findall(r'\b(?:nw|ne|sw|se)\b',raw)))
                parsed=[x.upper() for x in tokens] or None
            else:
                tokens=re.findall(r'\b(?:yes|no)\b',raw);parsed=tokens[0] if tokens else None
            require(parsed==row['parsed'],'Saved raw/parsed answer differs')
        require(all(row['parsed'] is not None for row in values),'Unexpected parse failure')
    real={row['id']:row for row in allrows['real']}
    report={'schema':'e0-bounded-independent-review-v0','checked_at':datetime.now(timezone.utc).isoformat(),
        'audit_code_sha256':own_sha,'audit_source_path':str(Path(__file__).resolve()),
        'analysis_status':'retrospective_reaggregation_not_preregistered',
        'scope':'Saved answers/code only, no inference or remote operation. E0 primary verdict unchanged. New disclosure: per-region effects and negative Q3 donor-versus-source contrast.',
        'arms':{},'checks':{},'source_hashes':{str(repo/name):pin for name,pin in PINS.items()}}
    for metric,arm,label in [('d_real','real','emb_gold'),('d_zero','zero_embedding','text_gold'),
                             ('d_swap','swap_within_tile','emb_gold'),('d_cross','swap_cross_tile','emb_gold')]:
        values=allrows[arm];value=difference(values,label);bounds=interval(values,label)
        require(agrees(value,scores[metric]['value']) and all(agrees(a,b) for a,b in zip(bounds,scores[metric]['ci95'])),'Recorded score/CI differs')
        require({row['id'] for row in values}==set(real),'Q1 arm identity support differs')
        require(all(row['text_gold']==real[row['id']]['text_gold'] and row['tile']==real[row['id']]['tile'] and row['fold']==real[row['id']]['fold'] for row in values),'Source identity differs')
        regions={fold:{'n_items':sum(row['fold']==fold for row in values),'n_tiles':len({row['tile'] for row in values if row['fold']==fold}),
                       'd':difference([row for row in values if row['fold']==fold],label)} for fold in sorted({row['fold'] for row in values})}
        record={'n':len(values),'d':value,'ci95_registered_tile_bootstrap':bounds,'parsed_counts':dict(Counter(row['parsed'] for row in values)),'region_descriptive':regions}
        if arm.startswith('swap'):
            require(all(row['emb_item'] in real and row['emb_gold']==real[row['emb_item']]['text_gold'] and row['emb_gold']!=row['text_gold'] for row in values),'Donor identity/label differs')
            distinct=len({row['emb_item'] for row in values})
            record.update(donor_label_identity_verified=True,unique_donors=distinct,unique_donor_question_ids=distinct,
                          donor_assignments=len(values),repeated_assignments_beyond_first_per_donor_question_id=len(values)-distinct,
                          same_answer_as_donor_real=sum(row['parsed']==real[row['emb_item']]['parsed'] for row in values))
            if arm=='swap_within_tile':
                require(all(row['tile']==real[row['emb_item']]['tile'] for row in values),'Within-tile donor mismatch')
            else:
                for row in values:
                    candidates=[x for x in real.values() if x['fold']==row['fold'] and x['tile']!=row['tile'] and x['text_gold']!=row['text_gold']]
                    require(row['emb_item']==min(candidates,key=lambda x:donor_rank(row,x))['id'],'Cross donor differs from fixed deterministic rule')
                record['deterministic_same_region_opposite_label_donors_verified']=True
        report['arms'][arm]=record
    q3=allrows['q3_swap_cross_tile'];ql={row['id']:row for row in q3}
    for row in q3:
        require(row['emb_item'] in ql and row['emb_gold']==ql[row['emb_item']]['text_gold'],'Q3 donor label differs')
        candidates=[x for x in q3 if x['fold']==row['fold'] and x['tile']!=row['tile'] and set(x['text_gold'])!=set(row['text_gold'])]
        require(row['emb_item']==min(candidates,key=lambda x:donor_rank(row,x))['id'],'Q3 deterministic donor differs')
    donor=sum(jaccard(x['parsed'],x['emb_gold']) for x in q3)/len(q3)
    source=sum(jaccard(x['parsed'],x['text_gold']) for x in q3)/len(q3)
    require(agrees(donor,scores['q3_secondary']['jaccard_vs_donor']) and agrees(source,scores['q3_secondary']['jaccard_vs_source']),'Q3 score differs')
    report['q3_secondary']={'n':len(q3),'donor_jaccard':donor,'source_jaccard':source,'donor_minus_source':donor-source,
        'donor_label_identity_verified':True,'interpretation':'No positive descriptive donor-following localization result; this is not itself evidence of source-text following, because class/spatial priors and mismatch remain uncontrolled.'}
    # Stored reproduction is a prerequisite, explicitly not independently recomputed.
    validity=scores['validity']
    require(validity['reproduction_rate']==1.0 and validity['reproduction_n']==384 and validity['max_parse_fail']==0,'Stored validity record differs')
    require(report['arms']['real']['d']>=.10 and report['arms']['swap_within_tile']['d']>=.10
            and report['arms']['swap_within_tile']['ci95_registered_tile_bootstrap'][0]>0,'Registered primary effect rule failed')
    require(scores['verdict']=='reads_embedding','Stored verdict differs')
    sums=base/'SHA256SUMS';sums_sha=sha(sums)
    require(all(sha(base/name)==pin for pin,name in (line.split() for line in sums.read_text().splitlines())),'Artifact SHA256SUMS differs')
    report['source_hashes'][str(sums)]=sums_sha
    report['checks']={'registered_numeric_scores_and_intervals_match':True,'current_code_sha_matches_manifest':True,
        'all_published_artifact_shas_match':True,'parse_failure_counts_all_zero':True,
        'saved_real_reproduction_independently_recomputed':False,'raw_parser_and_deterministic_donors_match':True}
    report['stored_not_reverified_reproduction']={'reproduction_rate':1.0,'reproduction_n':384,
        'reason':'Original MS131 answers_seed1.jsonl was not part of this bounded local input set.'}
    report['verdict']='reads_embedding'
    report['limits']=[
        'Within/cross swaps change EO features together with embedded observation timestamps; zero removes both. These controls do not isolate pixel semantics from time metadata encoded in EO features. Current extractor supplies real_timestamps into single-acquisition encoding, but E0 lacks pinned cache/source-input hashes to reproduce historical bytes here.',
        '192 source tiles are186 Hiroshima and6 Indonesia. Registered intervals resample tiles, not independent regions/events; they are conditional intervals and do not establish new-region generalization.',
        'Cross-swap uses384 donor assignments to246 distinct donor QUESTION IDs, hence138 assignments beyond each donor ID first use. This is not246 donor tiles or246 repeated assignments. Source-tile bootstrap does not model donor reuse or region-level shared dependence.',
        'Zero input still traverses the learned projector and retains token/type/sequence structure; all-no on zero is not a trained text-only control nor evidence that date cues cannot be learned.',
        'Observed same-source prompt/donor feature mismatches are interventions, not coherent physical-change ground truth. Donor labels are source QA labels, not newly verified physical scene labels.',
        'Original saved MS131 answers_seed1.jsonl was not read; stored reproduction_n384/rate1 cannot be independently rederived in this bounded review.',
        'E0 manifest pins projector and code, but not prereg, input QA, LLM/tokenizer bytes or caches. Modern E5-style full immutable lineage was not recorded. No evidence of actual tampering is implied.',
        'This script and its additional region/donor/Q3 interpretation are retrospective, not newly preregistered analyses. No decision rule, output, sample, or label was changed.'
    ]
    for path,pin in report['source_hashes'].items():require(sha(path)==pin,'Input changed during review: '+path)
    require(sha(__file__)==own_sha,'Audit script changed during review')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=DEFAULT_REPO)
    parser.add_argument('--out',type=Path,default=Path('/private/tmp/E0_BOUNDED_INDEPENDENT_REVIEW_20260925.json'))
    args=parser.parse_args();report=audit(args.repo)
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    cross=report['arms']['swap_cross_tile']
    print(json.dumps({'out':str(args.out),'verdict':report['verdict'],'audit_code_sha256':report['audit_code_sha256'],
                      'cross_donor_question_ids':cross['unique_donor_question_ids'],'cross_assignments':cross['donor_assignments'],
                      'cross_repeated_assignments':cross['repeated_assignments_beyond_first_per_donor_question_id'],
                      'MS131_original_reproduction_rechecked':False},ensure_ascii=False))


if __name__=='__main__':main()
