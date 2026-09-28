#!/usr/bin/env python3
"""Independent result recomputation; reuses only the pinned data-only decoder.

Does not import production export/join code, use imagery, access a server, or
read model outputs. All matching, field, temporal and summary checks below are
separate implementations. The parser reuse and local-only scope are reported.
"""
import argparse
import ast
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

JOIN_SHA = '2f1be237c796ef58e75437c6714387385f2b3e8d55a9d2e1e59663a1bcabe141'
EXPORT_SHA = '79d5e9b6d963229fbb28056544851884bfc633e050a38294af93dc6ea12f384e'
UPSTREAM_SHA = 'ef8d42582292ba285e7911ddd9d022bc92ee014a77f0d4b6c994006a6388695f'
PARSER_SHA = 'e7db187d984b1c75c795b9b310b44b2c6b2724112e8df262d0b080882f8c54ef'
SLOTS = [('pre_1','SL2'),('pre_2','SL1'),('post','MS1')]
EXPECTED_SPLITS = {'train':4000,'validation':1000,'test':2000}


def check(ok, message):
    if not ok:raise AssertionError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load(path):
    def pairs(values):
        out={}
        for key,value in values:
            check(key not in out,'Duplicate JSON key');out[key]=value
        return out
    def bad(value):raise AssertionError('Nonfinite JSON token '+value)
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=bad)


def jsonl(path):
    text=Path(path).read_text().splitlines()
    check(all(text),'Blank JSONL record')
    return [json.loads(line) for line in text]


def equal(actual, expected, where):
    if isinstance(expected,dict):
        check(isinstance(actual,dict) and set(actual)==set(expected),where+' keys')
        for key in expected:equal(actual[key],expected[key],where+'/'+str(key))
    elif isinstance(expected,list):
        check(isinstance(actual,list) and len(actual)==len(expected),where+' list')
        for i,(a,b) in enumerate(zip(actual,expected)):equal(a,b,where+'/'+str(i))
    elif type(expected) in (int,float):
        check(type(actual) in (int,float) and math.isfinite(actual) and math.isfinite(expected)
              and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-10),where+' numeric')
    else:check(type(actual)==type(expected) and actual==expected,where+' value')


def polygon_corners(wkt):
    check(wkt.startswith('POLYGON ((') and wkt.endswith('))'),'Upstream rectangle WKT')
    points=[tuple(Decimal(v) for v in p.split()) for p in wkt[10:-2].split(',')]
    check(len(points)==5 and points[0]==points[-1] and all(len(p)==2 for p in points),'Closed rectangle')
    check(all(n.is_finite() for p in points for n in p),'Finite rectangle')
    vertices=set(points[:4]);xs={p[0] for p in vertices};ys={p[1] for p in vertices}
    check(len(xs)==len(ys)==2 and vertices=={(x,y) for x in xs for y in ys},'Four rectangle corners')
    check(all(sum(a!=b for a,b in zip(points[i],points[i+1]))==1 for i in range(4)),'Axis-aligned edges')
    return tuple(sorted(vertices))


def distribution(values):
    ordered=sorted(values);n=len(ordered)
    check(n>0,'Empty distribution')
    def quantile(p):
        position=(n-1)*p;low=int(position);fraction=position-low
        return ordered[low] if low==n-1 else ordered[low]*(1-fraction)+ordered[low+1]*fraction
    return {'n':n,'min':ordered[0],'p10':quantile(.1),'median':quantile(.5),'p90':quantile(.9),'max':ordered[-1],
            'mean':sum(ordered)/n,'zero_count':sum(v==0 for v in ordered),'negative_count':sum(v<0 for v in ordered)}


def review(repo):
    repo=Path(repo).resolve();base=repo/'artifacts'
    join=base/'t0_source_date_join_v0_20260925';export=base/'t0_metadata_export_v0_20260925'
    upstream=base/'kuro_original_metadata_pin_20260925'
    tracked={}
    def pin(path,expected=None):
        value=digest(path)
        if expected is not None:check(value==expected,'Hash mismatch: '+str(path))
        tracked[str(path)]=value
    for path,wanted in [(join/'manifest.json',JOIN_SHA),(export/'manifest.json',EXPORT_SHA),(upstream/'MANIFEST.json',UPSTREAM_SHA)]:pin(path,wanted)
    jm,em,um=load(join/'manifest.json'),load(export/'manifest.json'),load(upstream/'MANIFEST.json')
    check(jm['valid'] is em['valid'] is True,'Manifest validity')
    check(jm['input_export']==str(export) and jm['input_upstream_package']==str(upstream),'Input roots')
    for root,files in [(join,jm['files_sha256']),(export,em['files_sha256']),
                       (upstream,{k:v['sha256'] for k,v in um['files'].items()})]:
        for name,wanted in files.items():
            p=Path(name);check(not p.is_absolute() and '..' not in p.parts,'Manifest relative path')
            check((root/p).resolve().is_relative_to(root),'Manifest root containment')
            pin(root/p,wanted)
    pin(export/'status.json')
    status=load(export/'status.json')
    check(status['status']=='complete' and status['valid'] is True and status['manifest_sha256']==EXPORT_SHA,'Export completion')
    check(em['source_stat_before']==em['source_stat_after'],'Recorded container stat changed')
    input_audit=load(join/'input_audit.json')
    check(input_audit['export_status_sha256']==tracked[str(export/'status.json')],'Pinned export status')
    for key,wanted in [('export_manifest_sha256',EXPORT_SHA),('upstream_package_manifest_sha256',UPSTREAM_SHA)]:
        check(jm[key]==wanted,'Join parent pin')
    for field,filename in [('metadata_sha256','metadata.jsonl'),('raw_rows_sha256','raw_rows.jsonl'),('selected_columns_sha256','selected_columns.jsonl'),
                           ('cache_meta_sha256','cache_meta.jsonl'),('code_sha256','source.py')]:
        check(em[field]==tracked[str(export/filename)],'Export internal pin')
    check(em['top_metadata']['sha256']==tracked[str(export/'footer.parquet')]
          and em['top_metadata']['header_sha256']==tracked[str(export/'header18.bin')],'Footer/header pins')
    parser_path=upstream/'inspect_grid_metadata.py';pin(parser_path,PARSER_SHA)
    spec=importlib.util.spec_from_file_location('t0_pinned_data_only_decoder_for_independent_review',parser_path)
    parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    primary=parser.data_only(upstream/'KuroV2_grid_dict_test_0_100.gz')
    check(len(primary)==67490,'Primary metadata support')
    index=defaultdict(list)
    for key,record in primary.items():
        info=record['info']
        check(info['grid_id'].replace('-','')==key,'Upstream key/grid correspondence')
        check(record['path']==f"{info['actid']}/{info['aoiid']:02d}/{key}",'Upstream path correspondence')
        index[(info['actid'],info['aoiid'],polygon_corners(info['geom']))].append(key)
    contract=load(upstream/'published_geobench_slot_contract.json')
    check([contract['published_mapping'][k]['upstream_key'] for k in ['pre_event_1','pre_event_2','post_event']]==['SL2','SL1','MS1'],'Slot mapping')
    splits={}
    for node in ast.parse((upstream/'geobench2_kuro_generate_20260925.py').read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name not in ('train_acts','val_acts','test_acts'):continue
            split={'train_acts':'train','val_acts':'validation','test_acts':'test'}[name]
            for event in ast.literal_eval(node.value):
                check(event not in splits,'Overlapping event splits');splits[event]=split
    rows=jsonl(export/'metadata.jsonl');raw_rows=jsonl(export/'raw_rows.jsonl');joined=jsonl(join/'joined_rows.jsonl')
    check(len(rows)==len(raw_rows)==len(joined)==7000,'Current full coverage')
    check({r['source_id'] for r in rows}==set(range(7000)),'Source ID coverage')
    check(len({r['id'] for r in rows})==7000 and len({r['id'] for r in joined})==7000,'Unique current IDs')
    check(Counter(r['partition'] for r in rows)==EXPECTED_SPLITS,'Partition coverage')
    records=[];grid_ids=[];identity_checks=0;geometry_checks=0;field_checks=0;upstream_source_date_entries=0
    for n,(row,raw,record) in enumerate(zip(rows,raw_rows,joined)):
        identity={k:row[k] for k in ['table_row_index','id','source_id','partition','event_id','aoi_id']}
        check(identity['table_row_index']==raw['table_row_index']==n,'Table order')
        check(row['id']==f"ks_{row['source_id']:05d}",'ID formatting')
        check(row['source_row']==raw['source_row'] and row['source_numeric_vectors']==raw['source_numeric_vectors'],'Raw/normalized correspondence')
        check(all(record[k]==v for k,v in identity.items()),'Joined row identity')
        check(row['valid'] is True and not row['mismatches'] and row['acquisition_dates'] is None,'Normalized validity/unverified acquisition')
        check(record['status']=='matched_metadata' and record['original_export_valid'] is True and not record['reasons'],'Matched output state')
        source=row['source_row'];vectors=row['source_numeric_vectors']
        check(source['tortilla:id']==str(row['source_id']) and source['tortilla:data_split']==row['partition'],'Raw identity')
        check(splits[row['event_id']]==row['partition'],'Published event split')
        check(record['source_identity_raw']=={k:source[k] for k in ['tortilla:id','tortilla:data_split','actid','aoiid']},'Output raw identity')
        identity_checks+=1
        check(source['stac:crs']==row['crs']=='EPSG:3857','CRS')
        parsed_vectors={}
        for key in ['stac:geotransform','stac:raster_shape']:
            text=source[key].strip();check(text.startswith('[') and text.endswith(']'),'Raw numeric vector')
            parsed_vectors[key]=[Decimal(v) for v in text[1:-1].split()]
            check(parsed_vectors[key]==[Decimal(v) for v in vectors[key]],'Raw string/vector values')
        c,a,b,f,d,e=parsed_vectors['stac:geotransform']
        check([a,b,d,e]==[10,0,0,-10] and parsed_vectors['stac:raster_shape']==[224,224],'Raster grid contract')
        check([float(v) for v in [c,a,b,f,d,e]]==row['gdal_geotransform'] and row['raster_shape']==[224,224],'Normalized affine/shape')
        corners=tuple(sorted((c+a*x+b*y,f+d*x+e*y) for x,y in [(0,0),(224,0),(224,224),(0,224)]))
        supplied=row['corners_epsg3857'];check(len(supplied)==5 and supplied[0]==supplied[-1],'Closed source corner ring')
        check(tuple(sorted(tuple(Decimal(str(v)) for v in p) for p in supplied[:-1]))==corners,'Source four corners')
        check(tuple(tuple(Decimal(str(v)) for v in p) for p in record['corners_epsg3857'])==corners,'Joined four corners')
        bounds=[min(x for x,y in corners),min(y for x,y in corners),max(x for x,y in corners),max(y for x,y in corners)]
        check([Decimal(str(v)) for v in row['bbox_epsg3857']]==bounds,'Source bounds')
        candidates=index[(row['event_id'],row['aoi_id'],corners)]
        check(len(candidates)==record['candidate_count']==1,'Exactly one exact full-geometry candidate')
        info=primary[candidates[0]]['info'];grid=info['grid_id'];grid_ids.append(grid)
        check(record['grid_id']==grid and record['candidate_grid_ids']==[grid],'Selected grid')
        geometry_checks+=1
        expected_checks={}
        for key in ['actid','aoiid','pcovered','pwater','pflood']:
            check(float(source[key])==info[key],'Raw upstream numeric '+key)
            expected_checks[key]={'current':source[key],'upstream':info[key],'equal':True}
        check(source['flood_date']==info['flood_date'],'Raw upstream flood_date')
        expected_checks['flood_date']={'current':source['flood_date'],'upstream':info['flood_date'],'equal':True}
        for key,normalized in [('actid',row['event_id']),('aoiid',row['aoi_id']),('pwater',row['pwater']),('pflood',row['pflood'])]:
            check(normalized==info[key],'Normalized upstream '+key)
            expected_checks['normalized_'+key]={'current':normalized,'upstream':info[key],'equal':True}
        check(row['event_date']==info['flood_date'][:10],'Event date')
        expected_checks['event_date']={'current':row['event_date'],'upstream':info['flood_date'][:10],'equal':True}
        equal(record['field_checks'],expected_checks,'field_checks/'+row['id']);field_checks+=6
        source_dates={}
        for slot,key in SLOTS:
            text=info['sources'][key]['source_date'];check(re.fullmatch(r'\d{4}-\d{2}-\d{2}',text),'Day precision')
            source_dates[slot]=date.fromisoformat(text)
            ids=info['sources'][key]['s1_ids'];check(isinstance(ids,list) and ids and all(isinstance(v,str) and v for v in ids),'Source identifier candidates')
            upstream_source_date_entries+=1
        event=date.fromisoformat(source['flood_date'][:10])
        synthetic={key:event-timedelta(days=days) for key,days in [('pre_1',24),('pre_2',12),('post',0)]}
        p1,p2,post=[source_dates[s] for s,k in SLOTS]
        gaps={'pre1_to_pre2_days':(p2-p1).days,'pre2_to_post_days':(post-p2).days,'pre1_to_post_days':(post-p1).days,'event_to_post_days':(post-event).days}
        gaps.update({s+'_source_minus_synthetic_days':(source_dates[s]-synthetic[s]).days for s,k in SLOTS})
        temporal={'raw_sources':info['sources'],'published_slot_dates':{k:v.isoformat() for k,v in source_dates.items()},
                  'old_synthetic_date_only':{k:v.isoformat() for k,v in synthetic.items()},'intervals_days':gaps,
                  'strictly_increasing_published_dates':p1<p2<post,'source_date_precision':'day',
                  'historical_raster_to_published_slot_lineage':'not_verified'}
        equal(record['temporal'],temporal,'temporal/'+row['id'])
        records.append({'id':row['id'],'partition':row['partition'],'event_id':row['event_id'],'temporal':temporal})
    check(len(set(grid_ids))==7000,'Unique exact matched grids')
    groups={'all':records}
    groups.update({'split:'+s:[r for r in records if r['partition']==s] for s in sorted(EXPECTED_SPLITS)})
    groups.update({'event:'+str(e):[r for r in records if r['event_id']==e] for e in sorted({r['event_id'] for r in records})})
    recalculated={}
    for key,members in groups.items():
        recalculated[key]={'source_rows':len(members),'status_counts':{'matched_metadata':len(members)},
            'date_triplets':dict(Counter('|'.join(r['temporal']['published_slot_dates'][s] for s,k in SLOTS) for r in members)),
            'strict_chronological_count':sum(r['temporal']['strictly_increasing_published_dates'] for r in members),
            'interval_distributions':{gap:distribution([r['temporal']['intervals_days'][gap] for r in members]) for gap in sorted(gaps)}}
    summary={'valid':True,'global_errors':[],'duplicate_current_grid_groups':[],'summary':recalculated}
    equal(load(join/'summary.json'),summary,'complete_summary')
    for path,value in tracked.items():check(digest(path)==value,'Input changed during independent review')
    headline={key:recalculated[key]['interval_distributions']['pre2_to_post_days'] for key in ['all','split:train','split:validation','split:test']}
    return {'schema':'t0-independent-result-review-v0','consistent':True,'created_at':datetime.now(timezone.utc).isoformat(),
        'n_rows':7000,'n_upstream_records':67490,'n_unique_grid_matches':7000,'identity_rows_checked':identity_checks,
        'four_corner_matches_checked':geometry_checks,'raw_six_field_comparisons':field_checks,'published_slot_dates_checked':upstream_source_date_entries,
        'n_event_groups':len(groups)-4,'n_summary_groups':len(groups),'split_counts':EXPECTED_SPLITS,
        'all_strictly_chronological_count':recalculated['all']['strict_chronological_count'],
        'pre2_to_post_days':headline,'verified_full_summary':recalculated,
        'source_hashes_verified':tracked,'audit_code_sha256':digest(__file__),
        'production_join_or_export_imported':False,
        'decoder_reuse':{'file':str(parser_path),'sha256':PARSER_SHA,'function':'data_only',
            'scope':'Reused only the pinned non-executing pickle-opcode decoder; metadata matching/date/summary logic is independently implemented.'},
        'limits':['This rechecks pinned local metadata and report bytes, not the remote 7.5GB archive or SAR pixels.',
            'Parquet footer/header bytes are hash-checked but not independently decoded; raw JSON rows are the source for this normalization cross-check.',
            'Day-level original source_date plus published slot mapping remain acquisition candidates; historical raster-slot lineage and Sentinel product UTC acquisition timestamps are unverified.',
            'All/split/event interval summaries weight tiles within each group; 7000 tiles are not 7000 independent events.',
            'Chronological source metadata does not establish pre-event absence, change onset, causal impact, temporal reasoning, or memory benefit.',
            'No E5 outputs were read; no model, training, labels, prompts, inputs, or thresholds were changed.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    check(not args.out.exists(),'No overwrite of an existing review')
    try:result=review(args.repo)
    except Exception as error:
        result={'schema':'t0-independent-result-review-v0','consistent':False,'error':str(error),'audit_code_sha256':digest(__file__)}
        args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        raise
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['consistent','n_rows','n_unique_grid_matches','n_event_groups','pre2_to_post_days']},indent=2))
