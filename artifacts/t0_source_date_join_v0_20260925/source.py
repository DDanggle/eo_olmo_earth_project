#!/usr/bin/env python3
"""Frozen T0 metadata audit. Exact joins only; no imagery/model/E5 access."""
import argparse
import ast
import collections
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import shutil
import sys

PIN_MANIFEST_SHA = 'ef8d42582292ba285e7911ddd9d022bc92ee014a77f0d4b6c994006a6388695f'
PRIMARY = 'KuroV2_grid_dict_test_0_100.gz'
SECONDARY = 'KuroV2_grid_dict.gz'
EXPECTED_SPLITS = {'train': 4000, 'validation': 1000, 'test': 2000}
SLOT_MAP = {'pre_1': 'SL2', 'pre_2': 'SL1', 'post': 'MS1'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def clean(x):
    if isinstance(x, float) and not math.isfinite(x): return {'upstream_nonfinite_float': repr(x)}
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    return x


def write_json(path, obj):
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def same_record(a, b):
    return json.dumps(clean(a), sort_keys=True, allow_nan=False) == json.dumps(clean(b), sort_keys=True, allow_nan=False)


def rectangle(wkt):
    m = re.fullmatch(r'POLYGON\s*\(\(([^()]+)\)\)', wkt)
    if not m: raise ValueError('single polygon ring required')
    pts = [tuple(float(n) for n in p.strip().split()) for p in m[1].split(',')]
    if len(pts) != 5 or pts[0] != pts[-1] or any(len(p) != 2 or not all(math.isfinite(n) for n in p) for p in pts):
        raise ValueError('closed finite four-corner ring required')
    xs, ys = {p[0] for p in pts}, {p[1] for p in pts}
    if len(xs) != 2 or len(ys) != 2 or set(pts[:-1]) != {(x, y) for x in xs for y in ys}:
        raise ValueError('axis aligned rectangle required')
    if any((pts[i][0] == pts[i+1][0]) == (pts[i][1] == pts[i+1][1]) for i in range(4)):
        raise ValueError('crossed or degenerate edge')
    return tuple(sorted(pts[:-1]))


def current_corners(row):
    if row['crs'] != 'EPSG:3857' or row['raster_shape'] != [224, 224]: raise ValueError('source CRS/shape')
    c, a, b, f, d, e = row['gdal_geotransform']
    if [a, b, d, e] != [10, 0, 0, -10] or not all(math.isfinite(v) for v in [a,b,c,d,e,f]):
        raise ValueError('source affine')
    pts = tuple(sorted((a*x+b*y+c, d*x+e*y+f) for x,y in [(0,0),(224,0),(224,224),(0,224)]))
    supplied = row['corners_epsg3857']
    if len(supplied) == 5:
        if supplied[0] != supplied[-1]: raise ValueError('source ring not closed')
        supplied = supplied[:-1]
    if len(supplied) != 4 or tuple(sorted(tuple(p) for p in supplied)) != pts: raise ValueError('source corner mismatch')
    vectors = row.get('source_numeric_vectors', {})
    if 'stac:geotransform' in vectors and [float(v) for v in vectors['stac:geotransform']] != row['gdal_geotransform']: raise ValueError('raw vector/affine mismatch')
    if 'stac:raster_shape' in vectors and [float(v) for v in vectors['stac:raster_shape']] != row['raster_shape']: raise ValueError('raw vector/shape mismatch')
    bb = [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]
    if row['bbox_epsg3857'] != bb: raise ValueError('source bounds mismatch')
    return pts


def build_index(upstream):
    index = collections.defaultdict(list)
    represented = {}
    for key, record in upstream.items():
        info = record['info']; grid = info['grid_id']
        if key != grid.replace('-', ''): raise ValueError('upstream key/grid mismatch')
        if record['path'] != f"{info['actid']}/{info['aoiid']:02d}/{key}": raise ValueError('upstream path mismatch')
        if grid in represented and not same_record(record, represented[grid]): raise ValueError('conflicting duplicate grid representation')
        represented[grid] = record
        index[(info['actid'], info['aoiid'], rectangle(info['geom']))].append(record)
    return index


def parse_day(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value): raise ValueError('invalid date precision')
    return dt.date.fromisoformat(value)


def temporal_info(info, source_row):
    raw = info['sources']
    if not all(k in raw for k in SLOT_MAP.values()): raise ValueError('missing SL1/SL2/MS1')
    actual = {slot: parse_day(raw[k]['source_date']) for slot,k in SLOT_MAP.items()}
    for key in SLOT_MAP.values():
        if not isinstance(raw[key]['s1_ids'], list) or not raw[key]['s1_ids'] or not all(isinstance(s, str) and s for s in raw[key]['s1_ids']):
            raise ValueError('missing source UUID list')
    event = dt.datetime.fromisoformat(source_row['flood_date']).date()
    old = {'pre_1': event-dt.timedelta(days=24), 'pre_2': event-dt.timedelta(days=12), 'post': event}
    intervals = {'pre1_to_pre2_days': (actual['pre_2']-actual['pre_1']).days,
                 'pre2_to_post_days': (actual['post']-actual['pre_2']).days,
                 'pre1_to_post_days': (actual['post']-actual['pre_1']).days,
                 'event_to_post_days': (actual['post']-event).days}
    intervals.update({f'{slot}_source_minus_synthetic_days': (actual[slot]-old[slot]).days for slot in actual})
    return {'raw_sources': raw, 'published_slot_dates': {k:v.isoformat() for k,v in actual.items()},
            'old_synthetic_date_only': {k:v.isoformat() for k,v in old.items()}, 'intervals_days': intervals,
            'strictly_increasing_published_dates': actual['pre_1'] < actual['pre_2'] < actual['post'],
            'source_date_precision': 'day', 'historical_raster_to_published_slot_lineage': 'not_verified'}


def compare_fields(row, info):
    raw = row['source_row']; checks = {}
    for key in ['actid', 'aoiid', 'pcovered', 'pwater', 'pflood']:
        a, b = float(raw[key]), info[key]
        checks[key] = {'current': raw[key], 'upstream': b, 'equal': math.isfinite(a) and math.isfinite(b) and a == b}
    checks['flood_date'] = {'current': raw['flood_date'], 'upstream': info['flood_date'], 'equal': raw['flood_date'] == info['flood_date']}
    for key,current in [('actid',row['event_id']), ('aoiid',row['aoi_id']), ('pwater',row['pwater']), ('pflood',row['pflood'])]:
        checks['normalized_'+key] = {'current': current, 'upstream': info[key], 'equal': current == info[key]}
    checks['event_date'] = {'current': row['event_date'], 'upstream': info['flood_date'][:10], 'equal': row['event_date'] == info['flood_date'][:10]}
    return checks


def summary_values(values):
    x = sorted(values)
    def q(p):
        k=(len(x)-1)*p; a=math.floor(k); b=math.ceil(k)
        return x[a] + (x[b]-x[a])*(k-a)
    return {'n':len(x), 'min':x[0], 'p10':q(.1), 'median':statistics.median(x), 'p90':q(.9), 'max':x[-1],
            'mean':statistics.mean(x), 'zero_count':x.count(0), 'negative_count':sum(v<0 for v in x)} if x else {'n':0}


def aggregate(rows):
    groups = {'all': rows}
    for key in sorted({r.get('partition') for r in rows if isinstance(r.get('partition'),str)}):
        groups['split:'+key] = [r for r in rows if r.get('partition')==key]
    for key in sorted({r.get('event_id') for r in rows if isinstance(r.get('event_id'),int)}):
        groups['event:'+str(key)] = [r for r in rows if r.get('event_id')==key]
    out={}
    for key,group in groups.items():
        matched=[r for r in group if r['status']=='matched_metadata']; interval_keys=set().union(*(r['temporal']['intervals_days'] for r in matched)) if matched else set()
        out[key]={'source_rows':len(group), 'status_counts':dict(collections.Counter(r['status'] for r in group)),
                  'date_triplets':dict(collections.Counter('|'.join(r['temporal']['published_slot_dates'][k] for k in SLOT_MAP) for r in matched)),
                  'strict_chronological_count':sum(r['temporal']['strictly_increasing_published_dates'] for r in matched),
                  'interval_distributions':{k:summary_values([r['temporal']['intervals_days'][k] for r in matched]) for k in sorted(interval_keys)}}
    return out


def audit_rows(rows, upstream, expected_count=7000, expected_splits=None, published_splits=None):
    expected_splits = EXPECTED_SPLITS if expected_splits is None else expected_splits
    global_errors=[]; ids=[r.get('id') for r in rows]; source_ids=[r.get('source_id') for r in rows]
    if len(rows)!=expected_count: global_errors.append('row_count')
    if len(set(ids))!=len(ids): global_errors.append('duplicate_current_id')
    if len(set(source_ids))!=len(source_ids) or set(source_ids)!=set(range(expected_count)): global_errors.append('source_id_coverage')
    if dict(collections.Counter(r.get('partition') for r in rows))!=expected_splits: global_errors.append('split_counts')
    index=build_index(upstream); output=[]
    for row in rows:
        entry={k:row.get(k) for k in ['table_row_index','id','source_id','partition','event_id','aoi_id']};entry['source_identity_raw']={k:row.get('source_row',{}).get(k) for k in ['tortilla:id','tortilla:data_split','actid','aoiid']};entry.update(status='invalid_source', reasons=[], original_export_valid=row.get('valid'), original_export_mismatches=row.get('mismatches',[]))
        output.append(entry)
        try:
            if row.get('valid') is not True: entry['reasons'].append('export_row_invalid'); continue
            if row['id']!=f"ks_{row['source_id']:05d}": raise ValueError('ID contract')
            if str(row['source_id']) != row['source_row']['tortilla:id']: raise ValueError('raw ID contract')
            if row['partition'] != row['source_row']['tortilla:data_split']: raise ValueError('raw split contract')
            if published_splits is not None and published_splits.get(row['event_id']) != row['partition']: raise ValueError('published event split mismatch')
            points=current_corners(row); candidates=index.get((row['event_id'],row['aoi_id'],points),[])
            entry.update(corners_epsg3857=points, candidate_count=len(candidates), candidate_grid_ids=[v['info']['grid_id'] for v in candidates])
            if not candidates: entry['status']='no_exact_match'; continue
            if len(candidates)!=1: entry['status']='ambiguous_exact_match'; continue
            info=candidates[0]['info']; entry.update(grid_id=info['grid_id'], field_checks=compare_fields(row,info))
            if not all(c['equal'] for c in entry['field_checks'].values()): entry['status']='metadata_field_mismatch'; continue
            try: entry['temporal']=temporal_info(info,row['source_row'])
            except (ValueError,KeyError,TypeError) as e: entry['status']='invalid_upstream_dates';entry['reasons'].append(str(e));continue
            entry['status']='matched_metadata'
        except (ValueError,KeyError,TypeError,IndexError) as e: entry['reasons'].append(str(e))
    grid_to_rows=collections.defaultdict(list)
    for r in output:
        if r['status']=='matched_metadata': grid_to_rows[r['grid_id']].append(r)
    duplicate_groups=[]
    for grid,members in grid_to_rows.items():
        if len(members)>1:
            duplicate_groups.append({'grid_id':grid,'source_ids':[r['id'] for r in members],'temporal_representations_equal':all(same_record(r['temporal'],members[0]['temporal']) for r in members)})
            for r in members:r['status']='duplicate_current_grid';r['reasons'].append('multiple_current_ids_share_exact_grid')
    valid = not global_errors and all(r['status']=='matched_metadata' for r in output)
    return {'valid':valid,'global_errors':global_errors,'rows':output,'duplicate_current_grid_groups':duplicate_groups,'summary':aggregate(output)}


def safe_relative(base, name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts: raise ValueError('nonrelative manifest path')
    target=(base/p).resolve()
    if not target.is_relative_to(base.resolve()): raise ValueError('manifest path escapes root')
    return target


def read_json(path):
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=lambda c: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def read_jsonl(path):
    rows=[]
    for line in path.read_text().splitlines():
        if not line.strip(): raise ValueError('blank JSONL line')
        rows.append(json.loads(line, parse_constant=lambda c: (_ for _ in ()).throw(ValueError('nonfinite JSON'))))
    return rows


def stat_signature(path):
    s=path.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def verify_files(base, files):
    result={}
    for name, digest in files.items():
        path=safe_relative(base,name)
        if sha(path)!=digest: raise ValueError('SHA mismatch: '+str(path))
        result[name]={'sha256':digest,'stat':stat_signature(path),'absolute_path':str(path)}
    return result


def load_inputs(export, upstream, export_manifest_sha256):
    if sha(upstream/'MANIFEST.json')!=PIN_MANIFEST_SHA: raise ValueError('frozen upstream package manifest differs')
    pin=read_json(upstream/'MANIFEST.json')
    upstream_files={k:v['sha256'] for k,v in pin['files'].items()}
    up_audit=verify_files(upstream,upstream_files)
    if sha(export/'manifest.json')!=export_manifest_sha256: raise ValueError('export manifest SHA differs')
    manifest=read_json(export/'manifest.json'); status_sha=sha(export/'status.json'); status=read_json(export/'status.json')
    if manifest['schema']!='kuro-t0-top-metadata-manifest-v0': raise ValueError('export schema differs')
    if status['status'] not in ['complete','invalid']: raise ValueError('unfinished export')
    if status['status']=='complete' and status.get('manifest_sha256')!=export_manifest_sha256: raise ValueError('status/manifest SHA differs')
    exp_audit=verify_files(export,manifest['files_sha256'])
    for key,name in [('metadata_sha256','metadata.jsonl'),('raw_rows_sha256','raw_rows.jsonl'),('selected_columns_sha256','selected_columns.jsonl'),('cache_meta_sha256','cache_meta.jsonl'),('code_sha256','source.py')]:
        if manifest[key]!=exp_audit[name]['sha256']: raise ValueError('export internal hash disagreement: '+key)
    if manifest['top_metadata']['sha256']!=exp_audit['footer.parquet']['sha256'] or manifest['top_metadata']['header_sha256']!=exp_audit['header18.bin']['sha256']: raise ValueError('header/footer internal hash disagreement')
    if manifest['source_stat_before']!=manifest['source_stat_after']: raise ValueError('source changed during export')
    rows=read_jsonl(export/'metadata.jsonl');raw=read_jsonl(export/'raw_rows.jsonl')
    if len(raw)!=len(rows): raise ValueError('raw/normalized row count mismatch')
    for i,(a,b) in enumerate(zip(rows,raw)):
        if a['table_row_index']!=i or b['table_row_index']!=i or a['source_row']!=b['source_row'] or a['source_numeric_vectors']!=b['source_numeric_vectors']: raise ValueError('raw/normalized correspondence mismatch')
    contract=read_json(upstream/'published_geobench_slot_contract.json')
    actual_map={k:contract['published_mapping'][k]['upstream_key'] for k in ['pre_event_1','pre_event_2','post_event']}
    if actual_map!={'pre_event_1':'SL2','pre_event_2':'SL1','post_event':'MS1'}: raise ValueError('published mapping contract differs')
    tree=ast.parse((upstream/'geobench2_kuro_generate_20260925.py').read_text());splits={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['train_acts','val_acts','test_acts']:
            part={'train_acts':'train','val_acts':'validation','test_acts':'test'}[node.targets[0].id]
            for event in ast.literal_eval(node.value):
                if event in splits: raise ValueError('overlapping published event splits')
                splits[event]=part
    if not splits: raise ValueError('missing published event splits')
    spec=importlib.util.spec_from_file_location('t0_fixed_data_parser',upstream/'inspect_grid_metadata.py')
    parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    primary=parser.data_only(upstream/PRIMARY)
    if len(primary)!=67490: raise ValueError('primary upstream support differs')
    secondary=parser.data_only(upstream/SECONDARY)
    if len(secondary)!=31707: raise ValueError('secondary upstream support differs')
    crosschecks={}
    for tile,key,expected_secondary in [('ks_06770','e2f98639b3ca5c438d3b79c11b55bc5c',True),('ks_05265','d762ee26ef585723b9f7d9d18f419bd8',False)]:
        if key not in primary or (key in secondary)!=expected_secondary: raise ValueError('fixed two-case metadata coverage changed')
        equal=same_record(primary[key],secondary[key]) if key in secondary else None
        if equal is False: raise ValueError('two-source grid metadata conflicts')
        crosschecks[tile]={'grid_hex':key,'primary_present':True,'secondary_present':key in secondary,'full_representations_equal':equal}
    del secondary
    return rows,primary,splits,{'export_manifest_sha256':export_manifest_sha256,'upstream_manifest_sha256':PIN_MANIFEST_SHA,
            'export_valid':manifest['valid'],'export_status':status['status'],'export_status_valid':status.get('valid'),'export_status_sha256':status_sha,'export_files':exp_audit,'upstream_files':up_audit,
            'fixed_two_case_crosscheck':crosschecks,'primary_records':67490,'secondary_records':31707,
            'published_slot_map':SLOT_MAP,'no_nearest_neighbor_or_other_metadata_fallback':True}


def run(export, upstream, out, export_manifest_sha256):
    export,upstream,out=map(lambda p:Path(p).resolve(),[export,upstream,out])
    if out.exists(): raise FileExistsError('refuse existing output')
    source_sha=sha(Path(__file__))
    rows,primary,splits,audit=load_inputs(export,upstream,export_manifest_sha256)
    result=audit_rows(rows,primary,published_splits=splits)
    if audit['export_valid'] is not True: result['valid']=False;result['global_errors'].append('source_export_invalid')
    if audit['export_status']!='complete' or audit['export_status_valid'] is not True: result['valid']=False;result['global_errors'].append('source_export_not_complete')
    # Full source representations stay in pinned inputs; every current ID gets an output row.
    for kind in ['export_files','upstream_files']:
        for item in audit[kind].values():
            p=Path(item['absolute_path'])
            if stat_signature(p)!=item['stat'] or sha(p)!=item['sha256']: raise ValueError('input changed while joining')
    if sha(export/'status.json')!=audit['export_status_sha256']: raise ValueError('export status changed while joining')
    if sha(export/'manifest.json')!=export_manifest_sha256 or sha(upstream/'MANIFEST.json')!=PIN_MANIFEST_SHA or sha(Path(__file__))!=source_sha: raise ValueError('manifest/code changed while joining')
    out.mkdir(parents=True)
    shutil.copyfile(Path(__file__),out/'source.py')
    with (out/'joined_rows.jsonl').open('x') as f:
        for row in result['rows']:f.write(json.dumps(clean(row),ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
    write_json(out/'summary.json',{k:v for k,v in result.items() if k!='rows'})
    write_json(out/'input_audit.json',audit)
    matched=sum(r['status']=='matched_metadata' for r in result['rows'])
    report=(f'# T0 원 날짜 metadata 감사\n\n검사 행 {len(rows)}개, 정확 metadata 조인 {matched}개. 전체 계약 통과: {result["valid"]}.\n\n'
            '이는 원저자 source_date와 공개 GEO-Bench 슬롯 계약의 조인이다. 현재 tortilla 생성의 exact revision, 원 영상 값 대응, Sentinel 제품 UTC 취득시각을 검증한 것은 아니다.\n\n'
            '모든 원 행을 보존하고 event/AOI/전체사각형이 정확히 같은 유일 후보에만 날짜를 연결했다. 원 필드가 다른 후보, 중복·누락·invalid는 summary와 joined_rows에 남긴다. 최근접·중심점·비율 기반 fallback은 없다.\n\n'
            '날짜는 day precision이며 source-minus-synthetic은 사건일−24/−12/0의 날짜와 비교한다. gap 분포는 개별 타일 가중이며 split/event별 값을 같이 제공한다. 경험적 분위수는 (n−1)p 선형 보간이다.\n\n'
            '시간순 flag는 진단값이며 pre 영상의 무침수나 변화시점·인과성을 입증하지 않는다. 모델/embedding/E5 결과를 읽거나 기존 입력·라벨·실험을 수정하지 않았다.\n')
    (out/'REPORT.md').write_text(report)
    files={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    manifest={'schema':'kuro-t0-exact-source-date-join-v0','created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'valid':result['valid'],
              'input_export':str(export),'input_upstream_package':str(upstream),'export_manifest_sha256':export_manifest_sha256,
              'upstream_package_manifest_sha256':PIN_MANIFEST_SHA,'files_sha256':files,'expected_rows':7000,'expected_splits':EXPECTED_SPLITS,
              'claims':'Exact upstream grid metadata and published slot contract; historical pixel lineage unverified.'}
    write_json(out/'manifest.json',manifest)
    return {'valid':result['valid'],'rows':len(rows),'matched':matched,'global_errors':result['global_errors'],'manifest_sha256':sha(out/'manifest.json')}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ['export','upstream','out']:ap.add_argument('--'+arg,type=Path,required=True)
    ap.add_argument('--export-manifest-sha256',required=True)
    args=ap.parse_args()
    try:r=run(args.export,args.upstream,args.out,args.export_manifest_sha256)
    except Exception as exc: print(f'T0 join refused: {type(exc).__name__}: {exc}',file=sys.stderr);return 1
    print(json.dumps(r,indent=2));return 0 if r['valid'] else 2

if __name__=='__main__': raise SystemExit(main())
