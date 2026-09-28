#!/usr/bin/env python3
"""Build a new v8 snapshot with audited day-level source-date candidates.

No server calls, model execution, E5 result reads, or input edits. Existing
catalog fields, reader records, source images/manifests and meta.json stay intact.
"""
import argparse
import copy
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

V7_SHA='4484210df09ca8d84046602dca3361476115cdf785de337568b4c33038ae9ce0'
T0_SHA='2f1be237c796ef58e75437c6714387385f2b3e8d55a9d2e1e59663a1bcabe141'
AUDIT_SHA='9f1ac9eb5f396e7b1357e6497181d0f1815d90735adef7f5c6842673e9e3f78f'
UPSTREAM_SHA='ef8d42582292ba285e7911ddd9d022bc92ee014a77f0d4b6c994006a6388695f'
E5_STATUS_AS_OF='2026-09-25T06:14:07Z'
SLOTS={'pre_1':'SL2','pre_2':'SL1','post':'MS1'}
CHANGED={'catalog.json','research_runs.json','research_sources.json','index.html'}
V7_EXTRA={'v7_build_manifest.json','browser_checks_v7.json','disk_api_readback_checks_v7.json','http_readback_checks_v7.json'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    def unique(entries):
        result={}
        for key,value in entries:
            require(key not in result,'Duplicate JSON key');result[key]=value
        return result
    def invalid(value):raise ValueError('Invalid JSON constant '+value)
    return json.loads(Path(path).read_text(),object_pairs_hook=unique,parse_constant=invalid)


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def child(root,name):
    p=Path(name)
    require(not p.is_absolute() and '..' not in p.parts,'Unsafe source path')
    target=root/p
    require(target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(root),'Source outside snapshot')
    return target


def pin(path,expected,tracking):
    require(sha(path)==expected,'Pinned source hash mismatch: '+str(path))
    tracking[str(path)]=expected


def provenance_for(catalog_row,joined,pins):
    require(catalog_row['id']==joined['id']==f"ks_{joined['source_id']:05d}",'Catalog/T0 ID mismatch')
    require(catalog_row['dataset']=='kurosiwo' and str(joined['event_id'])==catalog_row['aoi_id']
            and int(catalog_row['region_id'])==joined['aoi_id'] and catalog_row['split']==joined['partition'],'Catalog/T0 region or split mismatch')
    require(joined['status']=='matched_metadata' and joined['original_export_valid'] is True and not joined['reasons']
            and joined['candidate_count']==1 and joined['candidate_grid_ids']==[joined['grid_id']],'T0 row is not uniquely matched')
    checks=joined['field_checks']
    require(all(c.get('equal') is True for c in checks.values()),'T0 source field disagreement')
    for key in ['actid','aoiid','pcovered','pwater','pflood','flood_date']:
        require(key in checks and checks[key]['equal'] is True,'Missing source-field check')
    require(checks['event_date']['current']==checks['event_date']['upstream']==catalog_row['event_date'],'Event date mismatch')
    require(checks['pflood']['upstream']==catalog_row['flood_fraction_pct']
            and checks['pwater']['upstream']==catalog_row['permanent_water_fraction_pct'],'Reference fraction mismatch')
    temporal=joined['temporal']
    require(temporal['source_date_precision']=='day'
            and temporal['historical_raster_to_published_slot_lineage']=='not_verified','False acquisition/lineage provenance')
    require(set(temporal['published_slot_dates'])==set(SLOTS) and set(temporal['old_synthetic_date_only'])==set(SLOTS),'Date slot support differs')
    published={}
    for slot,key in SLOTS.items():
        value=temporal['published_slot_dates'][slot]
        require(isinstance(value,str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}',value),'Day-level date required')
        published[slot]=date.fromisoformat(value)
        require(value==temporal['raw_sources'][key]['source_date'],'Published/raw source date mismatch')
        ids=temporal['raw_sources'][key]['s1_ids']
        require(isinstance(ids,list) and ids and all(isinstance(v,str) and v for v in ids),'Missing published source identifiers')
    event=date.fromisoformat(catalog_row['event_date'])
    assumed={s:(event-timedelta(days=n)).isoformat() for s,n in [('pre_1',24),('pre_2',12),('post',0)]}
    require(assumed==temporal['old_synthetic_date_only'],'Original synthetic-date assumptions differ')
    a,b,c=[published[s] for s in SLOTS]
    gaps={'pre1_to_pre2_days':(b-a).days,'pre2_to_post_days':(c-b).days,'pre1_to_post_days':(c-a).days,'event_to_post_days':(c-event).days}
    gaps.update({s+'_source_minus_synthetic_days':(published[s]-date.fromisoformat(assumed[s])).days for s in SLOTS})
    require(gaps==temporal['intervals_days'] and temporal['strictly_increasing_published_dates'] is True and a<b<c,'Published date interval disagreement')
    require(set(pins)=={'join_manifest_sha256','joined_rows_sha256','independent_review_sha256','upstream_package_manifest_sha256'}
            and all(re.fullmatch('[0-9a-f]{64}',v) for v in pins.values()),'Incomplete source pins')
    return {'schema':'eo-source-date-provenance-v1','status':'matched_published_metadata',
            'source_role':'original_author_day_metadata_plus_published_slot_mapping',
            'event_date':catalog_row['event_date'],'published_slot_dates':copy.deepcopy(temporal['published_slot_dates']),
            'published_slot_mapping':dict(SLOTS),'intervals_days':gaps,'model_assumed_slot_dates':assumed,
            'date_precision':'day','sensor_utc_acquisition_timestamps':{s:None for s in SLOTS},
            'actual_acquisition_verified':False,'historical_raster_to_published_slot_lineage':'not_verified',
            'match_method':'unique_exact_event_aoi_four_corners_and_source_fields',
            'source_grid_id':joined['grid_id'],'source_row_index':joined['table_row_index'],
            'published_source_identifiers':{s:copy.deepcopy(temporal['raw_sources'][k]['s1_ids']) for s,k in SLOTS.items()},
            'source_title':'Kuro 원저자 날짜 메타데이터 · GEO-Bench 공개 슬롯 매핑 · T0 독립 검산',
            'source_metadata_file':'KuroV2_grid_dict_test_0_100.gz','source_hashes':dict(pins),
            'limitations':['published_day_metadata_not_verified_sensor_utc_acquisition',
                'historical_raster_to_published_slot_lineage_unverified','existing_model_synthetic_dates_unchanged',
                'date_gap_is_not_change_duration_or_damage','search_time_filter_remains_event_date']}


def attach_dates(catalog,joined,pins,expected_count=7000):
    require(catalog['schema']=='eo-evidence-catalog-v0.2','Catalog schema differs')
    require(len({r['id'] for r in catalog['records']})==len(catalog['records']),'Duplicate catalog IDs')
    require(len(joined)==expected_count and len({r['id'] for r in joined})==expected_count,'T0 coverage/ID duplicates')
    kuro=[r for r in catalog['records'] if r['dataset']=='kurosiwo']
    require(len(kuro)==expected_count and {r['id'] for r in kuro}=={r['id'] for r in joined},'Catalog/T0 full coverage mismatch')
    require(all('source_date_provenance' not in r for r in catalog['records']),'Provenance already exists; no replacement')
    output=copy.deepcopy(catalog);by_id={r['id']:r for r in joined}
    for row in output['records']:
        if row['dataset']=='kurosiwo':row['source_date_provenance']=provenance_for(row,by_id[row['id']],pins)
    restored=copy.deepcopy(output)
    for row in restored['records']:row.pop('source_date_provenance',None)
    require(restored==catalog,'Existing catalog fields changed')
    return output


DATE_UI=r'''
function sourceDateDetail(r){
const d=r.source_date_provenance;if(!d)return '';
if(d.status!=='matched_published_metadata'||d.actual_acquisition_verified!==false||d.historical_raster_to_published_slot_lineage!=='not_verified')return '<p class="single-note">날짜 출처 상태를 확인하지 못했습니다. 실제 취득일은 미확인입니다.</p>';
let html='<section class="reader-case source-dates" aria-label="관측 날짜의 출처"><h3>관측 날짜 · 문서 기록과 모델 가정</h3><p class="caption">사건일 '+esc(d.event_date)+' · 검색 기간은 이 사건일 기준입니다.</p><div class="research-table-wrap"><table class="research-table"><caption>공개 메타데이터의 날짜 후보이며, 실제 센서 취득시각으로 확정한 값은 아닙니다.</caption><thead><tr><th scope="col">관측 슬롯</th><th scope="col">문서에 적힌 날짜</th><th scope="col">기존 모델 가정 날짜</th></tr></thead><tbody>';
for(const [slot,title] of [['pre_1','이전 관측 1'],['pre_2','이전 관측 2'],['post','이후 관측']])html+='<tr><th scope="row">'+title+'</th><td>'+esc(d.published_slot_dates[slot])+'</td><td>'+esc(d.model_assumed_slot_dates[slot])+'</td></tr>';
html+='</tbody></table></div><p class="caption">문서상 간격: 이전 1 → 이전 2 '+esc(d.intervals_days.pre1_to_pre2_days)+'일 · 이전 2 → 이후 '+esc(d.intervals_days.pre2_to_post_days)+'일 · 사건일 → 이후 '+esc(d.intervals_days.event_to_post_days)+'일</p><p class="single-note">실제 UTC 취득시각과 이 영상 파일의 과거 슬롯 대응은 아직 검증되지 않았습니다. 날짜 간격은 변화가 지속된 기간이나 피해량이 아닙니다.</p><p class="caption">모델 가정일은 사건일−24일 / −12일 / 사건일입니다. 아래 저장 판독의 입력 날짜와 답변은 그대로 유지됩니다.</p><details><summary>날짜 기록의 출처 보기</summary><p class="caption">Kuro 원저자 격자 메타데이터와 GEO-Bench 공개 슬롯 매핑을 연결하고 T0에서 독립 검산했습니다.</p><pre>'+esc(JSON.stringify({grid_id:d.source_grid_id,published_mapping:d.published_slot_mapping,source_file:d.source_metadata_file,published_source_identifiers:d.published_source_identifiers,source_hashes:d.source_hashes,unverified:['sensor UTC acquisition','historical raster/slot lineage']},null,2))+'</pre></details></section>';
return html;
}
'''


def patch_html(html):
    require(html.count('function select(r){')==1 and html.count('if(r.evidence?.length)html+=')==1,'Unexpected v7 UI anchors')
    require('function sourceDateDetail(' not in html,'Date UI already attached')
    return html.replace('function select(r){',DATE_UI+'\nfunction select(r){').replace(
        'if(r.evidence?.length)html+=','html+=sourceDateDetail(r);\nif(r.evidence?.length)html+=')


def t0_card(summary):
    groups=summary['summary'];stats=groups['all']['interval_distributions']['pre2_to_post_days']
    require(summary['valid'] is True and not summary['global_errors'] and groups['all']['source_rows']==7000,'T0 summary invalid')
    return {'id':'T0-DATE','title':'관측 날짜 출처 확인','status':'완료 · 공개 날짜 메타데이터 연결',
        'findings':['7,000개 타일을 원저자 격자와 정확히 연결했고, 날짜와 전체·분할·사건별 요약을 독립 검산했습니다.',
            f"문서상 이전 2 → 이후 간격은 전체 중앙값 {stats['median']:g}일, 최댓값 {stats['max']:g}일입니다. 기존 12일 가정과 다릅니다.",
            '공개 문서의 날짜는 7,000개 모두 시간순입니다. 실제 영상 취득시각을 복구한 것으로 판정하지 않습니다.'],
        'metrics':{'caption':'이전 2 → 이후의 문서상 날짜 간격 · 타일 가중 요약',
            'columns':['자료','타일 수','중앙값(일)','최댓값(일)'],
            'rows':[[label,groups[key]['source_rows'],groups[key]['interval_distributions']['pre2_to_post_days']['median'],
                     groups[key]['interval_distributions']['pre2_to_post_days']['max']]
                    for key,label in [('all','전체'),('split:train','학습'),('split:validation','검증'),('split:test','평가')]]},
        'limitations':['일 단위 원문 기록과 공개 슬롯 매핑의 연결입니다. 실제 센서 UTC 취득시각과 역사적 영상 대응은 미검증입니다.',
            '기존 모델의 합성 날짜·라벨·답변은 바꾸지 않았습니다. 이 감사는 모델 성능이나 변화·피해 검증이 아닙니다.',
            '타일 수는 독립 사건 수가 아닙니다. 기존 E5 진행 표시는 원래 고정 시각의 기록입니다.'],
        'next_step':'원천 제품 기록과 영상 파일을 대조해 실제 취득시각과 슬롯 대응을 확인합니다.'}


def build(base,t0,audit_file,out):
    base,t0,audit_file,out=[Path(p).resolve() for p in (base,t0,audit_file,out)]
    require(not out.exists(),'New output required; no overwrite')
    require(all(not out.is_relative_to(p) and not p.is_relative_to(out) for p in (base,t0)),'Output overlaps an input tree')
    tracking={}
    pin(base/'v7_build_manifest.json',V7_SHA,tracking);v7=read(base/'v7_build_manifest.json')
    actual_files={str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()}
    require(not any(p.is_symlink() for p in base.rglob('*')),'Symlink in base snapshot')
    require(actual_files==set(v7['output_files_sha256'])|V7_EXTRA,'Unexpected v7 files; do not absorb newer outputs')
    for name,digest in v7['output_files_sha256'].items():pin(child(base,name),digest,tracking)
    base_hashes={name:sha(child(base,name)) for name in sorted(actual_files)}
    pin(t0/'manifest.json',T0_SHA,tracking);tm=read(t0/'manifest.json')
    require(tm['valid'] is True and tm['schema']=='kuro-t0-exact-source-date-join-v0','T0 is incomplete/invalid')
    require(tm['upstream_package_manifest_sha256']==UPSTREAM_SHA and tm['expected_rows']==7000
            and tm['expected_splits']=={'train':4000,'validation':1000,'test':2000},'T0 source/population pins differ')
    for name,digest in tm['files_sha256'].items():pin(child(t0,name),digest,tracking)
    pin(audit_file,AUDIT_SHA,tracking);independent=read(audit_file)
    require(independent['schema']=='t0-independent-result-review-v0' and independent['consistent'] is True
            and independent['n_rows']==independent['n_unique_grid_matches']==7000
            and independent['all_strictly_chronological_count']==7000,'Independent T0 audit did not pass')
    for name in ['manifest.json','joined_rows.jsonl','summary.json']:
        require(independent['source_hashes_verified'].get(str(t0/name))==sha(t0/name),'Independent audit refers to different T0 inputs')
    summary=read(t0/'summary.json')
    require(independent['verified_full_summary']==summary['summary'],'Independent summary differs')
    joined=[json.loads(line) for line in (t0/'joined_rows.jsonl').read_text().splitlines()]
    catalog=read(base/'catalog.json');old_research=read(base/'research_runs.json');old_sources=read(base/'research_sources.json')
    require(old_research['schema_version']=='eo_research_snapshot_v1' and not any(r['id']=='T0-DATE' for r in old_research['runs']),'Unexpected research snapshot')
    require(old_sources['source_grounding_v7']['e5_status_as_of']==E5_STATUS_AS_OF,'E5 fixed snapshot time differs')
    e5=[r for r in old_research['runs'] if r['id']=='E5-EB']
    require(len(e5)==1 and e5[0]['status']=='학습 진행 스냅샷 · 결과 없음'
            and E5_STATUS_AS_OF in e5[0]['findings'][0],'Do not substitute E5 outcomes')
    pins={'join_manifest_sha256':T0_SHA,'joined_rows_sha256':tm['files_sha256']['joined_rows.jsonl'],
          'independent_review_sha256':AUDIT_SHA,'upstream_package_manifest_sha256':UPSTREAM_SHA}
    updated=attach_dates(catalog,joined,pins)
    html=patch_html((base/'index.html').read_text())
    now=datetime.now(timezone.utc).isoformat();research=copy.deepcopy(old_research);sources=copy.deepcopy(old_sources)
    research['checked_at']=now;research['runs'].append(t0_card(summary))
    require(research['runs'][:-1]==old_research['runs'],'Existing experiment records changed')
    require('source_dates_t0_v8' not in sources,'Source-date provenance already present')
    sources['source_dates_t0_v8']={'checked_at':now,'source_hashes':pins,'e5_status_as_of_unchanged':E5_STATUS_AS_OF,
        'role':'published_day_date_candidates_not_verified_utc_acquisition','n_tiles':7000,'model_outputs_modified':False}
    shutil.copytree(base,out)
    try:
        write(out/'catalog.json',updated);write(out/'research_runs.json',research);write(out/'research_sources.json',sources)
        (out/'index.html').write_text(html)
        folder=out/'source_dates_t0_v8';folder.mkdir()
        for name in ('manifest.json','summary.json'):shutil.copyfile(t0/name,folder/name)
        shutil.copyfile(audit_file,folder/'independent_review.json')
        for relative,digest in base_hashes.items():
            require(sha(base/relative)==digest,'Original v7 changed during build')
            if relative not in CHANGED:require(sha(out/relative)==digest,'Inherited file changed: '+relative)
        for path,digest in tracking.items():require(sha(path)==digest,'Pinned input changed during build')
        output_hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
        report={'schema':'eo-v8-source-dates-build-v0','created_at':now,'base':str(base),'out':str(out),
            'n_source_dates_added':7000,'changed_existing_files':sorted(CHANGED),'e5_status_as_of':E5_STATUS_AS_OF,
            'existing_catalog_fields_unchanged':True,'existing_research_runs_unchanged':True,
            'meta_reader_cases_and_previews_byte_identical':True,'source_hashes':tracking,'base_files_sha256':base_hashes,
            'output_files_sha256':output_hashes,'builder_sha256':sha(__file__),
            'limits':['The attached dates are published day-level metadata candidates; UTC acquisition and historical raster lineage remain unverified.',
                'T0 results and the independent local metadata review are consumed; this builder does not repeat the upstream join or remote raster audit.',
                'Search continues using event dates. No model inputs, outputs, labels, or evaluation membership are changed.',
                'Inherited browser/HTTP checks describe earlier snapshots; new v8 browser/HTTP validation is still required.']}
        write(out/'v8_build_manifest.json',report);return report
    except BaseException as error:
        write(out/'v8_failure.json',{'error':str(error),'partial_copy_not_valid':True});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ['base','t0','audit','out']:parser.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args();result=build(args.base,args.t0,args.audit,args.out)
    print(json.dumps({'out':result['out'],'n_source_dates_added':result['n_source_dates_added'],
                      'meta_reader_cases_and_previews_byte_identical':True}))
