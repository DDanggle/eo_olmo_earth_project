#!/usr/bin/env python3
"""OE11 source-region preparation. No model/GPU; reserved payloads forbidden."""
import argparse, hashlib, json, math, time, importlib.util
from collections import Counter
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,indent=2,default=str)+'\n')
def key(s):return hashlib.sha256(('oe11-source-v0:'+str(s)).encode()).hexdigest()
def distance_m(a,b):
    lon1,lat1=map(math.radians,a['centroid_lon_lat']);lon2,lat2=map(math.radians,b['centroid_lon_lat'])
    h=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 2*6371008.8*math.asin(math.sqrt(min(1,max(0,h))))
def verify_nested(nested,r):
    names=list(nested['tortilla:id'])
    assert names==['s2','s1a','s1d','semantic','instance']
    for _,m in nested.iterrows():assert str(m['patch_id'])==r['patch_id'] and str(m['tile'])==r['parent_tile']
    assert [int(x) for x in nested.iloc[0]['dates']]==r['all_dates_yyyymmdd']
    return names
def main():
    a=argparse.ArgumentParser()
    a.add_argument('--root',type=Path,required=True);a.add_argument('--audit',type=Path,required=True)
    a.add_argument('--legacy',type=Path,required=True);a.add_argument('--computed',type=Path,required=True)
    a.add_argument('--transform',type=Path,required=True);a.add_argument('--policy',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True);a.add_argument('--extract',action='store_true')
    z=a.parse_args();z.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    policy=json.loads(z.policy.read_text())
    assert set(policy['source_parents'])=={'t31tfj','t32ulu','t31tfm'} and policy['reserved_parent']=='t30uxv'
    assert policy['reserved_parent'] not in policy['source_parents']
    spec=importlib.util.spec_from_file_location('oe8_transform',z.transform);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    audit=json.loads(z.audit.read_text());source=z.root/old.SHARD
    source_record=next(x for x in audit['shards'] if x['name']==old.SHARD)
    assert source_record['valid'] and source_record['stable_during_hash'] and source_record['actual_sha256']==old.SHARD_SHA
    st=source.stat();assert (st.st_size,st.st_mtime_ns)==(source_record['bytes'],source_record['mtime_ns'])
    assert sha(z.computed)==old.COMPUTED_SHA
    assert sha(z.legacy)==old.CANDIDATE_SHA
    legacy={str(r['patch_id']) for r in map(json.loads,z.legacy.read_text().splitlines())}
    assert len(legacy)==80
    import tacoreader
    frame=tacoreader.load([str(source)]);frame=frame[frame['tortilla:data_split']=='train'].reset_index(drop=True)
    rows=[];forbidden=0
    for i,r in frame.iterrows():
        tile=str(r['tile']);pid=str(r['patch_id'])
        if tile not in policy['source_parents']:
            forbidden+=1;continue
        if pid in legacy:continue
        lon,lat=float(r['lon']),float(r['lat'])
        # Common local equirectangular coordinates per source area, metres, not cadastral geometry.
        xm=111320*math.cos(math.radians(46))*lon;ym=111320*lat
        block=f"{tile}:{math.floor(xm/policy['block_m'])}:{math.floor(ym/policy['block_m'])}"
        split_bucket=int(key(block)[:8],16)%10
        part='calibration' if split_bucket<2 else 'source_bank' if split_bucket<4 else 'train_pool'
        rows.append({'patch_id':pid,'parent_tile':tile,'partial_catalog_index':int(i),
           'source_packed_reference':str(r['internal:subfile']),'all_dates_yyyymmdd':[int(d) for d in r['dates']],
           'centroid_lon_lat':[lon,lat],'xy_m':[xm,ym],'spatial_block':block,'training_partition':part})
    assert len({r['patch_id'] for r in rows})==len(rows)
    # Fixed split before reading any source labels. Guard across all assigned source roles.
    safe=[];dropped=[]
    for r in rows:
        distances=[distance_m(r,s) for s in rows if s['training_partition']!=r['training_partition']]
        d=min(distances,default=float('inf'));r['nearest_other_partition_centroid_m']=None if math.isinf(d) else d
        (safe if d>=policy['minimum_cross_partition_centroid_m'] else dropped).append(r)
    write(z.out/'selection_policy.json',policy)
    (z.out/'source_metadata.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(z.out/'before_label_split.json',{'counts':dict(Counter(r['parent_tile']+'/'+r['training_partition'] for r in safe)),
       'guard_dropped':len(dropped),'source_metadata_sha256':sha(z.out/'source_metadata.jsonl'),
       'legacy_excluded':len(legacy),'reserved_payload_reads':0})
    # Read labels from permitted source regions only; no S2 yet.
    for n,r in enumerate(safe):
        nested=frame.read(r['partial_catalog_index']);names=verify_nested(nested,r)
        sem,pr=old.read_payload(nested.read(names.index('semantic')),{source.resolve()})
        ins,pi=old.read_payload(nested.read(names.index('instance')),{source.resolve()})
        assert sem.shape==(3,128,128) and ins.shape==(128,128)
        assert np.array_equal(sem[0],sem[0].astype(np.int64)) and set(np.unique(sem[0]))<=set(range(20))
        assert np.array_equal(ins,ins.astype(np.int64)) and (ins>=0).all()
        counts=np.bincount(sem[0].astype(np.int64).ravel(),minlength=20)
        r['class_pixel_counts']=counts.tolist();r['eligible_objects']={}
        for c in policy['target_classes']:
            sizes=[]
            for obj in np.unique(ins[sem[0]==c]):
                if obj<=0:continue
                m=ins==obj;size=int(m.sum());purity=float((sem[0][m]==c).mean())
                if int((sem[0][m]==c).sum())>=policy['minimum_object_pixels'] and purity>=policy['minimum_object_purity']:sizes.append(size)
            r['eligible_objects'][str(c)]=len(sizes)
        r['label_source_provenance']={'semantic':pr,'instance':pi}
        if n%50==0:print(json.dumps({'phase':'source_labels','done':n+1,'total':len(safe)}),flush=True)
    (z.out/'availability.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in safe))
    # Calibration is hash-only. Train and support balance source labels under fixed class order.
    selected=[]
    for parent in policy['source_parents']:
        for part,cap in policy['caps_per_parent'].items():
            pool=sorted([r for r in safe if r['parent_tile']==parent and r['training_partition']==part],key=lambda r:key(r['patch_id']))
            chosen=[]
            if part!='calibration':
                target_quota=cap//len(policy['target_classes'])
                for round_index in range(target_quota):
                    for c in policy['target_classes']:
                        pick=next((r for r in pool if r not in chosen and r['eligible_objects'][str(c)]>0),None)
                        if pick is not None and len(chosen)<cap:chosen.append(pick)
            chosen.extend(r for r in pool if r not in chosen)
            selected.extend(chosen[:cap])
    (z.out/'selected_candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    summary={'status':'metadata_and_source_labels_audited','source_parent_counts':dict(Counter(str(x) for x in frame['tile'])),
       'source_candidates_after_legacy_exclusion':len(rows),'safe_candidates':len(safe),'guard_dropped':len(dropped),
       'selected_count':len(selected),'selected_partitions':dict(Counter(r['parent_tile']+'/'+r['training_partition'] for r in selected)),
       'all_source_presence':{p:{str(c):sum(r['class_pixel_counts'][c]>0 for r in safe if r['parent_tile']==p) for c in policy['target_classes']} for p in policy['source_parents']},
       'selected_presence':{p+'/'+part:{str(c):sum(r['class_pixel_counts'][c]>0 for r in selected if r['parent_tile']==p and r['training_partition']==part) for c in policy['target_classes']} for p in policy['source_parents'] for part in policy['caps_per_parent']},
       'selected_support_objects':{p:{str(c):sum(min(2,r['eligible_objects'][str(c)]) for r in selected if r['parent_tile']==p and r['training_partition']=='source_bank') for c in policy['target_classes']} for p in policy['source_parents']},
       'reserved_metadata_rows_skipped':forbidden,'reserved_payload_reads':0,'legacy_ids_selected':0,
       'exact_footprint_verified':False,'parcel_disjoint_verified':False,'cloud_quality_certified':False,
       'source_audit_sha256':sha(z.audit),'source_stat_before':{'bytes':st.st_size,'mtime_ns':st.st_mtime_ns},
       'policy_sha256':sha(z.policy),'script_sha256':sha(__file__),'transform_sha256':sha(z.transform),
       'candidate_sha256':sha(z.out/'selected_candidates.jsonl'),'GPU_used':False,'input_bands':old.INPUT,'native_bands':old.NATIVE,'band_mapping':old.MAPPING}
    write(z.out/'availability_summary.json',summary)
    if z.extract:
        for sub in ['inputs','labels','qa']:(z.out/sub).mkdir()
        stats=json.loads(z.computed.read_text())['sentinel2_l2a'];records=[]
        for n,r in enumerate(selected):
            nested=frame.read(r['partial_catalog_index']);names=verify_nested(nested,r);arrays={};prov={}
            for kind in ['s2','semantic','instance']:
                arrays[kind],prov[kind]=old.read_payload(nested.read(names.index(kind)),{source.resolve()})
            idx,inputs,labels=old.transform(arrays['s2'],r['all_dates_yyyymmdd'],arrays['semantic'],arrays['instance'],stats)
            ip='inputs/'+r['patch_id']+'.npz';lp='labels/'+r['patch_id']+'.npz'
            np.savez_compressed(z.out/ip,**inputs);np.savez_compressed(z.out/lp,**labels)
            valid=inputs['observation_valid']
            rec=dict(r,role='development' if r['training_partition']=='calibration' else 'train',
               npz_path=ip,npz_sha256=sha(z.out/ip),label_path=lp,label_sha256=sha(z.out/lp),
               selected_indices=idx,selected_dates=[r['all_dates_yyyymmdd'][j] for j in idx],source_payloads=prov,
               raw_full_shape=list(arrays['s2'].shape),normalized_shape=list(inputs['normalized_s2'].shape),strict_no_missing_input_eligible=bool(valid.all()),
               supervised_training_allowed=r['training_partition']=='train_pool',clean_training_eligible=bool(valid.all()) and r['training_partition']=='train_pool',
               exact_footprint_available=False,cloud_quality_certified=False,
               quality={'invalid_pixel_dates':int((~valid).sum()),'per_frame_valid_fraction':valid.mean((1,2)).tolist()})
            records.append(rec)
            with (z.out/'manifest.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            write(z.out/'progress.json',{'prepared':len(records),'total':len(selected),'wall_seconds':time.monotonic()-start})
            print(json.dumps({'phase':'extract','prepared':len(records),'total':len(selected),'patch_id':r['patch_id']}),flush=True)
        # Three predetermined training cases per region, no score/quality selection.
        for j,parent in enumerate(policy['source_parents']):
            samples=sorted([r for r in records if r['parent_tile']==parent and r['training_partition']=='train_pool'],key=lambda r:key(r['patch_id']))[:3]
            if samples:old.render_page(samples,z.out,j)
        summary['prepared']=len(records);summary['strict_no_missing_inputs']=sum(r['strict_no_missing_input_eligible'] for r in records)
        summary['status']='inputs_prepared_requires_independent_verification'
    after=source.stat();assert (st.st_size,st.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    summary['source_stat_unchanged']=True;summary['wall_seconds']=time.monotonic()-start
    write(z.out/'receipt.json',summary)
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
