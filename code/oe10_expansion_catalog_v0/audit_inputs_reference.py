"""CPU-only class8/14 source-input audit. Reads train/bank packets, never dev labels.
No model, GPU, score selection, or claim of cloud/parcel independence.
"""
import argparse, hashlib, json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

CLASSES={8:"grapevine",14:"leguminous_fodder"}
BANDS=['B02','B03','B04','B08','B05','B06','B07','B8A','B11','B12','B01','B09']
MAPPING=[0,1,2,6,3,4,5,7,8,9,0,7]
PROTECTED=['pilot_sen12_gp_heads.py','sen12_official_baselines.py','extract_sen12_fold_cache.py','audit_sen12_fold_cache.py']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def require(ok,msg):
    if not ok:raise ValueError(msg)

def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def safe(root,rel,category):
    root=Path(root).resolve();p=(root/rel).resolve()
    require(p.is_relative_to(root/category) and p.is_file(),"reference escapes allowed category")
    return p

def permit_source(row):
    require(row['training_partition'] in ('train_pool','source_bank'),"dev packet forbidden")
    require(row['parent_tile'] in ('t32ulu','t31tfj'),"unopened parent forbidden")

def protect(root):
    ref=json.loads((root/'oe10_p2_v0/training_v0/status.json').read_text())
    got={}
    for name in PROTECTED:
        p=root/'code'/name;st=p.stat();got[name]={'sha256':sha(p),'mtime_ns':st.st_mtime_ns,'bytes':st.st_size}
    require(got==ref['protected_before'],"protected source changed")
    snapshot=root/'oe10_p2_v0/code_snapshot/oe10_p2_v3'
    for name,digest in ref['source_hashes'].items():
        require(sha(snapshot/name)==digest,"running snapshot changed: "+name)
    return got

def packet_check(row,z,l,stats):
    permit_source(row)
    require(not set(z)&{'semantic','instances','target_mask','label_valid','crop_label_valid'},'labels in input packet')
    raw=z['raw_selected_s2'];valid=z['observation_valid']
    require(raw.shape==(8,10,128,128) and raw.dtype==np.int16,'raw shape/dtype')
    require(valid.shape==(8,128,128) and valid.dtype==np.bool_,'valid shape/dtype')
    require(np.array_equal(valid,~(raw==-10000).any(axis=1)),'nodata validity')
    dates=[datetime.strptime(str(d),'%Y%m%d') for d in row['selected_dates']]
    require(dates==sorted(set(dates)) and len(dates)==8,'date order')
    expected=np.asarray([[d.day,d.month-1,d.year] for d in dates])
    require(np.array_equal(z['timestamps'],expected),'timestamps')
    mu=np.array([stats[b]['mean'] for b in BANDS]);sd=np.array([stats[b]['std'] for b in BANDS])
    require(np.all(sd>0),'std')
    native=raw[:,MAPPING].astype(np.float32).transpose(2,3,0,1)
    norm=((native-(mu-2*sd))/(4*sd)).astype(np.float32)
    norm[~valid.transpose(1,2,0)]=0
    require(np.array_equal(norm,z['normalized_s2']),'normalization')
    require(np.array_equal(z['band_observed'],[True]*10+[False]*2),'band observed')
    sem=l['semantic'];ins=l['instances']
    require(sem.shape==ins.shape==(128,128),'label shape')
    require(np.array_equal(l['label_valid'],sem!=19),'label validity')
    return raw,valid,sem,ins

def object_check(o,sem,ins,valid,mask):
    body=ins==o['instance_id'];expected=body&(sem==o['class_id'])&(sem!=19)
    require(mask.dtype==np.bool_ and np.array_equal(mask,expected),'mask/source-label binding')
    count=int(expected.sum());total=int(body.sum())
    require(count==o['class_pixel_count'] and total==o['instance_pixel_count'],'mask counts')
    purity=count/total
    require(abs(purity-o['class_purity'])<1e-12 and purity>=.95 and count>=64,'purity/minimum pixels')
    require(bool(valid[:,body].all()),'object has missing observations')
    border=bool(body[0].any() or body[-1].any() or body[:,0].any() or body[:,-1].any())
    require(border==o['touches_image_border'],'border flag')
    ys,xs=np.where(expected)
    require([int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]==o['bbox_xyxy_exclusive'],'bbox')
    return expected

def capacity(objects,role,c,excluded=None,border_only=False):
    chosen=[o for o in objects if o['episode_role']==role and o['class_id']==c and o['patch_id']!=excluded and (not border_only or not o['touches_image_border'])]
    counts=Counter(o['patch_id'] for o in chosen)
    return {'objects':len(chosen),'patches':len(counts),'max2_per_patch_capacity':sum(min(2,v) for v in counts.values())}

def sheet(cases,out):
    from PIL import Image,ImageDraw
    w,h=128,170
    canvas=Image.new('RGB',(9*w,len(cases)*h),'white');d=ImageDraw.Draw(canvas)
    for j,case in enumerate(cases):
        row,raw,mask=case
        d.text((4,j*h+2),f"{row['name']} {row['role']} patch {row['patch_id']} | annual reference; cloud status unknown",fill='black')
        for i in range(8):
            rgb=np.rint(np.clip(raw[i,[2,1,0]].transpose(1,2,0)/3000,0,1)*255).astype(np.uint8)
            canvas.paste(Image.fromarray(rgb),(i*w,j*h+35))
            d.text((i*w+2,j*h+20),str(row['dates'][i]),fill='black')
        m=np.zeros((128,128,3),np.uint8);m[mask]=[255,220,0]
        canvas.paste(Image.fromarray(m),(8*w,j*h+35));d.text((8*w+2,j*h+20),'reference mask',fill='black')
    canvas.save(out/'contact_sheet.png')

def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    before=protect(a.experiment_root) if a.experiment_root else None
    manifest=a.prepared/'manifest.jsonl';objects_path=a.episodes/'source_objects.jsonl'
    allrows=rows(manifest);byid={r['patch_id']:r for r in allrows};objs=rows(objects_path)
    selected=[o for o in objs if o['class_id'] in CLASSES]
    require(all(o['episode_role'] in ('train_pool','source_bank') for o in selected),'object role')
    ids=sorted(set(o['patch_id'] for o in selected))
    sourcehashes={str(p):sha(p) for p in (manifest,objects_path,a.prepared/'computed.json',Path(__file__))}
    stats=json.loads((a.prepared/'computed.json').read_text())['sentinel2_l2a']
    examples={}
    for c in CLASSES:
        for role in ('train_pool','source_bank'):
            example=min((o for o in selected if o['class_id']==c and o['episode_role']==role),key=lambda o:o['object_key'])
            examples[c,role]=example['patch_id']
    patchreports=[];objectreports=[];cases=[]
    for pid in ids:
        r=byid[pid];permit_source(r)
        ip=safe(a.prepared,r['npz_path'],'inputs');lp=safe(a.prepared,r['label_path'],'labels')
        require(sha(ip)==r['npz_sha256'] and sha(lp)==r['label_sha256'],'packet hash')
        with np.load(ip,allow_pickle=False) as z,np.load(lp,allow_pickle=False) as l:
            raw,valid,sem,ins=packet_check(r,z,l,stats)
            chosen=[o for o in selected if o['patch_id']==pid]
            for o in chosen:
                require(o['episode_role']==r['training_partition'] and o['parent_tile']==r['parent_tile'],'role/parent disagreement')
                require(o['source_label_sha256']==r['label_sha256'],'source label sha')
                mp=safe(a.episodes,o['mask_npz'],'support_masks');require(sha(mp)==o['mask_sha256'],'mask hash')
                with np.load(mp,allow_pickle=False) as zmask:mask=zmask['mask']
                object_check(o,sem,ins,valid,mask)
                red=raw[:,2].astype(float);nir=raw[:,6].astype(float)
                denomin=nir+red;good=valid&mask[None]&(denomin>0)
                ndvi=np.divide(nir-red,denomin,out=np.zeros_like(denomin),where=denomin>0)
                vals=[float(np.median(ndvi[i][good[i]])) if good[i].any() else None for i in range(8)]
                objectreports.append(dict(o,median_ndvi_by_date=vals,ndvi_valid_pixels_by_date=good.sum(axis=(1,2)).tolist(),ndvi_is_descriptive_not_cloud_or_crop_validation=True))
            for (c,role),examplepid in examples.items():
                if examplepid==pid:
                    union=np.zeros((128,128),bool)
                    for o in chosen:
                        if o['class_id']==c:union|=(ins==o['instance_id'])&(sem==c)
                    cases.append((dict(name=CLASSES[c],class_id=c,role=role,patch_id=pid,dates=r['selected_dates']),raw.copy(),union))
            patchreports.append({'patch_id':pid,'role':r['training_partition'],'parent':r['parent_tile'],'dates':r['selected_dates'],'input_sha256':r['npz_sha256'],'label_sha256':r['label_sha256'],'observation_valid_fraction':valid.mean(axis=(1,2)).tolist(),'normalization_exact_match':True,'cloud_status':'unknown','parcel_global_id_available':False})
    trainqueries=[r['patch_id'] for r in allrows if r['training_partition']=='train_pool']
    require(len(trainqueries)==48 and len(set(trainqueries))==48,'expected 48 unique training queries')
    coverage=[]
    for c,name in CLASSES.items():
        exclusion=[dict(query_patch_id=q,**capacity(selected,'train_pool',c,q)) for q in trainqueries]
        coverage.append({'class_id':c,'name':name,'train':capacity(selected,'train_pool',c),'bank':capacity(selected,'source_bank',c),'bank_interior_only_sensitivity_not_selection':capacity(selected,'source_bank',c,border_only=True),'train_query_exclusion':exclusion,'all48_train_queries_have_k8':all(x['max2_per_patch_capacity']>=8 for x in exclusion),'parent_counts':{role:dict(Counter(o['parent_tile'] for o in selected if o['class_id']==c and o['episode_role']==role)) for role in ('train_pool','source_bank')}})
    sheet(cases,a.out)
    write(a.out/'packet_audit.json',patchreports);write(a.out/'object_audit.json',objectreports);write(a.out/'case_manifest.json',[x[0] for x in cases])
    report={'schema':'oe10_expansion_input_audit_v1','created_utc':datetime.now(timezone.utc).isoformat(),'prepared_packets_checked':len(ids),'objects_checked':len(selected),'source_hashes':sourcehashes,'coverage':coverage,'development_packets_opened':0,'new_gpu_seconds':0,'source_bank_used_for_encoder_training':False,'current_p2_changed':False if a.experiment_root else None,'protection_checked':bool(a.experiment_root),'source_shard_reread':False,'global_parcel_independence_verified':False,'cloud_quality_certified':False,'human_crop_or_expert_validation':False,'new_evaluation_cohort_frozen':False,'limitations':['Source-label consistency is not independent expert correctness.','Observation-valid is sentinel availability, not cloud-free.','Local instance IDs do not certify global parcel independence.','Only training/source-bank inspected; new-class evaluation eligibility remains open.']}
    after=protect(a.experiment_root) if a.experiment_root else None
    require(before==after,'protected source modified during audit');report['protected_before']=before;report['protected_after']=after
    write(a.out/'audit_summary.json',report)
    entries=[{'path':str(p.relative_to(a.out)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(a.out.rglob('*')) if p.is_file()]
    write(a.out/'export_manifest.json',{'files':entries})
    print(json.dumps({k:report[k] for k in ('prepared_packets_checked','objects_checked','development_packets_opened','global_parcel_independence_verified','cloud_quality_certified')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--episodes',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--experiment-root',type=Path);run(p.parse_args())

