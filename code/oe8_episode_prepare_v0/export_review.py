#!/usr/bin/env python3
"""Export bounded review material; keep large arrays on the preparation server."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def distance(a,b):
    x,y=map(math.radians,a);u,v=map(math.radians,b)
    z=math.sin((y-v)/2)**2+math.cos(y)*math.cos(v)*math.sin((x-u)/2)**2
    return 2*6371008.8*math.asin(min(1,math.sqrt(z)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    prep=a.run_root/'prepared_v0';episodes=a.run_root/'episodes_v0'
    a.out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(Path(__file__).with_name('io_schema.json'),a.out/'io_schema.json')
    rows=[json.loads(l) for l in (prep/'manifest.jsonl').read_text().splitlines()]
    for name in ['contract.json','manifest.jsonl','quality_policy.json','partition.json','progress.json','frozen_candidates.jsonl','computed.json']:
        shutil.copyfile(prep/name,a.out/name)
    shutil.copytree(prep/'qa',a.out/'qa')
    for name in ['source_verification_v0.json','prepare_v0.log','verify_v0.log','episode_v0.log']:
        if (a.run_root/name).exists():shutil.copyfile(a.run_root/name,a.out/name)
    if episodes.exists():shutil.copytree(episodes,a.out/'episodes')
    # One predeclared manifest-first packet from each partition for local inspection.
    sample_ids=[]
    for role in ['train_pool','source_bank','dev_query']:
        row=next(r for r in rows if r['training_partition']==role)
        sample_ids.append(row['patch_id'])
        for field in ['npz_path','label_path']:
            dst=a.out/'sample_packets'/row[field];dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(prep/row[field],dst)
    pairs=[]
    for bank in rows:
        if bank['training_partition']!='source_bank':continue
        for train in rows:
            if train['training_partition']=='train_pool' and bank['parent_tile']==train['parent_tile']:
                pairs.append({'bank':bank['patch_id'],'train':train['patch_id'],'parent_tile':bank['parent_tile'],
                              'centroid_distance_m':distance(bank['centroid_lon_lat'],train['centroid_lon_lat'])})
    pairs.sort(key=lambda r:r['centroid_distance_m'])
    geometry={'method':'haversine on provided centroid, R=6371008.8m','cross_partition_pairs':len(pairs),
              'nearest_pairs':pairs[:10],'under_2000m_count':sum(r['centroid_distance_m']<2000 for r in pairs),
              'exact_footprint_or_parcel_nonoverlap_certified':False}
    (a.out/'centroid_separation_diagnostic.json').write_text(json.dumps(geometry,indent=2)+'\n')
    inventory=[{'path':str(f.relative_to(a.out)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(a.out.rglob('*')) if f.is_file()]
    receipt={'status':'review_export_not_full_raw_dataset','server_prepared_root':str(prep),
             'local_sample_packets':sample_ids,'large_input_packets_on_server':len(rows),'files':inventory,
             'total_bytes':sum(f['bytes'] for f in inventory),'source_raw_packet_hashes':[{k:r[k] for k in ['patch_id','npz_path','npz_sha256','label_path','label_sha256']} for r in rows]}
    (a.out/'export_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['files','source_raw_packet_hashes']}))

if __name__=='__main__':main()
