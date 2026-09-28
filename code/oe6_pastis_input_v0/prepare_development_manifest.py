"""Select metadata only, from verified PASTIS shards, before comparative training."""
import argparse
import hashlib
import json
from pathlib import Path
import tacoreader

ap=argparse.ArgumentParser()
ap.add_argument('--audit',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
audit=json.loads(a.audit.read_text())
assert audit['source_shards_unchanged'] and 'partial_read_error' not in audit
root=Path('/home/work/data/olmoearth/geobench2/pastis')
valid=[s for s in audit['shards'] if s['valid'] and s['stable_during_hash']]
for s in valid:
    st=(root/s['name']).stat()
    assert (st.st_size,st.st_mtime_ns)==(s['bytes'],s['mtime_ns'])
df=tacoreader.load([str(root/s['name']) for s in valid])
df=df[df['tortilla:data_split']=='train'].reset_index(drop=True)
assignment={'t32ulu':'train','t31tfj':'train','t31tfm':'development','t30uxv':'reserved_unopened_this_step'}
assert set(df['tile'])==set(assignment)
rows=[]
for tile,role in assignment.items():
    if role=='reserved_unopened_this_step':continue
    indices=df.index[df['tile']==tile].tolist()
    indices.sort(key=lambda i:hashlib.sha256(('oe6-develop-v0:'+str(df.iloc[i]['patch_id'])).encode()).hexdigest())
    for i in indices[:32 if role=='train' else 16]:
        r=df.iloc[i]
        rows.append({'role':role,'parent_tile':tile,'patch_id':str(r['patch_id']),
            'partial_catalog_index':int(i),'source_packed_reference':str(r['internal:subfile']),
            'all_dates_yyyymmdd':[int(d) for d in r['dates']],
            'centroid_lon_lat':[float(r['lon']),float(r['lat'])]})
assert len(rows)==80 and len({r['patch_id'] for r in rows})==80
assert {r['parent_tile'] for r in rows if r['role']=='train'}.isdisjoint({r['parent_tile'] for r in rows if r['role']=='development'})
(a.out/'candidates.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
result={'status':'metadata_candidates_selected_raw_inputs_and_labels_not_prepared',
    'candidate_train':64,'candidate_development':16,'candidate_heldout_opened':0,
    'parent_assignment':assignment,'available_parent_counts':df['tile'].value_counts().to_dict(),
    'selection_rule':'sha256 of fixed namespace plus patch ID within each parent, no labels read',
    'manifest_sha256':hashlib.sha256((a.out/'candidates.jsonl').read_bytes()).hexdigest(),
    'source_shard_hashes':{s['name']:s['actual_sha256'] for s in valid},
    'prior_visual_inspection':{'t32ulu':['40188'],'t31tfj':['20451']},
    'reserved_is_new_scientifically_unexposed_test':False,
    'reserved_warning':'Unopened in this preparation step only; historical usage/pretraining exposure must be audited.',
    'training_ready':False,'gpu_used':False,
    'remaining':['raw/label extraction and source hash checks','observation quality policy','pixel/label/parent contracts',
                 'full reader weight identity','trained baseline recipe and checkpoint save/reload']}
(a.out/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
