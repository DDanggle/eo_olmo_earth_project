import ast,collections,gzip,hashlib,json
from pathlib import Path
from inspect_grid_metadata import data_only
P=Path(__file__).resolve().parent
out={'schema':'kuro-public-metadata-coverage-v1','files':{}}
generator=Path('/private/tmp/geobench2_kuro_generate_20260925.py')
tree=ast.parse(generator.read_text()); fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='create_split_mapper')
# Print function's source for review; do not execute upstream source code.
out['published_split_function_source']=ast.get_source_segment(generator.read_text(),fn)
for n in ['KuroV2_grid_dict.gz','KuroV2_grid_dict_test_0_100.gz']:
 path=P/n;rows=data_only(path); c=collections.Counter();zero=collections.Counter();ids=set();aoi=set()
 for k,v in rows.items():
  q=v['info']; e=str(q['actid']);c[e]+=1;zero[e]+=int(q['pflood']==0 and q['pwater']==0);ids.add(q['grid_id']);aoi.add((q['actid'],q['aoiid']))
 out['files'][n]={'compressed_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'records':len(rows),'distinct_grid_ids':len(ids),'distinct_events':len(c),'distinct_event_aois':len(aoi),'counts_by_event':dict(sorted(c.items(),key=lambda x:int(x[0]))),'zero_flood_and_water_counts_by_event':dict(sorted(zero.items(),key=lambda x:int(x[0])))}
 del rows
groups={}
for node in tree.body:
 if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['train_acts','val_acts','test_acts']:groups[node.targets[0].id]=ast.literal_eval(node.value)
for value in out['files'].values():
 value['counts_by_published_split']={group:sum(value['counts_by_event'].get(str(e),0) for e in ids) for group,ids in groups.items()}
 value['missing_event_ids_from_published_split_lists']={group:[e for e in ids if str(e) not in value['counts_by_event']] for group,ids in groups.items()}
out['published_split_event_ids']=groups
(P/'metadata_coverage_inventory.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
