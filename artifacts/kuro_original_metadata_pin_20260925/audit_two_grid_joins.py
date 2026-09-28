"""Two preselected source metadata joins; no raster inference or model access."""
import datetime, hashlib, json, math, re
from pathlib import Path
from inspect_grid_metadata import data_only
P=Path(__file__).resolve().parent
EXPORT=Path('/private/tmp/fixed_case_grounding_download_20260925/fixed_case_grounding_v0_20260925')
REV='4347ed173c4e48f5a9d578bb5fe8453706b08e5e'
FILES={'KuroV2_grid_dict.gz':'6235bf33fdf188ae23386134d48d70f57de90c2994e8fd9c33d000f9bd5420cc','KuroV2_grid_dict_test_0_100.gz':'dfc6e9a752d05aaa0a38c3fd6c18ce12e090f95635e5e4709025cb8512f020ad'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rectangle(wkt):
 m=re.fullmatch(r'POLYGON\s*\(\(([^()]+)\)\)',wkt)
 if not m:raise ValueError('single polygon ring required')
 ring=[tuple(float(n) for n in part.strip().split()) for part in m[1].split(',')]
 if len(ring)!=5 or ring[0]!=ring[-1] or any(len(p)!=2 or not all(math.isfinite(n) for n in p) for p in ring):raise ValueError('closed four-corner ring required')
 xs={p[0] for p in ring};ys={p[1] for p in ring}
 if len(xs)!=2 or len(ys)!=2 or set(ring[:-1])!={(x,y) for x in xs for y in ys}:raise ValueError('axis-aligned rectangle required')
 if any((ring[i][0]==ring[i+1][0])==(ring[i][1]==ring[i+1][1]) for i in range(4)):raise ValueError('crossing/degenerate edge')
 return tuple(sorted(ring[:-1]))
def raster_corners(asset):
 a,b,c,d,e,f,g,h,j=asset['transform']; rows,cols=asset['shape'][-2:]
 assert asset['crs']=='EPSG:3857' and [a,b,d,e,g,h,j]==[10,0,0,-10,0,0,1] and [rows,cols]==[224,224]
 return tuple(sorted((a*x+b*y+c,d*x+e*y+f) for x,y in [(0,0),(cols,0),(cols,rows),(0,rows)]))
def clean(obj):
 # Preserve upstream DEM nodata NaN as explicit text; never serialize non-standard JSON NaN.
 if isinstance(obj,float) and not math.isfinite(obj):return {'upstream_nonfinite_float':repr(obj)}
 if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
 if isinstance(obj,list):return [clean(v) for v in obj]
 return obj
sources=[];records={};totals={}
for name,expected in FILES.items():
 path=P/name; assert sha(path)==expected
 x=data_only(path); totals[name]=len(x)
 selected={k:v for k,v in x.items() if v['info'].get('actid')==562 and v['info'].get('aoiid')==13}
 records[name]=selected
 sources.append({'file':name,'bytes':path.stat().st_size,'sha256':expected,'github_revision':REV,'url':f'https://raw.githubusercontent.com/Orion-AI-Lab/KuroSiwo/{REV}/pickle/{name}','total_records':len(x),'event_aoi_candidates':len(selected)})
 del x
cases={}
for tile in ['ks_06770','ks_05265']:
 mp=EXPORT/tile/'source_manifest.json'; m=json.loads(mp.read_text()); sr=m['source_row'];corners=raster_corners(m['assets'][0])
 assert all(raster_corners(a)==corners for a in m['assets'])
 evidence=[]
 for name,rows in records.items():
  matches=[(key,v) for key,v in rows.items() if rectangle(v['info']['geom'])==corners]
  assert len(matches)<2
  for key,v in matches:
   info=v['info']; field_checks={k:{'source_row':sr[k],'catalogue':info[k],'equal':float(sr[k])==info[k]} for k in ['actid','aoiid','pcovered','pwater','pflood']}
   field_checks['flood_date']={'source_row':sr['flood_date'],'catalogue':info['flood_date'],'equal':sr['flood_date']==info['flood_date']}
   assert all(c['equal'] for c in field_checks.values())
   assert key==info['grid_id'].replace('-','') and v['path']==f'562/13/{key}'
   evidence.append({'source_file':name,'candidate_count_event_aoi':len(rows),'candidate_count_exact_geometry':len(matches),'grid_id':info['grid_id'],'path':v['path'],'corners_epsg3857':corners,'geometry_exact':True,'field_checks':field_checks,'upstream_record':clean(v)})
 assert evidence
 base=evidence[0]['upstream_record']
 assert all(e['upstream_record']==base for e in evidence)
 dates={k:v['source_date'] for k,v in base['info']['sources'].items()}
 ms=datetime.date.fromisoformat(dates['MS1'])
 cases[tile]={'source_manifest_path':str(mp),'source_manifest_sha256':sha(mp),'source_asset_sha256':{a['role']:a['sha256'] for a in m['assets']},'metadata_join_status':'unique_exact_rectangle_and_source_fields','joined_records':evidence,'upstream_date_precision':'day','upstream_date_and_id_sources':base['info']['sources'],'days_to_MS1':{k:(ms-datetime.date.fromisoformat(d)).days for k,d in dates.items()},'geobench_slot_mapping_status':'unverified','acquisition_dates_in_existing_export':m['acquisition_dates'],'raster_value_correspondence_to_upstream_archive':'not_verified','sentinel_product_name_and_precise_utc_time':'not_resolved_from_UUIDs'}
out={'schema':'kuro-original-source-metadata-two-case-audit-v1','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'metadata_dates_recovered_pixel_slot_mapping_unverified','geometry_contract':'All 5 exported assets are 224x224,10m,EPSG:3857; four full affine raster corners exactly equal closed axis-aligned upstream WKT polygon corners; no centroid rounding/tolerance/nearest match. Grid join additionally requires exact actid/aoiid/flood_date/pcovered/pwater/pflood.','sources':sources,'cases':cases,'no_model_or_E5_outputs_read':True,'existing_artifacts_unchanged':True,'parser_sha256':sha(P/'inspect_grid_metadata.py'),'audit_script_sha256':sha(Path(__file__))}
path=P/'two_case_metadata_join_audit.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':out['status'],'output':str(path),'sha256':sha(path),'cases':{k:{'grid':v['joined_records'][0]['grid_id'],'dates':{s:d['source_date'] for s,d in v['upstream_date_and_id_sources'].items()},'days_to_MS1':v['days_to_MS1']} for k,v in cases.items()}},indent=2))
