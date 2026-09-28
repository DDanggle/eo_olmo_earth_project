#!/usr/bin/env python3
"""Build a new evidence catalog version by attaching verified CPU measurements."""
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
from eo_query_core_v0 import query_catalog

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--measurement',type=Path,required=True);p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--preview',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 cat=read(a.base/'catalog.json');meta=read(a.base/'meta.json');m=read(a.measurement);s=read(a.source_manifest)
 if m['schema']!='eo-flood-landcover-overlap-v0' or m['status']!='measured':raise ValueError('Unsupported measurement')
 if sha(a.source_manifest)!=m['provenance']['flood_source_manifest_sha256']:raise ValueError('Source manifest hash mismatch')
 for source in s['assets']:
  if sha(a.source_manifest.parent/source['path'])!=source['sha256']:raise ValueError('Source raster mismatch')
 for filename,key in [('worldcover2020_aligned.tif','worldcover_aligned_sha256'),('evidence_arrays.npz','arrays_sha256')]:
  if sha(a.measurement.parent/filename)!=m['provenance'][key]:raise ValueError('Measurement asset mismatch')
 matches=[r for r in cat['records'] if r['id']==m['id'] and r['dataset']=='kurosiwo']
 if len(matches)!=1:raise ValueError('Unique catalog association required')
 r=matches[0]
 if (r['aoi_id'],r['event_date'],r['split'])!=(m['event_id'],m['event_date'],m['split']):raise ValueError('Catalog event mismatch')
 if r['split']!='train':raise ValueError('Train pilot only')
 r['land_cover_overlap']=m;r['geometry']=m['geometry'];r['location_name']='Larkana, Pakistan';r['location_name_source']='https://github.com/Orion-AI-Lab/KuroSiwo/blob/main/catalogue/catalogue.yaml'
 r['provenance']={'reference_catalog':cat['provenance'],'restored_geometry':'original GeoTIFF affine and CRS','measurement_sha256':sha(a.measurement),'measurement_sources':m['provenance']};r['acquisition_dates']=m['acquisition_dates'];r['limitations']=[x for x in r['limitations'] if x not in ('land_cover_overlap_unavailable','georeferenced_extent_unavailable','centroid_only_not_footprint')]+m['limitations']
 preview_name=m['id']+'_overlap.png'
 r['evidence'].append({'image_url':'/previews/'+preview_name,'kind':'georeferenced_flood_landcover_overlap','acquisition_dates':m['acquisition_dates'],'source':{'preview_sha256':sha(a.preview),'measurement_sha256':sha(a.measurement)},'note':'One selected train case. WorldCover 2020 plus reference flood mask; actual acquisition timestamps unknown.'})
 check=query_catalog(cat,{'dataset':'kurosiwo','land_cover':'cropland'})
 if check['status']!='partial_coverage' or check['coverage']['assessed_count']!=1:raise ValueError('Partial coverage contract failed')
 cat['schema']='eo-evidence-catalog-v0.2';cat['built_at']=datetime.now(timezone.utc).isoformat()
 cat['land_cover_provenance']={'measurement_file':str(a.measurement.resolve()),'measurement_sha256':sha(a.measurement),'attachment_code_sha256':sha(__file__)}
 meta.update(version='0.2 · 2026-09-25',land_cover_assessed_records=1,land_cover_example_event=m['event_id'],land_cover_example_id=m['id'],land_cover_baseline=m['baseline'],preview_count=sum(bool(r['evidence']) for r in cat['records'] if r['dataset']=='kurosiwo'))
 for e in meta['events']:
  if e['id']==m['event_id']:e['location_name']='Larkana, Pakistan (AOI 01 pilot)'
 meta['limits']=['Historical reference catalog, not live detection.','One selected train chip has recovered footprint and WorldCover overlap; other chips are unassessed.','Geographic search still tests centroids, not footprints.','Event date differs from unknown actual acquisition dates.','Land-cover reference overlap does not measure crop loss, damage or recovery.']
 shutil.copytree(a.base,a.out)
 for old in ['api_checks.json']:
  if (a.out/old).exists():(a.out/old).unlink()
 write(a.out/'catalog.json',cat);write(a.out/'meta.json',meta);shutil.copyfile(a.preview,a.out/'previews'/preview_name);shutil.copyfile(Path(__file__).with_name('eo_evidence_search_v0.html'),a.out/'index.html')
 write(a.out/'land_cover_attachment.json',{'source':cat['land_cover_provenance'],'coverage':check['coverage'],'status':check['status']})
 print(json.dumps({'out':str(a.out),'status':check['status'],'coverage':check['coverage']}))
if __name__=='__main__':main()
