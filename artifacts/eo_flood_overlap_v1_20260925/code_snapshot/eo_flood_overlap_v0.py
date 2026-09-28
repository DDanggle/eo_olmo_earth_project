#!/usr/bin/env python3
"""CPU spatial join of ONE georeferenced train chip and WorldCover 2020.

Not a model evaluation. Keeps flood labels, land-cover inference, observation validity,
source dates and geodesic area separate. No imputed SAR acquisition dates.
"""
import argparse, hashlib, json, math, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from pyproj import Geod, Transformer

WC_URL='https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_N27E066_Map.tif'
WC_CLASSES=[10,20,30,40,50,60,70,80,90,95,100]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def pixel_areas(transform, crs, h, w):
    """WGS84 geodesic quadrilateral area; supports affine rotation too."""
    tr=Transformer.from_crs(crs,'EPSG:4326',always_xy=True);geod=Geod(ellps='WGS84')
    area=np.zeros((h,w),dtype=np.float64)
    # One longitude-translation invariant column is exact for north-up EPSG3857.
    same_row=(str(crs)=='EPSG:3857' and transform.b==0 and transform.d==0)
    for row in range(h):
        for col in range(1 if same_row else w):
            pts=[transform*(col,row),transform*(col+1,row),transform*(col+1,row+1),transform*(col,row+1)]
            lon,lat=tr.transform(*zip(*pts));v=abs(geod.polygon_area_perimeter(lon,lat)[0])
            if same_row:area[row,:]=v
            else:area[row,col]=v
    if not np.all(np.isfinite(area)&(area>0)):raise ValueError('Invalid geodesic areas')
    return area

def measure(mask,valid,land,land_valid,area):
    if len({x.shape for x in [mask,valid,land,land_valid,area]})!=1:raise ValueError('Grid mismatch')
    if not set(np.unique(mask)).issubset({0,1,2,3}):raise ValueError('Unexpected flood class')
    if not np.all(np.isfinite(area)&(area>0)):raise ValueError('Invalid pixel area')
    fv=valid.astype(bool)&(mask!=3)
    lv=land_valid.astype(bool)&np.isin(land,WC_CLASSES)
    joint=fv&lv;flood=fv&(mask==2)
    A=lambda v:float(area[v].sum())
    out={'joint_valid_area_m2':A(joint),'unknown_area_m2':A(~joint),
         'footprint_area_m2':float(area.sum()),'flood_valid_area_m2':A(fv),
         'reference_flood_area_m2':A(flood),'flood_unknown_landcover_area_m2':A(flood&~lv),
         'flood_valid_pixels':int(fv.sum()),'joint_valid_pixels':int(joint.sum()),'unknown_pixels':int((~joint).sum()),'classes':{}}
    for name,code in [('cropland',40),('forest',10)]:
        target=joint&(land==code);hit=target&flood
        out['classes'][name]={'class_code':code,'source_class_name':'Cropland (annual herbaceous)' if code==40 else 'Tree cover','flood_overlap_m2':A(hit),'observed_class_area_m2':A(target),'overlap_pixels':int(hit.sum()),'observed_class_pixels':int(target.sum()),'flood_share_of_observed_class':A(hit)/A(target) if target.any() else None}
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    m=json.loads((a.case/'source_manifest.json').read_text())
    if m['id']!='ks_00276' or m['row']['tortilla:data_split']!='train':raise ValueError('Pilot scope mismatch')
    for asset in m['assets']:
        if sha(a.case/asset['path'])!=asset['sha256']:raise ValueError('Source hash mismatch')
    a.out.mkdir(parents=True)
    arr=[]
    with rasterio.open(a.case/'asset_4.tif') as ref:
        crs,transform,h,w=ref.crs,ref.transform,ref.height,ref.width;profile=ref.profile.copy()
        for k in range(6):
            with rasterio.open(a.case/f'asset_{k}.tif') as src:
                if (src.crs,src.transform,src.height,src.width)!=(crs,transform,h,w):raise ValueError('Source grids disagree')
                arr.append(src.read())
    with urllib.request.urlopen(urllib.request.Request(WC_URL,method='HEAD'),timeout=30) as res:
        headers={k.lower():v for k,v in res.headers.items()}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',GDAL_HTTP_TIMEOUT='45'):
        with rasterio.open(WC_URL) as src:
            wcmeta={'crs':str(src.crs),'transform':list(src.transform),'width':src.width,'height':src.height,'nodata':src.nodata,'tags':src.tags()}
            tr=Transformer.from_crs(crs,src.crs,always_xy=True)
            for c,r in [(0,0),(w,0),(w,h),(0,h)]:
                x,y=tr.transform(*(transform*(c,r)))
                if not(src.bounds.left<=x<=src.bounds.right and src.bounds.bottom<=y<=src.bounds.top):raise ValueError('Outside selected WC tile')
            with WarpedVRT(src,crs=crs,transform=transform,width=w,height=h,resampling=Resampling.nearest) as vrt:
                land=vrt.read(1);lv=vrt.read_masks(1)>0
    mask=arr[4][0];valid=arr[5][0]==1;areas=pixel_areas(transform,crs,h,w)
    metrics=measure(mask,valid,land,lv,areas)
    if abs(100*float((mask==2).mean())-float(m['row']['pflood']))>1e-8:raise ValueError('Reference pflood mismatch')
    profile.update(count=1,dtype='uint8',nodata=0,compress='deflate')
    with rasterio.open(a.out/'worldcover2020_aligned.tif','w',**profile) as dst:dst.write(land,1)
    np.savez_compressed(a.out/'evidence_arrays.npz',pre1=arr[0],pre2=arr[1],post=arr[2],flood_mask=mask,valid=valid,landcover=land,landcover_valid=lv,pixel_area_m2=areas)
    tx=Transformer.from_crs(crs,'EPSG:4326',always_xy=True)
    ring=[list(tx.transform(*(transform*(c,r)))) for c,r in [(0,0),(w,0),(w,h),(0,h),(0,0)]]
    published='2021-10-20';event=m['row']['flood_date'][:10]
    result={'schema':'eo-flood-landcover-overlap-v0','status':'measured','id':m['id'],'event_id':m['row']['actid'],'event_date':event,'split':'train','computed_at':datetime.now(timezone.utc).isoformat(),
      'baseline':{'dataset':'ESA WorldCover 2020 v100','reference_start':'2020-01-01','reference_end':'2020-12-31','product_publication_date':published,'current_object_last_modified':headers.get('last-modified'),'historical_byte_identity_verified':False,'retrospective_analysis':True},
      'source_grid':{'crs':str(crs),'transform':list(transform),'shape':[h,w]},'geometry':{'type':'Polygon','coordinates':[ring]},
      'acquisition_dates':{'pre_event_1':None,'pre_event_2':None,'post_event':None},
      'acquisition_audit':'All three tortilla sample rows repeat the event timestamp; original TIFF tags have no acquisition timestamps. No -24/-12-day imputation.',
      'area_method':'sum of WGS84 ellipsoidal geodesic four-corner pixel areas; EPSG3857 projected area is NOT ground area',
      'naive_projected_footprint_m2':abs(transform.a*transform.e-transform.b*transform.d)*h*w,
      'resampling':'nearest neighbor; original flood grid retained',
      'mask_contract':'raw 0 nonwater,1 permanent water,2 flood,3 nodata; valid asset5==1; TIFF mask nodata=0 ignored because class0 is semantic nonwater',
      'provenance':{'flood_source_manifest_sha256':sha(a.case/'source_manifest.json'),'flood_assets':m['assets'],'worldcover_url':WC_URL,'worldcover_http_headers':headers,'worldcover_source_metadata':wcmeta,'worldcover_aligned_sha256':sha(a.out/'worldcover2020_aligned.tif'),'arrays_sha256':sha(a.out/'evidence_arrays.npz'),'analysis_code_sha256':sha(__file__),'source_selection':m['selection'],'references':['https://esa-worldcover.org/en/data-access','https://eo4society.esa.int/2022/11/11/2021-worldcover-product/'],'attribution':'© ESA WorldCover project 2020 / Contains modified Copernicus Sentinel data (2020) processed by ESA WorldCover consortium'},
      'limitations':['single_selected_positive_train_case_not_model_evaluation','two_year_old_landcover_may_be_stale','class40_excludes_woody_crops_and_greenhouses','spatial_label_overlap_not_crop_loss_or_causal_damage','actual_SAR_acquisition_timestamps_unavailable','current_COG_historical_byte_identity_unverified','no_area_accuracy_confidence_interval','near_boundary_resampling_uncertainty_not_quantified'],**metrics}
    write(a.out/'overlap.json',result)
    write(a.out/'footprint.geojson',{'type':'Feature','geometry':result['geometry'],'properties':{'id':m['id'],'event_id':result['event_id'],'event_date':event,'role':'raster_footprint_not_flood_boundary'}})
    print(json.dumps({k:v for k,v in result.items() if k!='provenance'},indent=2))
if __name__=='__main__':main()
