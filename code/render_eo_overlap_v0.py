"""Render measured geospatial arrays; all rasters share the original chip grid."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
p=argparse.ArgumentParser();p.add_argument('--measurement',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
m=json.loads(a.measurement.read_text());fp=a.measurement.parent/'evidence_arrays.npz'
if hashlib.sha256(fp.read_bytes()).hexdigest()!=m['provenance']['arrays_sha256']:raise ValueError('Array hash mismatch')
z=np.load(fp);W,H=1320,1170;canvas=Image.new('RGB',(W,H),'#f5f6f1');d=ImageDraw.Draw(canvas)
font='/System/Library/Fonts/Supplemental/Arial.ttf'
def f(n):return ImageFont.truetype(font,n)
def txt(x,y,s,n=18,c='#173c35'):d.text((x,y),s,font=f(n),fill=c)
txt(30,25,'Larkana | Flood reference x 2020 cropland',31)
txt(30,73,'ks_00276  |  train development case  |  event reference: 2022-09-10  |  acquisition dates unavailable',17)
mask=z['flood_mask'];v=z['valid'];land=z['landcover'];lv=z['landcover_valid'];known=v&(mask!=3)&lv&np.isin(land,[10,20,30,40,50,60,70,80,90,95,100]);crop=known&(land==40);hit=crop&(mask==2)
colors={0:(229,234,229),1:(68,125,157),2:(34,173,193),3:(140,137,155)}
ref=np.zeros((*mask.shape,3),dtype=np.uint8)
for k,c in colors.items():ref[mask==k]=c
ref[~v]=(140,137,155)
lc=np.full((*mask.shape,3),[218,223,217],dtype=np.uint8)
for k,c in {40:(207,172,72),10:(44,109,62),80:(68,125,157),50:(146,93,84),30:(166,187,100),60:(214,195,167)}.items():lc[land==k]=c
lc[~known]=(140,137,155)
over=np.full((*mask.shape,3),[229,234,229],dtype=np.uint8);over[crop]=(207,172,72);over[known&(mask==2)&~crop]=(122,191,207);over[hit]=(234,123,62);over[~known]=(140,137,155)
panels=[]
for k,title in [('pre1','Pre-event SAR 1'),('pre2','Pre-event SAR 2'),('post','Post-event SAR')]:
 vv=z[k][0];finite=np.isfinite(vv)&(vv>0);db=10*np.log10(np.clip(np.nan_to_num(vv,nan=0),1e-6,None));g=(np.clip((db+25)/25,0,1)*255).astype('uint8');rgb=np.repeat(g[...,None],3,axis=2);rgb[~finite]=(140,137,155);panels.append((title,rgb,'VV / -25 to 0 dB display; actual date unknown'))
panels.extend([('Flood reference mask',ref,'Cyan: flood | blue: permanent water | gray: nonwater'),('WorldCover 2020 v100',lc,'Gold: class 40 cropland | other classes in context'),('Observed spatial intersection',over,'Orange: flood AND cropland | gold: other cropland')])
for i,(title,rgb,note) in enumerate(panels):
 x=30+(i%3)*430;y=130+(i//3)*450;txt(x,y,title,22);canvas.paste(Image.fromarray(rgb).resize((390,390),Image.Resampling.NEAREST),(x,y+38));txt(x,y+431,note,12)
txt(30,1037,f"Cropland overlap: {m['classes']['cropland']['flood_overlap_m2']/10000:.1f} ha | jointly valid: {m['joint_valid_area_m2']/10000:.1f} ha | mask-unknown: {m['unknown_area_m2']/10000:.1f} ha",22)
txt(30,1074,'Spatial reference overlap, not crop loss, damage attribution, recovery, or model accuracy.',17)
txt(30,1104,'WorldCover reprojected with nearest neighbor. WGS84 geodesic pixel areas; north is up.',15)
txt(30,1134,'© ESA WorldCover project 2020 / Contains modified Copernicus Sentinel data (2020) processed by ESA WorldCover consortium',12)
a.out.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.out);print(a.out)
