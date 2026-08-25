"""[AUDIT-ONLY/SUPERSEDED] 오프셋 (XO2,YO2) 탐색 기록.

공식 KMA KO/2 km lat/lon grid가 발견되어 operational 경로에서는 폐기했다. 고유 공간점 4개로
offset을 선택하고 같은 자료로 평가하므로 최고점 개수나 반복 시각 수가 외부 타당성을 주지 않는다.
"""
import gzip, json, math, os
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(os.environ.get("GK2A_ROOT", os.path.expanduser("~/dong/ai_projects/data/gk2a")))
RE, SLAT1, SLAT2, OLON, OLAT = 6371.00877, 30.0, 60.0, 126.0, 38.0

def ll2g(lon, lat, gk):
    D=math.pi/180; re=RE/gk
    s1,s2,ol,oa=SLAT1*D,SLAT2*D,OLON*D,OLAT*D
    sn=math.log(math.cos(s1)/math.cos(s2))/math.log(math.tan(math.pi*.25+s2*.5)/math.tan(math.pi*.25+s1*.5))
    sf=(math.tan(math.pi*.25+s1*.5)**sn)*math.cos(s1)/sn
    ro=re*sf/(math.tan(math.pi*.25+oa*.5)**sn)
    ra=re*sf/(math.tan(math.pi*.25+lat*D*.5)**sn)
    th=lon*D-ol
    th=th-2*math.pi if th>math.pi else (th+2*math.pi if th<-math.pi else th)
    th*=sn
    return ra*math.sin(th), ro-ra*math.cos(th)   # 오프셋 없는 순수 격자좌표

cache={}
for line in (ROOT/"_crs/area_anchors.jsonl").read_text(encoding="utf-8").splitlines():
    if line:
        r=json.loads(line)
        if r.get("resultType", "CLD") == "CLD":
            cache[(r["dateTime"],r["dong"])]=(r["lon"],r["lat"],r["value"])
print("캐시된 CLD 앵커 관측:", len(cache), " 고유 지점:", len({k[1] for k in cache}))

grids={}
for day in sorted(p for p in ROOT.glob("*/*/*") if p.is_dir()):
    stem=f"{day.parent.parent.name}{day.parent.name}{day.name}"
    for f in day.glob("getGk2acldAll_CLD_*.json.gz"):
        it=json.loads(gzip.open(f).read())["response"]["body"]["items"]["item"][0]
        grids[stem+f.stem.split("_")[-1].replace(".json","")]= it

obs=[]
for (dt,dong),(lon,lat,val) in cache.items():
    it=grids.get(dt)
    if not it: continue
    gx,gy=ll2g(lon,lat,float(it["gridKm"]))
    obs.append((gx,gy,val,int(float(it["xdim"])),int(float(it["ydim"])),
                float(it["x0"]),float(it["y0"]),it["value"].split(",")))
print("비교 가능 관측:", len(obs))

best=defaultdict(list)
for xo in range(-200, 401, 1):
    for yo in range(-100, 701, 1):
        for order in ("row","col"):
            for flip in (1,-1):
                hit=n=0
                for gx,gy,val,xd,yd,x0,y0,vals in obs:
                    i=int(round(gx+xo))-int(x0)
                    j=(int(round(gy+yo))-int(y0)) if flip==1 else (int(y0)-int(round(gy+yo)))
                    if not (0<=i<xd and 0<=j<yd): break
                    idx=j*xd+i if order=="row" else i*yd+j
                    if idx>=len(vals): break
                    n+=1
                    if vals[idx]==val: hit+=1
                else:
                    if n==len(obs):
                        best[hit/n].append((xo,yo,order,flip))
if not best:
    print("모든 오프셋에서 전 관측이 격자 안에 들어오지 않음")
else:
    top=max(best)
    print(f"\n최고 일치율 {top:.4f}  — 그 값을 내는 오프셋 조합 {len(best[top])}개")
    for c in best[top][:6]: print("   xo=%4d yo=%4d %s flip=%+d" % c)
    for r in sorted(best, reverse=True)[:5]:
        print("   일치율 %.4f → 조합 %d개" % (r, len(best[r])))
