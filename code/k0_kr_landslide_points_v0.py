#!/usr/bin/env python3
"""K0 step 1 (local): Korean landslide occurrence records -> points (lon/lat), one table.

Sources (data.go.kr file data, downloaded to data_kr/, free without login):
  15031926 경기도 가평군 산사태 발생현황   722 rows, 지번주소 + 피해규모(ha), 2006-2025 (2025-07: 405, 2020-08: 91)
  15122494 전북 남원시 산사태 발생현황     217 rows, 2020-08 집중호우, 발생위치 = 지번 (+ '일원')
  15134897 충남 산사태 발생이력           146 rows, 경도/위도 in EPSG:3857, 2011/2012/2017
Parcel addresses are geocoded with the VWorld address API (type=parcel; key from .env, never written out). A parcel
point is the parcel's representative point, not the scar: mountain parcels can be large -> positional error of
100s of m is expected and must be checked visually (K0 step 2).

  python3 -B code/k0_kr_landslide_points_v0.py        -> data_kr/k0_points.json
"""
import csv
import io
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
D = HERE / "data_kr"


def read(f):
    b = f.read_bytes()
    for e in ("utf-8-sig", "cp949"):
        try:
            return list(csv.DictReader(io.StringIO(b.decode(e))))
        except UnicodeDecodeError:
            pass
    raise ValueError(f)


def clean(addr):
    a = re.sub(r"\s+", " ", addr.strip())
    a = re.sub(r"\s*(외\s*\d*\s*(필지)?|일원|일대)\s*$", "", a)
    return a.strip()


def geocode(addr, key, dom):
    q = urllib.parse.urlencode({"service": "address", "request": "getcoord", "version": "2.0", "crs": "epsg:4326", "address": addr,
                                "refine": "true", "simple": "false", "format": "json", "type": "parcel", "key": key, "domain": dom})
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen("https://api.vworld.kr/req/address?" + q, timeout=20))["response"]
            if r.get("status") == "OK":
                p = r["result"]["point"]
                return float(p["x"]), float(p["y"]), r.get("refined", {}).get("text")
            return None
        except Exception:
            time.sleep(1)
    return None


def main():
    for line in (HERE / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    key, dom = os.environ["VWORLD_API_KEY"], os.environ.get("VWORLD_API_DOMAIN", "")
    pts = []
    for r in read(D / "kr_15031926.csv"):
        pts.append({"src": "gapyeong", "date": r["발생일자"], "addr": r["지번주소"], "ha": float(r["피해규모"]) if r["피해규모"].strip() else None})
    for r in read(D / "kr_15122494.csv"):
        pts.append({"src": "namwon", "date": "2020-08", "addr": r["산사태 발생위치"], "ha": None})
    out = []
    for p in pts:
        g = geocode(clean(p["addr"]), key, dom)
        out.append({**p, "addr": clean(p["addr"]), "lon": g[0] if g else None, "lat": g[1] if g else None, "geocoded": bool(g)})
    for r in read(D / "kr_15134897.csv"):
        x, y = float(r["경도"]), float(r["위도"])
        out.append({"src": "chungnam", "date": r["발생일시"], "addr": r["주소"], "ha": None, "geocoded": False,
                    "lon": x / 6378137 * 180 / math.pi, "lat": math.degrees(math.atan(math.sinh(y / 6378137)))})
    (D / "k0_points.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))
    from collections import Counter
    print(json.dumps({"n": len(out), "with_coords": sum(o["lon"] is not None for o in out),
                      "by_src_month": Counter(f"{o['src']} {o['date'][:7]}" for o in out if o["lon"] is not None).most_common(12)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
