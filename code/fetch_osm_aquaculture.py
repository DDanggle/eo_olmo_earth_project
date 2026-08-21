"""OSM에서 양식장 폴리곤 중심점을 받아온다 (프로토타입 쿼리용 공개 라벨).

로컬(맥)에서 실행 → JSON 저장 → 서버로 전송해서 farm_prototype.py가 사용.
어장정보도(해수부) 전수 폴리곤으로 교체하면 정밀도가 올라간다 (백로그).

사용법:
    python fetch_osm_aquaculture.py wando 34.15 126.55 34.50 126.95
    python fetch_osm_aquaculture.py jeju  33.15 126.10 33.60 127.00
"""

import json
import sys
import urllib.parse
import urllib.request

OVERPASS = "https://overpass-api.de/api/interpreter"


def fetch(south: float, west: float, north: float, east: float) -> list[tuple[float, float]]:
    bbox = f"{south},{west},{north},{east}"
    query = (
        "[out:json][timeout:60];("
        f'way["landuse"="aquaculture"]({bbox});'
        f'relation["landuse"="aquaculture"]({bbox});'
        ");out center 200;"
    )
    req = urllib.request.Request(
        OVERPASS, data=urllib.parse.urlencode({"data": query}).encode()
    )
    with urllib.request.urlopen(req) as resp:
        elements = json.load(resp).get("elements", [])
    pts = []
    for e in elements:
        if "center" in e:
            pts.append((e["center"]["lat"], e["center"]["lon"]))
        elif "lat" in e:
            pts.append((e["lat"], e["lon"]))
    return pts


if __name__ == "__main__":
    name, s, w, n, e = sys.argv[1], *map(float, sys.argv[2:6])
    pts = fetch(s, w, n, e)
    out = f"osm_aqua_{name}.json"
    json.dump(pts, open(out, "w"))
    print(f"{name}: {len(pts)} aquaculture features -> {out}")
