#!/usr/bin/env python3
"""Compile research artifacts into browser-safe GIS assets.

Python owns geospatial I/O and provenance. The deployed UI consumes only the
generated PNG/GeoJSON/JSON files, so the Cloudflare runtime never needs Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import transform_bounds


APP_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = APP_ROOT.parents[1]
SOURCE_ROOT = (
    WORK_ROOT
    / "artifacts/external_data/nepal_olmo_live_v1/materialized/baseline/dataset/windows/nepal/rasuwagadhi"
)
PUBLIC_DATA = APP_ROOT / "public/data"
ROUTE_WAY_IDS = [201928141, 809865767, 24624604]

POINTS = [
    {
        "id": "A",
        "name": "Rasuwagadhi impact AOI",
        "coordinates": [85.3780644, 28.2786794],
        "role": "impact_focus",
        "place": "Pasang Lhamu Highway, Rasuwagadhi, Rasuwa, Nepal",
        "source": "user coordinate + OSM Nominatim reverse lookup",
    },
    {
        "id": "B",
        "name": "Gyirong border checkpoint",
        "coordinates": [85.3763336, 28.2828546],
        "role": "border_checkpoint",
        "place": "G216, Gyirong Town, Tibet, China",
        "source": "user coordinate + OSM Nominatim reverse lookup",
    },
    {
        "id": "C",
        "name": "Rishing reference",
        "coordinates": [84.3103107, 27.8790412],
        "role": "distant_reference",
        "place": "Rishing-03, Tanahun, Gandaki Province, Nepal",
        "source": "user coordinate + OSM Nominatim reverse lookup",
    },
]

S2_LAYERS = [
    ("sentinel2_l2a.3", "2026-07-03T04:57:01Z"),
    ("sentinel2_l2a.2", "2026-07-23T04:57:01Z"),
    ("sentinel2_l2a.1", "2026-08-07T04:56:59Z"),
    ("sentinel2_l2a", "2026-08-12T04:57:01Z"),
]
S1_LAYERS = [
    ("sentinel1.3", "2026-07-11T12:21:39Z"),
    ("sentinel1.2", "2026-07-23T12:21:40Z"),
    ("sentinel1.1", "2026-08-04T12:21:40Z"),
    ("sentinel1", "2026-08-24T00:18:44Z"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def haversine_km(first: list[float], second: list[float]) -> float:
    lon1, lat1 = map(math.radians, first)
    lon2, lat2 = map(math.radians, second)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(value))


def image_coordinates(dataset: rasterio.io.DatasetReader) -> list[list[float]]:
    left, bottom, right, top = transform_bounds(dataset.crs, "EPSG:4326", *dataset.bounds)
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def stretch(channel: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    finite = channel[np.isfinite(channel) & (channel > 0)]
    if finite.size == 0:
        return np.zeros(channel.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        hi = lo + 1
    scaled = np.clip((channel - lo) / (hi - lo), 0, 1)
    return np.round(np.power(scaled, 0.86) * 255).astype(np.uint8)


def render_s2(source: Path, destination: Path) -> dict[str, Any]:
    with rasterio.open(source) as dataset:
        data = dataset.read([4, 3, 2]).astype(np.float32)
        rgb = np.stack([stretch(data[index]) for index in range(3)], axis=-1)
        alpha = np.where(np.any(data > 0, axis=0), 238, 0).astype(np.uint8)
        image = np.dstack([rgb, alpha])
        coordinates = image_coordinates(dataset)
        stats = {"shape": [dataset.count, dataset.height, dataset.width], "crs": str(dataset.crs)}
    Image.fromarray(image).save(destination, optimize=True)
    return {"coordinates": coordinates, "stats": stats}


def render_s1(source: Path, destination: Path) -> dict[str, Any]:
    with rasterio.open(source) as dataset:
        vv, vh = dataset.read([1, 2]).astype(np.float32)
        vv_rgb, vh_rgb = stretch(vv, 1, 99), stretch(vh, 1, 99)
        ratio_rgb = stretch(vv - vh, 2, 98)
        image = np.dstack([vv_rgb, vh_rgb, ratio_rgb, np.full(vv.shape, 225, dtype=np.uint8)])
        coordinates = image_coordinates(dataset)
        stats = {"shape": [dataset.count, dataset.height, dataset.width], "crs": str(dataset.crs)}
    Image.fromarray(image).save(destination, optimize=True)
    return {"coordinates": coordinates, "stats": stats}


def fetch_hydrography(destination: Path) -> dict[str, Any]:
    query = f'[out:json][timeout:25];way(id:{",".join(map(str, ROUTE_WAY_IDS))});out tags geom;'
    url = "https://overpass-api.de/api/interpreter?" + urllib.parse.urlencode({"data": query})
    request = urllib.request.Request(url, headers={"User-Agent": "olmoearth-live-twin/0.1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = json.load(response)

    by_id = {item["id"]: item for item in payload["elements"]}
    features = []
    route: list[list[float]] = []
    for way_id in ROUTE_WAY_IDS:
        element = by_id[way_id]
        coordinates = [[point["lon"], point["lat"]] for point in element["geometry"]]
        if route and route[-1] == coordinates[0]:
            route.extend(coordinates[1:])
        else:
            route.extend(coordinates)
        features.append(
            {
                "type": "Feature",
                "properties": {"osm_way_id": way_id, **element.get("tags", {})},
                "geometry": {"type": "LineString", "coordinates": coordinates},
            }
        )

    step = max(1, math.ceil(len(route) / 80))
    wasm_route = route[::step]
    if wasm_route[-1] != route[-1]:
        wasm_route.append(route[-1])
    geojson = {
        "type": "FeatureCollection",
        "name": "Bhote Koshi to Trishuli verified river centerline",
        "license": "OpenStreetMap ODbL",
        "fetched_at": datetime.now(UTC).isoformat(),
        "features": features,
        "simulation_route": wasm_route,
    }
    destination.write_text(json.dumps(geojson, indent=2, ensure_ascii=False) + "\n")
    return geojson


def build(refresh_osm: bool) -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Missing materialized source: {SOURCE_ROOT}")
    scenes_dir = PUBLIC_DATA / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    scene_records: list[dict[str, Any]] = []
    band_path = "B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12/geotiff.tif"
    for layer, timestamp in S2_LAYERS:
        source = SOURCE_ROOT / "layers" / layer / band_path
        destination = scenes_dir / f"s2-{timestamp[:10]}.png"
        rendered = render_s2(source, destination)
        scene_records.append(
            {
                "id": f"s2-{timestamp[:10]}",
                "sensor": "Sentinel-2 L2A",
                "acquired_at": timestamp,
                "state": "baseline_ready",
                "image": f"/data/scenes/{destination.name}",
                "coordinates": rendered["coordinates"],
                "source_sha256": sha256(source),
                **rendered["stats"],
            }
        )
    for layer, timestamp in S1_LAYERS:
        source = SOURCE_ROOT / "layers" / layer / "vv_vh/geotiff.tif"
        destination = scenes_dir / f"s1-{timestamp[:10]}.png"
        rendered = render_s1(source, destination)
        scene_records.append(
            {
                "id": f"s1-{timestamp[:10]}",
                "sensor": "Sentinel-1 RTC VV/VH",
                "acquired_at": timestamp,
                "state": "baseline_ready",
                "image": f"/data/scenes/{destination.name}",
                "coordinates": rendered["coordinates"],
                "source_sha256": sha256(source),
                **rendered["stats"],
            }
        )

    hydrography_path = PUBLIC_DATA / "hydrography.geojson"
    if refresh_osm or not hydrography_path.exists():
        hydrography = fetch_hydrography(hydrography_path)
    else:
        hydrography = json.loads(hydrography_path.read_text())

    montage_source = WORK_ROOT / "artifacts/nepal_olmo_live_v1/pre_event_input_montage.png"
    if montage_source.exists():
        shutil.copy2(montage_source, PUBLIC_DATA / "pre-event-input-montage.png")

    anchor_features = []
    nepal_root = SOURCE_ROOT.parent
    for anchor_root in sorted(path for path in nepal_root.iterdir() if path.is_dir()):
        raster = anchor_root / "layers/sentinel2_l2a" / band_path
        if not raster.exists():
            continue
        with rasterio.open(raster) as dataset:
            left, bottom, right, top = transform_bounds(dataset.crs, "EPSG:4326", *dataset.bounds)
        anchor_features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": anchor_root.name,
                    "status": "input_materialized",
                    "interpretation": "provisional" if anchor_root.name == "source_provisional" else "operational_anchor",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[left, top], [right, top], [right, bottom], [left, bottom], [left, top]]],
                },
            }
        )
    anchors_geojson = {"type": "FeatureCollection", "features": anchor_features}
    anchors_path = PUBLIC_DATA / "olmo-input-anchors.geojson"
    anchors_path.write_text(json.dumps(anchors_geojson, indent=2, ensure_ascii=False) + "\n")

    point_a = POINTS[0]["coordinates"]
    for point in POINTS:
        point["distance_from_a_km"] = round(haversine_km(point_a, point["coordinates"]), 2)

    manifest = {
        "schema": "olmoearth-nepal-live-twin/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "event": {
            "name": "2026 Rasuwa–Bhote Koshi flash flood",
            "occurred_at": "2026-08-26T03:15:00Z",
            "cause_status": "Glacier/ice collapse and temporary debris blockage are under investigation; not an earthquake forecast.",
            "evidence_status": "Post-event open satellite scene pending in this snapshot.",
        },
        "points": POINTS,
        "scene_records": sorted(scene_records, key=lambda item: item["acquired_at"]),
        "scheduled_scenes": [
            {"sensor": "Sentinel-2", "acquired_at": "2026-08-27T04:56:52Z", "state": "catalog_pending"},
            {"sensor": "Sentinel-1", "acquired_at": "2026-08-28T12:19:28Z", "state": "planned"},
        ],
        "olmoearth": {
            "model": "OLMoEarth Base",
            "input_contract": "S1 RTC VV/VH + S2 L2A 12-band, 10 m, 4 periods, 2.56 km windows",
            "anchors": 5,
            "rasuwagadhi_baseline": "materialized_and_sealed",
            "embedding_status": "not_run_in_this_web_snapshot",
            "post_event_delta": "blocked_until_post_scene",
            "anchor_geojson": "/data/olmo-input-anchors.geojson",
        },
        "simulation": {
            "engine": "Rust/WASM deterministic particle preview",
            "route_source": "OSM ways 201928141, 809865767, 24624604",
            "route_points": len(hydrography["simulation_route"]),
            "claim": "illustrative_kinematic_preview_not_hazard_forecast",
        },
        "provenance": {
            "source_root": str(SOURCE_ROOT),
            "metadata_sha256": sha256(SOURCE_ROOT / "metadata.json"),
            "items_sha256": sha256(SOURCE_ROOT / "items.json"),
            "hydrography_sha256": sha256(hydrography_path),
        },
    }
    (PUBLIC_DATA / "scenario.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"scenes": len(scene_records), "route_points": len(hydrography["simulation_route"]), "output": str(PUBLIC_DATA)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-osm", action="store_true", help="Refresh the three verified river ways from Overpass")
    args = parser.parse_args()
    build(refresh_osm=args.refresh_osm)


if __name__ == "__main__":
    main()
