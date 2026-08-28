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
MATERIALIZED_ROOT = WORK_ROOT / "artifacts/external_data/nepal_olmo_live_v1/materialized"
# 씬으로 렌더할 모드. placebo_*는 baseline의 shifted 사본이라 씬 목록에 넣지 않음
# (같은 관측이 중복 표기됨). 나중 모드가 같은 (센서, 촬영시각) 씬을 이기며 state가 live가 됨.
SCENE_MODES = ["baseline", "s2_live", "s1_live"]
DISPLAY_ANCHOR = "rasuwagadhi"
SOURCE_ROOT = MATERIALIZED_ROOT / "baseline/dataset/windows/nepal" / DISPLAY_ANCHOR
PUBLIC_DATA = APP_ROOT / "public/data"
CATALOG_ROOT = WORK_ROOT / "artifacts/external_data/nepal_olmo_live_v1/catalog"
PREFLIGHTS = {
    "s2_live": MATERIALIZED_ROOT / "s2_live/selection_preflight.json",
    "s1_live": MATERIALIZED_ROOT / "s1_live/selection_preflight.json",
}
MANIFESTS = {
    mode: MATERIALIZED_ROOT / mode / "materialization_manifest.json"
    for mode in PREFLIGHTS
}
DELTA_ROOT = WORK_ROOT / "artifacts/external_data/nepal_olmo_live_v1/delta"
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

import re as _re

_PRODUCT_TS = _re.compile(r"(\d{8})T(\d{6})")


def discover_layers(window_root: Path) -> dict[str, list[tuple[str, str]]]:
    """items.json에서 (레이어 디렉터리, ISO 촬영시각) 목록을 유도함.

    실측 규칙: serialized_item_groups[i] ↔ 레이어 디렉터리 `name`(i==0) / `name.i`(i>0).
    촬영시각은 그룹 첫 제품 이름의 YYYYMMDDTHHMMSS에서 옴 (예: S1D_..._20260824T001844_...).
    이전에는 이 대응이 하드코딩돼 있어 live 모드 장면을 넣을 수 없었음.
    """
    items = json.loads((window_root / "items.json").read_text())
    out: dict[str, list[tuple[str, str]]] = {}
    for entry in items:
        base = entry.get("layer_name")
        rows: list[tuple[str, str]] = []
        for i, group in enumerate(entry.get("serialized_item_groups") or []):
            if not group:
                continue
            m = _PRODUCT_TS.search(group[0].get("name", ""))
            if not m:
                continue
            d, t = m.group(1), m.group(2)
            iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}Z"
            layer_dir = base if i == 0 else f"{base}.{i}"
            rows.append((layer_dir, iso))
        rows.sort(key=lambda r: r[1])
        out[base] = rows
    return out


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


def load_live_observation() -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    """Join the immutable Copernicus snapshot to local rslearn readiness.

    The official catalogue and the provider used by rslearn are different
    systems.  Publication therefore never implies that OLMo input pixels have
    been selected or materialized.
    """
    latest_path = CATALOG_ROOT / "LATEST"
    if not latest_path.exists():
        return None, [], {}
    snapshot_id = latest_path.read_text().strip()
    snapshot = CATALOG_ROOT / snapshot_id
    catalog_path = snapshot / "catalog.json"
    status_path = snapshot / "acquisition_status.json"
    if not catalog_path.exists() or not status_path.exists():
        return None, [], {"catalog_snapshot": snapshot_id, "catalog_error": "snapshot_incomplete"}

    catalog = json.loads(catalog_path.read_text())
    status = json.loads(status_path.read_text())
    passes = status.get("passes", [])

    def find_product(expected_substring: str):
        return next((scene for scene in catalog.get("scenes", [])
                     if expected_substring and expected_substring in scene.get("name", "")), None)

    def readiness_for(sensor_name: str):
        """센서 live 모드의 selection과 완성된 OLMo 입력 seal을 함께 읽는다."""
        mode = "s1_live" if "1" in sensor_name.split()[0].replace("Sentinel-", "S") else "s2_live"
        if sensor_name.startswith("Sentinel-1"):
            mode = "s1_live"
        elif sensor_name.startswith("Sentinel-2"):
            mode = "s2_live"
        preflight_path = PREFLIGHTS.get(mode)
        manifest_path = MANIFESTS.get(mode)
        preflight = (json.loads(preflight_path.read_text())
                     if preflight_path and preflight_path.exists() else None)
        manifest = (json.loads(manifest_path.read_text())
                    if manifest_path and manifest_path.exists() else None)
        return mode, preflight, manifest

    # published인 pass들 중 가장 최근을 대표 live 관측으로 삼음. 이전에는 s2b_20260827이
    # 하드코딩돼 있어 S1D 등 이후 관측을 표시할 수 없었음.
    published = [row for row in passes if row.get("status") == "published"]
    published.sort(key=lambda r: r.get("start_utc", ""), reverse=True)
    live_observation = None
    if published:
        row = published[0]
        product = find_product(row.get("expected_product_substring", ""))
        mode, preflight, materialization = readiness_for(row.get("sensor", ""))
        selection_ready = bool(preflight and preflight.get("valid"))
        seal_ready = bool(materialization and materialization.get("valid"))
        if selection_ready and seal_ready:
            materialization_status = "sealed_olmo_input"
        elif selection_ready and materialization:
            materialization_status = "partial_cube_contract_failed"
        elif selection_ready:
            materialization_status = "selected_not_materialized"
        elif preflight:
            materialization_status = "blocked_provider_selection"
        else:
            materialization_status = "not_preflighted"
        period_readiness: dict[str, int] = {}
        period_audit = (materialization or {}).get("period_audit") or {}
        for layer_name in ("sentinel1", "sentinel2_l2a"):
            counts = [len((audit.get("completed_layers") or {}).get(layer_name, []))
                      for audit in period_audit.values()]
            if counts:
                period_readiness[layer_name] = min(counts)
        live_observation = {
            "sensor": row["sensor"],
            "acquired_at": (product or {}).get("sensing_start_utc", row["start_utc"]),
            "catalog_status": row["status"],
            "product_name": (product or {}).get("name"),
            "product_id": (product or {}).get("id"),
            "publication_utc": (product or {}).get("publication_utc"),
            "cloud_cover_tile_pct": (product or {}).get("cloud_cover"),
            "online": (product or {}).get("online"),
            "relative_orbit": (product or {}).get("relative_orbit"),
            "catalog_provider": "Copernicus Data Space OData",
            "materialization_provider": "Microsoft Planetary Computer STAC via rslearn",
            "materialization_mode": mode,
            "materialization_status": materialization_status,
            "selection_preflight_valid": selection_ready,
            "materialization_seal_valid": seal_ready,
            "period_readiness": period_readiness,
            "olmo_ready": selection_ready and seal_ready,
            "claim_boundary": "Tile cloud and B02-bright fractions are not AOI cloud-free coverage; no post-event embedding is claimed.",
        }

    scheduled = []
    for row in passes:
        state = row["status"]
        if state == "published":
            continue  # published는 live_observation/씬 쪽에서 다룸
        scheduled.append({"sensor": row["sensor"], "acquired_at": row["start_utc"], "state": state})
    scheduled = scheduled[:3]

    provenance = {
        "catalog_snapshot": snapshot_id,
        "catalog_generated_at_utc": catalog.get("generated_at_utc"),
        "catalog_sha256": sha256(catalog_path),
        "acquisition_status_sha256": sha256(status_path),
        "catalog_seal_sha256": sha256(snapshot / "SHA256SUMS"),
    }
    for mode, path in PREFLIGHTS.items():
        if path.exists():
            provenance[f"{mode}_selection_preflight_sha256"] = sha256(path)
    for mode, path in MANIFESTS.items():
        if path.exists():
            provenance[f"{mode}_materialization_manifest_sha256"] = sha256(path)
    return live_observation, scheduled, provenance


AOI_OBS = WORK_ROOT / "artifacts/aoi_observability_20260827.json"


def build_ops_log() -> list[dict[str, Any]]:
    """EarthRanger식 이벤트 레코드 피드 — 전부 실제 산출물의 타임스탬프에서 옴.

    이벤트 = {time_utc, source, type, priority(green|orange|blue), summary}.
    파이프라인이 한 일(그리고 거부한 일)을 감사 가능한 로그로 노출함.
    """
    events: list[dict[str, Any]] = []

    def add(t, source, etype, priority, summary, evidence_uri=None, time_basis="artifact_recorded_at"):
        if t:
            identity = hashlib.sha256(
                f"{t}|{source}|{etype}|{summary}|{evidence_uri or ''}".encode()
            ).hexdigest()[:16]
            events.append({"event_id": identity, "time_utc": t, "recorded_at_utc": t,
                           "time_basis": time_basis, "source": source, "type": etype,
                           "priority": priority, "summary": summary,
                           "evidence_uri": evidence_uri})

    # 카탈로그 스냅샷들
    if CATALOG_ROOT.exists():
        for snap in sorted(CATALOG_ROOT.iterdir()):
            st = snap / "acquisition_status.json"
            if not st.exists():
                continue
            d = json.loads(st.read_text())
            add(d.get("evaluated_at_utc"), "Copernicus OData", "CATALOG_SNAPSHOT", "blue",
                f"snapshot {snap.name}: " + ", ".join(
                    f"{r['id']}={r['status']}" for r in d.get("passes", [])[:3]),
                str(st.relative_to(WORK_ROOT)))
            for r in d.get("passes", []):
                if r.get("status") == "published" and r.get("catalog_matches"):
                    add(d.get("evaluated_at_utc"), "Copernicus OData", "SCENE_PUBLISHED",
                        "green", f"{r['sensor']} {r['id']} published "
                        f"(latency {r.get('publication_latency_minutes','?')} min)",
                        str(st.relative_to(WORK_ROOT)))

    # preflight / manifest / 임베딩
    for mode_dir in sorted(MATERIALIZED_ROOT.iterdir()) if MATERIALIZED_ROOT.exists() else []:
        if not mode_dir.is_dir():
            continue
        name = mode_dir.name
        pf = mode_dir / "selection_preflight.json"
        if pf.exists():
            d = json.loads(pf.read_text())
            ok = bool(d.get("valid"))
            found = d.get("anchor_count", len(d.get("anchors") or []))
            expected = d.get("expected_anchor_count", "?")
            event_time = d.get("checked_at_utc") or d.get("evaluated_at_utc")
            add(event_time or datetime.fromtimestamp(
                    pf.stat().st_mtime, UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "rslearn preflight", "PREFLIGHT_PASS" if ok else "PREFLIGHT_BLOCK",
                "green" if ok else "orange",
                f"{name}: {found}/{expected} anchors " + ("selected required scene" if ok
                 else "missing required scene — download refused"),
                str(pf.relative_to(WORK_ROOT)),
                "artifact_field" if event_time else "filesystem_mtime")
        mf = mode_dir / "materialization_manifest.json"
        if mf.exists():
            d = json.loads(mf.read_text())
            ok = bool(d.get("valid"))
            period_audit = d.get("period_audit") or {}
            layer_counts = {}
            for layer_name in ("sentinel1", "sentinel2_l2a"):
                counts = [len((audit.get("completed_layers") or {}).get(layer_name, []))
                          for audit in period_audit.values()]
                if counts:
                    layer_counts[layer_name] = min(counts)
            period_summary = (f"S1 {layer_counts.get('sentinel1', '?')}/4, "
                              f"S2 {layer_counts.get('sentinel2_l2a', '?')}/4")
            add(d.get("created_at_utc"), "rslearn materialize",
                "SEALED" if ok else "SEAL_INVALID", "green" if ok else "orange",
                f"{name}: {d.get('file_count','?')} files, "
                f"{d.get('total_bytes',0):,} B — " + ("sealed" if ok else
                 f"contract failed ({period_summary})"),
                str(mf.relative_to(WORK_ROOT)))
        emb = list(mode_dir.glob("dataset/windows/nepal/*/layers/embeddings/**/*.tif"))
        if len(emb) >= 5:
            t = datetime.fromtimestamp(max(e.stat().st_mtime for e in emb), UTC)
            add(t.isoformat(timespec="seconds").replace("+00:00", "Z"),
                "OLMoEarth v1", "EMBEDDED", "green",
                f"{name}: 5 anchors × 768-d cube",
                str(mode_dir.relative_to(WORK_ROOT)), "filesystem_mtime")

    # AOI 관측성 (밝기 휴리스틱 — SCL 없음을 명시)
    if AOI_OBS.exists():
        d = json.loads(AOI_OBS.read_text())
        ras = (d.get("anchors") or {}).get("rasuwagadhi")
        if ras:
            add(d.get("created_at_utc") or datetime.fromtimestamp(AOI_OBS.stat().st_mtime, UTC)
                    .isoformat(timespec="seconds").replace("+00:00", "Z"),
                "AOI heuristic", "OBSERVABILITY", "blue",
                f"8/27 S2: Rasuwagadhi AOI bright {ras['bright_frac_of_valid']*100:.1f}% "
                f"(tile cloud 78.5% ≠ AOI; not a cloud-free estimate)",
                str(AOI_OBS.relative_to(WORK_ROOT)),
                "artifact_field" if d.get("created_at_utc") else "filesystem_mtime")

    # Δz 리포트
    if DELTA_ROOT.exists():
        for rp in sorted(DELTA_ROOT.glob("*/nepal_delta_report.json")):
            d = json.loads(rp.read_text())
            add(d.get("created_at_utc"), "OLMoEarth Δz", "DELTA_REPORT", "green",
                f"live={d.get('live_mode')} placebo n={len(d.get('placebo_modes_available', []))}",
                str(rp.relative_to(WORK_ROOT)))

    events.sort(key=lambda e: e["time_utc"], reverse=True)
    return events[:30]


def build_decision(live_observation: dict[str, Any] | None,
                   scheduled_scenes: list[dict[str, Any]],
                   olmoearth: dict[str, Any]) -> dict[str, str]:
    """현재 허용되는 다음 action을 UI가 한 문장으로 답하게 한다."""
    ped = olmoearth.get("post_event_delta")
    # placebo 전용 리포트(live_mode=None)를 "post-event delta 있음"으로 승격하면 안 됨 —
    # 2026-08-28 실측: 그 오독으로 카드가 REVIEW CANDIDATE EVIDENCE 를 잘못 표시했음.
    if isinstance(ped, dict) and ped.get("live_mode") and ped.get("rasuwagadhi_live_mean") is not None:
        return {
            "status": "candidate_ready",
            "action": "REVIEW CANDIDATE EVIDENCE",
            "reason": "A sealed post-event cube and OLMoEarth delta report are available.",
            "next_gate": "Seek independent sensor, physical, or human corroboration.",
            "allowed_claim": "Candidate representation change; not damage probability or hazard extent.",
        }
    if live_observation and live_observation.get("olmo_ready"):
        return {
            "status": "embed_ready",
            "action": "QUEUE SEALED EMBEDDING",
            "reason": "Scene selection and the full 5-anchor input seal both passed.",
            "next_gate": "Run one immutable OLMoEarth recipe, then compare against placebo windows.",
            "allowed_claim": "Input contract ready; no representation change result yet.",
        }
    if live_observation:
        periods = live_observation.get("period_readiness") or {}
        reason = (f"Post-event scene selected, but cube seal failed: "
                  f"S1 {periods.get('sentinel1', '?')}/4; "
                  f"S2 {periods.get('sentinel2_l2a', '?')}/4 per anchor.")
        next_scene = scheduled_scenes[0] if scheduled_scenes else None
        next_gate = (f"Wait for {next_scene['sensor']} at {next_scene['acquired_at']}, materialize, then reseal 5/5 anchors."
                     if next_scene else "Acquire the missing modality period and reseal 5/5 anchors.")
        return {
            "status": "hold",
            "action": "DO NOT EMBED",
            "reason": reason,
            "next_gate": next_gate,
            "allowed_claim": "Post-event pixels exist; OLMoEarth evidence does not.",
        }
    return {
        "status": "wait_observation",
        "action": "WAIT FOR OBSERVATION",
        "reason": "No published post-event scene is joined to this snapshot.",
        "next_gate": "Refresh the immutable catalog snapshot after the next acquisition.",
        "allowed_claim": "Pre-event baseline only.",
    }


def olmoearth_block() -> dict[str, Any]:
    """임베딩·Δz 산출물이 실재하면 실측값으로, 없으면 정직한 대기 문구로 채움."""
    embedded_modes = []
    for mode_dir in sorted(MATERIALIZED_ROOT.iterdir()) if MATERIALIZED_ROOT.exists() else []:
        if not mode_dir.is_dir() or mode_dir.name.startswith("baseline_failed"):
            continue
        emb = list(mode_dir.glob("dataset/windows/nepal/*/layers/embeddings/**/*.tif"))
        if len(emb) >= 5:
            embedded_modes.append(mode_dir.name)
    latest_delta = None
    if DELTA_ROOT.exists():
        reports = sorted(DELTA_ROOT.glob("*/nepal_delta_report.json"))
        if reports:
            latest_delta = json.loads(reports[-1].read_text())
    block: dict[str, Any] = {
        "model": "OLMoEarth Base v1 (768-d)",
        "input_contract": "S1 RTC VV/VH + S2 L2A 12-band, 10 m, 4 periods, 2.56 km windows",
        "anchors": 5,
        "rasuwagadhi_baseline": "materialized_and_sealed",
        "embedding_status": (f"embedded: {', '.join(embedded_modes)}" if embedded_modes
                             else "not_run_in_this_web_snapshot"),
        "anchor_geojson": "/data/olmo-input-anchors.geojson",
    }
    if latest_delta:
        ras = latest_delta.get("anchors", {}).get(DISPLAY_ANCHOR, {})
        verdict = ras.get("verdict", {})
        block["post_event_delta"] = {
            "created_at_utc": latest_delta.get("created_at_utc"),
            "live_mode": latest_delta.get("live_mode"),
            "placebo_samples": len(latest_delta.get("placebo_modes_available", [])),
            "rasuwagadhi_live_mean": ras.get("live_delta", {}).get("mean"),
            "label": verdict.get("label"),
            "claim_boundary": "candidate change only; not damage probability",
        }
    else:
        block["post_event_delta"] = "blocked_until_olmo_ready_post_cube"
    return block


def build(refresh_osm: bool) -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Missing materialized source: {SOURCE_ROOT}")
    scenes_dir = PUBLIC_DATA / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    scene_records: list[dict[str, Any]] = []
    band_path = "B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12/geotiff.tif"
    seen_scene_ids: set[str] = set()
    for mode in SCENE_MODES:
        window_root = MATERIALIZED_ROOT / mode / "dataset/windows/nepal" / DISPLAY_ANCHOR
        manifest_path = MATERIALIZED_ROOT / mode / "materialization_manifest.json"
        if not window_root.exists() or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if mode == "baseline":
            if not manifest.get("valid"):
                continue
            state = "baseline_ready"
        else:
            # live 모드는 부분 완성도 보여줌 — 예: 8/27 S2는 물질화됐지만 S1 4기간이
            # 아직 안 차서 seal invalid인 상태. 장면 자체는 실측이므로 표시하되
            # state로 정직하게 구분함 (live_partial = 임베딩 게이트 미통과).
            state = "live_ready" if manifest.get("valid") else "live_partial"
        layers = discover_layers(window_root)
        for layer, timestamp in layers.get("sentinel2_l2a", []):
            scene_id = f"s2-{timestamp[:10]}"
            source = window_root / "layers" / layer / band_path
            if not source.exists():
                continue
            if scene_id in seen_scene_ids:
                # live 모드가 같은 관측을 다시 담으면 state만 승격함
                for rec in scene_records:
                    if rec["id"] == scene_id and state.startswith("live"):
                        rec["state"] = state
                continue
            seen_scene_ids.add(scene_id)
            destination = scenes_dir / f"{scene_id}.png"
            rendered = render_s2(source, destination)
            scene_records.append({
                "id": scene_id, "sensor": "Sentinel-2 L2A", "acquired_at": timestamp,
                "state": state, "image": f"/data/scenes/{destination.name}",
                "coordinates": rendered["coordinates"], "source_sha256": sha256(source),
                **rendered["stats"],
            })
        for layer, timestamp in layers.get("sentinel1", []):
            scene_id = f"s1-{timestamp[:10]}"
            source = window_root / "layers" / layer / "vv_vh/geotiff.tif"
            if not source.exists():
                continue
            if scene_id in seen_scene_ids:
                for rec in scene_records:
                    if rec["id"] == scene_id and state.startswith("live"):
                        rec["state"] = state
                continue
            seen_scene_ids.add(scene_id)
            destination = scenes_dir / f"{scene_id}.png"
            rendered = render_s1(source, destination)
            scene_records.append({
                "id": scene_id, "sensor": "Sentinel-1 RTC VV/VH", "acquired_at": timestamp,
                "state": state, "image": f"/data/scenes/{destination.name}",
                "coordinates": rendered["coordinates"], "source_sha256": sha256(source),
                **rendered["stats"],
            })

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

    live_observation, scheduled_scenes, live_provenance = load_live_observation()
    olmoearth = olmoearth_block()
    if live_observation and live_observation["catalog_status"] == "published":
        evidence_status = (
            "Post-event pixels and the full OLMoEarth input seal are ready; embedding has not run."
            if live_observation.get("olmo_ready")
            else "Post-event pixels exist, but the full OLMoEarth input contract is not sealed."
        )
    else:
        evidence_status = "Post-event open satellite scene pending in this snapshot."

    manifest = {
        "schema": "olmoearth-nepal-live-twin/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "event": {
            "name": "2026 Rasuwa–Bhote Koshi flash flood",
            "occurred_at": "2026-08-26T03:15:00Z",
            "cause_status": "Glacier/ice collapse and temporary debris blockage are under investigation; not an earthquake forecast.",
            "evidence_status": evidence_status,
        },
        "points": POINTS,
        "scene_records": sorted(scene_records, key=lambda item: item["acquired_at"]),
        "scheduled_scenes": scheduled_scenes,
        "live_observation": live_observation,
        "olmoearth": olmoearth,
        "decision": build_decision(live_observation, scheduled_scenes, olmoearth),
        "ops_log": build_ops_log(),
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
            **live_provenance,
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
