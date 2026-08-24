#!/usr/bin/env python3
"""Build a season-aligned, editable visual audit of Jeju change candidates.

Candidate selection is deterministic from the preserved v3/v6 rankings and is
performed before rendering any new RGB chips.  Every candidate is shown at the
same target season (closest acquisition to 15 May) and the same fixed reflectance
stretch for 2023--2026.  The output HTML keeps algorithm scores separate from
human labels and lets a reviewer export an override JSON file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __name__ == "__main__" and os.environ.get("ALLOW_HISTORICAL_INVALID_JEJU_CANDIDATES") != "1":
    raise SystemExit(
        "REFUSED: the preserved v3/v6 rankings contain overlapping and "
        "season-confounded time contracts. This renderer is historical-audit only. "
        "Set ALLOW_HISTORICAL_INVALID_JEJU_CANDIDATES=1 to reproduce it explicitly."
    )

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window


YEARS = ("2023", "2024", "2025", "2026")
YEAR_PREFIX = {
    "2023": "jeju23_",
    "2024": "jeju_",
    "2025": "jeju25_",
    "2026": "jeju26r_",
}
RGB_INDEXES = (4, 3, 2)
B02_INDEX = 2
CLOUD_DN = 1800
TARGET_MONTH = 5
TARGET_DAY = 15
CONTEXT_RADIUS = 64  # 1.28 km wide at 10 m.
DETAIL_RADIUS = 20  # 400 m wide at 10 m.
FIXED_RGB_MAX = 3000.0

LABEL_OPTIONS = [
    ["persistent_development_or_clearing", "개발·벌채처럼 지속되는 지표 변화"],
    ["oreum_or_vegetation_seasonality", "오름·초지·농경지의 계절/식생 변화"],
    ["cloud_haze_shadow", "구름·해무·그림자 오염"],
    ["coastal_water_reflectance", "바다·연안 반사 변화"],
    ["stable_or_unclear", "안정 또는 불명확"],
]


@dataclass(frozen=True)
class WindowRecord:
    year: str
    path: Path
    crs: Any
    transform: Any
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--windows-root",
        default="/home/work/data/olmoearth/embed_search/dataset/windows/default",
    )
    parser.add_argument(
        "--v3-json",
        default="/home/work/data/olmoearth/embed_search/jeju_change_v3_top.json",
    )
    parser.add_argument(
        "--v6-json",
        default="/home/work/data/olmoearth/embed_jeju_v2/jeju_change_v6_top.json",
    )
    parser.add_argument(
        "--out",
        default="/home/work/data/olmoearth/embed_jeju_v2/human_review_v1",
    )
    return parser.parse_args()


def distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(float(a["lat"]) - float(b["lat"]), float(a["lon"]) - float(b["lon"]))


def with_source(
    raw: dict[str, Any], cohort: str, source_name: str, source_rank: int, candidate_id: str
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "cohort": cohort,
        "lat": float(raw["lat"]),
        "lon": float(raw["lon"]),
        "algorithm": {
            "source": source_name,
            "rank": source_rank,
            "z": float(raw["z"]),
            "when": raw["when"],
            "landcover": raw["landcover"],
            "cloud_max": raw.get("cloud_max"),
        },
        "human_review": {
            "label": None,
            "is_persistent_change": None,
            "confidence": None,
            "notes": "",
        },
    }


def select_candidates(v3: dict[str, Any], v6: dict[str, Any]) -> list[dict[str, Any]]:
    """Pre-register candidates from ranking and coordinate rules only."""
    v3_top = v3["top"]
    v6_top = v6["top_12ts"]
    selected: list[dict[str, Any]] = []

    # Positive control recorded before this audit in GOAL/README.
    control_index = min(
        range(len(v3_top)),
        key=lambda i: math.hypot(v3_top[i]["lat"] - 33.5087, v3_top[i]["lon"] - 126.5747),
    )
    selected.append(
        with_source(
            v3_top[control_index],
            "development_controls",
            "v3_top",
            control_index + 1,
            f"dev_control_v3_r{control_index + 1:02d}",
        )
    )

    # First three built-class candidates in the full-year v6 ranking.
    built_count = 0
    for rank, raw in enumerate(v6_top, start=1):
        if raw["landcover"] != "built":
            continue
        selected.append(
            with_source(
                raw,
                "development_controls",
                "v6_top_12ts",
                rank,
                f"built_v6_r{rank:02d}",
            )
        )
        built_count += 1
        if built_count == 3:
            break

    # Eastern mid-mountain cluster, fixed before RGB review. This region contains
    # multiple mapped tree/grass candidates and is treated as an oreum/vegetation
    # cohort, not as known development.
    oreum_raw: list[tuple[int, dict[str, Any]]] = []
    for rank, raw in enumerate(v6_top, start=1):
        if not (33.30 <= raw["lat"] <= 33.40 and 126.64 <= raw["lon"] <= 126.72):
            continue
        if raw["landcover"] not in {"tree", "grass"}:
            continue
        if any(distance(raw, prior) < 0.008 for _, prior in oreum_raw):
            continue
        oreum_raw.append((rank, raw))
        if len(oreum_raw) == 6:
            break
    for rank, raw in oreum_raw:
        selected.append(
            with_source(
                raw,
                "eastern_midmountain_cluster",
                "v6_top_12ts",
                rank,
                f"oreum_v6_r{rank:02d}",
            )
        )

    # Spatially separated v3 high-score controls. The same 0.02 degree rule was
    # used by the preserved verify_candidates.py, but here we retain four after
    # excluding the already recorded development control.
    context_raw: list[tuple[int, dict[str, Any]]] = []
    for rank, raw in enumerate(v3_top, start=1):
        if rank == control_index + 1:
            continue
        if any(distance(raw, prior) < 0.02 for _, prior in context_raw):
            continue
        context_raw.append((rank, raw))
        if len(context_raw) == 4:
            break
    for rank, raw in context_raw:
        selected.append(
            with_source(
                raw,
                "v3_spatial_controls",
                "v3_top",
                rank,
                f"context_v3_r{rank:02d}",
            )
        )

    if len(selected) != 14:
        raise RuntimeError(f"expected 14 pre-registered candidates, got {len(selected)}")
    return selected


def natural_layer_index(path: Path) -> int:
    match = re.fullmatch(r"sentinel2_l2a(?:\.(\d+))?", path.name)
    return int(match.group(1) or 0) if match else 10_000


def first_tiff(layer_path: Path) -> Path:
    matches = sorted(layer_path.glob("*/geotiff.tif"))
    if not matches:
        raise FileNotFoundError(f"no geotiff in {layer_path}")
    return matches[0]


def index_windows(root: Path) -> dict[str, list[WindowRecord]]:
    output: dict[str, list[WindowRecord]] = {year: [] for year in YEARS}
    for year in YEARS:
        for path in sorted(root.glob(f"{YEAR_PREFIX[year]}*")):
            layers = sorted(
                (p for p in (path / "layers").glob("sentinel2_l2a*") if p.is_dir()),
                key=natural_layer_index,
            )
            if not layers:
                continue
            with rasterio.open(first_tiff(layers[0])) as src:
                output[year].append(
                    WindowRecord(year, path, src.crs, src.transform, src.width, src.height)
                )
    return output


def locate_window(records: list[WindowRecord], lat: float, lon: float) -> tuple[WindowRecord, int, int]:
    for record in records:
        transformer = Transformer.from_crs("EPSG:4326", record.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        col_float, row_float = ~record.transform * (x, y)
        if 0 <= row_float < record.height and 0 <= col_float < record.width:
            return record, int(row_float), int(col_float)
    raise RuntimeError(f"no window contains lat={lat} lon={lon}")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def select_period(window_path: Path, year: str) -> tuple[int, str, list[str]]:
    payload = json.loads((window_path / "items.json").read_text(encoding="utf-8"))
    layer = next(item for item in payload if item["layer_name"] == "sentinel2_l2a")
    target = datetime(int(year), TARGET_MONTH, TARGET_DAY, tzinfo=timezone.utc)
    options = []
    for period_index, group in enumerate(layer["serialized_item_groups"]):
        starts = [parse_time(item["geometry"]["time_range"][0]) for item in group]
        if not starts:
            continue
        representative = min(starts, key=lambda dt: abs((dt - target).total_seconds()))
        options.append((abs((representative - target).total_seconds()), period_index, representative, group))
    if not options:
        raise RuntimeError(f"no periods in {window_path}")
    _, period_index, representative, group = min(options, key=lambda item: item[0])
    return period_index, representative.date().isoformat(), [item["name"] for item in group]


def read_chip(
    path: Path, row: int, col: int, radius: int
) -> tuple[np.ndarray, dict[str, float], tuple[float, float]]:
    with rasterio.open(path) as src:
        r0, c0 = max(0, row - radius), max(0, col - radius)
        r1, c1 = min(src.height, row + radius), min(src.width, col + radius)
        window = Window(c0, r0, c1 - c0, r1 - r0)
        rgb = np.stack([src.read(index, window=window) for index in RGB_INDEXES]).astype(np.float32)
        b02 = src.read(B02_INDEX, window=window)
        invalid = src.read_masks(B02_INDEX, window=window) == 0
    zero = (b02 == 0) | invalid
    cloud = (b02 > CLOUD_DN) & ~zero
    fixed = np.moveaxis(np.clip(rgb / FIXED_RGB_MAX, 0, 1) ** 0.8, 0, -1)
    metrics = {
        "cloud_proxy": float(cloud.mean()),
        "zero_proxy": float(zero.mean()),
        "bad_proxy": float((cloud | zero).mean()),
    }
    center = (float(col - c0), float(row - r0))
    return fixed, metrics, center


def layer_path(window_path: Path, period_index: int) -> Path:
    name = "sentinel2_l2a" if period_index == 0 else f"sentinel2_l2a.{period_index}"
    return first_tiff(window_path / "layers" / name)


def render_candidate(
    candidate: dict[str, Any], window_index: dict[str, list[WindowRecord]], out_dir: Path
) -> dict[str, Any]:
    images: dict[str, dict[str, Any]] = {}
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.6))
    for column, year in enumerate(YEARS):
        record, row, col = locate_window(
            window_index[year], candidate["lat"], candidate["lon"]
        )
        period_index, acquisition_date, item_names = select_period(record.path, year)
        tif = layer_path(record.path, period_index)
        context, context_metrics, context_center = read_chip(
            tif, row, col, CONTEXT_RADIUS
        )
        detail, detail_metrics, detail_center = read_chip(tif, row, col, DETAIL_RADIUS)
        images[year] = {
            "window": record.path.name,
            "period_index": period_index,
            "acquisition_date": acquisition_date,
            "source_items": item_names,
            "context_metrics": context_metrics,
            "detail_metrics": detail_metrics,
        }
        for row_index, (chip, center, scale) in enumerate(
            ((context, context_center, "1.28 km context"), (detail, detail_center, "400 m detail"))
        ):
            ax = axes[row_index, column]
            ax.imshow(chip)
            ax.scatter(
                [center[0]], [center[1]], marker="+", s=50, linewidths=1.2, color="#00ffff"
            )
            ax.axis("off")
            if row_index == 0:
                ax.set_title(
                    f"{year} | {acquisition_date}\nbad={context_metrics['bad_proxy']:.3f}",
                    fontsize=9,
                )
            if column == 0:
                ax.text(
                    -0.04,
                    0.5,
                    scale,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=9,
                )
    algorithm = candidate["algorithm"]
    fig.suptitle(
        f"{candidate['candidate_id']} | {candidate['cohort']}\n"
        f"{candidate['lat']:.4f}, {candidate['lon']:.4f} | "
        f"{algorithm['source']} rank {algorithm['rank']} | z={algorithm['z']:.2f} | "
        f"{algorithm['when']} | {algorithm['landcover']}\n"
        "closest to 15 May each year | fixed RGB stretch 0–3000 DN | cyan + = candidate",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    filename = f"{candidate['candidate_id']}.png"
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    candidate["season_aligned_rgb"] = images
    candidate["image"] = f"candidates/{filename}"
    return candidate


def render_html(manifest: dict[str, Any]) -> str:
    embedded = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    options = "".join(
        f'<option value="{value}">{label}</option>' for value, label in LABEL_OPTIONS
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jeju change candidate review</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin:0; background:#0b0d12; color:#edf2f7; }}
header {{ position:sticky; top:0; z-index:2; padding:16px 22px; background:#111722ee; border-bottom:1px solid #334155; }}
h1 {{ margin:0 0 6px; font-size:21px; }}
.sub {{ color:#aab8ca; font-size:13px; line-height:1.5; max-width:1100px; }}
.toolbar {{ display:flex; gap:10px; align-items:center; margin-top:10px; flex-wrap:wrap; }}
button {{ background:#14b8a6; border:0; border-radius:7px; padding:8px 13px; color:#062c2a; font-weight:700; cursor:pointer; }}
#status {{ color:#93c5fd; font-size:13px; }}
main {{ max-width:1400px; margin:auto; padding:20px; }}
.cohort {{ margin:24px 0 10px; color:#67e8f9; font-size:18px; }}
.card {{ display:grid; grid-template-columns:minmax(500px, 2fr) minmax(280px, 1fr); gap:18px; background:#141a24; border:1px solid #273449; border-radius:10px; padding:14px; margin:0 0 18px; }}
.card img {{ width:100%; background:#fff; border-radius:6px; }}
.meta {{ font-size:13px; color:#bdc9d8; line-height:1.55; }}
.meta strong {{ color:#fff; }}
label {{ display:block; margin-top:12px; font-size:12px; color:#94a3b8; }}
select, textarea {{ width:100%; box-sizing:border-box; margin-top:5px; border:1px solid #3b4a61; border-radius:6px; padding:8px; color:#e5e7eb; background:#0e1420; }}
textarea {{ min-height:75px; resize:vertical; }}
.checks {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
@media(max-width:900px) {{ .card {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>제주 변화 후보 육안 감사</h1>
  <div class="sub">후보는 새 RGB를 보기 전에 v3/v6 순위와 공간 규칙으로 고정했습니다. 모든 패널은 각 연도 5월 15일에 가장 가까운 관측과 동일 0–3000 DN stretch입니다. cyan +가 후보 위치입니다. 이 판정은 조사 우선순위이며 개발 원인이나 생태 영향을 확정하지 않습니다.</div>
  <div class="toolbar"><button id="export">판정 JSON 내보내기</button><button id="clear">로컬 판정 초기화</button><span id="status"></span></div>
</header>
<main id="app"></main>
<script>
const manifest = {embedded};
const labelOptions = `{options}`;
const storageKey = "jeju-human-review-v1";
let saved = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
const app = document.getElementById("app");
let lastCohort = "";
function persist() {{ localStorage.setItem(storageKey, JSON.stringify(saved)); updateStatus(); }}
function updateStatus() {{
  const total = manifest.candidates.length;
  const done = Object.values(saved).filter(x => x.label).length;
  document.getElementById("status").textContent = `${{done}} / ${{total}} 판정 저장됨 (이 브라우저)`;
}}
for (const candidate of manifest.candidates) {{
  if (candidate.cohort !== lastCohort) {{
    const h = document.createElement("h2"); h.className="cohort"; h.textContent=candidate.cohort; app.appendChild(h); lastCohort=candidate.cohort;
  }}
  const state = saved[candidate.candidate_id] || {{label:"", persistent:"", confidence:"", notes:""}};
  const card = document.createElement("section"); card.className="card";
  card.innerHTML = `<div><img src="${{candidate.image}}" alt="${{candidate.candidate_id}}"></div>
  <div><div class="meta"><strong>${{candidate.candidate_id}}</strong><br>${{candidate.lat.toFixed(4)}}, ${{candidate.lon.toFixed(4)}}<br>${{candidate.algorithm.source}} rank ${{candidate.algorithm.rank}} · z=${{candidate.algorithm.z}} · ${{candidate.algorithm.when}} · ${{candidate.algorithm.landcover}}</div>
  <label>주 판정<select class="label"><option value="">미판정</option>${{labelOptions}}</select></label>
  <div class="checks"><label>지속 변화인가?<select class="persistent"><option value="">미판정</option><option value="yes">예</option><option value="no">아니오</option><option value="uncertain">불확실</option></select></label>
  <label>확신도<select class="confidence"><option value="">미판정</option><option value="high">높음</option><option value="medium">중간</option><option value="low">낮음</option></select></label></div>
  <label>메모<textarea class="notes" placeholder="어느 연도부터 무엇이 지속되는지, 구름/계절성 여부">${{state.notes || ""}}</textarea></label></div>`;
  for (const field of ["label","persistent","confidence"]) card.querySelector("."+field).value = state[field] || "";
  for (const field of ["label","persistent","confidence","notes"]) card.querySelector("."+field).addEventListener("change", () => {{
    saved[candidate.candidate_id] = {{label:card.querySelector(".label").value,persistent:card.querySelector(".persistent").value,confidence:card.querySelector(".confidence").value,notes:card.querySelector(".notes").value}}; persist();
  }});
  app.appendChild(card);
}}
document.getElementById("export").onclick = () => {{
  const output = {{schema:"jeju-human-review-v1", manifest_sha256:manifest.provenance.manifest_sha256, exported_at:new Date().toISOString(), reviews:saved}};
  const blob = new Blob([JSON.stringify(output,null,2)], {{type:"application/json"}}); const url=URL.createObjectURL(blob); const a=document.createElement("a"); a.href=url; a.download="jeju_human_review.json"; a.click(); URL.revokeObjectURL(url);
}};
document.getElementById("clear").onclick = () => {{ if(confirm("이 브라우저에 저장된 판정을 초기화할까요?")) {{ saved={{}}; localStorage.removeItem(storageKey); location.reload(); }} }};
updateStatus();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    root = Path(args.windows_root)
    out = Path(args.out)
    candidates_dir = out / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    v3 = json.loads(Path(args.v3_json).read_text(encoding="utf-8"))
    v6 = json.loads(Path(args.v6_json).read_text(encoding="utf-8"))
    selected = select_candidates(v3, v6)
    windows = index_windows(root)
    rendered = [render_candidate(candidate, windows, candidates_dir) for candidate in selected]
    manifest = {
        "schema": "jeju-human-review-v1",
        "status": "algorithm_candidates_fixed_human_labels_pending",
        "provenance": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "candidate_selection": "deterministic v3/v6 rank + coordinate cohort rules before new RGB rendering",
            "target_season": "closest acquisition to 15 May in each year",
            "rgb_stretch": "fixed 0-3000 DN, gamma 0.8",
            "cloud_proxy": f"B02 > {CLOUD_DN}",
            "manifest_sha256": None,
        },
        "review_options": LABEL_OPTIONS,
        "limitations": [
            "Sentinel-2 RGB supports visual triage but does not identify a causal driver.",
            "One May acquisition per year can still contain atmosphere, shadow, or phenology differences.",
            "The eastern mid-mountain coordinate rule is an oreum/vegetation audit cohort, not a verified named-oreum boundary.",
            "Human labels must remain separate from algorithm ranking scores.",
        ],
        "candidates": rendered,
    }
    hash_payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    manifest["provenance"]["manifest_sha256"] = hashlib.sha256(hash_payload).hexdigest()
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "dashboard.html").write_text(render_html(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidates": len(rendered),
                "cohorts": {cohort: sum(c["cohort"] == cohort for c in rendered) for cohort in sorted({c["cohort"] for c in rendered})},
                "out": str(out),
                "manifest_sha256": manifest["provenance"]["manifest_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
