#!/usr/bin/env python3
"""Render season-aligned RGB audit panels for selected oreum screens."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_jeju_human_review as review


review.YEAR_PREFIX = {
    "2023": "jeju23_",
    "2024": "jeju24_",
    "2025": "jeju25_",
    "2026": "jeju26r_",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_record(
    record: dict[str, Any],
    windows: dict[str, list[review.WindowRecord]],
    candidates_dir: Path,
) -> dict[str, Any]:
    location = record["location"]
    images = {}
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.6))
    for column, year in enumerate(review.YEARS):
        window, row, col = review.locate_window(
            windows[year], float(location["lat"]), float(location["lon"])
        )
        period_index, acquisition_date, item_names = review.select_period(window.path, year)
        tif = review.layer_path(window.path, period_index)
        context, context_metrics, context_center = review.read_chip(
            tif, row, col, review.CONTEXT_RADIUS
        )
        detail, detail_metrics, detail_center = review.read_chip(
            tif, row, col, review.DETAIL_RADIUS
        )
        images[year] = {
            "window": window.path.name,
            "period_index": period_index,
            "acquisition_date": acquisition_date,
            "source_items": item_names,
            "context_metrics": context_metrics,
            "detail_metrics": detail_metrics,
        }
        for row_index, (chip, center, scale) in enumerate(
            (
                (context, context_center, "1.28 km context"),
                (detail, detail_center, "400 m detail"),
            )
        ):
            axis = axes[row_index, column]
            axis.imshow(chip)
            axis.scatter(
                [center[0]], [center[1]], marker="+", s=55, linewidths=1.3, color="#00ffff"
            )
            axis.axis("off")
            if row_index == 0:
                axis.set_title(
                    f"{year} | {acquisition_date}\nbad={context_metrics['bad_proxy']:.3f}",
                    fontsize=9,
                )
            if column == 0:
                axis.text(
                    -0.04,
                    0.5,
                    scale,
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=9,
                )
    model = record["model_screen"]
    fig.suptitle(
        f"{record['oreum_id']} | {location['lat']:.5f}, {location['lon']:.5f}\n"
        "current OSM peak point, not an official oreum boundary\n"
        f"4-period p={model.get('percentile_4', '-')} ({model.get('split_4', '-')}) | "
        f"12-period p={model.get('percentile_12', '-')} ({model.get('split_12', '-')}) | "
        f"{model.get('screen_class', 'human-context-only')}\n"
        "closest to 15 May | fixed RGB 0–3000 DN | cyan + = current OSM peak",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    filename = f"{record['oreum_id']}.png"
    fig.savefig(candidates_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "oreum_id": record["oreum_id"],
        "official_record_no": record["official_record_no"],
        "name": record["name"],
        "city": record["city"],
        "address": record["address"],
        "lat": location["lat"],
        "lon": location["lon"],
        "model_screen": model,
        "candidate_links": record["candidate_links"],
        "image": f"candidates/{filename}",
        "season_aligned_rgb": images,
        "human_review": {
            "label": "pending",
            "persistent_change": "pending",
            "confidence": "pending",
            "notes": "",
        },
    }


def render_html(manifest: dict[str, Any]) -> str:
    cards = []
    for candidate in manifest["candidates"]:
        model = candidate["model_screen"]
        cards.append(
            f"""<article data-id="{candidate['oreum_id']}">
<img src="{html.escape(candidate['image'])}" alt="{html.escape(candidate['name'])} 4-year RGB">
<section><h2>{html.escape(candidate['name'])} <small>{candidate['oreum_id']}</small></h2>
<p>{html.escape(candidate['address'])}</p><p class="model">4기간 {model.get('percentile_4','-')} · 12기간 {model.get('percentile_12','-')} · {html.escape(model.get('screen_class','human-context-only'))}</p>
<label>육안 판정<select class="label"><option value="pending">미판정</option><option value="persistent_surface_change">지속 지표 변화</option><option value="seasonality_or_agriculture">계절·농경</option><option value="cloud_haze_shadow">구름·해무·그림자</option><option value="coast_or_reflectance">해안·반사</option><option value="stable_or_unclear">안정·불명확</option></select></label>
<label>확신도<select class="confidence"><option value="pending">미판정</option><option value="high">높음</option><option value="medium">중간</option><option value="low">낮음</option></select></label>
<label>메모<textarea class="notes"></textarea></label></section></article>"""
        )
    embedded = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Oreum screen RGB review</title><style>
:root{{--bg:#09110f;--panel:#12201b;--line:#2d463d;--text:#edf5f1;--muted:#9db0a9;--mint:#54e6b1}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}}header{{position:sticky;top:0;z-index:2;background:#09110fee;border-bottom:1px solid var(--line);padding:18px 24px}}h1{{margin:0 0 5px}}header p,p{{color:var(--muted)}}button{{background:var(--mint);border:0;border-radius:7px;padding:8px 12px;font-weight:800}}main{{max-width:1450px;margin:auto;padding:20px}}article{{display:grid;grid-template-columns:minmax(600px,2fr) minmax(280px,1fr);gap:18px;background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:14px;margin-bottom:18px}}img{{width:100%;border-radius:8px;background:white}}h2{{margin:0}}small{{color:var(--muted);font-size:12px}}label{{display:block;margin-top:12px;color:var(--muted)}}select,textarea{{width:100%;margin-top:4px;padding:8px;background:#09130f;color:var(--text);border:1px solid var(--line);border-radius:7px}}textarea{{min-height:85px}}.model{{color:var(--mint)}}@media(max-width:900px){{article{{grid-template-columns:1fr}}}}</style></head><body><header><h1>제주 오름 선택적 변화 screen · RGB 검수</h1><p>9개 조사 우선 후보만 표시합니다. 각 연도 5월 15일 최근접 관측과 고정 0–3000 DN stretch이며 OSM point는 공식 경계가 아닙니다.</p><button id="export">판정 JSON 내보내기</button> <span id="status"></span></header><main>{''.join(cards)}</main><script>
const manifest={embedded};const key='kearth-oreum-rgb-review-v1';let saved=JSON.parse(localStorage.getItem(key)||'{{}}');document.querySelectorAll('article').forEach(card=>{{const id=card.dataset.id;const s=saved[id]||{{label:'pending',confidence:'pending',notes:''}};card.querySelector('.label').value=s.label;card.querySelector('.confidence').value=s.confidence;card.querySelector('.notes').value=s.notes;card.querySelectorAll('select,textarea').forEach(x=>x.addEventListener('input',save))}});function save(){{document.querySelectorAll('article').forEach(card=>{{saved[card.dataset.id]={{label:card.querySelector('.label').value,confidence:card.querySelector('.confidence').value,notes:card.querySelector('.notes').value}}}});localStorage.setItem(key,JSON.stringify(saved));document.getElementById('status').textContent=`${{Object.values(saved).filter(x=>x.label!=='pending').length}} / ${{manifest.candidates.length}} 판정`}}save();document.getElementById('export').onclick=()=>{{const blob=new Blob([JSON.stringify({{schema:'kearth-oreum-rgb-review-v1',manifest_sha256:manifest.provenance.manifest_sha256,reviews:saved}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='kearth_oreum_rgb_review.json';a.click();URL.revokeObjectURL(a.href)}};
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--windows-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    selected = [
        record
        for record in registry["records"]
        if record["selective_decision"] == "investigate"
    ]
    if len(selected) != 9:
        raise ValueError(f"expected 9 frozen investigate records, got {len(selected)}")
    candidates_dir = args.out / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    windows = review.index_windows(args.windows_root)
    rendered = [render_record(record, windows, candidates_dir) for record in selected]
    manifest = {
        "schema": "kearth-oreum-rgb-review-manifest-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "nine_investigate_records_fixed_before_rgb_review",
        "provenance": {
            "source_registry": str(args.registry),
            "source_registry_sha256": sha256(args.registry),
            "selection_rule": "high_stable 4/12 screen OR existing high-confidence human candidate within 500m",
            "season_rule": "closest acquisition to 15 May per year",
            "rgb_stretch": "fixed 0-3000 DN, gamma 0.8",
            "coordinate_warning": "offline OSM point, not official oreum boundary",
        },
        "candidates": rendered,
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["provenance"]["manifest_sha256"] = sha256(manifest_path)
    (args.out / "dashboard.html").write_text(render_html(manifest), encoding="utf-8")
    print(json.dumps({"candidates": len(rendered), "out": str(args.out)}, ensure_ascii=False))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
