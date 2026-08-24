#!/usr/bin/env python3
"""Render the bounded Jeju API snapshot as a standalone evidence dashboard."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_LABELS = {
    "building_hub_basis": "건축HUB 기본개요",
    "building_hub_demolition": "건축HUB 철거·멸실",
    "eia_business_area": "환경영향평가 사업구역",
    "gk2a_cloud": "GK2A 과거 관측일",
    "gk2a_cloud_current": "GK2A 최신 허용시각",
    "mcee_landcover": "환경부 토지피복",
    "vworld_cadastral": "VWorld 연속지적",
}


def render(snapshot_dir: Path) -> str:
    summary = json.loads((snapshot_dir / "run_summary.json").read_text(encoding="utf-8"))
    requests = json.loads((snapshot_dir / "requests.json").read_text(encoding="utf-8"))["requests"]
    candidates = json.loads((snapshot_dir / "candidate_evidence.json").read_text(encoding="utf-8"))["records"]
    context = json.loads((snapshot_dir / "observation_context.json").read_text(encoding="utf-8"))
    candidate_count = len(candidates)
    candidate_ids = {str(record["target_id"]) for record in candidates}
    oreum_point_count = int(summary.get("scope", {}).get("resolved_oreum_points_available", 0))
    building_legal_dongs = len(
        {
            str(record.get("target_id"))
            for record in requests
            if record.get("source_id") == "building_hub_basis"
        }
    )
    eia_candidate_matches = sum(len(record["eia_polygon_matches"]) for record in candidates)
    landcover_target_count = len(
        {
            str(tile["target_id"])
            for tile in context["landcover_tiles"]
            if str(tile["target_id"]) in candidate_ids
        }
    )
    request_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in requests:
        request_groups[str(record["source_id"])].append(record)
    source_rows = []
    for source_id, group in sorted(request_groups.items()):
        semantic = Counter(str(item.get("semantic_status")) for item in group)
        semantic_errors = len(group) - semantic.get("api_success", 0) - semantic.get("api_no_features", 0)
        item_count = sum(int(item.get("semantic_item_count") or 0) for item in group)
        if source_id == "vworld_cadastral":
            vworld = summary.get("vworld", {})
            finding = (
                f"OK {semantic.get('api_success', 0)} · 무항목 {semantic.get('api_no_features', 0)} · "
                f"오류 {semantic_errors} · 후보 PNU {vworld.get('candidate_parcel_anchors', 0)}/{candidate_count} · "
                f"오름 PNU {vworld.get('oreum_parcel_anchors', 0)}/{oreum_point_count}"
            )
        elif source_id == "gk2a_cloud":
            finding = "과거 6시점 모두 최근 2일 제한"
        elif source_id == "gk2a_cloud_current":
            finding = f"최신 grid {summary['gk2a_current_grid_values']:,}값"
        elif source_id == "building_hub_basis":
            finding = f"{summary['building_event_rows']:,}행 · {building_legal_dongs} 법정동 · 페이지 소진"
        elif source_id == "building_hub_demolition":
            finding = f"{building_legal_dongs} 법정동 응답 · 조건 내 0행"
        elif source_id == "eia_business_area":
            finding = f"제주 bbox {summary['eia_feature_rows']} polygon · 후보 직접중첩 {eia_candidate_matches}"
        else:
            finding = f"후보 {landcover_target_count} × 2023–2025 = {summary['landcover_tile_rows']}장"
        cls = "ok" if semantic_errors == 0 else "bad"
        source_rows.append(
            f"<tr><td>{html.escape(SOURCE_LABELS.get(source_id, source_id))}</td>"
            f"<td>{len(group)}</td><td class='{cls}'>{semantic.get('api_success', 0)} / "
            f"{semantic.get('api_no_features', 0)} / {semantic_errors}</td>"
            f"<td>{item_count:,}</td><td>{html.escape(finding)}</td></tr>"
        )

    candidate_rows = []
    for record in candidates:
        parcel = record.get("representative_parcel") or {}
        pnu_values = record.get("parcel_pnu_values") or [parcel.get("pnu")]
        pnu_values = [str(pnu) for pnu in pnu_values if pnu]
        relation = str(record.get("parcel_pnu_relation") or "single")
        pnu_label = "<br>".join(html.escape(pnu) for pnu in pnu_values) or "미확보"
        relation_label = "출처 충돌" if relation == "conflict" else relation
        candidate_rows.append(
            "<tr>"
            f"<td><b>{html.escape(str(record['target_id']))}</b><div>{record['lat']:.4f}, {record['lon']:.4f}</div></td>"
            f"<td>{pnu_label}<div>{html.escape(relation_label)} · source {len(record.get('parcel_evidence') or [])}</div></td>"
            f"<td>{record['same_legal_dong_building_event_count']:,}<div>정확 PNU {record['exact_parcel_building_event_count']}</div></td>"
            f"<td>{len(record['eia_polygon_matches'])}</td>"
            f"<td><span class='grade {record['causal_evidence_grade']}'>{record['causal_evidence_grade']}</span></td>"
            f"<td class='decision'>{'조사' if record['decision'] == 'investigate' else '보류'}</td>"
            f"<td>{html.escape(str(record['warning']))}</td></tr>"
        )

    exact_candidate_count = sum(
        record["exact_parcel_building_event_count"] > 0 for record in candidates
    )
    aligned_candidate_count = sum(
        bool(record["time_aligned_exact_parcel_events"]) for record in candidates
    )
    corroboration_count = int(summary.get("candidate_official_corroboration_b", 0))
    conflict_count = int(summary.get("candidate_parcel_pnu_conflicts", 0))
    vworld = summary.get("vworld", {})

    tiles_by_target: dict[str, list[dict[str, object]]] = defaultdict(list)
    for tile in context["landcover_tiles"]:
        tiles_by_target[str(tile["target_id"])].append(tile)
    tile_cards = []
    for target_id, tiles in sorted(tiles_by_target.items()):
        tiles = sorted(tiles, key=lambda item: int(item["year"]))
        images = "".join(
            f"<img data-year='{tile['year']}' class='{'active' if tile['year'] == 2025 else ''}' "
            f"src='{html.escape(str(tile['raw_file']))}' alt='{html.escape(target_id)} {tile['year']} 토지피복'>"
            for tile in tiles
        )
        buttons = "".join(
            f"<button data-year='{tile['year']}' class='{'active' if tile['year'] == 2025 else ''}'>{tile['year']}</button>"
            for tile in tiles
        )
        tile_cards.append(
            f"<article class='tile-card' data-target='{html.escape(target_id)}'><h3>{html.escape(target_id)}</h3>"
            f"<div class='tile-images'>{images}</div><div class='year-buttons'>{buttons}</div></article>"
        )

    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>K-Earth API snapshot v3</title><style>
:root{{--bg:#07100e;--panel:#101d19;--line:#29423a;--text:#edf7f2;--muted:#99aea6;--mint:#54e6b1;--amber:#f2cb67;--red:#ff7f73;--blue:#70b7ff}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 10% 0,#173b30,transparent 29%),var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}}header,main,footer{{max-width:1450px;margin:auto;padding:24px}}header{{padding-top:38px}}h1{{font-size:clamp(32px,5vw,64px);line-height:1;margin:8px 0 14px;max-width:1050px}}h2{{margin:0 0 8px}}h3{{margin:0 0 9px}}p{{color:var(--muted)}}a{{color:var(--mint)}}.eyebrow{{color:var(--mint);font-weight:900;letter-spacing:.12em;text-transform:uppercase}}.mode{{display:inline-block;border:1px solid #775f2a;background:#2b2517;color:var(--amber);border-radius:99px;padding:8px 12px;font-weight:800}}main{{display:grid;gap:18px}}.panel,.metric{{background:#101d19e8;border:1px solid var(--line);border-radius:15px;padding:19px}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:9px}}.metric b{{display:block;color:var(--mint);font-size:27px}}.metric span,.sub{{color:var(--muted)}}.finding{{border-left:4px solid var(--amber);background:#211e14;padding:13px 15px;border-radius:8px}}.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:14px}}.flow div{{background:#14251f;border:1px solid var(--line);border-radius:10px;padding:12px}}.flow b{{display:block;color:var(--mint)}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:10px}}table{{border-collapse:collapse;width:100%;min-width:1000px}}th,td{{padding:9px 10px;border-bottom:1px solid #21372f;text-align:left;vertical-align:top}}th{{background:#162820;position:sticky;top:0}}td div{{color:var(--muted);font-size:12px}}.ok,.grade.B{{color:var(--mint)}}.bad,.grade.U{{color:var(--red)}}.grade{{font-weight:900}}.decision{{font-weight:800;color:var(--amber)}}.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:11px;margin-top:14px}}.tile-card{{background:#0b1613;border:1px solid var(--line);border-radius:12px;padding:12px}}.tile-images{{aspect-ratio:1;position:relative}}.tile-images img{{display:none;width:100%;height:100%;object-fit:cover;border-radius:8px;image-rendering:auto}}.tile-images img.active{{display:block}}button{{border:1px solid var(--line);background:#14251f;color:var(--text);padding:5px 9px;border-radius:99px;cursor:pointer}}button.active{{background:var(--mint);color:#062017;font-weight:900}}.year-buttons{{display:flex;gap:6px;margin-top:9px}}.links{{display:flex;flex-wrap:wrap;gap:9px}}.links a{{border:1px solid var(--line);padding:8px 10px;border-radius:9px;text-decoration:none}}@media(max-width:800px){{.flow{{grid-template-columns:1fr}}header,main,footer{{padding:15px}}}}
</style></head><body><header><div class="eyebrow">Korean Earth Intelligence · live official evidence</div><h1>OlmoEarth 후보를 한국 행정·환경 근거로 검증하면</h1><span class="mode">결론: {candidate_count - corroboration_count}/{candidate_count} 보류 · A/B급 공식 공간·시간 근거 {corroboration_count}</span><p>API가 많이 연결됐다는 사실과 원인 규명이 가능하다는 사실을 분리합니다. 이 스냅샷은 응답 원본·SHA·요청 hash를 보존하고, 미일치와 API 제한을 음성 증거로 바꾸지 않습니다.</p></header><main>
<section class="metrics"><div class="metric"><b>{summary['outcomes'].get('http_success', 0)}/{summary['request_count']}</b><span>HTTP 성공/요청</span></div><div class="metric"><b>{summary['semantic_statuses'].get('api_success', 0)}</b><span>의미상 성공</span></div><div class="metric"><b>{summary['building_event_rows']:,}</b><span>건축행정 사건행</span></div><div class="metric"><b>{summary['eia_feature_rows']}</b><span>EIA polygon</span></div><div class="metric"><b>{summary['landcover_tile_rows']}</b><span>연도별 토지피복 tile</span></div><div class="metric"><b>{summary['cross_source_pnu_with_building_event']}/{summary['cross_source_pnu_population']}</b><span>전체 PNU↔건축HUB</span></div><div class="metric"><b>{corroboration_count}/{candidate_count}</b><span>후보 A/B corroboration</span></div></section>
<section class="panel"><h2>이번 결합에서 실제로 달라진 것</h2><div class="finding"><b>VWorld 인증 병목은 풀렸지만 원인 근거는 아직 늘지 않았습니다.</b> 후보 {vworld.get('candidate_parcel_anchors', 0)}/{candidate_count}와 위치화 오름 {vworld.get('oreum_parcel_anchors', 0)}/{oreum_point_count}에서 대표 PNU를 얻었습니다. 후보의 BuildingHUB exact PNU는 {exact_candidate_count}건이지만 변화 관측구간 안에 든 사건은 {aligned_candidate_count}건입니다. 필지 출처 충돌 {conflict_count}건도 보류했습니다. 따라서 대표 필지는 생겼어도 “이 변화의 원인”은 아직 말할 수 없습니다.</div><div class="flow"><div><b>OlmoEarth</b>{candidate_count} 후보·관측일</div><div><b>공간 spine</b>VWorld PNU {vworld.get('parcel_anchor_features', 0)} · unique {vworld.get('unique_pnu', 0)}</div><div><b>행정 사건</b>BuildingHUB {summary['building_event_rows']:,} · EIA {summary['eia_feature_rows']}</div><div><b>독립 상태</b>토지피복 {summary['landcover_tile_rows']} · GK2A 최신 1</div><div><b>선택 판정</b>정확 spatial+time 없으면 보류</div></div></section>
<section class="panel"><h2>소스별 실행·coverage 감사</h2><p class="sub">HTTP 200이라도 API 본문 상태를 따로 봅니다. 표는 성공 / 유효한 무항목 / 오류를 분리합니다. VWorld NOT_FOUND는 인증오류가 아니라 해당 점의 무항목 coverage이며, 과거 GK2A 제한은 계속 실패로 남습니다.</p><div class="table-wrap"><table><thead><tr><th>소스</th><th>요청</th><th>성공 / 무항목 / 오류</th><th>반환 item</th><th>판정</th></tr></thead><tbody>{''.join(source_rows)}</tbody></table></div></section>
<section class="panel"><h2>OlmoEarth 후보 {candidate_count}개 결합 결과</h2><p class="sub">같은 법정동 건수는 탐색 문맥일 뿐 원인 근거가 아닙니다. B는 정확 필지와 변화 전후 시간축이 함께 맞을 때만 허용하며, PNU 출처가 충돌하면 해소 전까지 보류합니다.</p><div class="table-wrap"><table><thead><tr><th>후보</th><th>필지 근거</th><th>건축HUB</th><th>EIA 중첩</th><th>근거</th><th>판정</th><th>이유</th></tr></thead><tbody>{''.join(candidate_rows)}</tbody></table></div></section>
<section class="panel"><h2>환경부 토지피복 {landcover_target_count}곳 × 3개년</h2><p class="sub">공식 연도별 상태지도입니다. 색 변화는 분류 변화일 수 있으며 곧바로 실제 토지 변화나 원인을 뜻하지 않습니다.</p><div class="tiles">{''.join(tile_cards)}</div></section>
<section class="panel"><h2>재현 산출물</h2><div class="links"><a href="run_summary.json">run summary</a><a href="COMPLETE.json">completion manifest</a><a href="requests.json">secret-safe request manifest</a><a href="candidate_evidence.json">candidate evidence</a><a href="building_events.json">BuildingHUB {summary['building_event_rows']:,}행</a><a href="eia_features.json">EIA {summary['eia_feature_rows']} polygon</a><a href="observation_context.json">GK2A + landcover</a><a href="cross_source_pnu_links.json">PNU cross-source links</a><a href="../dashboard.html">오름 368 메인 보드</a></div></section>
</main><footer>스냅샷 시각 {html.escape(summary['retrieved_at'])} · 제주 공식 오름 분모 368 유지 · NGII 항공사진은 수동 신청 채널이라 API 수집에서 제외</footer><script>document.querySelectorAll('.tile-card').forEach(card=>card.querySelectorAll('button').forEach(button=>button.onclick=()=>{{const year=button.dataset.year;card.querySelectorAll('button,img').forEach(el=>el.classList.toggle('active',el.dataset.year===year))}}));</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.snapshot_dir / "dashboard.html"
    output.write_text(render(args.snapshot_dir), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
