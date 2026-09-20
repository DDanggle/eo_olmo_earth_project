#!/usr/bin/env python3
"""Map Publisher ▸ Publish: Run 선택 + 제목/설명 + Access level(기본 Restricted) → (dry-run이면 멈춤) → Save → 목록 확인.
Public 은 이 스크립트로 선택하지 않는다 (외부 공개는 사람이 직접)."""
import argparse, json, re, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]; BASE = "https://olmoearth.allenai.org"
ap = argparse.ArgumentParser(); ap.add_argument("--project", required=True); ap.add_argument("--run", required=True); ap.add_argument("--title", required=True); ap.add_argument("--desc", required=True)
ap.add_argument("--narration", default=""); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
OUT = ROOT/"artifacts/studio_audit/publish"/time.strftime("%H%M%S"); OUT.mkdir(parents=True)
DLG = "() => { const ds=[...document.querySelectorAll('[role=dialog]')].filter(d=>!/osano/i.test(d.className+d.id)); return (ds.length?ds[ds.length-1]:document.body).innerText.replace(/\\s+/g,' ') }"
def snap(pg, n): pg.screenshot(path=str(OUT/f"{n}.png"), full_page=True); t = pg.evaluate(DLG); print(f"[{n}] {t[:700]}", flush=True); return t
with sync_playwright() as p:
    b = p.chromium.launch(headless=True); ctx = b.new_context(storage_state=str(ROOT/".studio_session.json"), viewport={"width":1440,"height":1150}); pg = ctx.new_page()
    pg.goto(f"{BASE}/projects/{a.project}/map-publisher", wait_until="networkidle", timeout=45000); pg.wait_for_timeout(1000)
    pg.get_by_role("button", name="Publish").first.click(timeout=5000); pg.wait_for_timeout(1500); snap(pg, "01_open")
    dlg = pg.locator("[role=dialog]").last
    dlg.get_by_role("button", name="Open").first.click(timeout=5000); pg.wait_for_timeout(800)
    pg.get_by_role("option", name=re.compile(re.escape(a.run[:30]))).first.click(timeout=5000); pg.wait_for_timeout(800)
    dlg.locator("input[name=title]").fill(a.title); dlg.locator("input[name=description]").fill(a.desc)
    if a.narration: dlg.locator("textarea").first.fill(a.narration)
    # Access level: Restricted 고정
    dlg.locator("[role=combobox]").last.click(timeout=5000); pg.wait_for_timeout(600)
    pg.get_by_role("option", name=re.compile("^Restricted")).first.click(timeout=5000); pg.wait_for_timeout(600)
    t = snap(pg, "02_filled"); print("Restricted 표시:", "Restricted" in t, "| Public 표시:", "Public" in t, flush=True)
    if a.dry_run: print("DRY_RUN_STOP", flush=True); b.close(); raise SystemExit(0)
    dlg.get_by_role("button", name="Save").click(timeout=5000); pg.wait_for_timeout(3000); snap(pg, "03_after_save")
    pg.goto(f"{BASE}/projects/{a.project}/map-publisher", wait_until="networkidle", timeout=45000); pg.wait_for_timeout(1500)
    t = snap(pg, "04_list"); print("PUBLISHED_ROW:", a.title in t, flush=True)
    links = pg.evaluate("() => [...document.querySelectorAll('a[href]')].map(a=>a.href).filter(h=>/viewer|map|publish/i.test(h) && !/allenai.org\\/projects/.test(h))"); print("links:", links[:10], flush=True)
    b.close()
