#!/usr/bin/env python3
"""Areas ▸ Add area: GeoJSON 업로드 + 이름 → (dry-run이면 여기서 멈춤) → Save → 목록에 행이 생겼는지 확인."""
import argparse, json, re, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]; BASE = "https://olmoearth.allenai.org"
ap = argparse.ArgumentParser(); ap.add_argument("--project", required=True); ap.add_argument("--file", required=True); ap.add_argument("--name", required=True); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
OUT = ROOT/"artifacts/studio_audit/add_area"/time.strftime("%H%M%S"); OUT.mkdir(parents=True)
txt = lambda pg: pg.evaluate("() => { const ds=[...document.querySelectorAll('[role=dialog]')].filter(d=>!/osano/i.test(d.className+d.id)); return (ds.length?ds[ds.length-1]:document.body).innerText.replace(/\\s+/g,' ') }")
def snap(pg, n): pg.screenshot(path=str(OUT/f"{n}.png"), full_page=True); t = txt(pg); print(f"[{n}] {t[:500]}", flush=True); return t
with sync_playwright() as p:
    b = p.chromium.launch(headless=True); ctx = b.new_context(storage_state=str(ROOT/".studio_session.json"), viewport={"width":1440,"height":1150}); pg = ctx.new_page()
    pg.goto(f"{BASE}/projects/{a.project}/areas", wait_until="networkidle", timeout=45000); pg.wait_for_timeout(1000)
    pg.get_by_role("button", name="Add area").first.click(timeout=5000); pg.wait_for_timeout(1200); snap(pg, "01_open")
    with pg.expect_file_chooser(timeout=8000) as fc: pg.get_by_role("button", name="Upload GeoJSON").first.click(timeout=5000)
    fc.value.set_files(str(Path(a.file).resolve())); pg.wait_for_timeout(2000); snap(pg, "02_uploaded")
    pg.get_by_label(re.compile("Area name")).first.fill(a.name); pg.wait_for_timeout(600)
    save = pg.get_by_role("button", name="Save").first; print("Save enabled:", save.is_enabled(), flush=True); snap(pg, "03_filled")
    if a.dry_run: print("DRY_RUN_STOP", flush=True); b.close(); raise SystemExit(0)
    save.click(timeout=5000); pg.wait_for_timeout(2500); pg.goto(f"{BASE}/projects/{a.project}/areas", wait_until="networkidle", timeout=45000); pg.wait_for_timeout(1000)
    t = snap(pg, "04_list"); print("AREA_ROW_PRESENT:", a.name in t, flush=True); b.close()
