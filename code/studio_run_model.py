#!/usr/bin/env python3
"""ready 모델 ▸ Predictions ▸ Run model: 기간(기본값 유지) + Area 선택 → 비용/버튼 상태 기록 → (dry-run이면 멈춤) → Run model."""
import argparse, json, re, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]; BASE = "https://olmoearth.allenai.org"
ap = argparse.ArgumentParser(); ap.add_argument("--project", required=True); ap.add_argument("--model", required=True); ap.add_argument("--area", required=True); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
OUT = ROOT/"artifacts/studio_audit/run_model"/time.strftime("%H%M%S"); OUT.mkdir(parents=True)
DLG = "() => { const ds=[...document.querySelectorAll('[role=dialog]')].filter(d=>!/osano/i.test(d.className+d.id)); return (ds.length?ds[ds.length-1]:document.body).innerText.replace(/\\s+/g,' ') }"
def snap(pg, n): pg.screenshot(path=str(OUT/f"{n}.png"), full_page=True); t = pg.evaluate(DLG); print(f"[{n}] {t[:600]}", flush=True); return t
with sync_playwright() as p:
    b = p.chromium.launch(headless=True); ctx = b.new_context(storage_state=str(ROOT/".studio_session.json"), viewport={"width":1440,"height":1150}); pg = ctx.new_page()
    pg.goto(f"{BASE}/projects/{a.project}/models", wait_until="networkidle", timeout=45000); pg.wait_for_timeout(1000)
    pg.get_by_text(a.model, exact=True).first.click(timeout=5000); pg.wait_for_load_state("networkidle"); pg.wait_for_timeout(1200)
    pg.get_by_role("tab", name="Predictions").click(timeout=5000); pg.wait_for_timeout(1000)
    pg.get_by_role("button", name="Run model").first.click(timeout=5000); pg.wait_for_timeout(1500); snap(pg, "01_dialog")
    # Area: MUI Autocomplete ('Select existing area(s)') — 타이핑 → 옵션 클릭 (Escape 금지)
    sel = pg.get_by_label(re.compile(r"Select existing area", re.I)); sel = sel if sel.count() else pg.get_by_role("combobox").last
    sel.first.click(timeout=4000); sel.first.fill(a.area[:10]); pg.wait_for_timeout(1200)
    opts = pg.evaluate("() => [...document.querySelectorAll('[role=option]')].map(e=>e.innerText.trim())"); print("area options:", opts, flush=True)
    pg.get_by_role("option", name=re.compile(re.escape(a.area))).first.click(timeout=4000); pg.wait_for_timeout(1500)
    try: pg.get_by_text("Run Model", exact=True).first.click(timeout=1500)  # blur
    except Exception: pass
    pg.wait_for_timeout(1500); t = snap(pg, "02_area_selected")
    run = pg.get_by_role("button", name="Run model").last; print("Run model enabled:", run.is_enabled(), "| cost text:", re.findall(r"[^.]*(?:cost|unit)[^.]*", t, re.I)[:3], flush=True)
    if a.dry_run: print("DRY_RUN_STOP", flush=True); b.close(); raise SystemExit(0)
    run.click(timeout=5000); pg.wait_for_timeout(3000); snap(pg, "03_after_run")
    pg.goto(f"{BASE}/projects/{a.project}/models", wait_until="networkidle", timeout=45000); pg.get_by_text(a.model, exact=True).first.click(timeout=5000); pg.wait_for_load_state("networkidle"); pg.wait_for_timeout(1000)
    pg.get_by_role("tab", name="Predictions").click(timeout=5000); pg.wait_for_timeout(1500); snap(pg, "04_predictions_list"); b.close()
