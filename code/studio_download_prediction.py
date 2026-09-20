#!/usr/bin/env python3
"""Data Viewer ▸ Predictions 레이어의 'Download prediction results'로 결과 파일을 받아 내용을 요약한다 (읽기 전용)."""
import argparse, json, re, time, collections
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]; BASE = "https://olmoearth.allenai.org"
ap = argparse.ArgumentParser(); ap.add_argument("--project", required=True); ap.add_argument("--run", required=True); a = ap.parse_args()
OUT = ROOT/"artifacts/studio_audit/prediction_download"; OUT.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(headless=True); ctx = b.new_context(storage_state=str(ROOT/".studio_session.json"), viewport={"width":1440,"height":1000}, accept_downloads=True); pg = ctx.new_page()
    pg.goto(f"{BASE}/projects/{a.project}/data-viewer", wait_until="networkidle", timeout=45000); pg.wait_for_timeout(1500)
    pg.locator("text=Predictions >> visible=true").last.click(timeout=5000); pg.wait_for_timeout(1200)
    pg.get_by_text(re.compile(re.escape(a.run[:24]))).last.click(timeout=4000); pg.wait_for_timeout(1500)
    with pg.expect_download(timeout=60000) as dl: pg.get_by_role("button", name="Download prediction results").first.click(timeout=5000)
    d = dl.value; dest = OUT/(d.suggested_filename or "prediction.bin"); d.save_as(str(dest)); print("saved:", dest, dest.stat().st_size, "bytes", flush=True)
    b.close()
dest = sorted(OUT.iterdir(), key=lambda x: x.stat().st_mtime)[-1]; head = dest.read_bytes()[:400]
print("head:", head[:200])
try:
    g = json.loads(dest.read_text()); feats = g.get("features", []); print("features:", len(feats)); 
    if feats:
        print("props keys:", list(feats[0]["properties"].keys())); print("geom types:", collections.Counter(f["geometry"]["type"] for f in feats).most_common(3))
        for k in feats[0]["properties"]:
            vals = [f["properties"].get(k) for f in feats]; c = collections.Counter(map(str, vals))
            if len(c) <= 12: print(f"  {k}: {c.most_common(6)}")
            else:
                nums = [v for v in vals if isinstance(v,(int,float))]; print(f"  {k}: {len(c)} distinct" + (f", min {min(nums):.3g} max {max(nums):.3g}" if nums else ""))
except Exception as e: print("not JSON:", e)
