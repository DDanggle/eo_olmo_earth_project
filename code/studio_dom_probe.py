#!/usr/bin/env python3
"""페이지 하나의 DOM 구조를 덤프한다 (읽기 전용). 셀렉터를 쓰기 전에 실물을 본다."""
import argparse, json, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]; BASE = "https://olmoearth.allenai.org"
ap = argparse.ArgumentParser(); ap.add_argument("--path", required=True); ap.add_argument("--out", default="dom_probe"); a = ap.parse_args()
OUT = ROOT/"artifacts/studio_audit"/a.out; OUT.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(headless=True); ctx = b.new_context(storage_state=str(ROOT/".studio_session.json"), viewport={"width":1440,"height":1150}); pg = ctx.new_page()
    pg.goto(BASE + a.path, wait_until="networkidle", timeout=45000); pg.wait_for_timeout(1500)
    tag = time.strftime("%H%M%S"); pg.screenshot(path=str(OUT/f"{tag}.png"), full_page=True)
    info = pg.evaluate("""() => ({
      url: location.href,
      text: document.body.innerText.replace(/\\s+/g,' ').slice(0,3000),
      tables: document.querySelectorAll('table').length,
      grids: Array.from(document.querySelectorAll('[role=grid],[role=table],[role=rowgroup]')).map(e=>e.tagName+'.'+e.className.toString().slice(0,60)),
      rows: Array.from(document.querySelectorAll('[role=row], tr')).slice(0,20).map(r=>({cls:r.className.toString().slice(0,60), cells:Array.from(r.querySelectorAll('[role=cell],[role=gridcell],[role=columnheader],td,th')).map(c=>c.innerText.trim().slice(0,40)), href:(r.querySelector('a')||{}).getAttribute? (r.querySelector('a')?r.querySelector('a').getAttribute('href'):null):null})),
      links: Array.from(document.querySelectorAll('a[href*="/models/"]')).map(a=>({t:a.innerText.trim().slice(0,50), href:a.getAttribute('href')})).slice(0,20),
    })""")
    (OUT/f"{tag}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1)); print(json.dumps(info, ensure_ascii=False, indent=1)[:6000])
    b.close()
