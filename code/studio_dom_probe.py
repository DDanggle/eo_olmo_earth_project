#!/usr/bin/env python3
"""페이지 하나의 DOM 구조를 덤프한다 (읽기 전용). 셀렉터를 쓰기 전에 실물을 본다."""
import argparse, json, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]; BASE = "https://olmoearth.allenai.org"
ap = argparse.ArgumentParser(); ap.add_argument("--path", required=True); ap.add_argument("--out", default="dom_probe")
ap.add_argument("--click-text", help="페이지 로드 후 이 텍스트(정확일치)를 한 번 클릭하고 나서 덤프 (카드 진입용)")
ap.add_argument("--hover-text", help="클릭 후 이 텍스트에 마우스를 올려 툴팁을 덤프"); a = ap.parse_args()
OUT = ROOT/"artifacts/studio_audit"/a.out; OUT.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(headless=True); ctx = b.new_context(storage_state=str(ROOT/".studio_session.json"), viewport={"width":1440,"height":1150}); pg = ctx.new_page()
    pg.goto(BASE + a.path, wait_until="networkidle", timeout=45000); pg.wait_for_timeout(1500)
    if a.click_text:
        pg.get_by_text(a.click_text, exact=True).first.click(timeout=5000); pg.wait_for_load_state("networkidle"); pg.wait_for_timeout(1500)
    if a.hover_text:  # 상태 칩 등 툴팁 확인용
        pg.get_by_text(a.hover_text, exact=True).first.hover(timeout=5000); pg.wait_for_timeout(1200)
        print("tooltips:", pg.evaluate("() => Array.from(document.querySelectorAll('[role=tooltip], .MuiTooltip-tooltip, [aria-describedby]')).map(e=>(e.innerText||e.getAttribute('aria-describedby')||'').trim().slice(0,600))"))
        print("titles:", pg.evaluate("() => Array.from(document.querySelectorAll('[title],[aria-label]')).map(e=>(e.getAttribute('title')||e.getAttribute('aria-label')).slice(0,300)).filter(s=>s.length>30)"))
    tag = time.strftime("%H%M%S"); pg.screenshot(path=str(OUT/f"{tag}.png"), full_page=True)
    info = pg.evaluate("""() => ({
      url: location.href,
      buttons: Array.from(document.querySelectorAll('button, [role=button], [role=menuitem]')).map(b=>b.innerText.trim().slice(0,40)).filter(Boolean),
      tabs: Array.from(document.querySelectorAll('[role=tab]')).map(b=>b.innerText.trim().slice(0,40)),
      text: document.body.innerText.replace(/\\s+/g,' ').slice(0,3000),
      tables: document.querySelectorAll('table').length,
      grids: Array.from(document.querySelectorAll('[role=grid],[role=table],[role=rowgroup]')).map(e=>e.tagName+'.'+e.className.toString().slice(0,60)),
      rows: Array.from(document.querySelectorAll('[role=row], tr')).slice(0,20).map(r=>({cls:r.className.toString().slice(0,60), cells:Array.from(r.querySelectorAll('[role=cell],[role=gridcell],[role=columnheader],td,th')).map(c=>c.innerText.trim().slice(0,40)), href:(r.querySelector('a')||{}).getAttribute? (r.querySelector('a')?r.querySelector('a').getAttribute('href'):null):null})),
      links: Array.from(document.querySelectorAll('a[href*="/models/"]')).map(a=>({t:a.innerText.trim().slice(0,50), href:a.getAttribute('href')})).slice(0,20),
    })""")
    (OUT/f"{tag}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1)); print(json.dumps(info, ensure_ascii=False, indent=1)[:6000])
    b.close()
