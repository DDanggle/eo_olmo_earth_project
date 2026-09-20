#!/usr/bin/env python3
"""Data Viewer ▸ Layers 패널의 'Predictions' 탭을 열어 예측 결과 레이어가 어떻게 보이는지 캡처한다 (읽기 전용)."""
import argparse, json, re, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]; BASE = "https://olmoearth.allenai.org"
ap = argparse.ArgumentParser(); ap.add_argument("--project", required=True); ap.add_argument("--run", default=""); a = ap.parse_args()
OUT = ROOT/"artifacts/studio_audit/dataviewer_predictions"/time.strftime("%H%M%S"); OUT.mkdir(parents=True)
txt = lambda pg: pg.evaluate("() => document.body.innerText.replace(/\\s+/g,' ')")
def dump(pg, n):
    pg.screenshot(path=str(OUT/f"{n}.png"), full_page=True); t = txt(pg)
    ctrls = pg.evaluate("""() => Array.from(document.querySelectorAll('aside button, aside [role=tab], aside input, aside [role=combobox], aside [role=option], [role=listbox] [role=option], .MuiPopover-paper *, [role=tab]')).map(e=>({tag:e.tagName,role:e.getAttribute('role'),name:(e.innerText||e.getAttribute('aria-label')||e.placeholder||'').trim().slice(0,60),sel:e.getAttribute('aria-selected')})).filter(c=>c.name)""")
    print(f"[{n}] {t[:700]}\n  ctrls: {json.dumps(ctrls[:40], ensure_ascii=False)}", flush=True); return t
with sync_playwright() as p:
    b = p.chromium.launch(headless=True); ctx = b.new_context(storage_state=str(ROOT/".studio_session.json"), viewport={"width":1440,"height":1000}); pg = ctx.new_page()
    pg.goto(f"{BASE}/projects/{a.project}/data-viewer", wait_until="networkidle", timeout=45000); pg.wait_for_timeout(1500); dump(pg, "01_dataviewer")
    tab = pg.get_by_role("tab", name="Predictions")
    if not tab.count(): tab = pg.locator("text=Predictions >> visible=true").last  # 사이드바 nav 링크(숨김)가 아닌 패널 탭
    tab.first.click(timeout=5000); pg.wait_for_timeout(1500); t = dump(pg, "02_predictions_tab")
    # 실행 선택(콤보/체크/라디오 중 무엇이든) — 이름 일부로 시도
    key = a.run[:24] if a.run else "audit-nano"
    cand = pg.get_by_text(re.compile(re.escape(key)))
    if cand.count(): cand.last.click(timeout=4000); pg.wait_for_timeout(2500); dump(pg, "03_run_selected")
    else:
        cb = pg.locator("aside [role=combobox], [role=combobox]").last
        if cb.count():
            cb.click(timeout=4000); pg.wait_for_timeout(800); print("options:", pg.evaluate("() => [...document.querySelectorAll('[role=option]')].map(e=>e.innerText.trim())"), flush=True)
            o = pg.get_by_role("option", name=re.compile(key))
            if o.count(): o.first.click(timeout=4000); pg.wait_for_timeout(2500); dump(pg, "03_run_selected")
    pg.wait_for_timeout(3000); pg.screenshot(path=str(OUT/"04_map_settled.png"), full_page=True)
    # 레이어 행의 아이콘 버튼(확대·다운로드)을 덤프하고, 첫 번째(확대)를 눌러 결과 위치로 이동
    row = pg.get_by_text(re.compile(re.escape(key))).first.locator("xpath=ancestor::*[self::li or self::div][.//button][1]")
    btns = row.locator("button"); names = [btns.nth(i).get_attribute("aria-label") or btns.nth(i).get_attribute("title") or f"btn{i}" for i in range(btns.count())]
    print("row buttons:", names, flush=True)
    if btns.count(): btns.first.click(timeout=4000); pg.wait_for_timeout(4000); pg.screenshot(path=str(OUT/"05_zoomed.png"), full_page=True); print("zoomed", flush=True)
    b.close()
