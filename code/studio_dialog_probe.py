#!/usr/bin/env python3
"""페이지에서 버튼 하나를 눌러 열리는 대화상자의 구조를 덤프한다 (제출 안 함, 읽기 전용).
쿠키 배너(osano)의 [role=dialog]는 제외한다 — 2026-09-19 초판 결함."""
import argparse, json, re, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]; BASE = "https://olmoearth.allenai.org"
ap = argparse.ArgumentParser(); ap.add_argument("--path", required=True); ap.add_argument("--button", required=True)
ap.add_argument("--then", nargs="*", default=[], help="대화상자 안에서 순서대로 누를 버튼/텍스트 (제출 동사 금지)"); ap.add_argument("--out", default="dialog_probe"); ap.add_argument("--combobox", action="store_true", help="대화상자 안 마지막 [role=combobox]를 열어 옵션을 덤프"); a = ap.parse_args()
OUT = ROOT/"artifacts/studio_audit"/a.out/time.strftime("%H%M%S"); OUT.mkdir(parents=True)
JS = """() => { const ds = Array.from(document.querySelectorAll('[role=dialog],[role=menu],.MuiPopover-paper')).filter(d=>!/osano/i.test(d.className+d.id)); const root = ds.length ? ds[ds.length-1] : document.body;
  return { dialog: ds.length, text: root.innerText.replace(/\\s+/g,' ').slice(0,2500),
    options: Array.from(document.querySelectorAll('[role=option], [role=listbox] li')).map(e=>e.innerText.trim().slice(0,80)),
    controls: Array.from(root.querySelectorAll('button,input,select,textarea,[role=combobox],[role=radio],[role=checkbox],[role=switch],[role=tab],[role=slider],[role=option],[role=menuitem]')).map(e=>({tag:e.tagName,role:e.getAttribute('role'),type:e.type||'',name:(e.innerText||e.getAttribute('aria-label')||e.placeholder||e.name||'').trim().slice(0,70),value:(e.value||'').slice(0,40),disabled:!!e.disabled})).filter(c=>c.name||c.value) } }"""
def dump(pg, name):
    pg.screenshot(path=str(OUT/f"{name}.png"), full_page=True); info = pg.evaluate(JS)
    (OUT/f"{name}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1)); print(f"[{name}] dialogs={info['dialog']}\n  text: {info['text'][:1200]}\n  options: {info['options']}\n  controls: {json.dumps(info['controls'][:45], ensure_ascii=False)}", flush=True)
assert not re.search(r"^(submit|save|create|run model|build model|publish|delete|import data)$", " ".join(a.then), re.I)
with sync_playwright() as p:
    b = p.chromium.launch(headless=True); ctx = b.new_context(storage_state=str(ROOT/".studio_session.json"), viewport={"width":1440,"height":1150}); pg = ctx.new_page()
    pg.goto(BASE + a.path, wait_until="networkidle", timeout=45000); pg.wait_for_timeout(1200)
    btn = pg.get_by_role("button", name=a.button); btn = btn if btn.count() else pg.get_by_text(a.button, exact=True)
    btn.first.click(timeout=5000); pg.wait_for_timeout(1500); dump(pg, "01_open")
    for i, t in enumerate(a.then, 2):
        loc = pg.get_by_role("button", name=t); loc = loc if loc.count() else pg.get_by_text(t)
        loc.first.click(timeout=5000); pg.wait_for_timeout(1500); dump(pg, f"{i:02d}_{re.sub(r'[^\w]+','_',t)[:20]}")
    if a.combobox:
        pg.locator("[role=dialog] [role=combobox]").last.click(timeout=5000); pg.wait_for_timeout(1000); dump(pg, "99_combobox")
    b.close()
