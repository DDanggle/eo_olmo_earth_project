#!/usr/bin/env python3
"""ready 모델의 Predictions 탭 → 'Run model' 버튼을 눌러 나오는 대화상자의 구조만 덤프한다 (제출 안 함, 읽기 전용)."""
import argparse, json, re, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]; BASE = "https://olmoearth.allenai.org"
ap = argparse.ArgumentParser(); ap.add_argument("--project", required=True); ap.add_argument("--model", required=True); a = ap.parse_args()
OUT = ROOT/"artifacts/studio_audit/run_model_probe"/time.strftime("%H%M%S"); OUT.mkdir(parents=True)
txt = lambda pg: pg.evaluate("() => document.body.innerText.replace(/\\s+/g,' ')")
def dump(pg, name):
    pg.screenshot(path=str(OUT/f"{name}.png"), full_page=True)
    info = pg.evaluate("""() => { const ds = Array.from(document.querySelectorAll('[role=dialog]')).filter(d=>!/osano/i.test(d.className+d.id)); const root = ds.length ? ds[ds.length-1] : document.body;
      return { url: location.href, dialog: !!document.querySelector('[role=dialog]'), text: root.innerText.replace(/\\s+/g,' ').slice(0,2500),
        controls: Array.from(root.querySelectorAll('button,input,select,textarea,[role=combobox],[role=radio],[role=checkbox],[role=switch],[role=tab],[role=slider]')).map(e=>({tag:e.tagName,role:e.getAttribute('role'),type:e.type||'',name:(e.innerText||e.getAttribute('aria-label')||e.placeholder||e.name||'').trim().slice(0,60),value:(e.value||'').slice(0,40),disabled:!!e.disabled})).filter(c=>c.name||c.value) } }""")
    (OUT/f"{name}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1)); print(f"[{name}] dialog={info['dialog']}\n  text: {info['text'][:900]}\n  controls: {info['controls'][:40]}", flush=True); return info
with sync_playwright() as p:
    b = p.chromium.launch(headless=True); ctx = b.new_context(storage_state=str(ROOT/".studio_session.json"), viewport={"width":1440,"height":1150}); pg = ctx.new_page()
    pg.goto(f"{BASE}/projects/{a.project}/models", wait_until="networkidle", timeout=45000); pg.wait_for_timeout(1000)
    pg.get_by_text(a.model, exact=True).first.click(timeout=5000); pg.wait_for_load_state("networkidle"); pg.wait_for_timeout(1200)
    pg.get_by_role("tab", name="Predictions").click(timeout=5000); pg.wait_for_timeout(1200); dump(pg, "01_predictions_tab")
    pg.get_by_role("button", name="Run model").first.click(timeout=5000); pg.wait_for_timeout(1500); info = dump(pg, "02_run_model_open")
    # 마법사형이면 Next 를 눌러 단계만 캡처 (최종 제출 동사는 절대 누르지 않음)
    for i in range(3, 9):
        nb = pg.get_by_role("button", name=re.compile(r"^(Next|Continue)$"))
        if not nb.count() or not nb.first.is_visible() or not nb.first.is_enabled(): break
        nb.first.click(timeout=4000); pg.wait_for_timeout(1200); dump(pg, f"{i:02d}_step")
    b.close()
