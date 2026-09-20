#!/usr/bin/env python3
"""모델의 Predictions 탭을 주기적으로 읽어 예측 실행 상태를 기록한다 (읽기 전용). pending/running 을 벗어나면 캡처 후 종료."""
import argparse, json, re, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]; BASE = "https://olmoearth.allenai.org"
ap = argparse.ArgumentParser(); ap.add_argument("--project", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--interval", type=int, default=120); ap.add_argument("--max-min", type=int, default=120); a = ap.parse_args()
OUT = ROOT/"artifacts/studio_audit/predictions_poll"; OUT.mkdir(parents=True, exist_ok=True)
txt = lambda pg: pg.evaluate("() => document.body.innerText.replace(/\\s+/g,' ')")
with sync_playwright() as p:
    b = p.chromium.launch(headless=True); ctx = b.new_context(storage_state=str(ROOT/".studio_session.json"), viewport={"width":1440,"height":1150}); pg = ctx.new_page()
    t0 = time.time()
    while True:
        pg.goto(f"{BASE}/projects/{a.project}/models", wait_until="networkidle", timeout=45000); pg.wait_for_timeout(800)
        pg.get_by_text(a.model, exact=True).first.click(timeout=5000); pg.wait_for_load_state("networkidle"); pg.wait_for_timeout(800)
        pg.get_by_role("tab", name="Predictions").click(timeout=5000); pg.wait_for_timeout(1500)
        t = txt(pg); i = t.find("Creation date"); seg = t[i:i+400]
        m = re.search(r"View Areas (\w+) ", seg); status = m.group(1) if m else "?"
        now = time.strftime("%H:%M:%S"); print(now, status, "|", seg[:200], flush=True)
        with (OUT/"poll_log.jsonl").open("a") as f: f.write(json.dumps({"t": now, "status": status}) + "\n")
        if status not in ("pending", "running", "predicting", "queued", "?"):
            pg.screenshot(path=str(OUT/f"final_{status}_{time.strftime('%H%M%S')}.png"), full_page=True); (OUT/f"final_{status}.txt").write_text(t); print("DONE", status, flush=True); break
        if time.time() - t0 > a.max_min * 60: print("TIMEOUT", flush=True); break
        time.sleep(a.interval)
    b.close()
