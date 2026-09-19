#!/usr/bin/env python3
"""Build model 마법사를 끝까지 진행해 학습을 *실제로 시작*한다 (사용자 지시: '모든 거 다해보고').
Nano · Window classification · Small 320m · 'A state' 12개월 · 예상 ~2/100 compute units.
각 단계 스크린샷 + Summary 텍스트 기록. --dry-run 이면 Build Model 직전에서 멈춘다."""
import argparse, json, re, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]; BASE = "https://olmoearth.allenai.org"
ap = argparse.ArgumentParser(); ap.add_argument("--project", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--label", default="sample_category"); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
OUT = ROOT/"artifacts/studio_audit/build_model_launch"/time.strftime("%Y%m%d_%H%M%S"); OUT.mkdir(parents=True)
txt = lambda pg: pg.evaluate("() => document.body.innerText.replace(/\\s+/g,' ')")
rec = []
def snap(pg, name, key=None, n=1600):
    pg.screenshot(path=str(OUT/f"{name}.png"), full_page=True); t = txt(pg); i = max(t.find(key) if key else 0, 0); rec.append({"name": name, "text": t[i:i+n]}); print(f"[{name}] {t[i:i+500]}\n", flush=True)
def nxt(pg):
    b = pg.get_by_role("button", name="Next")
    for _ in range(12):
        if b.count() and b.first.is_visible() and b.first.is_enabled(): b.first.click(timeout=4000); pg.wait_for_timeout(1400); return True
        pg.wait_for_timeout(500)
    return False
with sync_playwright() as p:
    b = p.chromium.launch(headless=True); ctx = b.new_context(storage_state=str(ROOT/".studio_session.json"), viewport={"width":1440,"height":1150}); pg = ctx.new_page()
    pg.goto(f"{BASE}/projects/{a.project}/models", wait_until="networkidle", timeout=45000); pg.wait_for_timeout(800)
    pg.get_by_role("button", name="Build model").first.click(timeout=5000); pg.wait_for_timeout(1000)
    try: sl = pg.get_by_role("slider").first; sl.focus(); pg.keyboard.press("ArrowLeft"); pg.wait_for_timeout(300)
    except Exception: pass
    snap(pg, "s1", "What size model"); assert nxt(pg)
    pg.get_by_role("combobox").first.click(timeout=3000); pg.wait_for_timeout(500)
    pg.get_by_role("option", name=re.compile(rf"^{a.label}")).first.click(timeout=3000); pg.wait_for_timeout(900)
    pg.get_by_role("radio", name=re.compile("Window based", re.I)).first.check(timeout=3000); pg.wait_for_timeout(400)
    snap(pg, "s2", "Which label"); assert nxt(pg)
    pg.get_by_role("radio", name=re.compile(r"^Small \(320m", re.I)).first.check(timeout=3000); pg.wait_for_timeout(400)
    snap(pg, "s3", "Spatial context"); assert nxt(pg)
    c = pg.get_by_text(re.compile(r"^A state$")); (c if c.count() else pg.get_by_text("A state")).first.click(timeout=3000); pg.wait_for_timeout(800)
    snap(pg, "s4", "What does a label"); assert nxt(pg)
    name_in = pg.get_by_label(re.compile("Model name")); 
    if not name_in.count(): name_in = pg.locator("input[type=text]").last
    name_in.first.fill(a.name); pg.wait_for_timeout(500)
    snap(pg, "s5_summary", "Summary", 2600)
    if a.dry_run: print("DRY-RUN: Build Model 직전 중단"); b.close(); raise SystemExit(0)
    # 비용 추정('Estimating cost...')이 끝나야 Build Model 이 활성화된다. 최대 90초 대기.
    bm = pg.get_by_role("button", name=re.compile(r"^Build Model$"))
    for i in range(90):
        t = txt(pg)
        if bm.count() and bm.first.is_enabled() and "Estimating cost" not in t: break
        pg.wait_for_timeout(1000)
    t = txt(pg); j = t.find("Estimated cost"); print("비용:", t[j:j+90] if j > 0 else "(추정 문구 없음)", flush=True)
    snap(pg, "s5_summary_costed", "Summary", 2600)
    assert bm.count() and bm.first.is_enabled(), "Build Model 버튼 비활성 (90s 대기 후)"
    bm.first.click(timeout=5000); print("→ Build Model 클릭 (학습 시작 요청)", flush=True)
    pg.wait_for_timeout(4000)
    try: pg.wait_for_load_state("networkidle", timeout=30000)
    except Exception: pass
    snap(pg, "after_build", None, 1200)
    pg.goto(f"{BASE}/projects/{a.project}/models", wait_until="networkidle", timeout=45000); pg.wait_for_timeout(2500)
    snap(pg, "models_after", "Models", 1200)
    (OUT/"launch.json").write_text(json.dumps({"name": a.name, "steps": rec}, indent=1, ensure_ascii=False)); b.close()
print("OUT", OUT)
