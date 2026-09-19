#!/usr/bin/env python3
"""학습 재시도 (네팔 폴리곤): 라벨 status, 학습 데이터를 데이터셋 하나로 Filter, 시간 'A sighting'.
Filter UI 는 미지 → 단계마다 스크린샷·덤프. 필터를 못 걸면 학습을 시작하지 않고 보고한다."""
import argparse, json, re, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]; BASE = "https://olmoearth.allenai.org"
ap = argparse.ArgumentParser(); ap.add_argument("--project", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--label", default="status"); ap.add_argument("--dataset", default="nepal_rasuwa_studio"); ap.add_argument("--temporal", default="A sighting")
ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--label-index", type=int, default=0, help="같은 이름 라벨이 여러 개일 때 몇 번째(0=첫, -1=마지막). 라벨셋은 데이터셋별로 따로 생기며 드롭다운은 데이터셋 순서"); a = ap.parse_args()
OUT = ROOT/"artifacts/studio_audit/build_model_launch"/(time.strftime("%Y%m%d_%H%M%S")+"_nepal"); OUT.mkdir(parents=True)
txt = lambda pg: pg.evaluate("() => document.body.innerText.replace(/\\s+/g,' ')")
def snap(pg, name, key=None, n=1500):
    pg.screenshot(path=str(OUT/f"{name}.png"), full_page=True); t=txt(pg); i=max(t.find(key) if key else 0,0); print(f"[{name}] {t[i:i+600]}\n", flush=True); return t
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
    assert nxt(pg)
    pg.get_by_role("combobox").first.click(timeout=3000); pg.wait_for_timeout(500)
    opts_l = pg.get_by_role("option", name=re.compile(rf"^{re.escape(a.label)} \("))
    n_l = opts_l.count(); idx = a.label_index if a.label_index >= 0 else n_l + a.label_index
    chosen = opts_l.nth(idx); uuid = (chosen.get_attribute("data-value") or "")[:8]
    print(f"라벨 옵션 {n_l}개 중 index {idx} 선택: {chosen.inner_text().strip()} [{uuid}]", flush=True)
    chosen.click(timeout=3000); pg.wait_for_timeout(900)
    pg.get_by_role("radio", name=re.compile("Window based", re.I)).first.check(timeout=3000); pg.wait_for_timeout(400)
    # --- Filter: 데이터셋 하나로 제한 ---
    pg.get_by_role("button", name="Filter").first.click(timeout=4000); pg.wait_for_timeout(1200)
    t = snap(pg, "s2_filter_open", "Which data")
    ctrls = pg.evaluate("""() => Array.from(document.querySelectorAll('[role=dialog] *, [role=menu] *, .MuiPopover-paper *')).filter(e=>['BUTTON','INPUT','SELECT','LABEL'].includes(e.tagName)||e.getAttribute('role')).map(e=>({tag:e.tagName,role:e.getAttribute('role'),t:(e.innerText||e.value||'').trim().slice(0,60),type:e.type||''})).filter(x=>x.t||x.type).slice(0,40)""")
    print("filter controls:", ctrls, flush=True)
    # 'Choose specific datasets' 버튼 → 데이터셋 선택기. 구조를 덤프하고 대상만 선택.
    pg.get_by_role("button", name="Choose specific datasets").first.click(timeout=4000); pg.wait_for_timeout(1200)
    # MUI Autocomplete 표준 조작: 입력 클릭 → 이름 타이핑 → 옵션 확인 → 옵션 클릭/Enter. Escape 금지(모달이 닫힘).
    add = pg.get_by_label(re.compile("Add datasets", re.I))
    if not add.count(): add = pg.get_by_role("combobox", name=re.compile("Add datasets", re.I))
    if not add.count(): add = pg.locator("input[type=text]").last
    add.first.click(timeout=4000); pg.wait_for_timeout(500)
    add.first.fill(a.dataset[:12]); pg.wait_for_timeout(1300)
    opts = pg.evaluate("""() => Array.from(document.querySelectorAll('[role=option], [role=listbox] li, .MuiAutocomplete-option')).map(e=>e.innerText.trim())""")
    (OUT/"dataset_options.txt").write_text("\n".join(opts)); print("dataset options:", opts, flush=True)
    snap(pg, "s2_dataset_options", "Add datasets")
    applied = False
    opt = pg.get_by_role("option", name=re.compile(re.escape(a.dataset)))
    if opt.count(): opt.first.click(timeout=3000); applied = True; print("dataset 선택(click):", a.dataset, flush=True)
    elif opts: pg.keyboard.press("Enter"); applied = True; print("dataset 선택(Enter)", flush=True)
    pg.wait_for_timeout(1200)
    try: pg.get_by_text("Which data should be used for training?").first.click(timeout=2000)   # blur (Escape 대신)
    except Exception: pass
    pg.wait_for_timeout(800)
    snap(pg, "s2_filter_set", "Which data")
    t = snap(pg, "s2_after_filter", "Which data")
    m = re.search(r"(\d+) annotations", t); n_ann = int(m.group(1)) if m else -1
    print("학습 annotation 수:", n_ann, flush=True)
    if not applied or n_ann <= 0 or n_ann > 320:
        print("!! 데이터셋 필터가 적용되지 않음 → 학습 시작하지 않음"); b.close(); raise SystemExit(2)
    assert nxt(pg)
    pg.get_by_role("radio", name=re.compile(r"^Small \(320m", re.I)).first.check(timeout=3000); pg.wait_for_timeout(400); assert nxt(pg)
    c = pg.get_by_text(re.compile(rf"^{re.escape(a.temporal)}$")); (c if c.count() else pg.get_by_text(a.temporal)).first.click(timeout=3000); pg.wait_for_timeout(900)
    snap(pg, "s4_temporal", "What does a label"); assert nxt(pg)
    ni = pg.get_by_label(re.compile("Model name")); (ni if ni.count() else pg.locator("input[type=text]").last).first.fill(a.name); pg.wait_for_timeout(500)
    bm = pg.get_by_role("button", name=re.compile(r"^Build Model$"))
    for i in range(90):
        if bm.count() and bm.first.is_enabled() and "Estimating cost" not in txt(pg): break
        pg.wait_for_timeout(1000)
    t = snap(pg, "s5_summary", "Summary", 2600); j = t.find("Estimated cost"); print("비용:", t[j:j+90] if j>0 else "(없음)", flush=True)
    if a.dry_run: print("DRY-RUN 중단"); b.close(); raise SystemExit(0)
    assert bm.first.is_enabled(), "Build Model 비활성"
    bm.first.click(timeout=5000); print("→ Build Model 클릭", flush=True); pg.wait_for_timeout(5000)
    pg.goto(f"{BASE}/projects/{a.project}/models", wait_until="networkidle", timeout=45000); pg.wait_for_timeout(3000)
    snap(pg, "models_after", "Models", 800); b.close()
print("OUT", OUT)
