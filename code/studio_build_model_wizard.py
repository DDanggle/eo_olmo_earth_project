#!/usr/bin/env python3
"""Build model 마법사를 읽기 전용으로 단계별 캡처. Next 만 누르고 최종 Build/Start/Train 은 절대 누르지 않는다.
Advanced options / Hacker mode 는 별도 캡처(1단계에서만, 캡처 후 원상 복구 시도)."""
import json, re, time, sys
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]; BASE = "https://olmoearth.allenai.org"
PROJ = sys.argv[1] if len(sys.argv) > 1 else "7f7546ff-cbc4-4c6c-8b09-deb2f63b2bb0"
FINAL = re.compile(r"^(build|build model|start|start training|train|run|submit|create|finish)$", re.I)
out = ROOT / "artifacts/studio_audit/build_model_wizard" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True)
def dump(page):
    return page.evaluate("""() => { const vis=e=>{const r=e.getBoundingClientRect();return r.width>0&&r.height>0};
      const q=s=>Array.from(document.querySelectorAll(s)).filter(vis); const t=e=>(e.innerText||'').trim().replace(/\\s+/g,' ').slice(0,90);
      return { step:(document.querySelector('[aria-current=step],.Mui-active')||{}).innerText||'', buttons:q('button,[role=button]').map(t).filter(Boolean),
        radios:q('[role=radio],input[type=radio]').map(e=>({t:t(e.closest('label')||e.parentElement),on:e.checked||e.getAttribute('aria-checked')==='true'})),
        switches:q('[role=switch],input[type=checkbox]').map(e=>({t:t(e.closest('label')||e.parentElement),on:e.checked||e.getAttribute('aria-checked')==='true'})),
        selects:q('[role=combobox],select').map(t), sliders:q('[role=slider],input[type=range]').map(e=>e.getAttribute('aria-valuetext')||e.value),
        text:document.body.innerText.replace(/\\s+/g,' ').slice(0,2500) }; }""")
rec = []
with sync_playwright() as p:
    b = p.chromium.launch(headless=True); ctx = b.new_context(storage_state=str(ROOT/".studio_session.json"), viewport={"width":1440,"height":1000}); page = ctx.new_page()
    page.goto(f"{BASE}/projects/{PROJ}/models", wait_until="networkidle", timeout=45000); page.wait_for_timeout(800)
    try:
        l = page.locator("button:has-text('Reject Non-Essential')").first
        if l.count() and l.is_visible(): l.click(timeout=2000)
    except Exception: pass
    page.get_by_role("button", name="Build model").first.click(timeout=5000); page.wait_for_timeout(1200)
    d = dump(page); page.screenshot(path=str(out/"step1.png"), full_page=True); rec.append({"label":"step1", **d}); print("[step1]", d["text"][d["text"].find("What do you want"):][:300])
    # 1) Next 로 끝까지 (최종 버튼은 절대 클릭 안 함)
    for step in range(2, 8):
        nxt = page.get_by_role("button", name="Next")
        if not (nxt.count() and nxt.first.is_visible()): print("Next 없음 → 종료"); break
        if not nxt.first.is_enabled(): print(f"[step {step-1}] Next 비활성 (필수 선택 필요) → 종료"); break
        nxt.first.click(timeout=4000); page.wait_for_timeout(1300)
        d = dump(page); page.screenshot(path=str(out/f"step{step}.png"), full_page=True); rec.append({"label":f"step{step}", **d})
        finals = [x for x in d["buttons"] if FINAL.match(x.strip())]
        i0 = max(d["text"].find("Model Training data"), 0)
        print(f"[step{step}] finals={finals} buttons={[x for x in d['buttons'] if x not in ('D','Save','Accept All','Reject Non-Essential','Close this dialog','Close Cookie Preferences','Toy project - jeju')][:9]}")
        print("        ", d["text"][i0:i0+700])
        if finals: print("최종 제출 버튼 도달 — 클릭하지 않고 종료"); break
    # 2) 모달 닫고 다시 열어 Advanced options / Hacker mode 캡처
    page.keyboard.press("Escape"); page.wait_for_timeout(600)
    page.goto(f"{BASE}/projects/{PROJ}/models", wait_until="networkidle", timeout=45000); page.wait_for_timeout(800)
    page.get_by_role("button", name="Build model").first.click(timeout=5000); page.wait_for_timeout(1200)
    for lbl in ["Advanced options", "Hacker mode"]:
        l = page.get_by_role("button", name=lbl)
        if l.count() and l.first.is_visible():
            l.first.click(timeout=3000); page.wait_for_timeout(900)
            d = dump(page); page.screenshot(path=str(out/f"step1_{lbl.replace(' ','_')}.png"), full_page=True); rec.append({"label":f"step1:{lbl}", **d})
            i0 = max(d["text"].find("What do you want"), 0); print(f"[step1:{lbl}]", d["text"][i0:i0+900])
    (out/"wizard.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False)); b.close()
print("OUT", out)
