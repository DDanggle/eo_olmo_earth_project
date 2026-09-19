#!/usr/bin/env python3
"""OlmoEarth Studio 에 학습 데이터 파일을 임포트한다 (사용자 승인 하에 실행 — 계정 데이터를 바꾼다).

마법사 각 단계를 스크린샷·폼 덤프로 기록하며 Next 로 진행. 필수 입력이 비어 Next 가 막히면
추측하지 않고 멈춰서 보고한다(--dry-run 이면 최종 제출 직전에 멈춤). Delete 류는 절대 누르지 않는다.
"""
import argparse, json, re, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]; BASE = "https://olmoearth.allenai.org"
FINAL = re.compile(r"^(import|import data|finish|submit|create dataset|start import|confirm|done)$", re.I)

def dump(page):
    return page.evaluate("""() => { const vis=e=>{const r=e.getBoundingClientRect();return r.width>0&&r.height>0};
      const q=s=>Array.from(document.querySelectorAll(s)).filter(vis);
      const t=e=>(e.innerText||e.textContent||'').trim().replace(/\\s+/g,' ').slice(0,80);
      return { buttons:q('button,[role=button]').map(t).filter(Boolean),
        inputs:q('input,select,textarea').map(i=>({type:i.type||i.tagName,name:i.name||i.id||'',placeholder:i.placeholder||'',value:(i.type==='password'?'':String(i.value||'').slice(0,40)),required:!!i.required})),
        selects:q('[role=combobox]').map(t), radios:q('[role=radio],input[type=radio]').map(e=>({t:t(e.closest('label')||e),checked:e.checked||e.getAttribute('aria-checked')==='true'})),
        text:(document.querySelector('[role=dialog]')||document.body).innerText.replace(/\\s+/g,' ').slice(0,1200) }; }""")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--file", required=True); ap.add_argument("--project", required=True)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--merge", default="No", help="Merge with Existing Data 단계 선택: No(별도 데이터셋) / Yes"); ap.add_argument("--max-steps", type=int, default=8); a = ap.parse_args()
    f = Path(a.file).resolve(); assert f.exists(), f
    out = ROOT / "artifacts/studio_audit/jeju_import" / f"import_{time.strftime('%Y%m%d_%H%M%S')}"; out.mkdir(parents=True)
    steps = []
    def log(m): print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(storage_state=str(ROOT/".studio_session.json"), viewport={"width":1440,"height":1000})
        page = ctx.new_page()
        page.goto(f"{BASE}/projects/{a.project}/datasets", wait_until="networkidle", timeout=45000); page.wait_for_timeout(800)
        try:
            l = page.locator("button:has-text('Reject Non-Essential')").first
            if l.count() and l.is_visible(): l.click(timeout=2000)
        except Exception: pass
        page.get_by_role("button", name="Import training data").first.click(timeout=5000); page.wait_for_timeout(1000)
        fi = page.locator("input[type=file]")
        if not fi.count(): log("file input 없음 → dropzone 클릭 시도"); page.get_by_text("click to select").first.click(); page.wait_for_timeout(500); fi = page.locator("input[type=file]")
        fi.first.set_input_files(str(f)); page.wait_for_timeout(2500)
        log(f"파일 선택: {f.name} ({f.stat().st_size} bytes)")
        for step in range(1, a.max_steps + 1):
            page.wait_for_timeout(800)
            d = dump(page); shot = out / f"step{step:02d}.png"; page.screenshot(path=str(shot), full_page=True)
            steps.append({"step": step, "url": page.url, "screenshot": shot.name, **d})
            log(f"[step {step}] buttons={[x for x in d['buttons'] if x not in ('D','Save','Accept All','Reject Non-Essential','Close this dialog','Close Cookie Preferences')][:10]}")
            log(f"          text: {d['text'][:300]}")
            if "Merge with Existing" in d["text"] or page.get_by_text("Merge with Existing Data").count():
                rads = page.evaluate("""() => Array.from(document.querySelectorAll('[role=radio],input[type=radio]')).map(e=>({t:((e.closest('label')||e.parentElement)||e).innerText.trim().slice(0,90), on:e.checked||e.getAttribute('aria-checked')==='true'}))""")
                log(f"          merge step options: {rads}")
                # 질문은 "기존 데이터셋과 병합할까?" Yes(recommended)/No. 별도 데이터셋으로 두려면 No.
                merge_choice = getattr(a, "merge", "No")
                r = page.get_by_role("radio", name=re.compile(rf"^{re.escape(merge_choice)}", re.I))
                if not r.count():
                    r = page.get_by_text(re.compile(rf"^{re.escape(merge_choice)}\b", re.I))
                if r.count():
                    r.first.click(timeout=2500); page.wait_for_timeout(600); log(f"          merge: '{merge_choice}' 선택")
                else:
                    log("          merge 선택지를 못 찾음")
            sa = page.get_by_role("button", name="+ select all")
            if not sa.count(): sa = page.locator("text=+ select all")
            if sa.count() and sa.first.is_visible():
                sa.first.click(timeout=3000); page.wait_for_timeout(1500)
                d = dump(page); page.screenshot(path=str(out / f"step{step:02d}_selected.png"), full_page=True)
                steps.append({"step": f"{step}_selected", "url": page.url, "screenshot": f"step{step:02d}_selected.png", **d})
                # 선택된 필드와 추론된 타입 기록
                fields = page.evaluate(r"""() => Array.from(document.querySelectorAll('table tr, [role=row]')).map(r=>r.innerText.replace(/\s+/g,' ').trim()).filter(t=>t && !/Field Name/.test(t)).slice(0,20)""")
                log(f"          select all → fields: {fields}")
            finals = [x for x in d["buttons"] if FINAL.match(x.strip())]
            nxt = page.get_by_role("button", name="Next")
            if finals and not (nxt.count() and nxt.first.is_visible() and nxt.first.is_enabled()):
                if a.dry_run: log(f"DRY-RUN: 최종 버튼 {finals} 직전에서 중단"); break
                log(f"최종 제출 클릭: {finals[0]!r}")
                page.get_by_role("button", name=finals[0]).first.click(timeout=5000)
                page.wait_for_timeout(4000)
                try: page.wait_for_load_state("networkidle", timeout=30000)
                except Exception: pass
                d2 = dump(page); page.screenshot(path=str(out/"after_submit.png"), full_page=True)
                steps.append({"step": "after_submit", "url": page.url, "screenshot": "after_submit.png", **d2})
                log(f"제출 후: {d2['text'][:400]}"); break
            if nxt.count() and nxt.first.is_visible():
                for _ in range(20):
                    if nxt.first.is_enabled(): break
                    page.wait_for_timeout(1000)
                if not nxt.first.is_enabled():
                    log("Next 비활성(20s 대기 후) — 필수 입력이 비어 있음. 추측하지 않고 중단."); break
                nxt.first.click(timeout=5000); log("→ Next"); continue
            log("Next/최종 버튼 없음 — 중단"); break
        # 결과 확인: Datasets 테이블
        page.keyboard.press("Escape"); page.goto(f"{BASE}/projects/{a.project}/datasets", wait_until="networkidle", timeout=45000); page.wait_for_timeout(1500)
        page.screenshot(path=str(out/"datasets_after.png"), full_page=True)
        rows = page.evaluate("() => document.body.innerText.replace(/\\s+/g,' ').slice(0,800)")
        log(f"Datasets 페이지: {rows}")
        (out/"steps.json").write_text(json.dumps({"file": str(f), "steps": steps, "datasets_after": rows}, indent=1, ensure_ascii=False))
        b.close()
    print("OUT", out)

if __name__ == "__main__": main()
