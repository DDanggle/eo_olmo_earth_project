#!/usr/bin/env python3
"""OlmoEarth Studio 제품 감사 크롤러 — 로그인 후 도달 가능한 모든 페이지를 돌며
스크린샷 + UI 구조(제목·헤딩·네비·버튼·폼·링크)를 기록한다.

자격증명: .env 의 STUDIO_EMAIL / STUDIO_PASSWORD 만 읽는다. 절대 출력·로그·커밋하지 않는다.

로그인 전략
  A) 이메일/비밀번호 폼이 보이면 자동 입력.
  B) SSO(Google 등)/2FA 이면 자동화가 막힌다 → `--login-manual` 로 **헤드 브라우저**를 띄워
     사용자가 한 번 직접 로그인하면 세션(storage_state)을 저장하고, 이후 headless 로 재사용.

출력: artifacts/studio_audit/<ts>/
  screenshots/<n>_<slug>.png   전체 페이지 스크린샷
  site_map.json                페이지별 구조 (url, title, headings, nav, buttons, forms, links)
  crawl_log.txt

안전: 같은 origin 만, logout/delete/remove 류 링크는 클릭하지 않음(읽기 전용 탐색),
      페이지 수 상한, 페이지 간 지연. 파괴적 버튼은 기록만 하고 누르지 않는다.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://olmoearth.allenai.org"
STATE = ROOT / ".studio_session.json"          # gitignore 됨 (.env.* 아님 → 아래서 별도 추가)
DESTRUCTIVE = re.compile(r"log ?out|sign ?out|delete|remove|destroy|revoke|cancel subscription", re.I)


def load_env():
    from dotenv import dotenv_values
    v = dotenv_values(ROOT / ".env")
    # 사용자가 넣은 실제 키 이름 우선(OLMOEARTH_STUDIO_ID/_PASSWORD), 구 이름 폴백
    email = v.get("OLMOEARTH_STUDIO_ID") or v.get("STUDIO_EMAIL")
    pw = v.get("OLMOEARTH_STUDIO_PASSWORD") or v.get("STUDIO_PASSWORD")
    return email, pw


def slug(url: str) -> str:
    p = urlparse(url)
    s = (p.path + ("_" + p.query if p.query else "")).strip("/") or "home"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80]


def same_origin(url: str) -> bool:
    return urlparse(url).netloc == urlparse(BASE).netloc


def extract_structure(page) -> dict:
    """페이지의 UI 구조를 텍스트로 뽑는다 (스크린샷과 짝)."""
    return page.evaluate("""() => {
      const txt = el => (el.innerText || el.textContent || '').trim().replace(/\\s+/g,' ').slice(0,120);
      const vis = el => { const r = el.getBoundingClientRect(); return r.width>0 && r.height>0; };
      const q = (sel) => Array.from(document.querySelectorAll(sel)).filter(vis);
      return {
        title: document.title,
        h1: q('h1').map(txt), h2: q('h2').map(txt), h3: q('h3').map(txt).slice(0,30),
        nav: q('nav a, [role=navigation] a, aside a').map(a => ({text: txt(a), href: a.href})).slice(0,60),
        buttons: q('button, [role=button], input[type=submit]').map(txt).filter(Boolean).slice(0,80),
        forms: q('form').map(f => ({
          action: f.action, fields: Array.from(f.querySelectorAll('input,select,textarea'))
            .map(i => ({type: i.type||i.tagName.toLowerCase(), name: i.name||i.id||'', placeholder: i.placeholder||''})).slice(0,20)
        })),
        inputs_outside_forms: q('input:not(form input), select:not(form select)').map(i => ({type:i.type, name:i.name||i.id||'', placeholder:i.placeholder||''})).slice(0,20),
        links: Array.from(new Set(q('a[href]').map(a => a.href))).slice(0,200),
        tabs: q('[role=tab]').map(txt).slice(0,30),
        modals_dialogs: q('[role=dialog], dialog').length,
        body_excerpt: (document.body.innerText||'').replace(/\\s+/g,' ').slice(0,1500),
      };
    }""")


def dismiss_cookies(page):
    """쿠키 배너가 *보이면* 닫는다. DOM 에만 남은 숨은 버튼을 기다리다 멈추지 않는다."""
    for b in ["button:has-text('Reject Non-Essential')", "button:has-text('Accept All')"]:
        try:
            loc = page.locator(b).first
            if loc.count() and loc.is_visible():
                loc.click(timeout=3000); page.wait_for_timeout(400); return True
        except Exception:
            pass
    return False


def detect_login(page) -> dict:
    s = extract_structure(page)
    has_pw = any(f.get("type") == "password" for fm in s["forms"] for f in fm["fields"]) or \
             any(i.get("type") == "password" for i in s["inputs_outside_forms"])
    sso = [b for b in s["buttons"] if re.search(r"google|github|microsoft|sso|okta|continue with|sign in with", b, re.I)]
    body = s["body_excerpt"].lower()
    return {"has_password_form": has_pw, "sso_buttons": sso,
            "mentions_request_access": "request access" in body or "waitlist" in body,
            "structure": s}


def try_form_login(page, email, password, log) -> bool:
    for sel in ["input[type=email]", "input[name*=email i]", "input[autocomplete=username]", "input[type=text]"]:
        if page.locator(sel).count():
            page.locator(sel).first.fill(email); break
    else:
        log("no email field"); return False
    if not page.locator("input[type=password]").count():
        # 2단계 로그인(이메일→다음→비번)
        for b in ["button:has-text('Next')", "button:has-text('Continue')", "button[type=submit]"]:
            if page.locator(b).count():
                page.locator(b).first.click(); page.wait_for_timeout(1500); break
    if not page.locator("input[type=password]").count():
        log("no password field after email"); return False
    page.locator("input[type=password]").first.fill(password)
    for b in ["button[type=submit]", "button:has-text('Sign in')", "button:has-text('Log in')", "button:has-text('Continue')"]:
        if page.locator(b).count():
            page.locator(b).first.click(); break
    page.wait_for_load_state("networkidle", timeout=20000)
    page.wait_for_timeout(3000)
    # 성공 판정: 모달의 password 필드가 사라졌거나, 헤더에 로그인 사용자용 'OlmoEarth Studio' 버튼이 보임.
    # (probe 확인: 로그인 전 헤더='Sign in', 후='OlmoEarth Studio'. 모달이 늦게 닫혀 필드만 보면 거짓실패.)
    logged_in_header = page.locator("text=OlmoEarth Studio").count() > 0
    pw_gone = page.locator("input[type=password]").count() == 0
    log(f"login check: header_studio_btn={logged_in_header} password_gone={pw_gone}")
    return logged_in_header or pw_gone


# ---------------- SPA 클릭 탐색 (읽기 전용) ----------------
# 제출/파괴 동사만 막는다. 모달을 *여는* 버튼(Create new project, Add account, Edit user, Filter…)은
# 눌러서 폼을 기록하되, 모달 안의 짧은 제출 버튼(Create/Add/Save…)은 정확 일치로 차단된다.
SUBMIT_EXACT = re.compile(r"^(save|submit|confirm|create|add|update|apply|send|invite|ok|done|finish|next|"
                          r"start|run|train|publish|deploy|export|download|upload|import|pay|checkout)$", re.I)
DESTRUCTIVE_ANY = re.compile(r"delete|remove|archive|revoke|sign ?out|log ?out|cancel subscription|deactivate|"
                             r"reset password|accept all|reject non-essential|close cookie|close this dialog", re.I)
SKIP_TEXT = re.compile(r"^(d|rows per page:?|\d+|\s*)$", re.I)
# 사이드바 항목은 <a>, "All projects" 는 combobox — 버튼만 보면 놓친다
CLICKABLE = ("button:visible, [role=button]:visible, [role=tab]:visible, [role=menuitem]:visible, "
             "nav a:visible, aside a:visible, [role=navigation] a:visible, [role=combobox]:visible, "
             "[role=link]:visible, select:visible")
POPOVER = "[role=dialog], dialog, [role=menu], [role=listbox], .MuiPopover-paper, .MuiMenu-paper, .MuiDialog-paper"


def blocked(text: str) -> bool:
    t = text.strip()
    return (not t) or bool(SKIP_TEXT.match(t)) or bool(SUBMIT_EXACT.match(t)) or bool(DESTRUCTIVE_ANY.search(t))


def close_dialogs(page):
    """열린 모달/팝오버를 부작용 없이 닫는다: Esc → Cancel/Close 버튼."""
    for _ in range(2):
        page.keyboard.press("Escape"); page.wait_for_timeout(250)
    for b in ["[role=dialog] button:has-text('Cancel')", "[role=dialog] button:has-text('Close')",
              "[role=dialog] button[aria-label*=close i]"]:
        try:
            loc = page.locator(b).first
            if loc.count() and loc.is_visible():
                loc.click(timeout=2000); page.wait_for_timeout(250)
        except Exception:
            pass


def explore(page, out, log, max_states=80, max_depth=3, per_state=30):
    """상태(URL+제목)별로 보이는 클릭 요소를 *인덱스로* 순회하며 눌러보고 화면·구조를 기록한다."""
    states, edges, order = {}, [], [0]

    def state_key(s):
        return page.url + " | " + (s["h1"][:1] or s["h2"][:1] or [s["title"]])[0]

    def snap(label):
        order[0] += 1
        s = extract_structure(page)
        shot = out / "screenshots" / f"e{order[0]:03d}_{slug(page.url)}_{re.sub(r'[^A-Za-z0-9]+','_',label)[:40]}.png"
        page.screenshot(path=str(shot), full_page=True)
        return s, shot.name

    def visible_popover():
        try:
            loc = page.locator(POPOVER)
            for i in range(min(loc.count(), 6)):
                if loc.nth(i).is_visible(): return True
        except Exception:
            pass
        return False

    def visit(depth, via):
        if len(states) >= max_states: return
        s, shot = snap(via)
        key = state_key(s)
        if key in states: return
        states[key] = {"url": page.url, "via": via, "depth": depth, "screenshot": shot, **s}
        log(f"[state {len(states)} d{depth}] {page.url}  via={via!r}  buttons={len(s['buttons'])} tabs={s['tabs']}")
        if depth >= max_depth: return
        home = page.url
        n = min(page.locator(CLICKABLE).count(), per_state)
        seen_txt = set()
        for i in range(n):
            if len(states) >= max_states: return
            try:
                if page.url != home:
                    page.goto(home, wait_until="networkidle", timeout=30000); page.wait_for_timeout(600)
                    close_dialogs(page)
                loc = page.locator(CLICKABLE)
                if i >= loc.count(): break
                el = loc.nth(i)
                t = (el.inner_text(timeout=1500) or el.get_attribute("aria-label") or "").strip().replace("\n", " ")[:80]
                href = el.get_attribute("href") or ""
                if href and not same_origin(urljoin(page.url, href)):
                    log(f"   skip[{i}] {t!r} (external)"); continue
                if t in seen_txt or blocked(t):
                    log(f"   skip[{i}] {t!r} ({'dup' if t in seen_txt else 'blocked'})"); continue
                seen_txt.add(t)
                el.click(timeout=4000); page.wait_for_timeout(900)
                try: page.wait_for_load_state("networkidle", timeout=8000)
                except Exception: pass
                if visible_popover():
                    ds, dshot = snap(f"dialog:{t}")
                    edges.append({"from": home, "click": t, "kind": "dialog", "screenshot": dshot,
                                  "forms": ds["forms"], "inputs": ds["inputs_outside_forms"],
                                  "buttons": ds["buttons"][:20], "text": ds["body_excerpt"][:600]})
                    log(f"   dialog ← [{i}] {t!r}: forms={len(ds['forms'])} inputs={len(ds['inputs_outside_forms'])}")
                    close_dialogs(page)
                    if page.url != home:
                        page.goto(home, wait_until="networkidle", timeout=30000); page.wait_for_timeout(500)
                    continue
                if page.url != home:
                    edges.append({"from": home, "click": t, "kind": "navigate", "to": page.url})
                    log(f"   navigate ← [{i}] {t!r} → {page.url}")
                    visit(depth + 1, t)
                    page.goto(home, wait_until="networkidle", timeout=30000); page.wait_for_timeout(600)
                else:
                    s2 = extract_structure(page); k2 = state_key(s2)
                    if k2 != key and k2 not in states:
                        edges.append({"from": home, "click": t, "kind": "inpage", "to": page.url})
                        log(f"   inpage ← [{i}] {t!r}")
                        visit(depth + 1, t)
                    else:
                        log(f"   noop  [{i}] {t!r}")
            except Exception as e:
                log(f"   [err click {i} {t if 't' in dir() else ''!r}] {type(e).__name__}: {str(e).splitlines()[0][:90]}")
                close_dialogs(page)

    visit(0, "start")
    (out / "explore_map.json").write_text(json.dumps(
        {"states": list(states.values()), "edges": edges}, indent=1, ensure_ascii=False))
    log(f"explore 완료: 상태 {len(states)}, 엣지 {len(edges)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="로그인 페이지 방식만 탐지(자격증명 불필요)")
    ap.add_argument("--login-manual", action="store_true", help="헤드 브라우저 띄워 사용자가 직접 로그인 → 세션 저장")
    ap.add_argument("--public-only", action="store_true", help="로그인 없이 공개 페이지만 크롤")
    ap.add_argument("--explore", action="store_true", help="SPA 클릭 탐색(읽기 전용): 버튼·탭·카드를 눌러 깊이 탐색")
    ap.add_argument("--max-pages", type=int, default=120)
    ap.add_argument("--delay-ms", type=int, default=800)
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / "artifacts" / "studio_audit" / ts
    (out / "screenshots").mkdir(parents=True, exist_ok=True)
    logf = open(out / "crawl_log.txt", "a")
    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not (a.headed or a.login_manual))
        ctx_kw = {"viewport": {"width": 1440, "height": 900}}
        if STATE.exists() and not a.login_manual:
            ctx_kw["storage_state"] = str(STATE); log("세션 재사용")
        ctx = browser.new_context(**ctx_kw)
        page = ctx.new_page()
        page.goto(BASE, wait_until="networkidle", timeout=45000)
        page.wait_for_timeout(1500)

        # ---- 로그인 방식 탐지 ----
        det = detect_login(page)
        page.screenshot(path=str(out / "screenshots" / "00_landing.png"), full_page=True)
        (out / "login_detect.json").write_text(json.dumps(
            {k: v for k, v in det.items() if k != "structure"} | {"url": page.url,
             "buttons": det["structure"]["buttons"], "forms": det["structure"]["forms"],
             "body_excerpt": det["structure"]["body_excerpt"][:600]}, indent=1, ensure_ascii=False))
        log(f"landing url={page.url} password_form={det['has_password_form']} sso={det['sso_buttons']}")
        if a.probe:
            dismiss_cookies(page)   # Sign in 눌러 실제 인증 방식을 본다 (자격증명 불필요)
            if page.locator("text=Sign in").count():
                page.locator("text=Sign in").first.click()
                page.wait_for_load_state("networkidle", timeout=30000); page.wait_for_timeout(1500)
                det2 = detect_login(page)
                page.screenshot(path=str(out / "screenshots" / "01_signin.png"), full_page=True)
                (out / "signin_detect.json").write_text(json.dumps(
                    {"url": page.url, "has_password_form": det2["has_password_form"],
                     "sso_buttons": det2["sso_buttons"], "buttons": det2["structure"]["buttons"],
                     "forms": det2["structure"]["forms"], "inputs": det2["structure"]["inputs_outside_forms"],
                     "body_excerpt": det2["structure"]["body_excerpt"][:800]}, indent=1, ensure_ascii=False))
                log(f"signin url={page.url} password_form={det2['has_password_form']} sso={det2['sso_buttons']}")
            log("probe 완료"); browser.close(); return 0

        # ---- 로그인 ----
        dismiss_cookies(page)   # 배너가 버튼을 가려 클릭을 막으므로 먼저 닫는다

        if a.login_manual:
            if page.locator("text=Sign in").count():
                page.locator("text=Sign in").first.click(); page.wait_for_timeout(1000)
            log("헤드 브라우저에서 직접 로그인하세요(Google 포함). 끝나면 이 터미널에서 Enter.")
            input()
            ctx.storage_state(path=str(STATE)); log(f"세션 저장 → {STATE.name}")
        elif a.public_only:
            log("public-only: 로그인 없이 공개 페이지만 크롤")
        elif not STATE.exists():
            # 폼은 랜딩이 아니라 'Sign in' 클릭 후 모달에 있다 (probe 로 확인: email+password + Google SSO)
            if page.locator("text=Sign in").count():
                page.locator("text=Sign in").first.click()
                page.wait_for_timeout(1500)
            det = detect_login(page)
            if det["has_password_form"]:
                email, pw = load_env()
                if not (email and pw):
                    log("STUDIO_EMAIL/STUDIO_PASSWORD 가 .env 에 없음"); browser.close(); return 2
                ok = try_form_login(page, email, pw, log)
                log(f"폼 로그인 {'성공' if ok else '실패'} url={page.url}")
                if not ok:
                    page.screenshot(path=str(out / "screenshots" / "01_login_failed.png"), full_page=True)
                    log("실패 시: Google 로 가입한 계정이면 `--login-manual` 로 한 번 직접 로그인하세요.")
                    browser.close(); return 3
                ctx.storage_state(path=str(STATE)); log(f"세션 저장 → {STATE.name}")
            elif det["sso_buttons"]:
                log(f"SSO 만 감지 {det['sso_buttons']} → `--login-manual` 로 한 번 직접 로그인 필요.")
                browser.close(); return 4
        # 크롤 시작점: 로그인 직후 앱 URL(/projects). 세션 재사용이면 직접 /projects 로 간다.
        if "/projects" not in page.url and not a.public_only:
            page.goto(BASE + "/projects", wait_until="networkidle", timeout=45000); page.wait_for_timeout(1200)
        dismiss_cookies(page)
        log(f"크롤 시작 url={page.url}")

        if a.explore:
            explore(page, out, log, max_states=a.max_pages)
            browser.close(); return 0

        # ---- BFS 크롤 (읽기 전용) ----
        seen, queue, site = set(), deque([page.url]), []
        n = 0
        while queue and n < a.max_pages:
            url = queue.popleft()
            if url in seen or not same_origin(url) or DESTRUCTIVE.search(url):
                continue
            seen.add(url)
            try:
                page.goto(url, wait_until="networkidle", timeout=45000)
                page.wait_for_timeout(a.delay_ms)
                s = extract_structure(page)
                n += 1
                shot = out / "screenshots" / f"{n:02d}_{slug(page.url)}.png"
                page.screenshot(path=str(shot), full_page=True)
                site.append({"n": n, "url": page.url, "requested": url, "screenshot": shot.name, **s})
                log(f"[{n}] {page.url}  h1={s['h1'][:2]} buttons={len(s['buttons'])} links={len(s['links'])}")
                for l in s["links"]:
                    l2 = urljoin(page.url, l).split("#")[0]
                    if same_origin(l2) and l2 not in seen and not DESTRUCTIVE.search(l2):
                        queue.append(l2)
            except Exception as e:
                log(f"[skip] {url} — {type(e).__name__}: {str(e)[:100]}")
        (out / "site_map.json").write_text(json.dumps(site, indent=1, ensure_ascii=False))
        log(f"완료: {n} 페이지, 미방문 큐 {len(queue)}")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
