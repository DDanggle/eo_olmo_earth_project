#!/usr/bin/env python3
"""Studio 모델 목록을 주기적으로 읽어 상태 변화를 기록한다 (읽기 전용).
training 이 아닌 상태로 바뀐 모델은 상세 페이지를 스크린샷·텍스트 덤프한다.
종료 조건: 감시 대상(--watch) 전부가 training 을 벗어나거나 --max-min 경과."""
import argparse, json, re, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]; BASE = "https://olmoearth.allenai.org"
ap = argparse.ArgumentParser(); ap.add_argument("--project", required=True)
ap.add_argument("--watch", nargs="+", required=True, help="감시할 모델 이름")
ap.add_argument("--interval", type=int, default=120); ap.add_argument("--max-min", type=int, default=120); a = ap.parse_args()
OUT = ROOT/"artifacts/studio_audit/model_poll"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT/"poll_log.jsonl"
txt = lambda pg: pg.evaluate("() => document.body.innerText.replace(/\\s+/g,' ')")

def read_models(pg):
    pg.goto(f"{BASE}/projects/{a.project}/models", wait_until="networkidle", timeout=45000); pg.wait_for_timeout(1200)
    # 모델 목록은 <table>이 아니라 카드(링크 없음, 2026-09-19 DOM 확인). 카드 텍스트 패턴:
    #   "<name> Fine-tuned model <task> <source> Created M/D/YYYY <status>"
    t = txt(pg); out = {}
    for m in re.finditer(r"(\S+) (Fine-tuned model|Embeddings model) (.*?) Created (\d+/\d+/\d+) (\w+)", t):
        out[m.group(1)] = {"status": m.group(5), "href": None, "cells": [m.group(1), m.group(3), m.group(4), m.group(5)]}
    return out

def capture_detail(pg, name, m, tag):
    if m["href"]:
        pg.goto(BASE + m["href"] if m["href"].startswith("/") else m["href"], wait_until="networkidle", timeout=45000)
    else:
        pg.get_by_text(name, exact=True).first.click(timeout=5000); pg.wait_for_load_state("networkidle")
    pg.wait_for_timeout(1500)
    safe = re.sub(r"[^\w.-]", "_", name)
    pg.screenshot(path=str(OUT/f"{safe}_{tag}.png"), full_page=True)
    t = txt(pg); (OUT/f"{safe}_{tag}.txt").write_text(t)
    # 탭이 있으면 하나씩 눌러 캡처 (읽기 전용)
    tabs = pg.get_by_role("tab")
    for i in range(min(tabs.count(), 8)):
        try:
            tb = tabs.nth(i); tn = re.sub(r"[^\w.-]", "_", tb.inner_text().strip())[:30]
            tb.click(timeout=3000); pg.wait_for_timeout(1200)
            pg.screenshot(path=str(OUT/f"{safe}_{tag}_tab{i}_{tn}.png"), full_page=True)
            (OUT/f"{safe}_{tag}_tab{i}_{tn}.txt").write_text(txt(pg))
        except Exception as e: print("tab fail", i, e, flush=True)
    return t

with sync_playwright() as p:
    b = p.chromium.launch(headless=True); ctx = b.new_context(storage_state=str(ROOT/".studio_session.json"), viewport={"width":1440,"height":1150}); pg = ctx.new_page()
    t0 = time.time(); done = set(); last = {}
    while True:
        models = read_models(pg); now = time.strftime("%H:%M:%S")
        line = {k: v["status"] for k, v in models.items()}
        print(now, json.dumps(line, ensure_ascii=False), flush=True)
        with LOG.open("a") as f: f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": line}) + "\n")
        for name in a.watch:
            m = models.get(name)
            if not m: print(f"  !! {name} 목록에 없음", flush=True); continue
            if m["status"].lower() != "training" and name not in done:
                print(f"  -> {name}: {m['status']} — 상세 캡처", flush=True)
                t = capture_detail(pg, name, m, time.strftime("%H%M%S")); print("  ", t[:700], flush=True); done.add(name)
        if not models: print("  !! 모델 카드 0개 — 파싱 실패 또는 세션 만료. 계속 재시도", flush=True)
        if all(n in done for n in a.watch): print("ALL_DONE", flush=True); break
        if time.time() - t0 > a.max_min * 60: print("TIMEOUT", flush=True); break
        time.sleep(a.interval)
    b.close()
