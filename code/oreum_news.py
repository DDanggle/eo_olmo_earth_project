#!/usr/bin/env python3
"""상위 오름의 한국 뉴스 기사 후보 — 타임라인의 **맥락**이다. 근거가 아니다.

Google News RSS(키 불필요)로 "<오름명> 제주" 를 검색해 2021~ 기사 제목·날짜·출처·링크만 모은다.
기사 본문은 저장하지 않는다(저작권·검증 불가). 제목에 훼손·개발·복원 관련 낱말이 있으면 `topic` 태그를 붙이되
**관련성은 사람이 판단한다** — 오름 이름은 지명·상호와 겹치는 경우가 많다(예: 가마오름, 세미소).

절대 하지 않는 것: 기사와 깃발을 인과로 연결, 기사 내용 요약·재작성, 없는 기사 만들기.
산출물: <WEB_DATA_ROOT>/series/<oreum_id>_news.json
"""
from __future__ import annotations

import json
import re
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime

from jeju_paths import WEB_DATA_ROOT, display_path

TOPICS = {
    "훼손·개발": ("훼손", "개발", "공사", "벌채", "절토", "허가", "태양광", "골프", "채석", "매립", "불법", "훼손지"),
    "복원·보전": ("복원", "보전", "보호", "지정", "국립공원", "생태", "람사르", "천연기념물", "휴식년"),
    "재해": ("산불", "화재", "산사태", "폭우", "태풍", "붕괴"),
    "탐방": ("탐방", "관광", "축제", "둘레길", "정상"),
}
SINCE = datetime(2021, 1, 1)


def fetch(name: str) -> list[dict]:
    q = urllib.parse.quote(f'"{name}" 제주')
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    ctx = ssl._create_unverified_context()
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60, context=ctx).read()
    root = ET.fromstring(raw)
    out = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = it.findtext("link") or ""
        pub = it.findtext("pubDate")
        src = it.find("source"); source = src.text if src is not None else ""
        try:
            dt = parsedate_to_datetime(pub).replace(tzinfo=None)
        except Exception:
            continue
        if dt < SINCE:
            continue
        title_clean = re.sub(r"\s*-\s*[^-]+$", "", title)      # "제목 - 출처" 꼬리 제거
        topics = [k for k, ws in TOPICS.items() if any(w in title_clean for w in ws)]
        out.append({"date": dt.strftime("%Y-%m-%d"), "month": dt.strftime("%Y-%m"), "title": title_clean,
                    "source": source, "link": link, "topics": topics})
    out.sort(key=lambda r: r["date"])
    return out


def main() -> None:
    series = WEB_DATA_ROOT / "series"
    idx = json.loads((series / "index.json").read_text())
    total = 0
    for oid in idx["oreum_ids"]:
        meta = json.loads((series / f"{oid}.json").read_text())
        name = meta["name"]
        try:
            items = fetch(name)
        except Exception as e:
            items, err = [], type(e).__name__
        else:
            err = None
        rec = {"oreum_id": oid, "name": name, "query": f'"{name}" 제주', "source": "Google News RSS",
               "fetched_at": datetime.now().strftime("%Y-%m-%d"), "since": "2021-01-01", "n": len(items), "error": err,
               "caveat": "맥락 자료. 오름 이름은 지명·상호와 겹칠 수 있어 관련성은 사람이 판단한다. 기사와 점수를 인과로 연결하지 않는다.",
               "items": items}
        (series / f"{oid}_news.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        by_month = {}
        for r in items:
            by_month[r["month"]] = by_month.get(r["month"], 0) + 1
        print(f"  {oid} {name}: 기사 {len(items)}건" + (f" ({err})" if err else "") +
              (f" · 훼손·개발 {sum('훼손·개발' in r['topics'] for r in items)}" if items else ""))
        total += len(items); time.sleep(1.5)
    print(f"총 {total}건 → {display_path(series)}")


if __name__ == "__main__":
    main()
