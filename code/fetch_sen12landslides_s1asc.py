#!/usr/bin/env python3
"""Sen12Landslides harmonized S2 수신 — 핵심 경로(Italy → Korea)의 첫 블로커.

왜 이것부터인가: 2026-08-25 자체 감사 결과 최근 측정 8건 중 7건이 `E_live` 배관(KMA API)이었고
핵심 경로는 **수신 0 · 물질화 0 · probe 0** 이었음. 큰 그림은 `Italy → Korea 임베딩 강화`이며
그 첫 블로커는 데이터 수신임.

받는 것: `data_harmonized/s2/` 28 파트, 약 39.42 GB.
  - **harmonized만** 받음. `data_raw`와 섞으면 안 됨 — harmonized는 ESA Baseline 04.00의
    +1000 DN offset을 보정했고 raw는 안 했음 (M11).
  - 파트가 지역별로 묶여 있으므로(part01 = chimanimani 500개, M11) 어느 파트에 어느 지역이
    있는지 모름. 필요한 지역이 Italy(T-x arm) + Höhn 11지역(headline)이라 대부분이 필요하므로
    28 파트를 전부 받고 **수신 후 지역 인덱스를 만듦**.

산출물
  raw/          받은 tar.gz (해시와 함께 manifest에 기록)
  extracted/    풀어놓은 .nc
  region_index_s1asc.jsonl   파일별 (region, sensor, id, 파트) — 지역 단위 LOCO의 기반
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

REPO = "paulhoehn/Sen12Landslides"
PREFIX = "data_harmonized/s1asc/"
ROOT = Path(os.environ.get("S12_ROOT", "/home/work/data/sen12landslides"))
RAW = ROOT / "raw"
EXT = ROOT / "extracted"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    RAW.mkdir(parents=True, exist_ok=True)
    EXT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.dataset_info(REPO, files_metadata=True)
    sizes = {s.rfilename: (s.size or (s.lfs.size if s.lfs else None))
             for s in info.siblings}
    parts = sorted(k for k in sizes if k.startswith(PREFIX) and k.endswith(".tar.gz"))
    total = sum(sizes[k] or 0 for k in parts)
    print(f"파트 {len(parts)}개, 합계 {total/1e9:.2f} GB", flush=True)

    manifest_p = ROOT / "manifest_s1asc.jsonl"
    done = set()
    if manifest_p.exists():
        for line in manifest_p.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                if r.get("extracted"):
                    done.add(r["part"])

    index_p = ROOT / "region_index_s1asc.jsonl"
    region_counter = Counter()
    with manifest_p.open("a", encoding="utf-8") as mf, \
         index_p.open("a", encoding="utf-8") as xf:
        for i, part in enumerate(parts, 1):
            name = Path(part).name
            if part in done:
                print(f"[{i}/{len(parts)}] {name} 건너뜀 (이미 완료)", flush=True)
                continue
            t0 = time.time()
            local = Path(hf_hub_download(REPO, part, repo_type="dataset"))
            dl = time.time() - t0
            digest = sha256_file(local)
            # 전개하면서 파일명에서 지역을 뽑음: <region>_<sensor>_<id>.nc (M11 확인)
            members, regions = 0, Counter()
            with tarfile.open(local, "r:gz") as tf:
                for m in tf:
                    if not m.name.endswith(".nc"):
                        continue
                    fn = Path(m.name).name
                    region = fn.split("_")[0]
                    regions[region] += 1
                    members += 1
                    xf.write(json.dumps({"file": fn, "region": region,
                                         "part": name}, ensure_ascii=False) + "\n")
                tf.extractall(EXT)   # noqa: S202  (공개 CC BY 4.0 데이터셋)
            xf.flush()
            region_counter.update(regions)
            mf.write(json.dumps({
                "part": part, "name": name, "bytes": sizes[part],
                "sha256": digest, "nc_files": members,
                "regions": dict(regions.most_common()),
                "download_s": round(dl, 1), "extracted": True}, ensure_ascii=False) + "\n")
            mf.flush()
            print(f"[{i}/{len(parts)}] {name} {sizes[part]/1e9:.2f} GB "
                  f"{dl:.0f}s · nc {members} · 지역 {dict(regions.most_common(3))}",
                  flush=True)

    summary = {"schema": "sen12landslides-s1asc-fetch-v1", "repo": REPO,
               "license": "CC BY 4.0", "prefix": PREFIX,
               "parts": len(parts), "bytes": total,
               "regions_seen": dict(region_counter.most_common())}
    (ROOT / "fetch_summary_s1asc.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
