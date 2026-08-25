#!/usr/bin/env python3
"""G-0 — Sen12Landslides 접근 감사. MountainShift 전체의 첫 게이트.

설계 문서(docs/MOUNTAINSHIFT_EXPERIMENT_DESIGN.md) §4:
  "G-0 실패 시 MountainShift 중단. 여기서 막히면 없다."

여기서 판정할 것 (전부 실물 기준. 논문·검색 요약을 근거로 쓰지 않는다):
  G0-1 다운로드 가능한가 — repo 파일 목록을 실제로 받을 수 있는가
  G0-2 라이선스가 무엇인가 — 재배포·파생물 조건
  G0-3 **지역이 실제로 몇 개이고 파일 단위로 분리되는가** — 15지역 LOCO의 전제
  G0-4 split 정의가 제공되는가 (S12LS-LD / S12LS-AD)
  G0-5 밴드·시점 구성이 OlmoEarth 입력 계약과 맞는가 (S2 10밴드 B02–B12, 15 timestep)
  G0-6 event date와 confidence 필드가 실재하는가 (cutoff replay의 전제)

전체를 내려받지 않는다. 파일 목록과 표본 1개만 본다.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

REPO = "paulhoehn/Sen12Landslides"
OUT = Path("/home/work/data/olmoearth/sen12landslides/audit")


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    result: dict = {"schema": "sen12landslides-access-audit-v1", "repo": REPO}

    # ---- G0-1 / G0-2 : repo 메타 ----
    info = api.dataset_info(REPO, files_metadata=False)
    card = getattr(info, "card_data", None)
    result["G0_2_license"] = (card.get("license") if isinstance(card, dict)
                              else getattr(card, "license", None))
    result["last_modified"] = str(getattr(info, "last_modified", None))
    result["tags"] = list(getattr(info, "tags", []) or [])[:20]

    files = api.list_repo_files(REPO, repo_type="dataset")
    result["G0_1_file_count"] = len(files)
    result["extensions"] = dict(Counter(
        f.rsplit(".", 1)[-1].lower() if "." in f else "(none)" for f in files).most_common(10))
    result["top_level"] = dict(Counter(f.split("/")[0] for f in files).most_common(20))
    result["sample_paths"] = files[:15]

    # ---- G0-3 : 지역 분리 ----
    # 경로에서 지역명 후보를 뽑는다. 규칙을 가정하지 않고 계층을 훑는다.
    seg_counts = {}
    for depth in range(4):
        vals = Counter(f.split("/")[depth] for f in files if len(f.split("/")) > depth + 1)
        seg_counts[f"depth_{depth}"] = {"unique": len(vals), "top": dict(vals.most_common(20))}
    result["G0_3_path_segments"] = seg_counts

    # NetCDF 파일명에서 지역 접두 추출 시도
    nc = [f for f in files if f.lower().endswith((".nc", ".nc4"))]
    result["netcdf_count"] = len(nc)
    if nc:
        stems = [Path(f).stem for f in nc]
        prefix = Counter(re.split(r"[_\-\d]", s)[0] for s in stems if s)
        result["G0_3_filename_prefixes"] = {"unique": len(prefix),
                                            "top": dict(prefix.most_common(25))}

    # ---- G0-4 : split 정의 파일 ----
    split_like = [f for f in files
                  if re.search(r"(split|train|val|test|fold|S12LS[-_](LD|AD))", f, re.I)]
    result["G0_4_split_like_files"] = split_like[:40]
    result["G0_4_split_like_count"] = len(split_like)

    # ---- G0-5 / G0-6 : 표본 1개 구조 ----
    sample_info = None
    if nc:
        target = sorted(nc)[0]
        try:
            p = Path(hf_hub_download(REPO, target, repo_type="dataset"))
            sample_info = {"path": target, "bytes": p.stat().st_size}
            try:
                import xarray as xr
                ds = xr.open_dataset(p)
                sample_info["dims"] = {k: int(v) for k, v in ds.sizes.items()}
                sample_info["data_vars"] = list(ds.data_vars)[:30]
                sample_info["coords"] = list(ds.coords)[:20]
                sample_info["attrs"] = {k: str(v)[:200] for k, v in list(ds.attrs.items())[:30]}
                # event date / confidence 후보
                sample_info["G0_6_date_like"] = [
                    k for k in list(ds.attrs) + list(ds.data_vars) + list(ds.coords)
                    if re.search(r"(date|time|event|confid)", str(k), re.I)]
                ds.close()
            except ImportError:
                sample_info["note"] = "xarray 없음 — 구조 미확인"
        except Exception as exc:  # noqa: BLE001
            sample_info = {"error": repr(exc)[:300]}
    result["G0_5_sample"] = sample_info

    result["gates"] = {
        "G0_1_downloadable": result["G0_1_file_count"] > 0,
        "G0_2_license_declared": bool(result["G0_2_license"]),
        "G0_3_regions_separable": None,   # 아래 출력 보고 사람이 판정 후 기록
        "G0_4_split_provided": result["G0_4_split_like_count"] > 0,
        "G0_5_sample_readable": bool(sample_info and "dims" in (sample_info or {})),
    }

    (OUT / "access_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)[:6000])
    print("DONE")


if __name__ == "__main__":
    main()
