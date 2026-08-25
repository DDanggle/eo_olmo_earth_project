#!/usr/bin/env python3
"""AI-Hub 71363 데이터 계약 봉인 — split 해시만으로는 부족하다.

M10은 **타일 ID 목록**만 동결했다. 그런데 다음이 바뀌어도 타일 ID는 그대로다.

  - 원본 zip이 다른 버전으로 교체됨
  - 라벨 폴리곤 내용이 바뀜
  - 인벤토리 생성 로직(좌표 해석 등)이 바뀜
  - split builder 코드가 바뀜

따라서 "이 수치는 이 데이터로 나왔다"를 증명하려면 네 층을 같이 묶어야 한다.

  L0 원본      각 zip의 SHA-256과 바이트 수
  L1 파생      inventory.jsonl / tile_assignment.jsonl / spatial_holdout.json 해시
  L2 내용      타일별 geometry 해시, 라벨 클래스 해시 (ID가 같아도 내용이 바뀌면 잡힌다)
  L3 코드      생성에 쓰인 스크립트들의 해시와 git commit

논문 공개 대상은 원본이 아니라 이 계약 파일이다 (AIHUB_INQUIRY.md 질문 2).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

RAW = Path("/home/work/data/olmoearth/aihub/raw/71363")
SPLITS = Path("/home/work/data/olmoearth/aihub/splits")
INVDIR = Path("/home/work/data/olmoearth/aihub/inventory")
CODE = Path("/home/work/data/code")
OUT = Path("/home/work/data/olmoearth/aihub/contract")

CODE_FILES = [
    "build_aihub_inventory.py",
    "audit_aihub_split_leakage.py",
    "build_aihub_spatial_holdout.py",
    "audit_aihub_71363.py",
    "aihub_setup.sh",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(data: bytes):
    for enc in ("utf-8-sig", "utf-8", "cp949", "latin1"):
        try:
            return json.loads(data.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError("디코딩 실패")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    contract: dict = {"schema": "aihub-71363-data-contract-v1",
                      "dataset": {"portal": "aihub.or.kr", "datasetkey": 71363,
                                  "subset": "Sentinel-2 10m only"}}

    # ---- L0 원본 zip ----
    zips = sorted(RAW.rglob("*.zip"))
    contract["L0_source_zips"] = [
        {"name": z.name, "relpath": str(z.relative_to(RAW)),
         "bytes": z.stat().st_size, "sha256": sha256_file(z)}
        for z in zips
    ]

    # ---- L1 파생 산출물 ----
    derived = {}
    for p in [INVDIR / "inventory.jsonl", INVDIR / "inventory_audit.json",
              INVDIR / "split_leakage_audit.json", SPLITS / "spatial_holdout.json",
              SPLITS / "tile_assignment.jsonl", SPLITS / "loco_folds.json"]:
        if p.exists():
            derived[p.name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    contract["L1_derived"] = derived

    # ---- L2 내용 해시 (ID가 같아도 내용이 바뀌면 잡힌다) ----
    # geometry: 인벤토리의 (타일, 날짜, bbox)를 정렬해 한 덩어리로 해싱
    geo_lines = []
    for line in (INVDIR / "inventory.jsonl").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        r = json.loads(line)
        geo_lines.append(f'{r["key"]}|{r["platform"]}|{",".join(map(str, r["utm52n_bbox"]))}')
    contract["L2_geometry_sha256"] = hashlib.sha256(
        "\n".join(sorted(geo_lines)).encode()).hexdigest()
    contract["L2_geometry_rows"] = len(geo_lines)

    # label: 타일별 (클래스, 폴리곤 좌표수)를 정렬해 해싱. 폴리곤이 바뀌면 값이 바뀐다.
    label_lines = []
    for pattern in ("TL_02.JSON_03*.zip", "VL_02.JSON_03*.zip"):
        hits = sorted(RAW.rglob(pattern))
        if not hits:
            continue
        with zipfile.ZipFile(hits[0]) as zf:
            for name in sorted(n for n in zf.namelist() if n.lower().endswith(".json")):
                obj = read_json(zf.read(name))
                if not isinstance(obj, dict):
                    continue
                key = str(obj.get("name") or Path(name).stem)
                for feat in obj.get("features") or []:
                    props = (feat or {}).get("properties") or {}
                    n_coord = len(json.dumps((feat or {}).get("geometry") or {}))
                    label_lines.append(f'{key}|{props.get("ANN_CD")}|{n_coord}')
    contract["L2_label_sha256"] = hashlib.sha256(
        "\n".join(sorted(label_lines)).encode()).hexdigest()
    contract["L2_label_features"] = len(label_lines)

    # ---- L3 코드 ----
    code = {}
    for name in CODE_FILES:
        p = CODE / name
        if p.exists():
            code[name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    contract["L3_code"] = code
    try:
        contract["L3_git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(CODE), stderr=subprocess.DEVNULL,
            text=True).strip()
    except Exception:  # noqa: BLE001
        contract["L3_git_commit"] = None  # 서버 code/ 는 git 저장소가 아니다

    # ---- 최상위 봉인 해시 ----
    body = json.dumps({k: v for k, v in contract.items() if k != "seal"},
                      ensure_ascii=False, sort_keys=True)
    contract["seal_sha256"] = hashlib.sha256(body.encode()).hexdigest()

    (OUT / "data_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "L0_zips": len(contract["L0_source_zips"]),
        "L0_total_bytes": sum(z["bytes"] for z in contract["L0_source_zips"]),
        "L1_files": sorted(derived),
        "L2_geometry_rows": contract["L2_geometry_rows"],
        "L2_geometry_sha256": contract["L2_geometry_sha256"][:16],
        "L2_label_features": contract["L2_label_features"],
        "L2_label_sha256": contract["L2_label_sha256"][:16],
        "L3_code": sorted(code),
        "seal_sha256": contract["seal_sha256"],
    }, ensure_ascii=False, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
