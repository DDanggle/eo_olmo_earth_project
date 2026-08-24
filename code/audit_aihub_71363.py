#!/usr/bin/env python3
"""AI-Hub 71363 (10m Sentinel-2) 수신 후 검증 — 4항목.

docs/AIHUB_RUNBOOK.md에 사전 등록한 4개 확인 항목을 실제 데이터로 판정한다.
설계 가설(9 클래스를 성질이 다른 3 task로 분리)이 성립하는지가 핵심이다.

  V1 촬영시점(YYYYMMDD)이 실재하는가        -> 실패 시 STAC 조회 불가, 12밴드 물질화 불가
  V2 EPSG:32652 좌표가 타일별로 유효한가     -> 같음
  V3 land-cover / 산사태·토석류 / 벌목지가
     같은 타일에서 겹치는가                  -> 실패 시 multi-task 구성 불가
  V4 희소 클래스 양이 head 학습에 충분한가   -> 실패 시 탐지 -> 존재여부 분류로 강등

zip을 전개하지 않고 zipfile로 직접 읽는다 (759 MB + 86 MB).
인코딩은 cp949다 (UTF-8 아님 — 실측 확인).
"""
from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

RAW = Path("/home/work/data/olmoearth/aihub/raw/71363")
OUT = Path("/home/work/data/olmoearth/aihub/audit")
# 희소 클래스 판정 기준을 사전에 고정한다 (L4: 실험 전에 판정 기준을 정한다).
MIN_TILES_FOR_TASK = 100      # 한 task가 성립하려면 최소 이만큼의 타일에 등장해야 한다
MIN_OVERLAP_TILES = 50        # 두 task를 같은 캐시로 묶으려면 최소 이만큼 겹쳐야 한다


def find_zip(pattern: str) -> Path:
    hits = sorted(RAW.rglob(pattern))
    if not hits:
        raise SystemExit(f"zip을 찾지 못했다: {pattern}")
    return hits[0]


def load_json(data: bytes) -> object:
    # 라벨(GeoJSON)은 UTF-8, 메타데이터는 cp949다 (실측). UTF-8을 먼저 시도한다.
    for enc in ("utf-8-sig", "utf-8", "cp949", "latin1"):
        try:
            return json.loads(data.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError("어떤 인코딩으로도 읽지 못했다")


def walk_labels(zip_path: Path, limit: int | None = None) -> tuple[Counter, dict, list]:
    """라벨 JSON을 순회해 (클래스 카운트, 타일->클래스집합, 스키마 샘플)을 만든다."""
    class_counts: Counter = Counter()
    tile_classes: dict[str, set] = defaultdict(set)
    schema_samples: list = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".json")]
        if limit:
            names = names[:limit]
        for i, name in enumerate(names):
            try:
                obj = load_json(zf.read(name))
            except ValueError:
                continue
            if len(schema_samples) < 2:
                schema_samples.append({"file": name, "content": obj})
            # 타일 키는 파일명의 <타일ID>_<날짜> 부분이다.
            stem = Path(name).name.rsplit("_", 1)[0] if "_" in Path(name).name else Path(name).stem
            # 실측 스키마: GeoJSON FeatureCollection.
            #   name = "<타일ID>_<YYYYMMDD>"
            #   features[].properties.ANN_CD (숫자 코드) / ANN_NM (한글 클래스명)
            if not isinstance(obj, dict) or "features" not in obj:
                continue
            key = obj.get("name") or stem
            for feat in obj.get("features") or []:
                props = (feat or {}).get("properties") or {}
                cd, nm = props.get("ANN_CD"), props.get("ANN_NM")
                if cd is None and not nm:
                    continue
                cls = f"{cd}:{nm}" if nm else str(cd)
                class_counts[cls] += 1
                tile_classes[key].add(cls)
    return class_counts, tile_classes, schema_samples


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict = {"schema": "aihub-71363-audit-v1",
                    "gates_preregistered": {
                        "MIN_TILES_FOR_TASK": MIN_TILES_FOR_TASK,
                        "MIN_OVERLAP_TILES": MIN_OVERLAP_TILES}}

    # ---- V1 / V2: 메타데이터 ----
    meta_zip = find_zip("01.메타데이터_03*.zip")
    times, crs, res, sats, coords_ok, tiles = Counter(), Counter(), Counter(), Counter(), 0, set()
    with zipfile.ZipFile(meta_zip) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".json")]
        for name in names:
            for rec in (load_json(zf.read(name)) or []):
                if not isinstance(rec, dict):
                    continue
                times[rec.get("img_time", "")] += 1
                crs[rec.get("img_coordinate", "")] += 1
                res[str(rec.get("img_resolution", ""))] += 1
                sats[rec.get("img_type", "")] += 1
                tiles.add(str(rec.get("img_id", "")).rsplit("_", 1)[0])
                try:
                    x, y = [float(v) for v in str(rec.get("coordinates", "")).split(",")]
                    # EPSG:32652 (UTM 52N) 한반도 범위 개략 검증
                    if 1e5 < x < 1e6 and 3.5e6 < y < 4.6e6:
                        coords_ok += 1
                except (ValueError, TypeError):
                    pass
    result["metadata"] = {
        "file_count": len(names),
        "unique_tiles": len(tiles),
        "unique_acquisition_dates": len([t for t in times if t]),
        "date_range": [min(t for t in times if t), max(t for t in times if t)],
        "crs": dict(crs), "resolution": dict(res), "satellite": dict(sats),
        "coords_within_utm52n_korea": coords_ok,
    }
    result["V1_acquisition_time_present"] = all(
        len(str(t)) == 8 and str(t).isdigit() for t in times if t)
    result["V2_coords_valid"] = coords_ok == len(names)

    # ---- V3 / V4: 라벨 ----
    for split, pattern in (("train", "TL_02.JSON_03*.zip"), ("valid", "VL_02.JSON_03*.zip")):
        zp = find_zip(pattern)
        counts, tile_classes, samples = walk_labels(zp)
        # 타일 단위 클래스 등장 횟수
        tiles_per_class = Counter()
        for cls_set in tile_classes.values():
            for c in cls_set:
                tiles_per_class[c] += 1
        # 클래스 쌍 동시등장
        co = Counter()
        for cls_set in tile_classes.values():
            ordered = sorted(cls_set)
            for i, a in enumerate(ordered):
                for b in ordered[i + 1:]:
                    co[f"{a} + {b}"] += 1
        result[split] = {
            "zip": zp.name,
            "tiles_seen": len(tile_classes),
            "annotation_counts_by_class": dict(counts.most_common()),
            "tiles_per_class": dict(tiles_per_class.most_common()),
            "top_cooccurrence": dict(co.most_common(20)),
            "schema_sample": samples[:1],
        }

    tr = result.get("train", {})
    tpc = tr.get("tiles_per_class", {})
    result["V4_classes_meeting_min_tiles"] = {
        c: n for c, n in tpc.items() if n >= MIN_TILES_FOR_TASK}
    result["V3_pairs_meeting_min_overlap"] = {
        k: v for k, v in tr.get("top_cooccurrence", {}).items() if v >= MIN_OVERLAP_TILES}
    result["verdict"] = {
        "V1": result["V1_acquisition_time_present"],
        "V2": result["V2_coords_valid"],
        "V3": len(result["V3_pairs_meeting_min_overlap"]) > 0,
        "V4": len(result["V4_classes_meeting_min_tiles"]) >= 2,
    }
    (OUT / "aihub_71363_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # 콘솔에는 요약만 낸다 (스키마 샘플은 파일에만).
    slim = {k: v for k, v in result.items() if k not in ("train", "valid")}
    for split in ("train", "valid"):
        if split in result:
            slim[split] = {k: v for k, v in result[split].items() if k != "schema_sample"}
    print(json.dumps(slim, ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
