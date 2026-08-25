#!/usr/bin/env python3
"""C2-S — AI-Hub 71363 AOI 군집 단위 spatial holdout 구축 및 동결. 네트워크 미사용.

M9에서 공식 split이 사용 불가로 판정됐다 (valid 타일 110/110이 train과 공간 중첩).
여기서 우리 split을 만든다. 규칙은 **결정적**이어야 한다 — 같은 입력이면 같은 분할이 나오고,
사람이 고르는 여지가 없어야 한다. cherry-picking이 들어가면 test가 test가 아니다.

설계
  - 단위는 타일이 아니라 **AOI 군집**이다. 군집은 서로 20.48 km(타일 두 변) 밖에 있으므로
    중첩도 인접도 불가능하다.
  - 희소 클래스(벌목지·산사태)가 test에 없으면 Task-Logging / Task-Landslide를 평가할 수 없다.
    따라서 클래스 커버리지를 제약으로 건다.
  - 군집이 13개뿐이므로 leave-one-cluster-out(LOCO) 폴드도 같이 만들어 둔다.

결정적 선택 규칙 (사전 등록)
  1. 군집을 (타일 수 내림차순, 군집ID 오름차순)으로 정렬한다.
  2. test 목표 비율 20%를 넘지 않는 선에서, **아직 test에 없는 희소 클래스를 가장 많이
     추가하는 군집**을 고른다. 동점이면 타일 수가 작은 쪽, 그래도 동점이면 군집ID 순.
  3. 두 희소 클래스가 모두 test에 들어오면 중단한다.
  4. 남은 군집에 같은 규칙으로 val 15%를 채운다. 나머지가 train이다.

동결
  - 산출물에 각 split의 타일 목록 SHA-256을 박는다. 이후 이 해시가 바뀌면 split이 바뀐 것이다.
  - 탐색에 이미 쓴 공식 valid 300은 test로 쓰지 않는다.
    주의: S4는 **타일 ID 교집합 0**만 검사한다. 기탐색 타일과의 모든 공간적 근접까지
    0이라는 뜻은 아니다. 게이트 이름을 그 범위에 맞췄다.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

RAW = Path("/home/work/data/olmoearth/aihub/raw/71363")
INV = Path("/home/work/data/olmoearth/aihub/inventory/inventory.jsonl")
OUT = Path("/home/work/data/olmoearth/aihub/splits")
CLUSTER_LINK_M = 20480.0
TEST_FRAC, VAL_FRAC = 0.20, 0.15
RARE = ("70:벌목지", "80:산사태및토석류피해지")
MIN_RARE_TILES = 10   # val/test 각각이 희소 클래스마다 최소 이만큼은 가져야 평가가 성립한다


def read_json(data: bytes) -> object:
    for enc in ("utf-8-sig", "utf-8", "cp949", "latin1"):
        try:
            return json.loads(data.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError("디코딩 실패")


def overlaps(a, b, pad=0.0) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0]
                or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in INV.read_text(encoding="utf-8").splitlines() if l]

    # ---- 타일별 bbox / 관측쌍 ----
    tile_bbox, tile_rows = {}, defaultdict(list)
    for r in rows:
        tile_bbox.setdefault(r["tile_id"], r["utm52n_bbox"])
        tile_rows[r["tile_id"]].append(r)

    # ---- 타일별 클래스 (라벨 zip에서 재계산) ----
    tile_classes = defaultdict(set)
    for pattern in ("TL_02.JSON_03*.zip", "VL_02.JSON_03*.zip"):
        zp = sorted(RAW.rglob(pattern))[0]
        with zipfile.ZipFile(zp) as zf:
            for name in [n for n in zf.namelist() if n.lower().endswith(".json")]:
                obj = read_json(zf.read(name))
                if not isinstance(obj, dict):
                    continue
                key = str(obj.get("name") or Path(name).stem)
                tile_id = key.rpartition("_")[0]
                for feat in obj.get("features") or []:
                    props = (feat or {}).get("properties") or {}
                    cd, nm = props.get("ANN_CD"), props.get("ANN_NM")
                    if cd is not None or nm:
                        tile_classes[tile_id].add(f"{cd}:{nm}" if nm else str(cd))

    # ---- 군집 (union-find) ----
    keys = sorted(tile_bbox)
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if overlaps(tile_bbox[a], tile_bbox[b], pad=CLUSTER_LINK_M):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    members = defaultdict(list)
    for k in keys:
        members[find(k)].append(k)
    # 군집ID를 안정적으로 부여한다: 대표 타일ID 오름차순
    cluster_ids = {root: f"C{idx:02d}" for idx, root in enumerate(sorted(members), start=1)}
    clusters = {cluster_ids[root]: sorted(ms) for root, ms in members.items()}

    def cluster_stats(cid):
        ms = clusters[cid]
        cls = Counter()
        for t in ms:
            for c in tile_classes.get(t, ()):
                cls[c] += 1
        pairs = sum(len(tile_rows[t]) for t in ms)
        dates = {r["date"] for t in ms for r in tile_rows[t]}
        plats = Counter(r["platform"] for t in ms for r in tile_rows[t])
        xs = [tile_bbox[t] for t in ms]
        return {"tiles": len(ms), "pairs": pairs, "dates": len(dates),
                "platforms": dict(plats), "tiles_per_class": dict(cls.most_common()),
                "utm_extent": [min(b[0] for b in xs), min(b[1] for b in xs),
                               max(b[2] for b in xs), max(b[3] for b in xs)]}

    stats = {cid: cluster_stats(cid) for cid in clusters}
    total_tiles = len(keys)

    # ---- 군집 간 최소 이격 (게이트) ----
    # 군집 extent(bounding box)로 재면 안 된다. 군집들이 지리적으로 맞물려 있어 extent는
    # 겹치지만 실제 타일은 떨어져 있을 수 있다. **타일 단위**로 잰다.
    cids = sorted(clusters)
    tile_cluster = {t: cid for cid in cids for t in clusters[cid]}
    min_gap = float("inf")
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if tile_cluster[a] == tile_cluster[b]:
                continue
            ba, bb = tile_bbox[a], tile_bbox[b]
            dx = max(0.0, bb[0] - ba[2], ba[0] - bb[2])
            dy = max(0.0, bb[1] - ba[3], ba[1] - bb[3])
            min_gap = min(min_gap, (dx * dx + dy * dy) ** 0.5)

    # ---- 결정적 선택 ----
    # 첫 시도는 test가 희소 클래스가 있는 작은 군집을 전부 먼저 가져가 val의 산사태가 4타일만
    # 남았다. 산사태 90타일 중 56이 대형 군집 C11·C13에 몰려 있어 val 예산으로는 못 받는다.
    # 따라서 **두 split의 쿼터를 동시에** 채운다. 예산을 특정 군집에 맞춰 바꾸는 것은
    # cherry-picking이므로 하지 않는다.
    #
    #  1. test·val 각각 희소 클래스마다 MIN_RARE_TILES를 쿼터로 둔다.
    #  2. 미충족량이 큰 split이 먼저 고른다 (동점이면 test).
    #  3. 그 split은 예산 안에서 미충족 쿼터를 가장 많이 줄이는 군집을 고른다.
    #     동점이면 타일 수가 작은 쪽, 그래도 동점이면 군집ID.
    #  4. 어느 split도 쿼터를 줄일 수 없으면 중단하고, 남은 예산을 작은 군집부터 채운다.
    #  5. 나머지는 train.
    budgets = {"test": int(round(TEST_FRAC * total_tiles)),
               "val": int(round(VAL_FRAC * total_tiles))}
    chosen = {"test": [], "val": []}
    used = {"test": 0, "val": 0}
    have = {sp: {r: 0 for r in RARE} for sp in ("test", "val")}
    pool = sorted(cids, key=lambda c: (stats[c]["tiles"], c))

    def unmet(sp):
        return sum(max(0, MIN_RARE_TILES - have[sp][r]) for r in RARE)

    def best_for(sp):
        best, best_key = None, None
        for cid in pool:
            if cid in chosen["test"] or cid in chosen["val"]:
                continue
            if used[sp] + stats[cid]["tiles"] > budgets[sp]:
                continue
            gain = sum(min(stats[cid]["tiles_per_class"].get(r, 0),
                           max(0, MIN_RARE_TILES - have[sp][r])) for r in RARE)
            if gain <= 0:
                continue
            key = (-gain, stats[cid]["tiles"], cid)
            if best_key is None or key < best_key:
                best, best_key = cid, key
        return best

    while unmet("test") > 0 or unmet("val") > 0:
        order = sorted(("test", "val"), key=lambda sp: (-unmet(sp), sp != "test"))
        moved = False
        for sp in order:
            if unmet(sp) == 0:
                continue
            cid = best_for(sp)
            if cid is None:
                continue
            chosen[sp].append(cid)
            used[sp] += stats[cid]["tiles"]
            for r in RARE:
                have[sp][r] += stats[cid]["tiles_per_class"].get(r, 0)
            moved = True
            break
        if not moved:
            break

    # 남은 예산 채우기 (작은 군집부터, 결정적)
    for sp in ("test", "val"):
        for cid in pool:
            if cid in chosen["test"] or cid in chosen["val"]:
                continue
            if used[sp] + stats[cid]["tiles"] <= budgets[sp]:
                chosen[sp].append(cid)
                used[sp] += stats[cid]["tiles"]
                for r in RARE:
                    have[sp][r] += stats[cid]["tiles_per_class"].get(r, 0)

    test_c, val_c = sorted(chosen["test"]), sorted(chosen["val"])
    train_c = [c for c in cids if c not in test_c and c not in val_c]
    test_cov = {r for r in RARE if have["test"][r] > 0}
    val_cov = {r for r in RARE if have["val"][r] > 0}

    official_valid_tiles = {r["tile_id"] for r in rows if r["split"] == "valid"}
    assign = {}
    for cid, split in [(c, "test") for c in test_c] + [(c, "val") for c in val_c] \
                      + [(c, "train") for c in train_c]:
        for t in clusters[cid]:
            # 이미 탐색(집계 감사)에 쓴 공식 valid 타일이 우리 test에 들어오면 제외한다.
            # train으로 옮기면 공간 분리가 깨지므로 어느 split에도 넣지 않는다.
            if split == "test" and t in official_valid_tiles:
                assign[t] = {"split": "excluded", "cluster": cid,
                             "reason": "previously_explored_official_valid"}
            else:
                assign[t] = {"split": split, "cluster": cid}

    def digest(split):
        ts = sorted(t for t, v in assign.items() if v["split"] == split)
        return hashlib.sha256("\n".join(ts).encode()).hexdigest(), len(ts)

    splits_meta = {}
    for sp in ("train", "val", "test", "excluded"):
        h, n = digest(sp)
        pairs = sum(len(tile_rows[t]) for t, v in assign.items() if v["split"] == sp)
        cls = Counter()
        for t, v in assign.items():
            if v["split"] == sp:
                for c in tile_classes.get(t, ()):
                    cls[c] += 1
        splits_meta[sp] = {"clusters": sorted({assign[t]["cluster"] for t in assign
                                               if assign[t]["split"] == sp}),
                           "tiles": n, "pairs": pairs, "sha256": h,
                           "tiles_per_class": dict(cls.most_common())}

    # 공식 valid(탐색에 사용)와 우리 test의 겹침 (제외 처리 후이므로 0이어야 한다)
    our_test_tiles = {t for t, v in assign.items() if v["split"] == "test"}
    overlap_with_explored = sorted(official_valid_tiles & our_test_tiles)

    result = {
        "schema": "aihub-71363-spatial-holdout-v1",
        "rules": {"cluster_link_m": CLUSTER_LINK_M, "test_frac": TEST_FRAC,
                  "val_frac": VAL_FRAC, "rare_classes": list(RARE), "min_rare_tiles": MIN_RARE_TILES,
                  "selection": "결정적 그리디: 희소클래스 커버 우선 -> 타일수 오름차순 -> 군집ID"},
        "cluster_count": len(clusters),
        "total_tiles": total_tiles,
        "min_inter_cluster_gap_m": round(min_gap, 1),
        "cluster_stats": stats,
        "splits": splits_meta,
        "test_rare_coverage": sorted(test_cov),
        "val_rare_coverage": sorted(val_cov),
        "overlap_with_previously_explored_official_valid": {
            "count": len(overlap_with_explored), "sample": overlap_with_explored[:10]},
    }
    def rare_counts(sp):
        return {r: splits_meta[sp]["tiles_per_class"].get(r, 0) for r in RARE}

    result["rare_tiles_per_split"] = {sp: rare_counts(sp) for sp in ("train", "val", "test")}
    result["gates"] = {
        "S1_inter_cluster_gap_ge_one_tile": min_gap >= 10240.0,
        "S2_test_covers_both_rare": set(test_cov) >= set(RARE),
        "S3_val_covers_both_rare": set(val_cov) >= set(RARE),
        "S4_test_has_no_explored_valid_tile_ids": len(overlap_with_explored) == 0,
        "S5_all_tiles_assigned": len(assign) == total_tiles,
        "S6_val_test_rare_ge_min": all(
            rare_counts(sp)[r] >= MIN_RARE_TILES for sp in ("val", "test") for r in RARE),
    }
    result["verdict"] = ("holdout 동결 가능" if all(result["gates"].values())
                         else "게이트 미통과 — 규칙 조정 필요")

    (OUT / "spatial_holdout.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "tile_assignment.jsonl").write_text(
        "".join(json.dumps({"tile_id": t, **v}, ensure_ascii=False) + "\n"
                for t, v in sorted(assign.items())), encoding="utf-8")
    # LOCO 폴드 (군집 13개이므로 13폴드)
    (OUT / "loco_folds.json").write_text(json.dumps(
        {"folds": [{"fold": i, "held_out_cluster": c,
                    "held_out_tiles": len(clusters[c]),
                    "held_out_pairs": sum(len(tile_rows[t]) for t in clusters[c])}
                   for i, c in enumerate(cids, start=1)]},
        ensure_ascii=False, indent=2), encoding="utf-8")

    slim = {k: v for k, v in result.items() if k != "cluster_stats"}
    slim["cluster_stats_summary"] = {
        c: {"tiles": s["tiles"], "pairs": s["pairs"], "dates": s["dates"],
            "rare": {r: s["tiles_per_class"].get(r, 0) for r in RARE}}
        for c, s in sorted(stats.items())}
    print(json.dumps(slim, ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
