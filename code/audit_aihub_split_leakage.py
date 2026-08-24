#!/usr/bin/env python3
"""P0 1.5단계 — AI-Hub 71363 split 누수 감사. 네트워크 미사용.

가장 위험한 실패 모드다. 2,399/300 분할이 단순 랜덤이면 같은 공원·인접 chip·같은
acquisition이 train과 test에 함께 들어가 성능이 부풀려진다. 1024×1024(10.24 km) 타일이면
인접 중첩도 가능하다. 이걸 먼저 재지 않고 head를 학습하면 전부 무효다.

사전 등록 게이트:
  L1 tile_id 누수      train/valid가 같은 tile_id를 공유하는가            -> 0이어야 안전
  L2 날짜 누수         같은 acquisition date가 양쪽에 있는가              -> 있으면 date-block split 필요
  L3 공간 중첩         bbox가 실제로 겹치는 train-valid 쌍이 있는가        -> 0이어야 안전
  L4 근접             중첩은 없어도 buffer 이내로 인접한 쌍이 있는가       -> 많으면 spatial holdout 필요
  L5 AOI 군집         타일이 몇 개의 연결 군집(공원)으로 뭉치는가          -> 군집 단위 split의 기반

L3/L4에 걸리면 우리가 직접 spatial holdout을 만든다. 제공된 split을 쓰지 않는다.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

INV = Path("/home/work/data/olmoearth/aihub/inventory/inventory.jsonl")
OUT = Path("/home/work/data/olmoearth/aihub/inventory")
BUFFER_M = 10240.0   # 타일 한 변. 이 거리 안이면 "인접"으로 본다.
CLUSTER_LINK_M = 20480.0  # 군집 연결 기준: 타일 두 변.


def load() -> list[dict]:
    return [json.loads(line) for line in INV.read_text(encoding="utf-8").splitlines() if line]


def overlaps(a: list, b: list, pad: float = 0.0) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0]
                or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def main() -> None:
    rows = load()
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "valid"]

    # 타일 단위로 축약한다 (같은 tile_id의 여러 날짜는 같은 공간이다).
    def tile_boxes(rs):
        d = {}
        for r in rs:
            d.setdefault(r["tile_id"], r["utm52n_bbox"])
        return d

    tr_tiles, va_tiles = tile_boxes(tr), tile_boxes(va)

    # ---- L1 tile_id 누수 ----
    shared_tiles = sorted(set(tr_tiles) & set(va_tiles))
    # ---- L2 날짜 누수 ----
    tr_dates, va_dates = {r["date"] for r in tr}, {r["date"] for r in va}
    shared_dates = sorted(tr_dates & va_dates)

    # ---- L3/L4 공간 중첩·근접 ----
    overlap_pairs, near_pairs = [], []
    for vt, vb in va_tiles.items():
        for tt, tb in tr_tiles.items():
            if vt == tt:
                continue
            if overlaps(vb, tb):
                overlap_pairs.append((vt, tt))
            elif overlaps(vb, tb, pad=BUFFER_M):
                near_pairs.append((vt, tt))
    va_tiles_overlapping = {p[0] for p in overlap_pairs}
    va_tiles_near = {p[0] for p in near_pairs} - va_tiles_overlapping

    # ---- L5 AOI 군집 (union-find, CLUSTER_LINK_M 이내 연결) ----
    all_tiles = {**tr_tiles, **va_tiles}
    keys = sorted(all_tiles)
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if overlaps(all_tiles[a], all_tiles[b], pad=CLUSTER_LINK_M):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    clusters = defaultdict(list)
    for k in keys:
        clusters[find(k)].append(k)
    cluster_sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    # 각 군집이 train/valid 중 어디에 걸치는지
    mixed_clusters = 0
    for members in clusters.values():
        has_tr = any(m in tr_tiles for m in members)
        has_va = any(m in va_tiles for m in members)
        if has_tr and has_va:
            mixed_clusters += 1

    result = {
        "schema": "aihub-71363-split-leakage-v1",
        "constants": {"buffer_m": BUFFER_M, "cluster_link_m": CLUSTER_LINK_M},
        "counts": {"rows": len(rows), "train_rows": len(tr), "valid_rows": len(va),
                   "train_tiles": len(tr_tiles), "valid_tiles": len(va_tiles)},
        "L1_tile_id_leak": {"shared_tile_count": len(shared_tiles),
                            "sample": shared_tiles[:10]},
        "L2_date_leak": {"train_dates": len(tr_dates), "valid_dates": len(va_dates),
                         "shared_date_count": len(shared_dates), "sample": shared_dates[:10]},
        "L3_spatial_overlap": {"overlapping_pair_count": len(overlap_pairs),
                               "valid_tiles_affected": len(va_tiles_overlapping),
                               "sample": overlap_pairs[:10]},
        "L4_adjacency": {"near_pair_count": len(near_pairs),
                         "valid_tiles_affected": len(va_tiles_near),
                         "sample": near_pairs[:10]},
        "L5_clusters": {"cluster_count": len(clusters),
                        "largest_sizes": cluster_sizes[:10],
                        "clusters_spanning_train_and_valid": mixed_clusters},
    }
    result["gates"] = {
        "L1_no_tile_id_leak": len(shared_tiles) == 0,
        "L2_no_date_leak": len(shared_dates) == 0,
        "L3_no_spatial_overlap": len(overlap_pairs) == 0,
        "L4_no_adjacency": len(near_pairs) == 0,
        "L5_no_mixed_cluster": mixed_clusters == 0,
    }
    result["verdict"] = (
        "제공된 split 사용 가능"
        if all(result["gates"].values())
        else "제공된 split 사용 불가 — 군집 단위 spatial holdout을 직접 만들어야 한다"
    )
    (OUT / "split_leakage_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
