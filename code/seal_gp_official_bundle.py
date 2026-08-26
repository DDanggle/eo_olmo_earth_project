#!/usr/bin/env python3
"""M30 4-방식 결과를 **독립 재계산 가능한 증거 번들**로 봉인한다 (E0).

문서에 표만 적어두면 그건 기록이지 증거가 아니다. 표의 모든 수치를 per-sample 파일에서
다시 계산할 수 있어야 한다. 체크포인트는 용량 때문에 저장소에 넣지 않고 SHA-256만 봉인한다.
"""
from __future__ import annotations
import hashlib, json, pathlib, shutil

SRC = pathlib.Path("/home/work/data/olmoearth/sen12_gp_official")
LOG = pathlib.Path("/home/work/data/olmoearth/logs/gp_official_full.log")
OUT = pathlib.Path("/home/work/data/olmoearth/gp_official_bundle")
FOLD = "holdout_chimanimani"


def sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_sample").mkdir(exist_ok=True)
    files = {}

    for p in sorted((SRC / "per_sample" / FOLD).glob("*.jsonl")):
        shutil.copy2(p, OUT / "per_sample" / p.name)
        files[f"per_sample/{p.name}"] = {"sha256": sha(p), "bytes": p.stat().st_size}

    for p in (SRC / f"{FOLD}_pilot.json", LOG):
        if p.exists():
            shutil.copy2(p, OUT / p.name)
            files[p.name] = {"sha256": sha(p), "bytes": p.stat().st_size}

    # 체크포인트는 저장소에 넣지 않는다. 해시만 봉인해 나중에 동일성을 증명한다.
    ckpt = {}
    for p in sorted((SRC / "checkpoints" / FOLD).glob("*.pt")):
        ckpt[p.name] = {"sha256": sha(p), "bytes": p.stat().st_size,
                        "server_path": str(p)}

    # per-sample 에서 test 지표를 **다시 계산**해 문서의 표와 대조한다.
    # per-sample 스키마는 평면이다: tp/fp/fn (tn 없음). 처음에 confusion 중첩 키로
    # 추측해서 전부 0이 나왔다. 파일을 읽고 고쳤다.
    recomputed = {}
    group_keys = {"ann_id": set(), "event_date": set(), "region": set()}
    for p in sorted((OUT / "per_sample").glob("*_test.jsonl")):
        arm = p.name.split("_")[0]
        tp = fp = fn = n = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            r = json.loads(line)
            tp += r["tp"]; fp += r["fp"]; fn += r["fn"]
            n += 1
            for k in group_keys:
                group_keys[k].add(r.get(k))
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else None
        recomputed[arm] = {"n_samples": n, "tp": tp, "fp": fp, "fn": fn,
                           "iou_micro_recomputed": round(iou, 6) if iou else None}

    # 부트스트랩 가능성 진단: 묶을 단위가 실제로 몇 개인가.
    # test region이 1개면 region-level CI는 원리상 불가하다 (n=1).
    bootstrap_units = {k: {"n_distinct": len([x for x in v if x not in (None, "")]),
                           "n_blank": len([x for x in v if x in (None, "")])}
                       for k, v in group_keys.items()}

    manifest = {
        "schema": "gp-official-evidence-bundle-v1",
        "fold": FOLD,
        "what_this_proves": "M30 표의 test 지표를 per-sample 파일에서 재계산할 수 있음",
        "what_this_does_not_prove": [
            "체크포인트 자체는 저장소에 없음 (SHA-256만 봉인)",
            "신뢰구간 없음 — 아직 부트스트랩 미실행",
            "test region이 Chimanimani 하나이므로 region-level CI는 원리상 불가 (n=1)",
        ],
        "files": files,
        "checkpoints_sha256_only": ckpt,
        "recomputed_from_per_sample": recomputed,
        "bootstrap_units_available": bootstrap_units,
    }
    (OUT / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
