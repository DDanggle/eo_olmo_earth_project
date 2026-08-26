#!/usr/bin/env python3
"""E7a — R-event retrieval: 같은 frozen 캐시가 **다른 task**에서는 다른 순위를 내는가. CPU 전용.

RQ2(task별 action 이질성)의 public twin 최소형이다. 학습 없이 잰다:
train 양성 타일들의 평균 임베딩(prototype)에 대한 코사인 유사도로 test 타일을
순위 매기고 Recall@K / nDCG@K 를 계산한다.

비교 arm (segmentation E1과 같은 축):
  tiled_4x64   기존 serving 계약 캐시
  full_1x128   통짜 인코딩 캐시 (segmentation에서는 이게 **나빴다** — M37)
  raw_spectral 밴드별 시간·공간 평균 10차원 (모델 없는 하한)
  random       기대값 하한

판정: segmentation 순위(tiled > full)와 retrieval 순위가 **다르면** task 이질성의
공개 증거가 된다. 같으면 그 주장은 AI-Hub 3-task 전까지 보류한다(사전 등록 kill gate).

주의: prototype은 **train split에서만** 만든다. test 라벨은 순위 평가에만 쓴다.
"""
from __future__ import annotations
import json, pathlib
import numpy as np

BASE = pathlib.Path("/home/work/data/olmoearth")
CACHES = {
    "tiled_4x64": BASE / "sen12_pilot/holdout_chimanimani",
    "full_1x128": BASE / "sen12_pilot_full128/holdout_chimanimani",
}
CONTRACT = BASE / "sen12_gp_contract/sample_contract.jsonl"
FOLDS = BASE / "sen12_gp_contract/loco_folds.json"
OUT = BASE / "gp_official_bundle/r_event_probe.json"
KS = (50, 100, 200)
SEED = 20260826


def load_split():
    import hashlib
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    fold = next(f for f in folds["folds"] if f["fold"] == "holdout_chimanimani")
    records = {}
    for line in CONTRACT.read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line); records[r["sample_id"]] = r
    out = {}
    for split in ("train", "val", "test"):
        regions = (fold["train_regions"] if split == "train"
                   else [fold["val_region"]] if split == "val" else [fold["test_region"]])
        ids = sorted(s for s, r in records.items()
                     if r["region"] in regions and not r.get("error")
                     and r.get("s15_eligible", True))
        got = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        assert got == fold["sample_sha256"][split], f"{split} 해시 불일치"
        out[split] = ids
    return out, records


def feat_cache(root, sid):
    a = np.load(root / "emb_fp16" / f"{sid}.npy").astype("float32")   # 768,32,32
    return a.mean(axis=(1, 2))


def feat_raw(root, sid):
    a = np.load(root / "raw_u16" / f"{sid}.npy", mmap_mode="r")       # 10,T,H,W
    return np.asarray(a, dtype="float32").mean(axis=(1, 2, 3))


def rank_metrics(scores, labels, ks):
    order = np.argsort(-scores)
    lab = labels[order]
    n_pos = int(labels.sum())
    out = {}
    for k in ks:
        rec = float(lab[:k].sum()) / max(n_pos, 1)
        gains = lab[:k] / np.log2(np.arange(2, k + 2))
        ideal = np.ones(min(k, n_pos)) / np.log2(np.arange(2, min(k, n_pos) + 2))
        out[f"recall@{k}"] = round(rec, 4)
        out[f"ndcg@{k}"] = round(float(gains.sum() / max(ideal.sum(), 1e-9)), 4)
    # 평균 정밀도(전체)
    hits = np.cumsum(lab)
    prec = hits / np.arange(1, len(lab) + 1)
    out["average_precision"] = round(float((prec * lab).sum() / max(n_pos, 1)), 4)
    return out


def main():
    splits, records = load_split()
    y_test = np.array([1 if records[s]["mask_positive_pixels"] > 0 else 0
                       for s in splits["test"]])
    print(f"test {len(y_test)} · 양성 {int(y_test.sum())}", flush=True)
    res = {"schema": "r-event-probe-v1",
           "evidence_status": "development_only_not_confirmatory",
           "task": "R-event: 양성(산사태 존재) 타일 검색",
           "prototype": "train 양성 타일 평균 임베딩 (test 라벨 미사용)",
           "n_test": len(y_test), "n_test_positive": int(y_test.sum()),
           "arms": {}}

    for name, root in CACHES.items():
        tr_pos = [s for s in splits["train"] if records[s]["mask_positive_pixels"] > 0]
        proto = np.mean([feat_cache(root, s) for s in tr_pos], axis=0)
        proto /= np.linalg.norm(proto) + 1e-12
        scores = []
        for s in splits["test"]:
            f = feat_cache(root, s)
            scores.append(float(f @ proto / (np.linalg.norm(f) + 1e-12)))
        res["arms"][name] = rank_metrics(np.array(scores), y_test, KS)
        print(name, res["arms"][name], flush=True)

    # raw spectral 하한 — tiled 캐시 디렉터리의 raw_u16을 쓴다 (동일 파일)
    root = CACHES["tiled_4x64"]
    tr_pos = [s for s in splits["train"] if records[s]["mask_positive_pixels"] > 0]
    proto = np.mean([feat_raw(root, s) for s in tr_pos], axis=0)
    proto /= np.linalg.norm(proto) + 1e-12
    scores = []
    for s in splits["test"]:
        f = feat_raw(root, s)
        scores.append(float(f @ proto / (np.linalg.norm(f) + 1e-12)))
    res["arms"]["raw_spectral_mean"] = rank_metrics(np.array(scores), y_test, KS)
    print("raw_spectral_mean", res["arms"]["raw_spectral_mean"], flush=True)

    rng = np.random.default_rng(SEED)
    rand = [rank_metrics(rng.permutation(len(y_test)).astype("float64"), y_test, KS)
            for _ in range(20)]
    res["arms"]["random_mean_of_20"] = {
        k: round(float(np.mean([r[k] for r in rand])), 4) for k in rand[0]}

    seg = {"tiled_4x64": 0.130582, "full_1x128": 0.116565}
    ap = {k: res["arms"][k]["average_precision"] for k in CACHES}
    res["task_heterogeneity_check"] = {
        "segmentation_iou_small_decoder": seg,
        "retrieval_average_precision": ap,
        "segmentation_winner": max(seg, key=seg.get),
        "retrieval_winner": max(ap, key=ap.get),
        "rank_reversed": max(seg, key=seg.get) != max(ap, key=ap.get),
        "note": "역전이면 같은 캐시 자원에 대해 task별 최적 선택이 다르다는 공개 증거"}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res["task_heterogeneity_check"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
