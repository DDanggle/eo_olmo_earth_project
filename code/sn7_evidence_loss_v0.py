#!/usr/bin/env python3
"""Stage L of the D1 decision experiment (config/decision_experiment_d1_prereg_v0.json).

Measures, WITHOUT any model, whether fixed-budget selectors keep the observations that human
annotators actually needed. Inputs are the v0.5 pack and the agreed targets written by
`sn7_visible_pack_v05.py finalize`. CPU only.

Preservation tiers (registered):
  tierA_change_exists : selection contains last_clear_no_change_id AND first_change_id
  tierB_first_claim   : tierA and every observation from the reference through first_change_id is
                        present, so the selection cannot hide an earlier change
  tierC_human_set     : selection is a superset of the annotator-agreed evidence_ids

Selectors: latest_k, uniform_k, change_topk (mean absolute grey difference to the previous frame,
computed from the same PNGs the annotators saw). All selections are causal: they only ever see
observations up to the episode cutoff, which is the whole frame list of an episode.

Usage
  python3 code/sn7_evidence_loss_v0.py --pack labeling_pack/visible_contract_v05_20260922/annotator_pack \
      --review <finalize_output_dir> --out artifacts/sn7_evidence_loss_v0
  python3 code/sn7_evidence_loss_v0.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BUDGETS = (2, 3, 4, 6)
SELECTORS = ("latest_k", "uniform_k", "change_topk")


# ---------------------------------------------------------------- selectors

def latest_k(frame_ids: list[str], k: int, _scores=None) -> list[str]:
    return list(frame_ids[-k:]) if k < len(frame_ids) else list(frame_ids)


def uniform_k(frame_ids: list[str], k: int, _scores=None) -> list[str]:
    n = len(frame_ids)
    if k >= n:
        return list(frame_ids)
    idx = sorted({int(round(i * (n - 1) / max(k - 1, 1))) for i in range(k)})
    return [frame_ids[i] for i in idx]


def change_topk(frame_ids: list[str], k: int, scores: dict[str, float] | None) -> list[str]:
    """Top-k frames by change score. Frame 0 has no predecessor and scores 0 (as in v0.x)."""
    if k >= len(frame_ids):
        return list(frame_ids)
    scores = scores or {}
    ranked = sorted(frame_ids, key=lambda f: (-scores.get(f, 0.0), frame_ids.index(f)))
    chosen = set(ranked[:k])
    return [f for f in frame_ids if f in chosen]


SELECTOR_FN = {"latest_k": latest_k, "uniform_k": uniform_k, "change_topk": change_topk}


# ---------------------------------------------------------------- tiers

def tier_flags(frame_ids: list[str], selection: list[str], target: dict) -> dict[str, bool | None]:
    """Registered preservation tiers. Returns None for tiers that do not apply to this target."""
    chosen = set(selection)
    first = target.get("first_change_id")
    last_clear = target.get("last_clear_no_change_id")
    evidence = set(target.get("evidence_ids") or [])

    if target.get("answer") != "change_supported" or not first:
        tier_a = tier_b = None
    else:
        tier_a = first in chosen and bool(last_clear) and last_clear in chosen
        span = frame_ids[: frame_ids.index(first) + 1] if first in frame_ids else []
        tier_b = bool(tier_a) and all(f in chosen for f in span)

    tier_c = evidence.issubset(chosen) if evidence else None
    return {"tierA_change_exists": tier_a, "tierB_first_claim": tier_b, "tierC_human_set": tier_c}


def min_k_for_tier_a(frame_ids, scores, target, selector, max_k=None) -> int | None:
    """Smallest budget at which this selector reaches tierA; None if never (up to len(frames))."""
    max_k = max_k or len(frame_ids)
    for k in range(1, max_k + 1):
        sel = SELECTOR_FN[selector](frame_ids, k, scores)
        if tier_flags(frame_ids, sel, target).get("tierA_change_exists"):
            return k
    return None


# ---------------------------------------------------------------- change scores

def change_scores(pack_dir: Path, episode: dict) -> dict[str, float]:
    """Mean absolute difference of the downsampled grey image against the previous observation."""
    import numpy as np
    from PIL import Image

    prev = None
    out: dict[str, float] = {}
    for frame in episode["frames"]:
        path = (pack_dir / frame["path"]).resolve()
        if not str(path).startswith(str(pack_dir.resolve())):
            raise ValueError(f"frame path escapes the pack: {frame['path']}")
        arr = np.asarray(Image.open(path).convert("L").resize((128, 128)), dtype="float32")
        out[frame["id"]] = 0.0 if prev is None else float(np.abs(arr - prev).mean())
        prev = arr
    return out


# ---------------------------------------------------------------- report

def evaluate(pack: dict, pack_dir: Path, targets: dict[str, dict]) -> dict:
    episodes = {ep["id"]: ep for ep in pack["episodes"]}
    per_episode, rows = [], []
    for eid, target in sorted(targets.items()):
        ep = episodes.get(eid)
        if ep is None:
            raise ValueError(f"target references unknown episode {eid}")
        frame_ids = [f["id"] for f in ep["frames"]]
        scores = change_scores(pack_dir, ep)
        entry = {"episode_id": eid, "aoi": ep["aoi"], "n_frames": len(frame_ids),
                 "answer": target.get("answer"), "first_change_id": target.get("first_change_id"),
                 "last_clear_no_change_id": target.get("last_clear_no_change_id"),
                 "n_evidence": len(target.get("evidence_ids") or []), "by_selector": {}}
        for selector in SELECTORS:
            by_k = {}
            for k in BUDGETS:
                sel = SELECTOR_FN[selector](frame_ids, k, scores)
                flags = tier_flags(frame_ids, sel, target)
                by_k[str(k)] = {"selection": sel, **flags}
                rows.append({"episode_id": eid, "selector": selector, "k": k, **flags})
            entry["by_selector"][selector] = {"by_k": by_k,
                                              "min_k_tierA": min_k_for_tier_a(frame_ids, scores, target, selector)}
        per_episode.append(entry)

    summary = {}
    for selector in SELECTORS:
        summary[selector] = {}
        for k in BUDGETS:
            cell = {}
            for tier in ("tierA_change_exists", "tierB_first_claim", "tierC_human_set"):
                vals = [r[tier] for r in rows if r["selector"] == selector and r["k"] == k and r[tier] is not None]
                cell[tier] = {"n": len(vals), "preserved": (sum(vals) / len(vals)) if vals else None}
            summary[selector][str(k)] = cell

    changed = [e for e in per_episode if e["answer"] == "change_supported"]
    best_at_4 = max((summary[s]["4"]["tierA_change_exists"]["preserved"] or 0.0) for s in SELECTORS) if changed else None
    reading = None
    if changed:
        if best_at_4 < 0.70:
            reading = "loss_present"
        elif best_at_4 >= 0.90:
            reading = "no_loss"
        else:
            reading = "inconclusive_between_registered_thresholds"
    return {"schema": "sn7-evidence-loss-v0", "pack_id": pack.get("pack_id"),
            "prereg": "config/decision_experiment_d1_prereg_v0.json", "budgets": list(BUDGETS),
            "n_episodes_scored": len(per_episode), "n_change_supported": len(changed),
            "summary": summary, "best_tierA_at_k4": best_at_4, "registered_reading": reading,
            "per_episode": per_episode,
            "not_a_claim": "budget selections only; no model accuracy, small exposed development sample"}


# ---------------------------------------------------------------- selftest

def selftest() -> None:
    frames = [f"F{i:03d}" for i in range(9)]
    target = {"answer": "change_supported", "first_change_id": "F006",
              "last_clear_no_change_id": "F005", "evidence_ids": ["F005", "F006"]}

    assert latest_k(frames, 4) == frames[-4:]
    assert latest_k(frames, 99) == frames
    assert uniform_k(frames, 3) == ["F000", "F004", "F008"]
    assert uniform_k(frames, 1) == ["F000"]
    scores = {f: 0.0 for f in frames}
    scores.update({"F006": 9.0, "F002": 5.0, "F008": 1.0})
    assert change_topk(frames, 2, scores) == ["F002", "F006"]
    assert change_topk(frames, 3, scores) == ["F002", "F006", "F008"]

    # latest_k=4 keeps F005..F008 -> tierA yes, tierB no (F000..F004 missing), tierC yes
    f = tier_flags(frames, latest_k(frames, 4), target)
    assert f == {"tierA_change_exists": True, "tierB_first_claim": False, "tierC_human_set": True}, f
    # change_topk=2 keeps F002,F006 -> misses the before-frame
    f = tier_flags(frames, change_topk(frames, 2, scores), target)
    assert f["tierA_change_exists"] is False and f["tierC_human_set"] is False
    # full prefix reaches tierB
    f = tier_flags(frames, frames, target)
    assert f["tierA_change_exists"] and f["tierB_first_claim"] and f["tierC_human_set"]
    # non-change targets leave A/B undefined but still check the human set
    f = tier_flags(frames, latest_k(frames, 2), {"answer": "no_visible_change", "evidence_ids": ["F008"]})
    assert f["tierA_change_exists"] is None and f["tierC_human_set"] is True
    # minimum budget that first reaches tierA for latest_k is 4 (needs F005 and F006)
    assert min_k_for_tier_a(frames, scores, target, "latest_k") == 4
    assert min_k_for_tier_a(frames, scores, target, "uniform_k") in (None, 5, 6, 7, 8, 9)
    # a target whose change is at the very first observation cannot reach tierA without a before-frame
    odd = {"answer": "change_supported", "first_change_id": "F000", "last_clear_no_change_id": None,
           "evidence_ids": []}
    assert tier_flags(frames, frames, odd)["tierA_change_exists"] is False
    print("SELFTEST OK (12 checks)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack")
    ap.add_argument("--review", help="directory written by sn7_visible_pack_v05.py finalize")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.pack and a.review and a.out):
        raise SystemExit("--pack, --review and --out are required unless --selftest")
    pack_dir = Path(a.pack)
    pack = json.loads((pack_dir / "pack.json").read_text())
    report = json.loads((Path(a.review) / "review_report.json").read_text())
    if report.get("pack_id") != pack.get("pack_id"):
        raise SystemExit("review report belongs to a different pack")
    targets = report.get("targets") or {}
    if not targets:
        raise SystemExit("no agreed targets yet; stage H must finish first")
    out = evaluate(pack, pack_dir, targets)
    dest = Path(a.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "evidence_loss.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ("n_episodes_scored", "n_change_supported",
                                          "best_tierA_at_k4", "registered_reading")}, indent=1))
    print("EVIDENCE LOSS DONE")


if __name__ == "__main__":
    main()
