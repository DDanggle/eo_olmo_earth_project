#!/usr/bin/env python3
"""X1 (EarthDial arm): E8/E8b protocol on the untouched region italy, zero-shot.

Prereg: config/x1_italy_external_prereg_v0.json. Wraps unchanged code/e8_earthdial_protocol_v0.py and
code/e8b_pair_controls_v0.py; only the item file changes (x1_italy_items_v0, PNG renders included).
Arms: real, swap_within_tile, pair_post_post, pair_reversed.

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.../third_party/earthdial_py python -B code/x1_earthdial_italy_v0.py --out x1_italy_earthdial
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e8_earthdial_protocol_v0 as e8  # noqa: E402
import e8b_pair_controls_v0 as e8b  # noqa: E402
from earthtalk_content_controls_v0 import bootstrap, d_stat  # noqa: E402

ARMS = ("real", "swap_within_tile", "pair_post_post", "pair_reversed")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="x1_italy_earthdial")
    a = ap.parse_args()
    e8.ITEMS = e8.ROOT / "x1_italy_items_v0/items.jsonl"
    e8.plan_arms = e8b.plan_arms
    e8.run(argparse.Namespace(selftest=False, probe=False, ckpt="RGB", model_dir=None, earthdial_src=str(e8.EARTHDIAL_SRC),
                              template="earthdial", sensor="hr_temp", arms=",".join(ARMS), out=a.out))
    root = e8.ROOT / a.out
    jl = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    rows = {x: jl(root / f"answers_{x}.jsonl") for x in ARMS}
    scores = {"d_real": {"value": d_stat(rows["real"], "emb"), "ci95": bootstrap(rows["real"], "emb")},
              "d_swap": {"value": d_stat(rows["swap_within_tile"], "emb"), "ci95": bootstrap(rows["swap_within_tile"], "emb")},
              "parse_fail": {k: sum(r["parsed"] is None for r in v) / len(v) for k, v in rows.items()},
              "p_yes_pos": {k: sum(r["parsed"] == "yes" for r in v if r["text_gold"] == "yes") / max(1, sum(r["text_gold"] == "yes" for r in v)) for k, v in rows.items()},
              "p_yes_neg": {k: sum(r["parsed"] == "yes" for r in v if r["text_gold"] == "no") / max(1, sum(r["text_gold"] == "no" for r in v)) for k, v in rows.items()},
              "compare_real_vs_postpost_pos": e8b.paired_diff(e8b.yes_by_tile(rows["real"], "yes"), e8b.yes_by_tile(rows["pair_post_post"], "yes")),
              "compare_real_vs_reversed_pos": e8b.paired_diff(e8b.yes_by_tile(rows["real"], "yes"), e8b.yes_by_tile(rows["pair_reversed"], "yes"))}
    (root / "scores.json").write_text(json.dumps(scores, indent=1))
    print(json.dumps(scores, indent=1))
    print("X1 EARTHDIAL DONE")
