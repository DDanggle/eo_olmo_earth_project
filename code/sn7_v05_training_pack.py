#!/usr/bin/env python3
"""Build a small SN7 v0.5 calibration pack from AOIs that are NOT in the D1 evaluation pack,
then compare two annotators' practice exports frame by frame.

Purpose: stage H annotators learn the tool and align on "clearly visible new structure"
before the registered evaluation. Nothing here is gold, and nothing here is evaluated.

- Same source file, same crops, same review UI as the evaluation pack; only the AOIs differ.
- Refuses to build if any selected AOI appears in the evaluation pack.
- Legacy silver labels were used only to pick varied episodes (see TRAINING_EPISODES);
  they are not written into the public pack and must not be shown to annotators.
- The pack_id prefix is ``sn7v05train-`` so the UI stores practice records under a
  different key from the evaluation pack.

Usage:
  python3 -B code/sn7_v05_training_pack.py build \
      --source-items labeling_pack/_source_diag_lite_v0_4_nontarget/items.jsonl \
      --source-root labeling_pack/_source_diag_lite_v0_4_nontarget \
      --eval-pack labeling_pack/visible_contract_v05_20260922/annotator_pack/pack.json \
      --output labeling_pack/visible_contract_v05_training_20260924
  python3 -B code/sn7_v05_training_pack.py compare \
      --pack labeling_pack/visible_contract_v05_training_20260924/annotator_pack/pack.json \
      --annotations /abs/practice_a.json /abs/practice_b.json
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from sn7_visible_contract_v05 import validate_episode, derive_target
from sn7_visible_pack_v05 import digest, read_jsonl, safe_image

SCHEMA = "sn7-visible-pack-v0.5"          # same schema so the unchanged review UI accepts it
SERVER_PREFIX = "/home/work/data/olmoearth/spacenet7/diag_lite_v0_4"
EXPECTED_SOURCE_SHA256 = "062b2babf4adca507e76d101accbbacdf20d91512108a6909f89caa6b4b820b1"

# (aoi, cutoff, quadrant, why). 13 frames each, 39 frames total.
# Variety was chosen from legacy silver, which is NOT the v0.5 question and is never shown.
TRAINING_EPISODES = [
    ("L15-1389E-1284N_5557_3054_13", "2019-02", "SW", "legacy Q1 silver 'no' (likely little change)"),
    ("L15-1276E-1107N_5105_3761_13", "2019-01", "SW", "legacy Q2 silver early change (2018-03)"),
    ("L15-1669E-1160N_6678_3548_13", "2018-08", "NW", "legacy Q1 silver 'yes' (change late in prefix)"),
]


def build(args):
    source = Path(args.source_items).resolve()
    if digest(source) != EXPECTED_SOURCE_SHA256:
        raise ValueError("source items differ from the file the evaluation pack was built from")
    root = Path(args.source_root).resolve()
    eval_aois = {ep["aoi"] for ep in json.loads(Path(args.eval_pack).read_text())["episodes"]}
    overlap = {aoi for aoi, *_ in TRAINING_EPISODES} & eval_aois
    if overlap:
        raise ValueError(f"training AOIs overlap the evaluation pack: {sorted(overlap)}")
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")

    rows = read_jsonl(source)
    episodes, copies = [], []
    for number, (aoi, cutoff, quadrant, _why) in enumerate(TRAINING_EPISODES, 1):
        row = next(r for r in rows if r["aoi"] == aoi and r["cutoff"] == cutoff)
        months = row["conds"]["full_prefix"]
        if months[-1] != cutoff:
            raise ValueError("Source cutoff must equal the last full-prefix observation")
        eid = f"practice_{number:02d}"
        frames = []
        for index, month in enumerate(months):
            path = f"frames/{eid}/{month}.png"
            src = row["png"][month][quadrant].replace(SERVER_PREFIX, str(root), 1)
            if not Path(src).is_file():
                raise FileNotFoundError(src)
            frames.append({"id": f"F{index:03d}", "date": month, "path": path})
            copies.append({"source": src, "destination": path, "episode_id": eid})
        episode = {"id": eid, "aoi": aoi, "region": quadrant, "cutoff": cutoff,
                   "reference_id": "F000", "frames": frames}
        validate_episode(episode)
        episodes.append(episode)

    pack_dir = out / "annotator_pack"
    pack_dir.mkdir(parents=True)
    hashes = {}
    for entry in copies:
        dst = safe_image(pack_dir, entry["destination"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry["source"], dst)  # byte-identical original crop
        hashes[entry["destination"]] = digest(dst)
    payload = {"schema": SCHEMA, "split": "practice_calibration_not_evaluation",
               "episodes": episodes, "image_sha256": dict(sorted(hashes.items()))}
    payload["pack_id"] = "sn7v05train-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    (pack_dir / "pack.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    template = Path(__file__).with_name("sn7_visible_review_v05.html").read_text()
    if template.count("__PACK_JSON__") != 1:
        raise ValueError("Review template placeholder invalid")
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    (pack_dir / "index.html").write_text(template.replace("__PACK_JSON__", encoded))
    manifest = {"schema": "sn7-visible-training-provenance-v0.5", "source": str(source),
                "source_sha256": EXPECTED_SOURCE_SHA256, "evaluation_aois_excluded": sorted(eval_aois),
                "selection": [{"aoi": a, "cutoff": c, "quadrant": q, "why": w}
                              for a, c, q, w in TRAINING_EPISODES],
                "frame_count": len(copies), "pack_id": payload["pack_id"],
                "source_code_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                    ("sn7_v05_training_pack.py", "sn7_visible_contract_v05.py",
                     "sn7_visible_review_v05.html")}}
    (out / "build_provenance.private.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("pack_id", "frame_count")}, indent=2))


def compare(args):
    """Frame-by-frame side-by-side of two practice exports, for the calibration talk."""
    pack = json.loads(Path(args.pack).read_text())
    if not pack["pack_id"].startswith("sn7v05train-"):
        raise ValueError("compare is for the practice pack only; evaluation exports go to finalize")
    exports = [json.loads(Path(p).read_text()) for p in args.annotations]
    if len(exports) != 2 or any(e.get("pack_id") != pack["pack_id"] for e in exports):
        raise ValueError("need exactly two exports from this practice pack")
    names = [e["annotator_id"] for e in exports]
    by = [{a["episode_id"]: a for a in e["annotations"]} for e in exports]
    lines = [f"# 연습 판독 비교 — {names[0]} vs {names[1]}", ""]
    for ep in pack["episodes"]:
        rows = [b.get(ep["id"]) for b in by]
        lines.append(f"## {ep['id']}")
        answers = []
        for who, row in zip(names, rows):
            if not row or row.get("status") != "complete":
                answers.append(f"{who}: 미완료")
                continue
            t = derive_target(ep, row)
            answers.append(f"{who}: {t['answer']} / 첫 변화 {t.get('first_change_id')}")
        lines += ["- " + " · ".join(answers), "", "| 관측 | 날짜 | " + " | ".join(names) + " | |",
                  "|---|---|---|---|---|"]
        for frame in ep["frames"]:
            states = [(r or {}).get("states", {}).get(frame["id"], "-") for r in rows]
            mark = "" if states[0] == states[1] else "← 다름"
            lines.append(f"| {frame['id']} | {frame['date']} | {states[0]} | {states[1]} | {mark} |")
        lines.append("")
    text = "\n".join(lines)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


def render_ui(args):
    """Write the split-layout review page into an existing pack dir (practice or evaluation).

    Only the page changes: pack.json, images, pack_id, labels and export format stay the same.
    The first page found is kept as index.v05_original.html so the build-time UI is preserved.
    """
    pack_dir = Path(args.pack_dir).resolve()
    payload = json.loads((pack_dir / "pack.json").read_text())
    template = Path(__file__).with_name("sn7_visible_review_v05_split.html").read_text()
    if template.count("__PACK_JSON__") != 1:
        raise ValueError("Review template placeholder invalid")
    original = pack_dir / "index.v05_original.html"
    if not original.exists():
        shutil.copyfile(pack_dir / "index.html", original)
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    (pack_dir / "index.html").write_text(template.replace("__PACK_JSON__", encoded))
    print(json.dumps({"pack_id": payload["pack_id"], "index_sha256": digest(pack_dir / "index.html"),
                      "original_kept": str(original.name)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source-items", required=True)
    b.add_argument("--source-root", required=True)
    b.add_argument("--eval-pack", required=True)
    b.add_argument("--output", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--pack", required=True)
    c.add_argument("--annotations", nargs=2, required=True)
    c.add_argument("--output")
    r = sub.add_parser("render-ui")
    r.add_argument("--pack-dir", required=True)
    args = parser.parse_args()
    {"build": build, "compare": compare, "render-ui": render_ui}[args.cmd](args)


if __name__ == "__main__":
    main()
