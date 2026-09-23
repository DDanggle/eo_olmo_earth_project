#!/usr/bin/env python3
"""Prepare a blind, small, non-expert SN7 review pack, then validate manual exports.

No GPU, model imports, network or automatic annotation. Existing v0.4 outputs are
read only; new outputs must use an unused directory. Source gold is never used.
"""
import argparse
import hashlib
import itertools
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from sn7_visible_contract_v05 import validate_episode, derive_target, consensus_target, make_control_pair

SCHEMA = "sn7-visible-pack-v0.5"
SEED = "sn7-visible-v05-20260922"
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rank(value):
    return hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest()


def select_episodes(rows, max_episodes=12):
    """One cutoff per AOI, opposite pair chosen independently of legacy gold."""
    if max_episodes < 2 or max_episodes % 2:
        raise ValueError("max_episodes must be a positive even number >= 2")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["aoi"]].append(row)
    selected, files = [], []
    for aoi in sorted(grouped, key=rank)[:max_episodes // 2]:
        # Source question, region, answer, and privileged choices are ignored.
        row = min(grouped[aoi], key=lambda item: (item["cutoff"], item["id"]))
        months = row["conds"]["full_prefix"]
        if not months or len(months) < 2 or months[-1] != row["cutoff"]:
            raise ValueError("Source cutoff must equal the last full-prefix observation")
        quads = ("NW", "SE") if int(rank(aoi), 16) % 2 == 0 else ("NE", "SW")
        for quadrant in quads:
            eid = f"region_{len(selected)+1:03d}"
            frames = []
            for index, month in enumerate(months):
                path = f"frames/{eid}/{month}.png"
                source = row["png"][month][quadrant]
                frames.append({"id": f"F{index:03d}", "date": month, "path": path})
                files.append({"source": source, "destination": path, "episode_id": eid})
            episode = {"id": eid, "aoi": aoi, "region": quadrant,
                       "cutoff": row["cutoff"], "reference_id": "F000", "frames": frames}
            validate_episode(episode)
            selected.append(episode)
    if not selected:
        raise ValueError("No episodes in source")
    # Separate paired quadrants in review order; do not tell reviewers their
    # counterpart outcome. One cutoff per AOI still prevents future-prefix reuse.
    return selected[::2] + selected[1::2], files


def make_public_pack(episodes, hashes):
    whitelist = ("id", "aoi", "region", "cutoff", "reference_id", "frames")
    public = [{key: ep[key] for key in whitelist} for ep in episodes]
    for ep in public:
        ep["frames"] = [{key: frame[key] for key in ("id", "date", "path")} for frame in ep["frames"]]
    for ep in public:
        validate_episode(ep)
    payload = {"schema": SCHEMA, "split": "exposed_development", "episodes": public,
               "image_sha256": dict(sorted(hashes.items()))}
    payload["pack_id"] = "sn7v05-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return payload


def safe_image(pack_dir, relative):
    destination = (pack_dir / relative).resolve()
    if not destination.is_relative_to(pack_dir.resolve()):
        raise ValueError("Image escapes pack")
    return destination


def build(args):
    source = Path(args.source_items).resolve()
    out = Path(args.output).resolve()
    episodes, copies = select_episodes(read_jsonl(source), args.max_episodes)
    for entry in copies:
        if not Path(entry["source"]).is_file():
            raise FileNotFoundError(entry["source"])
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    pack_dir = out / "annotator_pack"
    pack_dir.mkdir(parents=True)
    hashes = {}
    for entry in copies:
        dst = safe_image(pack_dir, entry["destination"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry["source"], dst)  # byte-identical original crop
        hashes[entry["destination"]] = digest(dst)
    payload = make_public_pack(episodes, hashes)
    (pack_dir / "pack.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    template = Path(__file__).with_name("sn7_visible_review_v05.html").read_text()
    if template.count("__PACK_JSON__") != 1:
        raise ValueError("Review template placeholder invalid")
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    (pack_dir / "index.html").write_text(template.replace("__PACK_JSON__", encoded))
    manifest = {"schema": "sn7-visible-build-provenance-v0.5", "source": str(source),
                "source_sha256": digest(source), "seed": SEED,
                "selection_uses_legacy_gold": False, "episode_count": len(episodes),
                "aoi_count": len({ep["aoi"] for ep in episodes}), "frame_count": len(copies),
                "pack_id": payload["pack_id"], "source_files": copies,
                "source_code_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                    ("sn7_visible_pack_v05.py", "sn7_visible_contract_v05.py", "sn7_visible_review_v05.html")}}
    (out / "build_provenance.private.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ("episode_count", "aoi_count", "frame_count", "pack_id")}, indent=2))


def load_pack(path):
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError("Wrong pack schema")
    for ep in payload["episodes"]:
        validate_episode(ep)
    expected = make_public_pack(payload["episodes"], payload["image_sha256"])
    if expected != payload:
        raise ValueError("Pack content/identifier mismatch")
    ids = [ep["id"] for ep in payload["episodes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate episode ID")
    referenced = {f["path"] for ep in payload["episodes"] for f in ep["frames"]}
    if referenced != set(payload["image_sha256"]):
        raise ValueError("Image hash coverage mismatch")
    for relative, expected_hash in payload["image_sha256"].items():
        if digest(safe_image(path.parent, relative)) != expected_hash:
            raise ValueError(f"Image changed: {relative}")
    return payload


def prompt(ep):
    listed = ", ".join(f"{f['id']}={f['date']}" for f in ep["frames"])
    return (
        "These observations show the same fixed region. "
        f"The reference is {ep['reference_id']}. Observations in chronological order: {listed}. "
        "Compare each observation with the reference. Is new structural construction clearly visible? "
        "Identify the first PROVIDED observation that clearly shows this change, not the physical onset date, "
        "not a building-count threshold. If the reference is unreadable or ambiguous, answer "
        "insufficient_evidence even if a later image appears changed; current_state is unreadable when "
        "the latest image cannot be read, otherwise ambiguous because comparison lacks a usable reference. "
        "If no clear changed observation exists and any observation is unreadable or ambiguous, "
        "answer insufficient_evidence. "
        "If all observations are readable and show no clear structural change, answer no_visible_change. "
        "If the images are missing or blank, do not infer change from dates; answer insufficient_evidence "
        "with null first/last IDs, empty evidence_ids and unreadable current_state. "
        "If change is clear in a prior observation but the latest one is obscured, preserve the prior observation "
        "and report the latest state separately. Return JSON only with answer (change_supported, "
        "no_visible_change, insufficient_evidence), first_change_id (an observation ID or null), "
        "last_clear_no_change_id (last readable no-change observation before first change; if none changes, "
        "use the latest readable no-change observation; null if the reference cannot be read), "
        "evidence_ids, and current_state (no_visible_change, visible_change, unreadable, ambiguous)."
    )


def withheld_target(ep):
    """No pixels cannot support a visual claim; keep world-label comparison separate."""
    return {"episode_id": ep["id"], "answer": "insufficient_evidence", "first_change_id": None,
            "first_change_date": None, "last_clear_no_change_id": None, "last_clear_no_change_date": None,
            "evidence_ids": [], "current_state": "unreadable", "label_tier": "input_withheld_control",
            "reference_state": "unreadable", "raw_current_state": "unreadable",
            "evidence_ids_role": "supporting_observations_not_global_or_onset_proof",
            "source": "deterministic absence of image content; not a human scene annotation"}


def review_exports(pack, exports):
    episodes = {ep["id"]: ep for ep in pack["episodes"]}
    by_episode = defaultdict(list)
    seen = set()
    incomplete = []
    for export in exports:
        if export.get("schema") != "sn7-visible-annotations-v0.5" or export.get("pack_id") != pack["pack_id"]:
            raise ValueError("Annotation export comes from another contract/pack")
        who = export.get("annotator_id")
        if not isinstance(who, str) or not who.strip():
            raise ValueError("Missing annotator identity")
        for annotation in export["annotations"]:
            eid = annotation["episode_id"]
            if eid not in episodes or annotation.get("annotator_id") != who:
                raise ValueError("Unknown episode or inconsistent annotator identity")
            key = (eid, who)
            if key in seen:
                raise ValueError("Repeated annotator/episode; cannot count as independent agreement")
            seen.add(key)
            if annotation["status"] == "timeout":
                incomplete.append({"episode_id": eid, "annotator_id": who, "reason": "timeout"})
                continue
            derive_target(episodes[eid], annotation)  # invalid labels fail closed
            by_episode[eid].append(annotation)
    targets, pending = {}, []
    for eid, ep in episodes.items():
        try:
            targets[eid] = consensus_target(ep, by_episode[eid])
        except ValueError as error:
            pending.append({"episode_id": eid, "reason": str(error)})
    pairs = []
    for left, right in itertools.combinations(targets, 2):
        if episodes[left]["aoi"] != episodes[right]["aoi"]:
            continue
        try:
            pairs.append(make_control_pair(episodes[left], targets[left], episodes[right], targets[right]))
        except ValueError:
            pass  # same-outcome or temporally unmatched donors are not counterfactuals
    counts = Counter(t["answer"] for t in targets.values())
    ready = all(counts.get(k, 0) > 0 for k in ("change_supported", "no_visible_change", "insufficient_evidence")) and bool(pairs)
    return {"schema": "sn7-visible-reviewed-v0.5", "pack_id": pack["pack_id"],
            "targets": targets, "pending": pending, "incomplete": incomplete, "control_pairs": pairs,
            "answer_counts": dict(counts), "diagnostic_manifest_ready": ready,
            "model_launch_allowed": False, "scientific_gate_passed": False,
            "note": "Readiness is schema/coverage only; no model inference or G1 success. Resolve a separate evaluation preregistration before launching."}


def finalize(args):
    payload = load_pack(args.pack)
    exports = [json.loads(Path(path).read_text()) for path in args.annotations]
    result = review_exports(payload, exports)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    (out / "review_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    episodes = {ep["id"]: ep for ep in payload["episodes"]}
    matrix = []
    for eid, target in result["targets"].items():
        ep = episodes[eid]
        for mode in ("real", "metadata_only", "blank"):
            matrix.append({"id": f"{eid}|{mode}", "episode_id": eid, "aoi": ep["aoi"],
                           "condition": mode, "input": {"prompt": prompt(ep), "frames": ep["frames"], "image_mode": mode},
                           "target": target if mode == "real" else withheld_target(ep),
                           "real_image_reference_target": target, "allowed_for_model_run": False})
    for pair in result["control_pairs"]:
        for source_key, donor_key in (("source_episode_id", "donor_episode_id"), ("donor_episode_id", "source_episode_id")):
            source, donor = episodes[pair[source_key]], episodes[pair[donor_key]]
            # Dates and Fxxx IDs are aligned. The substituted content has its OWN target.
            if [f["id"] for f in source["frames"]] != [f["id"] for f in donor["frames"]]:
                raise ValueError("Swap prompt/target frame IDs must align; reindex explicitly before export")
            matrix.append({"id": f"{source['id']}|swap|{donor['id']}", "episode_id": source["id"],
                           "aoi": source["aoi"], "condition": "different_outcome_swap", "donor_episode_id": donor["id"],
                           "input": {"prompt": prompt(source), "frames": donor["frames"], "image_mode": "real"},
                           "target": result["targets"][donor["id"]], "allowed_for_model_run": False})
    with (out / "diagnostic_matrix.jsonl").open("w") as handle:
        for row in matrix:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"agreed": len(result["targets"]), "pending": len(result["pending"]),
                      "verified_control_pairs": len(result["control_pairs"]), "diagnostic_manifest_ready": result["diagnostic_manifest_ready"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--source-items", required=True)
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--max-episodes", type=int, default=12)
    build_parser.set_defaults(func=build)
    review_parser = sub.add_parser("finalize")
    review_parser.add_argument("--pack", required=True)
    review_parser.add_argument("--annotations", nargs="+", required=True)
    review_parser.add_argument("--output", required=True)
    review_parser.set_defaults(func=finalize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
