#!/usr/bin/env python3
"""Freeze 20 train-only metadata candidates. Does not fetch images or run models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

SALT = "oe2-train-metadata-candidates-v0|20260926|"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(ok, text):
    if not ok:
        raise ValueError(text)


def build(repo, output):
    repo, output = Path(repo).resolve(strict=True), Path(output)
    require(not output.exists(), "Refuse to overwrite candidate manifest")
    base = repo / "artifacts/oe1_bentxt_v0_20260926"
    data_dir = base / "data"
    sources = {
        "prepared_manifest": data_dir / "manifest.json",
        "train_questions": data_dir / "train.jsonl",
        "oe1_run_config": base / "pilot_v1/training/run_config.json",
        "data_source_contract": repo / "code/oe1_bentxt_v0/data_source_contract.json",
        "candidate_builder": Path(__file__).resolve(),
    }
    source_pins = {key: {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
                   for key, path in sources.items()}
    manifest = json.loads(sources["prepared_manifest"].read_text())
    config = json.loads(sources["oe1_run_config"].read_text())
    questions = [json.loads(line) for line in sources["train_questions"].read_text().splitlines() if line.strip()]
    require(manifest["status"] == "complete" and manifest["seed"] == 20260926, "Unexpected source preparation")
    require(manifest["splits"]["train"]["patches"] == 512 and len(questions) == 1024, "Unexpected train source population")
    require(source_pins["train_questions"]["sha256"] == manifest["splits"]["train"]["question_file_sha256"], "Training question SHA differs")
    require(source_pins["prepared_manifest"]["sha256"] == config["data_manifest_sha256"], "Prepared manifest SHA differs from run receipt")
    all_patches = {row["patch_id"]: row for row in manifest["patches"]}
    require(len(all_patches) == len(manifest["patches"]), "Duplicate source patch IDs")
    train = [row for row in manifest["patches"] if row["split"] == "train"]
    require(len(train) == 512 and all(row["official_split"] == "train" for row in train), "Only official train patches are eligible")
    by_patch = {}
    require(len({row["id"] for row in questions}) == len(questions), "Duplicate question IDs")
    for row in questions:
        require(row["patch_id"] in all_patches and row["split"] == row["official_split"] == "train", "Question is outside train")
        patch = all_patches[row["patch_id"]]
        require(row["date"] == patch["timestamp_utc"] and row["mgrs"] == patch["mgrs"]
                and row["bands"] == patch["bands"] and row["image_path"] == patch["image_path"], "Image/question source metadata mismatch")
        by_patch.setdefault(row["patch_id"], []).append(row)
    require(set(by_patch) == {row["patch_id"] for row in train}, "Missing training question coverage")
    require(all(sorted(q["output"] for q in rows) == ["no", "yes"] for rows in by_patch.values()), "Expected original yes/no pair per patch")
    rank = lambda patch: hashlib.sha256((SALT + patch["patch_id"]).encode()).hexdigest()
    chosen, used_mgrs = [], set()
    for patch in sorted(train, key=lambda row: (rank(row), row["patch_id"])):
        if patch["mgrs"] in used_mgrs:
            continue
        chosen.append(patch); used_mgrs.add(patch["mgrs"])
        if len(chosen) == 20:
            break
    require(len(chosen) == 20, "Fewer than twenty train MGRS groups available")
    candidates = []
    for patch in chosen:
        relative = Path(patch["image_path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe source image reference")
        local_path = (data_dir / relative).resolve()
        require(local_path.is_relative_to(data_dir.resolve()), "Image reference outside data directory")
        server_reference = str(Path(config["data_dir"]) / relative)
        expected_file_sha = config["data_files_sha256"][server_reference]
        require(re.fullmatch("[0-9a-f]{64}", expected_file_sha) is not None, "Missing expected source image hash")
        available = local_path.is_file()
        if available:
            require(sha(local_path) == expected_file_sha, "Local raw image does not match source receipt")
        candidates.append({
            "case_id": patch["patch_id"], "selection_sha256": rank(patch),
            "original_patch_metadata": patch,
            "original_questions": by_patch[patch["patch_id"]],
            "raw_image": {"local_path": str(local_path), "local_present": available,
                "server_path_from_historical_receipt": server_reference,
                "expected_npy_file_sha256": expected_file_sha,
                "expected_array_sha256": patch["image_array_sha256"]},
            "source_label_scope": "scene_presence", "dense_mask": None,
            "ready_for_inference": False,
            "readiness_reason": "metadata_only_preparation; model runtime not configured" if available else "raw_image_not_local; model runtime not configured",
        })
    for key, source in sources.items():
        require(sha(source) == source_pins[key]["sha256"], "Source changed during candidate preparation: " + key)
    result = {
        "schema": "oe2-train-candidate-manifest-v0", "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "metadata_only", "ready_for_inference": False,
        "sampling_rule": {"eligible_source": "OE1 original official-train 512 patches only",
            "rank": "lexicographic sha256(UTF8(salt+patch_id)), patch_id tie-break",
            "salt": SALT, "selection": "first ranked patch per previously unselected MGRS group; stop at 20",
            "model_predictions_used": False, "labels_used_for_ranking": False},
        "counts": {"candidates": 20, "original_questions": 40, "mgrs_groups": len(used_mgrs),
            "locally_present_raw_images": sum(row["raw_image"]["local_present"] for row in candidates)},
        "source_files": source_pins, "candidates": candidates,
        "limitations": ["Candidate metadata preparation only: no image transfer, inference, training, or scoring was performed.",
            "All candidates were part of OE1 training; they are not an unseen evaluation population.",
            "Source class-presence annotations are scene labels, not dense masks, localization, changed area, or expert pixel validation.",
            "Use original_patch_metadata outer-index 10 m geotransform; nested TIFF 60 m transform is known inconsistent.",
            "Server paths and expected hashes are historical receipts; current server availability was not rechecked."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    return {"output": str(output.resolve()), "sha256": sha(output), "counts": result["counts"], "ready_for_inference": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
