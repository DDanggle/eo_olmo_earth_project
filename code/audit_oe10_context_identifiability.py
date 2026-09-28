"""Reproduce the context/region audit from recovered metadata only.

No image, label array, server, model, or held-out-region payload is opened.
This reports potential confounds, not measured model reliance or performance.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

DEFAULTS = {
    "prepared": "artifacts/oe8_pastis_prepare_20260927/review_bundle_v0/manifest.jsonl",
    "episodes": "artifacts/oe10_expansion_catalog_20260928/oe10_expansion_catalog_v0/runtime_v2/episodes_train.jsonl",
    "contexts": "artifacts/oe10_text_mask_prepare_20260928/contexts_v0/contexts.jsonl",
}
CLASSES = {1: "meadow", 2: "winter_wheat", 3: "corn", 4: "winter_barley", 8: "grapevine", 14: "leguminous_fodder"}

def audit(root):
    paths = {k: root / v for k, v in DEFAULTS.items()}
    data = {k: [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            for k, p in paths.items()}
    prepared = data["prepared"]
    train = [r for r in prepared if r["role"] == "train" and r["training_partition"] == "train_pool"]
    by_id = {str(r["patch_id"]): r for r in train}
    episodes = data["episodes"]
    contexts = data["contexts"]
    if len(by_id) != len(train):
        raise ValueError("Duplicate train patch")
    if any(e["split"] != "train" or e["query_patch_id"] not in by_id for e in episodes):
        raise ValueError("Non-train or unmatched episode")
    if {e["query_patch_id"] for e in episodes} != set(by_id):
        raise ValueError("Episode queries differ from train_pool")
    if len({e["episode_id"] for e in episodes}) != len(episodes):
        raise ValueError("Duplicate episode")
    by_episode = {r["episode_id"]: r for r in contexts}
    if len(by_episode) != len(contexts) or set(by_episode) != {r["episode_id"] for r in episodes}:
        raise ValueError("Contexts/episodes mismatch")
    module_path = root / "code/oe10_text_mask_v2/contracts.py"
    spec = importlib.util.spec_from_file_location("oe10_context_contracts", module_path)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    strings = defaultdict(set)
    for row in contexts:
        for condition, value in row["model_context_by_condition"].items():
            strings[condition].add(contract.render_context(value))
    dates_to_parent = defaultdict(set)
    dates_by_parent = defaultdict(set)
    for e in episodes:
        obs = {o["observation_id"]: o["date_yyyymmdd"] for o in e["query_observations"]}
        dates = tuple(obs[o] for o in e["initial_observation_ids"])
        dates_to_parent[dates].add(e["query_parent_tile"])
        dates_by_parent[e["query_parent_tile"]].add(dates)
        if list(dates) != by_episode[e["episode_id"]]["audit_only"]["acquired_dates"]:
            raise ValueError("Context audit dates mismatch")
    present = Counter()
    for r in train:
        a, b = r["class_pixel_counts"][8] > 0, r["class_pixel_counts"][14] > 0
        present["both" if a and b else "grapevine_only" if a else "fodder_only" if b else "neither"] += 1
    target_presence = Counter()
    source_parent_k1 = {role: Counter() for role in ("positive", "counterexample")}
    for e in episodes:
        c = by_episode[e["episode_id"]]["audit_only"]["support_class_ids_audit_only"]["positive"]
        target_presence["present" if by_id[e["query_patch_id"]]["class_pixel_counts"][c] > 0 else "absent"] += 1
        if e["k_pairs"] == 1:
            for role in source_parent_k1:
                source_parent_k1[role][e["support_pairs"][0][role]["parent_tile"]] += 1
    source_paths = {**paths, "context_renderer": module_path, "audit_script": Path(__file__).resolve()}
    return {
        "schema": "oe10_context_identifiability_metadata_audit_v1",
        "scope": "Already recovered metadata only; no NPZ, server, model, or unopened evaluation payload.",
        "inputs": {k: {"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in source_paths.items()},
        "prepared_role_counts": dict(Counter(r["role"] for r in prepared)),
        "prepared_partition_counts": dict(Counter(r["training_partition"] for r in prepared)),
        "train_parent_counts": dict(sorted(Counter(r["parent_tile"] for r in train).items())),
        "train_class_present_patch_counts": {
            parent: {name: sum(r["class_pixel_counts"][cid] > 0 for r in train if r["parent_tile"] == parent)
                     for cid, name in CLASSES.items()} for parent in sorted(dates_by_parent)},
        "expansion_episode_rows": len(episodes),
        "unique_query_patches": len({e["query_patch_id"] for e in episodes}),
        "k_counts": dict(sorted(Counter(str(e["k_pairs"]) for e in episodes).items())),
        "distinct_rendered_strings_by_condition": {k: len(v) for k,v in sorted(strings.items())},
        "distinct_rendered_strings_total": len(set().union(*strings.values())),
        "query_dates_by_parent": {k: [list(x) for x in sorted(v)] for k,v in sorted(dates_by_parent.items())},
        "date_pair_identifies_source_parent_in_this_pack": all(len(v) == 1 for v in dates_to_parent.values()),
        "query_grapevine_fodder_presence": dict(sorted(present.items())),
        "directed_target_presence_over_all_k": dict(sorted(target_presence.items())),
        "support_parent_k1": {role: dict(sorted(v.items())) for role,v in source_parent_k1.items()},
        "interpretation": [
            "Seven deterministic frozen-text vectors may be cached; this is not proof that language semantics are useless.",
            "Class, parent, and date schedules are confounded; model reliance has not been measured.",
            "Repeated K/role rows are not independent imagery or independent regions.",
            "Annual masks do not label temporal state, observed evidence, or human correction validity."
        ]
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ["expansion_episode_rows", "unique_query_patches",
        "distinct_rendered_strings_total", "date_pair_identifies_source_parent_in_this_pack",
        "query_grapevine_fodder_presence", "support_parent_k1"]}, ensure_ascii=False))

if __name__ == "__main__":
    main()
