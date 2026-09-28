"""OE8 input boundary. CPU NPZ reads are full-eight; encoder outputs are acquired only.

No model framework dependency. Do not pass the returned ``audit`` to a model.
``episodes`` returns defensive copies of the canonical public catalog, keyed by ID.
Acquisition budget/selection belongs to the caller; initial positions 2, 5 are mandatory.
This loader does not certify cloud quality or geographic parcel independence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


PROMPT = ("Find regions matching the positive examples and exclude the counterexamples. "
          "Return a mask and the acquired observation dates supporting it.")
PUBLIC_PROMPT = ("Find regions matching the positive examples and exclude the counterexamples. "
                 "Return a mask reference and the observation IDs you used.")
INITIAL = (2, 5)
PACKET_KEYS = {"raw_selected_s2", "normalized_s2", "timestamps", "band_observed",
               "nodata_observed", "observation_valid"}


class ContractError(ValueError):
    """Input provenance, role, acquisition, or array contract was violated."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path, key):
    result = {}
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            require(row[key] not in result, f"duplicate {key}")
            result[row[key]] = row
    return result


def safe_path(root, relative, category):
    relative = Path(relative)
    require(not relative.is_absolute() and relative.parts[0] == category
            and ".." not in relative.parts, "path outside permitted category")
    root = Path(root).resolve()
    resolved = (root / relative).resolve()
    require(resolved.is_relative_to(root / category), "path or symlink escapes root")
    require(resolved.suffix == ".npz", "expected NPZ path")
    return resolved


class EpisodeLoader:
    def __init__(self, prepared_root, episodes_root, split):
        require(split in {"train", "development"}, "invalid split")
        self.prepared_root = Path(prepared_root).resolve()
        self.episodes_root = Path(episodes_root).resolve()
        self.split = split
        contract = json.loads((self.episodes_root / "episode_contract.json").read_text())
        manifest = self.prepared_root / "manifest.jsonl"
        catalog = self.episodes_root / f"episodes_{split}.jsonl"
        require(digest(manifest) == contract["prepared_manifest_sha256"], "manifest hash")
        require(digest(catalog) == contract["public_catalog_sha256"][split], "catalog hash")
        require(contract["initial_candidate_positions"] == list(INITIAL), "initial dates")
        require(contract["all_eight_support_observations_must_be_counted_in_cost_ledger"]
                is True, "support observation policy")
        self._manifest = rows(manifest, "patch_id")
        self._episodes = rows(catalog, "episode_id")
        self._scoring = None  # Never opened by __init__ or load.
        for episode in self._episodes.values():
            self._validate_episode(episode)

    @property
    def episodes(self):
        """ID -> defensive public metadata copy; routing only, never model input."""
        return copy.deepcopy(self._episodes)

    def _episode(self, episode):
        eid = episode if isinstance(episode, str) else episode.get("episode_id")
        require(eid in self._episodes, "episode absent from selected split")
        canonical = self._episodes[eid]
        require(isinstance(episode, str) or episode == canonical,
                "supplied episode differs from pinned public catalog")
        return canonical

    def _validate_episode(self, e):
        qid = e["query_patch_id"]
        require(qid in self._manifest, "unknown query patch")
        q = self._manifest[qid]
        expected_query = "train_pool" if self.split == "train" else "dev_query"
        expected_support = "train_pool" if self.split == "train" else "source_bank"
        require(e["split"] == self.split and q["training_partition"] == expected_query,
                "query partition violation")
        require(q["role"] == ("train" if self.split == "train" else "development"),
                "query role violation")
        train_allowed = self.split == "train"
        require(q["supervised_training_allowed"] is train_allowed
                and e["supervised_training_allowed"] is train_allowed, "training role flag")
        require(e["query_label_supplied"] is False, "query gold supplied")
        require(e["prompt"] == PUBLIC_PROMPT, "prompt must be generic fixed template")
        require(e["query_input_npz"] == q["npz_path"]
                and e["query_input_sha256"] == q["npz_sha256"]
                and e["query_parent_tile"] == q["parent_tile"], "query lineage mismatch")
        safe_path(self.prepared_root, q["npz_path"], "inputs")
        expected_obs = [{"observation_id": f"{qid}:obs:{i}", "date_yyyymmdd": date}
                        for i, date in enumerate(q["selected_dates"])]
        require(len(expected_obs) == 8 and e["query_observations"] == expected_obs,
                "query observation order/date mismatch")
        require(e["initial_observation_ids"] == [f"{qid}:obs:{i}" for i in INITIAL],
                "initial observation mismatch")
        require(e["k_pairs"] in {1, 2, 4, 8}
                and len(e["support_pairs"]) == e["k_pairs"], "support count mismatch")
        objects, per_class_patch = set(), {"positive": {}, "counterexample": {}}
        for pair in e["support_pairs"]:
            require(set(pair) == {"positive", "counterexample"}, "support pair keys")
            for kind, s in pair.items():
                require(s["patch_id"] in self._manifest, "unknown support patch")
                m = self._manifest[s["patch_id"]]
                require(m["training_partition"] == expected_support and m["role"] == "train",
                        "support partition violation")
                require(m["supervised_training_allowed"] is (expected_support == "train_pool"),
                        "support training flag violation")
                require(s["patch_id"] != qid, "support/query same patch")
                require(s["object_key"] not in objects, "duplicate support object")
                objects.add(s["object_key"])
                require(s["object_key"].split(":")[0] == s["patch_id"], "object patch mismatch")
                require(s["input_npz"] == m["npz_path"] and s["input_sha256"] == m["npz_sha256"]
                        and s["parent_tile"] == m["parent_tile"], "support lineage mismatch")
                require(s["dates_yyyymmdd"] == m["selected_dates"]
                        and s["observation_ids"] == [f"{s['patch_id']}:obs:{i}" for i in range(8)],
                        "support observation mismatch")
                require(s["mask_npz"] == f"support_masks/{s['object_key'].replace(':', '_')}.npz",
                        "mask object lineage mismatch")
                safe_path(self.prepared_root, s["input_npz"], "inputs")
                safe_path(self.episodes_root, s["mask_npz"], "support_masks")
                counts = per_class_patch[kind]
                counts[s["patch_id"]] = counts.get(s["patch_id"], 0) + 1
                require(counts[s["patch_id"]] <= 2, "support per-patch cap exceeded")

    def _read_packet(self, patch_id):
        """Full CPU read, explicitly not an encoder-facing interface."""
        m = self._manifest[patch_id]
        path = safe_path(self.prepared_root, m["npz_path"], "inputs")
        require(digest(path) == m["npz_sha256"], "input hash mismatch")
        with np.load(path, allow_pickle=False) as z:
            require(set(z.files) == PACKET_KEYS, "unexpected input keys")
            a = {key: z[key] for key in PACKET_KEYS}
        specs = {"raw_selected_s2": ((8, 10, 128, 128), np.int16),
                 "normalized_s2": ((128, 128, 8, 12), np.float32),
                 "timestamps": ((8, 3), np.int64), "band_observed": ((12,), np.bool_),
                 "nodata_observed": ((8, 10, 128, 128), np.bool_),
                 "observation_valid": ((8, 128, 128), np.bool_)}
        for key, (shape, dtype) in specs.items():
            require(a[key].shape == shape and a[key].dtype == dtype, f"input shape/dtype: {key}")
        require(np.isfinite(a["normalized_s2"]).all(), "nonfinite normalized input")
        require(np.array_equal(a["nodata_observed"], a["raw_selected_s2"] == -10000), "nodata")
        valid = ~a["nodata_observed"].any(axis=1)
        require(np.array_equal(a["observation_valid"], valid), "validity mismatch")
        require(np.all(a["normalized_s2"][~valid.transpose(1, 2, 0)] == 0), "invalid fill")
        require(a["band_observed"].tolist() == [True] * 10 + [False] * 2, "band imputation flags")
        expected_dates = [[int(str(d)[6:8]), int(str(d)[4:6]) - 1, int(str(d)[:4])]
                          for d in m["selected_dates"]]
        require(a["timestamps"].tolist() == expected_dates, "timestamp/date mismatch")
        return a, {"path": str(path), "compressed_bytes": path.stat().st_size,
                   "decoded_array_bytes": sum(x.nbytes for x in a.values()),
                   "cpu_decoded_observations": 8}

    @staticmethod
    def _model_packet(a, positions, dates):
        # Explicit copies are essential: views can expose unacquired frames via .base.
        return {"s2": a["normalized_s2"][:, :, positions, :].copy(),
                "raw_s2": a["raw_selected_s2"][positions].copy(),
                "timestamps": a["timestamps"][positions].copy(),
                "dates_yyyymmdd": np.asarray(dates, dtype=np.int64)[positions].copy(),
                "observation_valid": a["observation_valid"][positions].copy(),
                "band_observed": a["band_observed"].copy()}

    def load(self, episode, acquired_positions=None):
        e = self._episode(episode)
        positions = list(INITIAL) if acquired_positions is None else list(acquired_positions)
        require(all(type(i) is int and 0 <= i < 8 for i in positions), "invalid acquisition index")
        require(len(positions) == len(set(positions)) and set(INITIAL).issubset(positions),
                "acquisition must uniquely include initial positions 2,5")
        positions = sorted(positions)  # Encoder sequence is always chronological.
        packets, ledger = {}, {}

        def packet(pid, selected):
            if pid not in packets:
                packets[pid], ledger[pid] = self._read_packet(pid)
            return self._model_packet(packets[pid], selected, self._manifest[pid]["selected_dates"])

        query = packet(e["query_patch_id"], positions)
        pairs, masks = [], []
        for pair in e["support_pairs"]:
            output_pair = {}
            for kind, s in pair.items():
                value = packet(s["patch_id"], list(range(8)))
                path = safe_path(self.episodes_root, s["mask_npz"], "support_masks")
                require(digest(path) == s["mask_sha256"], "support mask hash")
                with np.load(path, allow_pickle=False) as z:
                    require(z.files == ["mask"], "support mask keys")
                    mask = z["mask"]
                require(mask.shape == (128, 128) and mask.dtype == np.bool_ and mask.any(),
                        "support mask shape/dtype/empty")
                value["mask"] = mask.copy()
                output_pair[kind] = value
                masks.append({"path": str(path), "compressed_bytes": path.stat().st_size,
                              "decoded_array_bytes": mask.nbytes})
            pairs.append(output_pair)
        support_patches = {s["patch_id"] for p in e["support_pairs"] for s in p.values()}
        audit = {"episode_id": e["episode_id"], "split": self.split,
                 "query_patch_id": e["query_patch_id"], "support_patch_ids": sorted(support_patches),
                 "acquired_query_positions": positions,
                 "acquired_query_observation_ids": [f"{e['query_patch_id']}:obs:{i}" for i in positions],
                 "returned_query_observations": len(positions),
                 "returned_support_observation_instances": 2 * e["k_pairs"] * 8,
                 "unique_support_patch_observations": len(support_patches) * 8,
                 "cpu_decoded_observations": 8 * len(packets),
                 "input_file_reads": list(ledger.values()), "support_mask_reads": masks,
                 "npz_bytes_read_for_decode": sum(v["compressed_bytes"] for v in ledger.values())
                     + sum(v["compressed_bytes"] for v in masks),
                 "hash_bytes_read": sum(v["compressed_bytes"] for v in ledger.values())
                     + sum(v["compressed_bytes"] for v in masks),
                 "decoded_array_bytes": sum(v["decoded_array_bytes"] for v in ledger.values())
                     + sum(v["decoded_array_bytes"] for v in masks),
                 "spatial_shape": [128, 128], "downsampling_applied": False,
                 "query_gold_read": False,
                 "cpu_io_note": "Every referenced input NPZ is fully read and decoded at all 8 dates; "
                     "hashing reads the compressed bytes again. Counts describe logical reads, not OS disk I/O. "
                     "Unique input packets reused only inside this load; no cross-call cache."}
        return {"model_input": {"prompt": PROMPT, "query": query, "support_pairs": pairs},
                "audit": audit}

    def training_target(self, episode, *, training=False):
        require(training is True and self.split == "train", "explicit train-only target access required")
        e = self._episode(episode)
        m = self._manifest[e["query_patch_id"]]
        require(m["training_partition"] == "train_pool" and m["supervised_training_allowed"] is True,
                "bank/development supervised training forbidden")
        return self._target(e, "training")

    def evaluation_target(self, episode):
        require(self.split == "development", "evaluation accessor requires development split")
        return self._target(self._episode(episode), "evaluation")

    def _target(self, e, purpose):
        if self._scoring is None:
            self._scoring = rows(self.episodes_root / "scoring" / f"scoring_{self.split}.jsonl", "episode_id")
        s = self._scoring[e["episode_id"]]
        m = self._manifest[e["query_patch_id"]]
        for key in ("episode_id", "base_id", "pair_id", "query_patch_id", "k_pairs"):
            require(s[key] == e[key], f"scoring join mismatch: {key}")
        require(s["query_label_npz"] == m["label_path"]
                and s["expected_query_label_sha256"] == m["label_sha256"]
                and s["query_label_sha256"] == m["label_sha256"], "scoring label lineage")
        require(1 <= s["target_class"] <= 18 and 1 <= s["counter_class"] <= 18
                and s["target_class"] != s["counter_class"], "scoring class range")
        path = safe_path(self.prepared_root, m["label_path"], "labels")
        require(digest(path) == m["label_sha256"], "label hash")
        with np.load(path, allow_pickle=False) as z:
            semantic, valid = z["semantic"], z["label_valid"]
        require(semantic.shape == (128, 128) and semantic.dtype == np.int64
                and valid.shape == (128, 128) and valid.dtype == np.bool_, "label shape/dtype")
        require(np.array_equal(valid, semantic != 19), "label validity")
        target, counter = (semantic == s["target_class"]) & valid, (semantic == s["counter_class"]) & valid
        require(int(target.sum()) == s["target_pixels"] and int(counter.sum()) == s["counter_pixels"]
                and int(valid.sum()) == s["label_valid_pixels"], "scoring pixel counts")
        return {"target_mask": target.copy(), "counter_mask": counter.copy(), "label_valid": valid.copy(),
                "audit": {"purpose": purpose, "episode_id": e["episode_id"], "query_patch_id": e["query_patch_id"],
                          "label_path": str(path), "label_sha256": m["label_sha256"]}}


def main():
    p = argparse.ArgumentParser(description="CPU input/role/acquisition guard audit; no model/GPU")
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--episodes-root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = {"status": "running", "model_executed": False, "splits": {}}
    seen = set()
    for split in ("train", "development"):
        loader = EpisodeLoader(args.prepared_root, args.episodes_root, split)
        # Every query once at K1: all80 patch packets are read across train/dev/support.
        selected = {}
        for e in loader.episodes.values():
            if e["k_pairs"] == 1:
                selected.setdefault(e["query_patch_id"], e)
        for e in selected.values():
            item = loader.load(e)
            require(item["model_input"]["query"]["s2"].shape == (128, 128, 2, 12), "query acquisition leak")
            seen.update(Path(x["path"]).stem for x in item["audit"]["input_file_reads"])
            target = loader.training_target(e, training=True) if split == "train" else loader.evaluation_target(e)
            require(target["target_mask"].shape == (128, 128), "target shape")
        # Read any support-bank patch not selected in K1 to cover all prepared packets.
        for pid in loader._manifest:
            if pid not in seen:
                loader._read_packet(pid)
                seen.add(pid)
        report["splits"][split] = {"metadata_episodes_checked": len(loader.episodes),
                                   "distinct_query_loads_checked": len(selected)}
    report.update(status="passed", prepared_packets_checked=len(seen),
                  limitation="CPU contract checks only; no model numeric parity, GPU, cloud or parcel certification")
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
