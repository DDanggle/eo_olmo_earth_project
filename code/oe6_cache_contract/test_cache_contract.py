"""Unit tests for provenance and gradient guards, not actual Qwen parity."""

import copy
from dataclasses import replace
import unittest

from cache_contract import Entry, Request, Stage, Training, check_reuse


def parts():
    return dict(
        raw=dict(content_hashes=["sha:a", "sha:b"], crop_crs="box:EPSG32652",
                 band_order=["B02", "B03"], calibration="L2A-v1",
                 normalization="official-v1", validity="mask-sha",
                 preprocessor="prep-sha", augmentation="identity", dtype="f32"),
        eo=dict(encoder_weights="encoder-sha", encoder_config="cfg-sha",
                input_window=[dict(source_id="a", acquired_at="2026-01-01",
                                   sensor="S2"),
                              dict(source_id="b", acquired_at="2026-02-01",
                                   sensor="S2")],
                attention_mask="bidirectional-valid", mode="eval",
                dtype="f32", runtime="torch-pinned"),
        projection=dict(adapter_weights="adapter-sha", resampler_weights="res-sha",
                        codebook_version="continuous-no-codebook",
                        codebook_weights="not-applicable", eo_embedding_weights="emb-sha",
                        projector_config="cfg-sha", selector_revision="selector-sha",
                        selector_mode="static", query_identity=None,
                        dtype="f32", runtime="torch-pinned"),
        kv=dict(reader_weights="qwen-sha", reader_config="cfg-sha",
                causal_prefix_identity="immutable-prefix-sha", position_ids=[0, 1],
                rope_config="mrope-pinned", attention_mask="causal-mask-sha",
                multimodal_layout="rgb-eo-layout-sha", dtype="f32",
                cache_format="runtime-kv-v1", runtime="torch-pinned"))


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.parts = parts()
        self.request = Request.build(**self.parts)
        self.entries = {s: Entry.capture(self.request, s) for s in Stage}

    def test_all_dependency_fields_invalidate_only_their_stage_and_descendants(self):
        # Every declared identity has an observable dependency edge.
        for index, (part_name, values) in enumerate(self.parts.items()):
            for key in values:
                if key in {"input_window", "selector_mode", "query_identity", "mode"}:
                    continue  # Structural cases below exercise their valid forms.
                with self.subTest(part=part_name, key=key):
                    changed = copy.deepcopy(self.parts)
                    value = changed[part_name][key]
                    changed[part_name][key] = value + ["different"] if isinstance(
                        value, list) else str(value) + ":different"
                    request = Request.build(**changed)
                    for stage in Stage:
                        decision = check_reuse(self.entries[stage], request)
                        self.assertEqual(decision.reusable, int(stage) < index)
                        if not decision.reusable:
                            self.assertEqual(decision.first_changed_stage,
                                             Stage(index).name)

    def test_unchanged_requests_and_suffix_reuse_immutable_prefix(self):
        request = Request.build(**self.parts, query_suffix="new question")
        for entry in self.entries.values():
            self.assertTrue(check_reuse(entry, request).reusable)

    def test_temporal_order_membership_and_timestamp_each_invalidate(self):
        variants = []
        for mutation in ("reverse", "append", "time", "sensor"):
            changed = copy.deepcopy(self.parts)
            window = changed["eo"]["input_window"]
            if mutation == "reverse":
                window.reverse()
            elif mutation == "append":
                window.append(dict(source_id="c", acquired_at="2026-03-01",
                                   sensor="S1"))
            elif mutation == "time":
                window[0]["acquired_at"] = "2026-01-02"
            else:
                window[0]["sensor"] = "S1"
            variants.append(Request.build(**changed))
        for request in variants:
            self.assertTrue(check_reuse(self.entries[Stage.RAW], request).reusable)
            for stage in (Stage.EO, Stage.PROJECTION, Stage.KV):
                self.assertFalse(check_reuse(self.entries[stage], request).reusable)

    def test_query_conditioned_selector_and_prefix_correction(self):
        p = copy.deepcopy(self.parts)
        p["projection"].update(selector_mode="query_conditional", query_identity="q1")
        request1 = Request.build(**p)
        entries = {s: Entry.capture(request1, s) for s in Stage}
        p["projection"]["query_identity"] = "q2"
        request2 = Request.build(**p)
        for stage in Stage:
            self.assertEqual(check_reuse(entries[stage], request2).reusable,
                             stage < Stage.PROJECTION)
        p = copy.deepcopy(self.parts)
        p["kv"]["causal_prefix_identity"] = "correction-before-eo"
        self.assertFalse(check_reuse(self.entries[Stage.KV], Request.build(**p)).reusable)

    def test_mutating_caller_inputs_does_not_mutate_existing_request(self):
        before = self.request.dependencies(Stage.KV)
        self.parts["eo"]["input_window"].reverse()
        self.assertEqual(self.request.dependencies(Stage.KV), before)

    def test_trainable_upstream_stage_rejects_detached_cache(self):
        policies = [(Training(raw_requires_grad=True), 0),
                    (Training(encoder_trainable=True), 1),
                    (Training(projector_trainable=True), 2),
                    (Training(reader_trainable=True), 3)]
        for policy, first in policies:
            for stage in Stage:
                self.assertEqual(check_reuse(self.entries[stage], self.request,
                                            policy).reusable, int(stage) < first)

    def test_frozen_reader_still_requires_input_gradient(self):
        for policy in (Training(encoder_trainable=True),
                       Training(projector_trainable=True)):
            with self.assertRaisesRegex(ValueError, "training gradients"):
                policy.assert_reader_forward(no_grad=True)
            policy.assert_reader_forward(no_grad=False)
        Training().assert_reader_forward(no_grad=True)

    def test_approximation_and_malformed_entries_fail_closed(self):
        entry = self.entries[Stage.KV]
        for bad, reason in [(replace(entry, fidelity="merged"), "approximate_cache_unsupported"),
                            (replace(entry, schema="old"), "schema_mismatch"),
                            (replace(entry, dependencies=()), "invalid_dependency_count"),
                            (replace(entry, stage=99), "invalid_stage")]:
            decision = check_reuse(bad, self.request)
            self.assertFalse(decision.reusable)
            self.assertEqual(decision.reason, reason)

    def test_missing_fields_and_invalid_structures_rejected(self):
        for part_name, values in self.parts.items():
            for key in values:
                p = copy.deepcopy(self.parts)
                del p[part_name][key]
                with self.assertRaises(ValueError):
                    Request.build(**p)
        for key, value in [("input_window", []), ("input_window", [{}])]:
            p = copy.deepcopy(self.parts)
            p["eo"][key] = value
            with self.assertRaises(ValueError):
                Request.build(**p)

    def test_nonfinite_and_empty_identities_rejected(self):
        for part_name, key, value in [("raw", "content_hashes", []),
                                      ("raw", "calibration", None),
                                      ("kv", "reader_weights", ""),
                                      ("projection", "extra_scale", float("nan"))]:
            p = copy.deepcopy(self.parts)
            p[part_name][key] = value
            with self.assertRaises(ValueError):
                Request.build(**p)
        p = copy.deepcopy(self.parts)
        p["eo"]["mode"] = "train"
        with self.assertRaisesRegex(ValueError, "deterministic eval"):
            Request.build(**p)
        for mode, query in [("static", "q"), ("query_conditional", None),
                            ("unknown", None)]:
            p = copy.deepcopy(self.parts)
            p["projection"].update(selector_mode=mode, query_identity=query)
            with self.assertRaises(ValueError):
                Request.build(**p)


if __name__ == "__main__":
    unittest.main()
