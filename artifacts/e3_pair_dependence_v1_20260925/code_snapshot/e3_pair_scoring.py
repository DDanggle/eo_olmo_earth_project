"""E3 source-label agreement scorer; interventions have no transformed-scene gold.

Only NumPy and the standard library are required. Real-arm reproduction, input
finiteness and checkpoint provenance are runner responsibilities. This scorer
validates expected IDs, item metadata, pairing, arm coverage and parse failure.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Any
import numpy as np

ARMS = ("real", "earlier_only", "later_only", "repeat_earlier", "repeat_later", "no_delta", "reverse")
HARD_ARMS = ("real", "later_only", "repeat_later")
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260925
MAX_PARSE_FAIL = 0.01
EPS = 1e-12


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _cluster(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("cluster must be a nonempty string or integer identifier")
    return str(value)


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _bootstrap_delta(event_deltas):
    values = np.asarray(event_deltas, dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    sampled = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(sampled, [0.025, 0.975])]


def _paired_metrics(items, answers):
    groups = defaultdict(list)
    for item in items:
        groups[item["cluster"]].append(item)
    events = {}
    tiles = {}
    for cluster in sorted(groups):
        current = groups[cluster]
        positive = [i for i in current if i["answer"] == "yes"]
        negative = [i for i in current if i["answer"] == "no"]
        pos_correct = sum(answers[i["id"]]["parsed"] == "yes" for i in positive)
        neg_correct = sum(answers[i["id"]]["parsed"] == "no" for i in negative)
        ba = (pos_correct / len(positive) + neg_correct / len(negative)) / 2
        events[cluster] = {"n_items": len(current), "n_tiles": len({i["tile"] for i in current}),
                           "n_yes": len(positive), "n_no": len(negative), "ba": ba,
                           "yes_recall": pos_correct / len(positive),
                           "no_recall": neg_correct / len(negative),
                           "fpr_yes": sum(answers[i["id"]]["parsed"] == "yes" for i in negative) / len(negative),
                           "parse_fail_count": sum(answers[i["id"]]["parsed"] is None for i in current)}
        by_tile = defaultdict(list)
        for item in current:
            by_tile[item["tile"]].append(item)
        for tile, pair in sorted(by_tile.items()):
            tiles[tile] = {"cluster": cluster, "ba": sum(answers[i["id"]]["parsed"] == i["answer"] for i in pair) / len(pair)}
    return {"macro_ba": float(np.mean([e["ba"] for e in events.values()])) if events else None,
            "event_count": len(events), "n_items": len(items), "n_tiles": len(tiles),
            "events": events, "tile_ba": tiles,
            "parse_fail_count": sum(answers[i["id"]]["parsed"] is None for i in items)}


def _hard_metrics(items, answers):
    groups = defaultdict(list)
    for item in items:
        groups[item["cluster"]].append(item)
    events = {}
    for cluster, current in sorted(groups.items()):
        count = sum(answers[i["id"]]["parsed"] == "yes" for i in current)
        events[cluster] = {"n_items": len(current), "false_positive_yes_count": count,
                           "fpr": count / len(current),
                           "parse_fail_count": sum(answers[i["id"]]["parsed"] is None for i in current)}
    positives = sum(e["false_positive_yes_count"] for e in events.values())
    return {"n_items": len(items), "event_count": len(events), "events": events,
            "fpr_pooled": _rate(positives, len(items)),
            "fpr_event_macro": float(np.mean([e["fpr"] for e in events.values()])) if events else None,
            "parse_fail_count": sum(e["parse_fail_count"] for e in events.values()),
            "note": "hard_negative_only; unparsed outputs reported separately and not counted as yes"}


def _behaviour(items, real, intervention):
    common = [i for i in items if real[i["id"]]["parsed"] is not None and intervention[i["id"]]["parsed"] is not None]
    changed = sum(real[i["id"]]["parsed"] != intervention[i["id"]]["parsed"] for i in common)
    real_yes_pos = [i for i in items if i["answer"] == "yes" and real[i["id"]]["parsed"] == "yes"]
    persisted = sum(intervention[i["id"]]["parsed"] == "yes" for i in real_yes_pos)
    transitions = {f"{a}_to_{b}": sum(real[i["id"]]["parsed"] == a and intervention[i["id"]]["parsed"] == b for i in common)
                   for a in ("yes", "no") for b in ("yes", "no")}
    return {"n_items": len(items), "n_both_parsed": len(common),
            "n_excluded_parse": len(items) - len(common),
            "decision_flip": _rate(changed, len(common)),
            "answer_agreement_with_real": _rate(len(common) - changed, len(common)),
            "yes_rate_all_source_pairs": _rate(sum(intervention[i["id"]]["parsed"] == "yes" for i in items), len(items)),
            "n_original_positives_answered_yes_in_real": len(real_yes_pos),
            "yes_persistence_on_real_yes_positives": _rate(persisted, len(real_yes_pos)),
            "persistence_unparsed_count": sum(intervention[i["id"]]["parsed"] is None for i in real_yes_pos),
            "transitions": transitions}


def score_run(rows, expected_items, seeds=(1, 2, 3)):
    """Score complete E3 inference output; invalid data return verdict=invalid.

    Paired pos/neg items require seven arms. Flood hard negatives require only
    real/later_only/repeat_later and never enter the primary paired BA or CI.
    The bootstrap unit is a whole flood event, with arm differences paired.
    """
    errors = []
    result = {"schema": "e3-pair-dependence-scores-v0", "valid": False, "verdict": "invalid",
              "invalid_reasons": errors, "coverage": {}, "metrics": {}, "seed_decisions": {},
              "interpretation": "Source-label agreement under interventions, not transformed-scene accuracy or proof of temporal reasoning.",
              "runner_validity_checks_required": ["real_reproduction", "finite_inputs_and_outputs", "provenance_and_checkpoint_hashes"]}
    if not isinstance(rows, (list, tuple)) or not isinstance(expected_items, (list, tuple)):
        errors.append("rows and expected_items must be lists or tuples")
        return result
    if not seeds or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds) or len(set(seeds)) != len(seeds):
        errors.append("seeds must be distinct integer identifiers")
        return result
    if not expected_items:
        errors.append("expected_items is empty")
        return result
    expected = {}
    pairs = defaultdict(list)
    for index, original in enumerate(expected_items):
        try:
            if not isinstance(original, dict):
                raise ValueError("item must be an object")
            item = dict(original)
            for key in ("id", "tile", "phen", "kind", "answer"):
                _text(item.get(key), key)
            item["cluster"] = _cluster(item.get("cluster"))
            if item["id"] in expected:
                raise ValueError(f"duplicate expected id {item['id']}")
            if item["phen"] not in ("flood", "landslide"):
                raise ValueError("phen must be flood or landslide")
            if item["kind"] not in ("pos", "neg", "hard_neg"):
                raise ValueError("unknown item kind")
            wanted = "yes" if item["kind"] == "pos" else "no"
            if item["answer"] != wanted:
                raise ValueError("item kind/answer mismatch")
            if item["kind"] == "hard_neg" and item["phen"] != "flood":
                raise ValueError("hard_neg is supported for flood only")
            expected[item["id"]] = item
            if item["kind"] != "hard_neg":
                pairs[(item["phen"], item["tile"])].append(item)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"expected item {index}: {exc}")
    for (phen, tile), pair in sorted(pairs.items()):
        if len(pair) != 2 or {p["kind"] for p in pair} != {"pos", "neg"} or len({p["cluster"] for p in pair}) != 1:
            errors.append(f"invalid within-tile yes/no pair: {phen}/{tile}")
    for item in expected.values():
        if item["kind"] == "hard_neg" and (item["phen"], item["tile"]) in pairs:
            errors.append(f"hard negative overlaps a paired positive tile: {item['tile']}")
    if errors:
        return result
    expected_keys = {(seed, arm, item["id"]) for seed in seeds for item in expected.values()
                     for arm in (HARD_ARMS if item["kind"] == "hard_neg" else ARMS)}
    by_key = {}
    duplicates = []
    unexpected = []
    parse_groups = defaultdict(list)
    for index, row in enumerate(rows):
        try:
            if not isinstance(row, dict):
                raise ValueError("row must be an object")
            seed = row.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed not in seeds:
                raise ValueError("unexpected seed")
            arm, sid = row.get("arm"), row.get("id")
            if not isinstance(arm, str) or not isinstance(sid, str):
                raise ValueError("arm/id must be strings")
            key = (seed, arm, sid)
            if key not in expected_keys:
                unexpected.append(key)
                continue
            if key in by_key:
                duplicates.append(key)
                continue
            item = expected[sid]
            for name in ("tile", "phen", "kind"):
                if row.get(name) != item[name]:
                    raise ValueError(f"{name} differs from expected item")
            if _cluster(row.get("cluster")) != item["cluster"]:
                raise ValueError("cluster differs from expected item")
            if row.get("source_gold") != item["answer"]:
                raise ValueError("source_gold differs from original item answer")
            if "parsed" not in row or row["parsed"] not in ("yes", "no", None):
                raise ValueError("parsed must be yes, no, or null")
            by_key[key] = row
            parse_groups[(seed, item["phen"], arm)].append(row)
        except (ValueError, TypeError) as exc:
            errors.append(f"output row {index}: {exc}")
    missing = sorted(expected_keys - set(by_key))
    result["coverage"] = {"expected_rows": len(expected_keys), "received_rows": len(rows),
                          "accepted_unique_rows": len(by_key), "missing_count": len(missing),
                          "duplicate_count": len(duplicates), "unexpected_count": len(unexpected),
                          "missing_examples": [list(x) for x in missing[:20]],
                          "duplicate_examples": [list(x) for x in duplicates[:20]],
                          "unexpected_examples": [list(x) for x in unexpected[:20]],
                          "parse_fail_by_seed_phen_arm": {}}
    if missing:
        errors.append(f"missing {len(missing)} expected rows")
    if duplicates:
        errors.append(f"duplicate {len(duplicates)} output keys")
    if unexpected:
        errors.append(f"unexpected {len(unexpected)} output keys")
    for key, group in sorted(parse_groups.items()):
        failures = sum(r["parsed"] is None for r in group)
        rate = failures / len(group)
        result["coverage"]["parse_fail_by_seed_phen_arm"]["|".join(map(str, key))] = {
            "n": len(group), "failed": failures, "rate": rate}
        if rate > MAX_PARSE_FAIL + EPS:
            errors.append(f"parse failure >1% for {key}: {failures}/{len(group)}")
    if errors:
        return result
    for seed in seeds:
        per_phen = {}
        for phen in sorted({i["phen"] for i in expected.values()}):
            paired = sorted((i for i in expected.values() if i["phen"] == phen and i["kind"] != "hard_neg"), key=lambda i: i["id"])
            hard = sorted((i for i in expected.values() if i["phen"] == phen and i["kind"] == "hard_neg"), key=lambda i: i["id"])
            answers = {arm: {i["id"]: by_key[(seed, arm, i["id"])] for i in paired} for arm in ARMS}
            pm = {arm: _paired_metrics(paired, answers[arm]) for arm in ARMS}
            contrasts = {}
            event_names = sorted(pm["real"]["events"])
            for arm in ARMS:
                deltas = {c: pm[arm]["events"][c]["ba"] - pm["real"]["events"][c]["ba"] for c in event_names}
                contrasts[arm] = {"delta": float(np.mean(list(deltas.values()))) if deltas else None,
                                  "event_deltas": deltas,
                                  "ci95_delta": _bootstrap_delta(list(deltas.values())) if phen == "flood" else None,
                                  "uncertainty_unit": "flood_event" if phen == "flood" else "descriptive_regions_only",
                                  "bootstrap_draws": BOOTSTRAP_DRAWS if phen == "flood" else None,
                                  "bootstrap_seed": BOOTSTRAP_SEED if phen == "flood" else None}
            per_phen[phen] = {"paired": pm, "contrasts": contrasts,
                              "behaviour": {arm: _behaviour(paired, answers["real"], answers[arm]) for arm in ARMS},
                              "hard_neg": {arm: _hard_metrics(hard, {i["id"]: by_key[(seed, arm, i["id"])] for i in hard}) for arm in HARD_ARMS} if hard else {},
                              "scope": "paired_positive_tiles_only_for_primary_metrics; hard negatives descriptive and separate"}
        result["metrics"][str(seed)] = per_phen
    sufficient_seeds, sensitive_seeds = [], []
    for seed in seeds:
        flood = result["metrics"][str(seed)].get("flood")
        support = flood["paired"]["real"]["event_count"] if flood else 0
        real_ba = flood["paired"]["real"]["macro_ba"] if flood else None
        eligible = support >= 5 and real_ba is not None and real_ba >= 0.60 - EPS
        sufficient, sensitive = False, False
        if eligible:
            sufficient = all(flood["paired"][arm]["macro_ba"] >= 0.60 - EPS and
                             flood["contrasts"][arm]["ci95_delta"][0] >= -0.05 - EPS
                             for arm in ("later_only", "repeat_later"))
            sensitive = all(flood["contrasts"][arm]["delta"] <= -0.10 + EPS and
                            flood["contrasts"][arm]["ci95_delta"][1] < -EPS
                            for arm in ("later_only", "repeat_later"))
        if sufficient:
            sufficient_seeds.append(seed)
        if sensitive:
            sensitive_seeds.append(seed)
        result["seed_decisions"][str(seed)] = {"flood_events": support, "real_macro_ba": real_ba,
                                               "eligible": eligible, "both_arms_sufficient": sufficient,
                                               "both_arms_history_sensitive": sensitive}
    # The preregistration fixes a two-seed requirement; a seeds subset does not
    # relax this threshold. This also supports testing a single seed's metrics.
    if len(sufficient_seeds) >= 2:
        verdict = "single_second_view_sufficient_under_intervention"
    elif len(sensitive_seeds) >= 2:
        verdict = "history_sensitive_under_intervention"
    else:
        verdict = "mixed_or_inconclusive"
    result.update(valid=True, verdict=verdict, sufficient_seeds=sufficient_seeds,
                  history_sensitive_seeds=sensitive_seeds)
    return result
