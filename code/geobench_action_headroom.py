#!/usr/bin/env python3
"""Gate G0: decide whether an Earth-cache selector has measurable headroom.

Split into two sub-gates so we never confuse a normalization artifact with value.

    G0-A (action heterogeneity): across independent episodes, do the native-metric
        winners actually differ, beyond within-episode seed noise?
    G0-B (operational value): with FIXED per-episode anchors and MEASURED cost, does
        oracle routing beat the best static policy by the declared margin?

Design rules that this file enforces (each one is a lesson from a real bug):

* Cross-episode aggregation requires **fixed anchors** (``lower_anchor``/``upper_anchor``
  supplied per episode before outcomes are seen). We never min-max normalize over the
  eligible actions, because removing a dominated action (e.g. by a budget filter) would
  then silently change the relative score of the survivors — a violation of
  independence of irrelevant alternatives (IIA). ``check_iia`` proves invariance.
* Native-metric per-episode margins are always reported and need no anchor.
* Action names carry no meaning. There is no assumption that any particular action
  (such as re-embedding) is best or worst; that must be measured.
* Rows from folds, scales, seeds, or readouts are not independent episodes.

Input JSON: ``rows`` (+ optional ``budgets``). Each row:

    episode_id, task, action, seed, score, higher_is_better,
    gpu_seconds, raw_bytes, cache_bytes
    [lower_anchor, upper_anchor]  -- fixed reference scores for this episode
    [support_label_count]         -- labels the action consumed (adaptation cost)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REQUIRED = {
    "episode_id", "task", "action", "seed", "score", "higher_is_better",
    "gpu_seconds", "raw_bytes", "cache_bytes",
}
OPTIONAL_COST = ("support_label_count",)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    value = _finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def validate_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list")

    seen: set[tuple[str, str, str]] = set()
    direction: dict[str, bool] = {}
    anchors: dict[str, tuple[float, float]] = {}
    clean: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} must be an object")
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"row {index} missing {sorted(missing)}")
        item = dict(row)
        for name in ("episode_id", "task", "action"):
            if not isinstance(item[name], str) or not item[name].strip():
                raise ValueError(f"row {index} {name} must be a non-empty string")
        if not isinstance(item["higher_is_better"], bool):
            raise ValueError(f"row {index} higher_is_better must be boolean")
        item["score"] = _finite(item["score"], f"row {index} score")
        for name in ("gpu_seconds", "raw_bytes", "cache_bytes"):
            item[name] = _finite_nonnegative(item[name], f"row {index} {name}")
        for name in OPTIONAL_COST:
            if name in item:
                item[name] = _finite_nonnegative(item[name], f"row {index} {name}")
        episode = item["episode_id"]
        key = (episode, item["action"], str(item["seed"]))
        if key in seen:
            raise ValueError(f"duplicate episode/action/seed: {key}")
        seen.add(key)
        if episode in direction and direction[episode] != item["higher_is_better"]:
            raise ValueError(f"mixed score direction in episode {episode}")
        direction[episode] = item["higher_is_better"]
        if "lower_anchor" in item or "upper_anchor" in item:
            if "lower_anchor" not in item or "upper_anchor" not in item:
                raise ValueError(f"row {index}: provide both lower_anchor and upper_anchor")
            lo = _finite(item["lower_anchor"], f"row {index} lower_anchor")
            hi = _finite(item["upper_anchor"], f"row {index} upper_anchor")
            if hi <= lo:
                raise ValueError(f"row {index}: upper_anchor must exceed lower_anchor")
            if episode in anchors and anchors[episode] != (lo, hi):
                raise ValueError(f"episode {episode}: inconsistent anchors")
            anchors[episode] = (lo, hi)
            item["lower_anchor"], item["upper_anchor"] = lo, hi
        clean.append(item)
    return clean


def _within_budget(row: dict[str, Any], budget: dict[str, Any]) -> bool:
    limits = {
        "gpu_seconds": budget.get("max_gpu_seconds"),
        "raw_bytes": budget.get("max_raw_bytes"),
        "cache_bytes": budget.get("max_cache_bytes"),
        "support_label_count": budget.get("max_support_labels"),
    }
    for name, limit in limits.items():
        if limit is None:
            continue
        value = row.get(name)
        if value is None:
            continue
        if value > float(limit):
            return False
    return True


def _aggregate(rows: list[dict[str, Any]], budget: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    episode_tasks: dict[str, str] = {}
    episode_anchor: dict[str, tuple[float, float]] = {}
    for row in rows:
        episode_tasks.setdefault(row["episode_id"], row["task"])
        if episode_tasks[row["episode_id"]] != row["task"]:
            raise ValueError(f"episode {row['episode_id']} maps to multiple tasks")
        if "lower_anchor" in row:
            episode_anchor[row["episode_id"]] = (row["lower_anchor"], row["upper_anchor"])
        grouped[(row["episode_id"], row["action"])].append(row)

    episodes = sorted(episode_tasks)
    available: dict[str, set[str]] = {episode: set() for episode in episodes}
    native_mean: dict[tuple[str, str], float] = {}     # sign-corrected native mean
    seed_scores: dict[tuple[str, str], dict[str, float]] = {}
    cost_mean: dict[tuple[str, str], dict[str, float]] = {}
    for (episode, action), group in grouped.items():
        agg_cost = {
            name: mean(row[name] for row in group)
            for name in ("gpu_seconds", "raw_bytes", "cache_bytes")
            if all(name in row for row in group)
        }
        for name in OPTIONAL_COST:
            if all(name in row for row in group):
                agg_cost[name] = mean(row[name] for row in group)
        # Apply the budget to the complete repeated estimate, not to individual seeds.
        budget_probe = {**{k: 0.0 for k in ("gpu_seconds", "raw_bytes", "cache_bytes")}, **agg_cost}
        if not _within_budget(budget_probe, budget):
            continue
        available[episode].add(action)
        sign = 1.0 if group[0]["higher_is_better"] else -1.0
        native_mean[(episode, action)] = sign * mean(row["score"] for row in group)
        seed_scores[(episode, action)] = {str(row["seed"]): sign * row["score"] for row in group}
        cost_mean[(episode, action)] = agg_cost

    empty = [episode for episode in episodes if not available[episode]]
    if empty:
        raise ValueError(f"budget leaves episodes without an eligible action: {empty}")
    common = sorted(set.intersection(*(available[episode] for episode in episodes)))
    if not common:
        raise ValueError("no static action is eligible in every episode")

    # ---- Per-episode NATIVE analysis (no normalization, IIA-safe) ----
    episode_details = []
    for episode in episodes:
        values = {a: native_mean[(episode, a)] for a in common}
        ordered = sorted(values, key=lambda a: (-values[a], a))
        best, runner = ordered[0], (ordered[1] if len(ordered) > 1 else ordered[0])
        shared = sorted(set(seed_scores[(episode, best)]) & set(seed_scores[(episode, runner)]))
        seed_wins = sum(seed_scores[(episode, best)][s] > seed_scores[(episode, runner)][s] for s in shared)
        episode_details.append({
            "episode_id": episode, "task": episode_tasks[episode],
            "native_best_action": best, "native_runner_up": runner,
            "native_margin": values[best] - values[runner],
            "shared_seed_count": len(shared), "best_seed_wins": seed_wins,
            "eligible_actions": sorted(available[episode]), "gate_actions": common,
            "has_anchor": episode in episode_anchor,
        })

    # ---- Cross-episode NORMALIZED analysis (requires fixed anchors; IIA-safe) ----
    anchored = all(e in episode_anchor for e in episodes)
    normalized_block: dict[str, Any]
    if anchored:
        norm: dict[tuple[str, str], float] = {}
        for episode in episodes:
            lo, hi = episode_anchor[episode]
            for a in common:
                norm[(episode, a)] = (native_mean[(episode, a)] - lo) / (hi - lo)
        static = {a: mean(norm[(e, a)] for e in episodes) for a in common}
        best_static = sorted(static, key=lambda a: (-static[a], a))[0]
        oracle = mean(max(norm[(e, a)] for a in common) for e in episodes)
        normalized_block = {
            "normalization": "fixed per-episode anchors",
            "best_static_action": best_static,
            "best_static_normalized_score": static[best_static],
            "oracle_normalized_score": oracle,
            "oracle_headroom": oracle - static[best_static],
        }
    else:
        normalized_block = {
            "normalization": "UNAVAILABLE — fixed anchors required for cross-episode headroom",
            "oracle_headroom": None,
            "note": "supply lower_anchor/upper_anchor per episode; min-max over eligible "
                    "actions is forbidden (violates IIA under budget filtering).",
        }

    # per-task native winner (for heterogeneity), collapsing episodes of same task
    task_native: dict[tuple[str, str], list[float]] = defaultdict(list)
    for episode in episodes:
        for a in common:
            task_native[(episode_tasks[episode], a)].append(native_mean[(episode, a)])
    task_best = {}
    for task in sorted(set(episode_tasks.values())):
        cand = {a: mean(v) for (t, a), v in task_native.items() if t == task}
        task_best[task] = sorted(cand, key=lambda a: (-cand[a], a))[0]

    return {
        "budget": budget,
        "n_episodes": len(episodes),
        "n_tasks": len(task_best),
        "common_static_actions": common,
        "task_native_best_actions": task_best,
        "n_unique_task_best_actions": len(set(task_best.values())),
        "episodes": episode_details,
        **normalized_block,
    }


def analyze(
    payload: dict[str, Any],
    headroom_threshold: float = 0.02,
    native_margin_threshold: float = 0.0,
    min_reversal_groups: int = 2,
) -> dict[str, Any]:
    """G0 = G0-A (heterogeneity) AND G0-B (value). See module docstring."""
    rows = validate_payload(payload)
    budgets = payload.get("budgets") or [{"name": "unlimited"}]
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("budgets must be a non-empty list")
    results = []
    for index, budget in enumerate(budgets):
        if not isinstance(budget, dict):
            raise ValueError(f"budget {index} must be an object")
        budget = dict(budget)
        budget.setdefault("name", f"budget_{index}")
        for name in ("max_gpu_seconds", "max_raw_bytes", "max_cache_bytes", "max_support_labels"):
            if name in budget:
                budget[name] = _finite_nonnegative(budget[name], f"budget {index} {name}")
        results.append(_aggregate(rows, budget))

    # G0-A: heterogeneity from NATIVE winners (anchor-free, IIA-safe by construction).
    # A task group counts only if its native best beats its runner beyond seed noise
    # (all shared seeds agree) in at least one budget.
    task_winner_beyond_noise: dict[str, set[str]] = defaultdict(set)
    for res in results:
        for ep in res["episodes"]:
            if ep["shared_seed_count"] > 0 and ep["best_seed_wins"] == ep["shared_seed_count"] \
               and ep["native_margin"] > native_margin_threshold:
                task_winner_beyond_noise[ep["task"]].add(ep["native_best_action"])
    distinct_winners = {next(iter(v)) for v in task_winner_beyond_noise.values() if len(v) == 1}
    n_robust_groups = len(task_winner_beyond_noise)
    g0a_pass = len(distinct_winners) >= 2 and n_robust_groups >= min_reversal_groups

    # G0-B: value requires anchored headroom in >= 1 budget track.
    anchored_headrooms = [r["oracle_headroom"] for r in results if r["oracle_headroom"] is not None]
    g0b_pass = any(h is not None and h >= headroom_threshold for h in anchored_headrooms)
    g0b_computable = len(anchored_headrooms) > 0

    return {
        "schema": "earthcache-action-headroom-v1",
        "status": "DIAGNOSTIC_ONLY_NOT_A_SELECTOR_RESULT",
        "thresholds": {
            "oracle_headroom": headroom_threshold,
            "native_margin": native_margin_threshold,
            "minimum_reversal_groups": min_reversal_groups,
        },
        "budgets": results,
        "G0A_action_heterogeneity": {
            "pass": g0a_pass,
            "distinct_robust_winners": sorted(distinct_winners),
            "n_task_groups_with_robust_winner": n_robust_groups,
            "explanation": "native-metric winners that beat their runner on every shared seed",
        },
        "G0B_operational_value": {
            "pass": g0b_pass,
            "computable": g0b_computable,
            "anchored_headrooms": anchored_headrooms,
            "explanation": "anchor-normalized oracle headroom over best static; None if anchors absent",
        },
        "G0_pass": bool(g0a_pass and g0b_pass),
        "interpretation": (
            "G0 needs BOTH heterogeneity (G0-A) and anchored+measured value (G0-B). "
            "Native margins and cost are still estimates until measured; seeds, LOTO, and "
            "LOFO validation remain required before any selector claim."
        ),
    }


def check_iia(payload: dict[str, Any], dominated_action: str = "__iia_probe__") -> dict[str, Any]:
    """Prove IIA: injecting a strictly dominated action into every episode must not
    change G0-A winners or G0-B anchored headroom. Returns the two reports + verdict."""
    base = analyze(payload)
    rows = validate_payload(payload)
    # worst native score per episode (sign-aware), then a dominated probe below it
    by_ep: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_ep[r["episode_id"]].append(r)
    injected = list(payload["rows"])
    for ep, group in by_ep.items():
        sign = 1.0 if group[0]["higher_is_better"] else -1.0
        worst_native = min(sign * g["score"] for g in group)
        probe_score = (worst_native - 1.0) * sign  # strictly worse in native space
        template = dict(group[0])
        probe = {k: template[k] for k in REQUIRED}
        probe.update({"action": dominated_action, "seed": "iia", "score": probe_score,
                      "gpu_seconds": 0.0, "raw_bytes": 0.0, "cache_bytes": 0.0})
        if "lower_anchor" in template:
            probe["lower_anchor"], probe["upper_anchor"] = template["lower_anchor"], template["upper_anchor"]
        injected.append(probe)
    perturbed = analyze({**payload, "rows": injected})
    same_winners = (base["G0A_action_heterogeneity"]["distinct_robust_winners"]
                    == perturbed["G0A_action_heterogeneity"]["distinct_robust_winners"])
    same_headroom = base["G0B_operational_value"]["anchored_headrooms"] \
        == perturbed["G0B_operational_value"]["anchored_headrooms"]
    return {"iia_holds": bool(same_winners and same_headroom),
            "same_winners": same_winners, "same_headroom": same_headroom,
            "base": base, "perturbed": perturbed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--headroom-threshold", type=float, default=0.02)
    parser.add_argument("--min-reversal-groups", type=int, default=2)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = analyze(payload, args.headroom_threshold, min_reversal_groups=args.min_reversal_groups)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
