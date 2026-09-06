#!/usr/bin/env python3
"""Gate G0: decide whether an Earth-cache selector has measurable headroom.

The input is a JSON object with ``rows`` and optional ``budgets``. Each row is one
seed-level outcome and must contain:

    episode_id, task, action, seed, score, higher_is_better,
    gpu_seconds, raw_bytes, cache_bytes

Rows from folds, scales, or readouts are not treated as independent tasks. The
script collapses seeds within episode/action, normalizes scores only *within* an
episode, and compares the per-episode oracle with the best action that is static
and eligible across all episodes. It is a diagnostic gate, not a selector.
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
    "episode_id",
    "task",
    "action",
    "seed",
    "score",
    "higher_is_better",
    "gpu_seconds",
    "raw_bytes",
    "cache_bytes",
}


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
        key = (item["episode_id"], item["action"], str(item["seed"]))
        if key in seen:
            raise ValueError(f"duplicate episode/action/seed: {key}")
        seen.add(key)
        episode = item["episode_id"]
        if episode in direction and direction[episode] != item["higher_is_better"]:
            raise ValueError(f"mixed score direction in episode {episode}")
        direction[episode] = item["higher_is_better"]
        clean.append(item)
    return clean


def _within_budget(row: dict[str, Any], budget: dict[str, Any]) -> bool:
    limits = {
        "gpu_seconds": budget.get("max_gpu_seconds"),
        "raw_bytes": budget.get("max_raw_bytes"),
        "cache_bytes": budget.get("max_cache_bytes"),
    }
    return all(limit is None or row[name] <= float(limit) for name, limit in limits.items())


def _aggregate(rows: list[dict[str, Any]], budget: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    episode_tasks: dict[str, str] = {}
    for row in rows:
        episode_tasks.setdefault(row["episode_id"], row["task"])
        if episode_tasks[row["episode_id"]] != row["task"]:
            raise ValueError(f"episode {row['episode_id']} maps to multiple tasks")
        grouped[(row["episode_id"], row["action"])].append(row)

    episodes = sorted(episode_tasks)
    available: dict[str, set[str]] = {episode: set() for episode in episodes}
    means: dict[tuple[str, str], float] = {}
    seed_scores: dict[tuple[str, str], dict[str, float]] = {}
    for key, group in grouped.items():
        # Apply the budget to the complete repeated estimate. Filtering individual
        # seeds by their runtime would bias both cost and performance upward.
        mean_cost = {
            name: mean(row[name] for row in group)
            for name in ("gpu_seconds", "raw_bytes", "cache_bytes")
        }
        if not _within_budget(mean_cost, budget):
            continue
        episode, action = key
        available[episode].add(action)
        sign = 1.0 if group[0]["higher_is_better"] else -1.0
        means[key] = sign * mean(row["score"] for row in group)
        seed_scores[key] = {str(row["seed"]): sign * row["score"] for row in group}

    empty = [episode for episode in episodes if not available[episode]]
    if empty:
        raise ValueError(f"budget leaves episodes without an eligible action: {empty}")
    common_actions = sorted(set.intersection(*(available[episode] for episode in episodes)))
    if not common_actions:
        raise ValueError("no static action is eligible in every episode")

    normalized: dict[tuple[str, str], float] = {}
    episode_details = []
    for episode in episodes:
        # G0 compares actions that were eligible everywhere. Otherwise a missing
        # action can manufacture oracle headroom without any prediction problem.
        values = {action: means[(episode, action)] for action in common_actions}
        lo, hi = min(values.values()), max(values.values())
        span = hi - lo
        for action, value in values.items():
            normalized[(episode, action)] = 1.0 if span == 0 else (value - lo) / span
        ordered = sorted(values, key=lambda action: (-values[action], action))
        best = ordered[0]
        runner = ordered[1] if len(ordered) > 1 else ordered[0]
        shared_seeds = sorted(set(seed_scores[(episode, best)]) & set(seed_scores[(episode, runner)]))
        seed_wins = sum(
            seed_scores[(episode, best)][seed] > seed_scores[(episode, runner)][seed]
            for seed in shared_seeds
        )
        margin = 0.0 if span == 0 else (values[best] - values[runner]) / span
        episode_details.append(
            {
                "episode_id": episode,
                "task": episode_tasks[episode],
                "best_action": best,
                "runner_up": runner,
                "normalized_margin": margin,
                "shared_seed_count": len(shared_seeds),
                "best_seed_wins": seed_wins,
                "eligible_actions": sorted(available[episode]),
                "gate_actions": common_actions,
            }
        )

    static_scores = {
        action: mean(normalized[(episode, action)] for episode in episodes)
        for action in common_actions
    }
    best_static = sorted(static_scores, key=lambda action: (-static_scores[action], action))[0]
    oracle_score = mean(max(normalized[(episode, action)] for action in common_actions) for episode in episodes)
    headroom = oracle_score - static_scores[best_static]

    task_action_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for episode in episodes:
        for action in common_actions:
            task_action_scores[(episode_tasks[episode], action)].append(normalized[(episode, action)])
    task_best = {}
    for task in sorted(set(episode_tasks.values())):
        candidates = {
            action: mean(scores)
            for (candidate_task, action), scores in task_action_scores.items()
            if candidate_task == task
        }
        task_best[task] = sorted(candidates, key=lambda action: (-candidates[action], action))[0]

    nonstatic_groups = sum(action != best_static for action in task_best.values())
    return {
        "budget": budget,
        "n_episodes": len(episodes),
        "n_tasks": len(task_best),
        "common_static_actions": common_actions,
        "best_static_action": best_static,
        "best_static_normalized_score": static_scores[best_static],
        "oracle_normalized_score": oracle_score,
        "oracle_headroom": headroom,
        "task_best_actions": task_best,
        "n_unique_task_best_actions": len(set(task_best.values())),
        "n_task_groups_with_nonstatic_best": nonstatic_groups,
        "episodes": episode_details,
    }


def analyze(
    payload: dict[str, Any], headroom_threshold: float = 0.02, min_reversal_groups: int = 2
) -> dict[str, Any]:
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
        for name in ("max_gpu_seconds", "max_raw_bytes", "max_cache_bytes"):
            if name in budget:
                budget[name] = _finite_nonnegative(budget[name], f"budget {index} {name}")
        result = _aggregate(rows, budget)
        result["headroom_pass"] = result["oracle_headroom"] >= headroom_threshold
        result["rank_variation_pass"] = (
            result["n_unique_task_best_actions"] >= 2
            and result["n_task_groups_with_nonstatic_best"] >= min_reversal_groups
        )
        result["diagnostic_pass"] = result["headroom_pass"] and result["rank_variation_pass"]
        results.append(result)

    budget_winners = {result["best_static_action"] for result in results}
    return {
        "schema": "earthcache-action-headroom-v0",
        "status": "DIAGNOSTIC_ONLY_NOT_A_SELECTOR_RESULT",
        "thresholds": {
            "oracle_headroom": headroom_threshold,
            "minimum_nonstatic_task_groups": min_reversal_groups,
        },
        "budget_static_winner_crossover": len(budget_winners) >= 2,
        "budgets": results,
        "G0_pass": any(result["diagnostic_pass"] for result in results),
        "interpretation": (
            "G0_pass only establishes selector headroom. Seed uncertainty, leave-one-task-out, "
            "leave-one-family-out, and measured-cost validation are still required."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--headroom-threshold", type=float, default=0.02)
    parser.add_argument("--min-reversal-groups", type=int, default=2)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = analyze(payload, args.headroom_threshold, args.min_reversal_groups)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
