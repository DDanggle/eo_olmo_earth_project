"""First-token yes/no scoring for new runs; never rewrites historical predictions.

The primary parser accepts a whole first token after whitespace/case normalization.
The constrained decoder compares canonical yes/no logits DIRECTLY, and resolves an
exact logit tie by the smaller tokenizer ID, matching full-vocabulary argmax on
these two tokens. A rounded probability of 0.5 is NOT an exact-logit-tie test.
Different output spaces can still disagree (e.g. 'Yes', 'No', or non-answer tokens).
"""
import math
from collections import Counter

POLICY_VERSION = "yes-no-direct-logit-smallest-token-id-v1"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _finite(value, name):
    _require(type(value) in (int, float), name + " must be an int or float")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(name + " must be finite") from exc
    _require(math.isfinite(value), name + " must be finite")
    return value


def validate_token_ids(token_ids):
    _require(isinstance(token_ids, dict) and set(token_ids) == {"yes", "no"}, "Exactly yes/no token IDs required")
    _require(all(type(value) is int and value >= 0 for value in token_ids.values()), "Token IDs must be nonnegative integers")
    _require(token_ids["yes"] != token_ids["no"], "Canonical answer token IDs must differ")


def parse_first_token(raw_token):
    """Accept only a normalized complete first token, not a yes/no substring."""
    _require(isinstance(raw_token, str), "Raw first token must be a string")
    normalized = raw_token.strip().lower()
    return normalized if normalized in ("yes", "no") else None


def binary_decision(yes_logit, no_logit, token_ids):
    """Canonical binary argmax with an explicit token-ID rule for exact ties."""
    validate_token_ids(token_ids)
    yes, no = _finite(yes_logit, "yes_logit"), _finite(no_logit, "no_logit")
    tied = yes == no
    if tied:
        prediction = min(token_ids, key=token_ids.get)
    else:
        prediction = "yes" if yes > no else "no"
    # Stable diagnostic probability. It is never used to select the label.
    difference = yes - no
    if difference >= 0:
        probability = 1.0 / (1.0 + math.exp(-difference))
    else:
        exponential = math.exp(difference)
        probability = exponential / (1.0 + exponential)
    return {"prediction": prediction, "selected_token_id": token_ids[prediction],
            "exact_logit_tie": tied, "yes_probability": probability,
            "probability_equals_half": probability == 0.5, "policy": POLICY_VERSION}


def score_answers(golds, predictions):
    """Invalid answers stay in the denominator, including invalid negatives."""
    golds, predictions = list(golds), list(predictions)
    _require(len(golds) == len(predictions), "Gold/prediction lengths differ")
    _require(all(gold in ("yes", "no") for gold in golds), "Invalid gold label")
    _require(all(pred in ("yes", "no", None) for pred in predictions), "Invalid parsed label")
    support = dict(Counter(golds))
    recalls = {gold: sum(pred == gold for target, pred in zip(golds, predictions) if target == gold) / support[gold]
               for gold in ("yes", "no") if support.get(gold)}
    return {"n": len(golds), "support": support,
            "accuracy": sum(gold == pred for gold, pred in zip(golds, predictions)) / len(golds) if golds else None,
            "balanced_accuracy": sum(recalls.values()) / 2 if len(recalls) == 2 else None,
            "recall_by_label": recalls, "invalid_count": sum(pred is None for pred in predictions)}


def analyze_historical_record(record, token_ids):
    """Return a separate policy comparison; input record is never mutated.

    Only a saved argmax token and two canonical logits are available in OE1.
    Therefore this validates the saved parser/canonical consistency, not a fresh
    reconstruction of the complete vocabulary argmax or higher-precision logits.
    """
    validate_token_ids(token_ids)
    primary = parse_first_token(record["raw_token"])
    _require(record["parsed"] == primary, "Saved primary parse differs from raw token")
    _require(record["eval_gold"] in ("yes", "no"), "Invalid evaluation gold")
    decision = binary_decision(record["yes_logit"], record["no_logit"], token_ids)
    probability = _finite(record["constrained_yes_probability"], "saved probability")
    _require(0 <= probability <= 1, "Saved probability outside [0,1]")
    _require(math.isclose(probability, decision["yes_probability"], rel_tol=0, abs_tol=2e-6), "Saved probability differs from logits")
    legacy = "yes" if probability >= .5 else "no"
    _require(record["constrained_prediction"] == legacy, "Saved constrained prediction differs from historical probability rule")
    winner = record["argmax_token_id"]
    _require(type(winner) is int and winner >= 0, "Invalid saved argmax token ID")
    if winner in token_ids.values():
        winner_label = next(label for label, token in token_ids.items() if token == winner)
        _require(primary == winner_label, "Canonical token ID/text disagree")
        _require(winner == decision["selected_token_id"], "Saved canonical argmax contradicts canonical logits or token-ID tie rule")
    return {"primary_unchanged": primary, "historical_constrained": legacy,
            "prospective_constrained": decision["prediction"],
            "constrained_label_changes": legacy != decision["prediction"],
            "exact_logit_tie": decision["exact_logit_tie"],
            "saved_probability_half_without_logit_tie": probability == .5 and not decision["exact_logit_tie"],
            "full_vocabulary_argmax_is_canonical": winner in token_ids.values(),
            "prospective_binary_differs_from_primary": decision["prediction"] != primary}
