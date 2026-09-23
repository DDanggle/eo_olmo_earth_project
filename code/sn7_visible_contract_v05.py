"""Prefix-visible human annotation contract for SpaceNet7 v0.5.

The helpers in this module are deliberately independent from the existing
SpaceNet7 runners.  They validate an episode, derive a tri-state target from a
single completed annotation, form an agreement target, and build a
date-matched content-control pair without reusing either episode's gold.
"""

from __future__ import annotations

import copy
import re
from pathlib import PurePosixPath
from urllib.parse import urlsplit


_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")
_STATES = {
    "no_visible_change",
    "visible_change",
    "unreadable",
    "ambiguous",
}
_ANSWERS = {
    "change_supported",
    "no_visible_change",
    "insufficient_evidence",
}


def _require_mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")


def _require_nonempty_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_month(value, name):
    _require_nonempty_string(value, name)
    match = _MONTH_RE.fullmatch(value)
    if match is None or not 1 <= int(match.group(2)) <= 12:
        raise ValueError(f"{name} must be an ISO YYYY-MM month")


def _validate_relative_path(value, name):
    _require_nonempty_string(value, name)
    if "\x00" in value:
        raise ValueError(f"{name} contains a null byte")

    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or value.startswith(("/", "\\")):
        raise ValueError(f"{name} must be a relative path, not a URL or absolute path")
    # Backslashes are separators on Windows but ordinary characters to
    # PurePosixPath. Reject them so the same contract is safe on every host.
    if "\\" in value:
        raise ValueError(f"{name} must use forward-slash relative paths")

    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must not escape its relative root")


def validate_episode(ep):
    """Validate one prefix-visible episode, raising ``ValueError`` on failure."""

    _require_mapping(ep, "episode")
    required = {"id", "aoi", "region", "cutoff", "reference_id", "frames"}
    missing = sorted(required - set(ep))
    if missing:
        raise ValueError(f"episode missing required fields: {', '.join(missing)}")

    for key in ("id", "aoi", "region", "reference_id"):
        _require_nonempty_string(ep[key], f"episode.{key}")
    _validate_month(ep["cutoff"], "episode.cutoff")

    frames = ep["frames"]
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("episode.frames must be a list with at least two frames")

    frame_ids = []
    dates = []
    for index, frame in enumerate(frames):
        _require_mapping(frame, f"episode.frames[{index}]")
        frame_missing = {"id", "date", "path"} - set(frame)
        if frame_missing:
            fields = ", ".join(sorted(frame_missing))
            raise ValueError(f"episode.frames[{index}] missing required fields: {fields}")
        _require_nonempty_string(frame["id"], f"episode.frames[{index}].id")
        _validate_month(frame["date"], f"episode.frames[{index}].date")
        _validate_relative_path(frame["path"], f"episode.frames[{index}].path")
        if frame["date"] > ep["cutoff"]:
            raise ValueError("frame date must not be after episode.cutoff")
        frame_ids.append(frame["id"])
        dates.append(frame["date"])

    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("episode frame IDs must be unique")
    if len(dates) != len(set(dates)):
        raise ValueError("episode frame dates must be unique")
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("episode frame dates must be strictly ascending")
    if ep["reference_id"] != frame_ids[0]:
        raise ValueError("episode.reference_id must equal the first frame ID")


def _validate_annotation(ep, ann):
    _require_mapping(ann, "annotation")
    required = {"episode_id", "annotator_id", "status", "states"}
    missing = sorted(required - set(ann))
    if missing:
        raise ValueError(f"annotation missing required fields: {', '.join(missing)}")
    _require_nonempty_string(ann["episode_id"], "annotation.episode_id")
    _require_nonempty_string(ann["annotator_id"], "annotation.annotator_id")
    if ann["episode_id"] != ep["id"]:
        raise ValueError("annotation.episode_id does not match episode.id")
    if ann["status"] not in {"complete", "timeout"}:
        raise ValueError("annotation.status must be 'complete' or 'timeout'")
    if ann["status"] == "timeout":
        raise ValueError("timeout annotations are incomplete and excluded from evaluation")

    states = ann["states"]
    _require_mapping(states, "annotation.states")
    frame_ids = [frame["id"] for frame in ep["frames"]]
    expected = set(frame_ids)
    supplied = set(states)
    if supplied != expected or len(states) != len(frame_ids):
        missing_ids = sorted(expected - supplied, key=str)
        extra_ids = sorted(supplied - expected, key=str)
        details = []
        if missing_ids:
            details.append("missing=" + ",".join(missing_ids))
        if extra_ids:
            details.append("extra=" + ",".join(extra_ids))
        raise ValueError("annotation.states must label every frame exactly once" + (
            ": " + "; ".join(details) if details else ""
        ))
    invalid = {frame_id: state for frame_id, state in states.items() if state not in _STATES}
    if invalid:
        raise ValueError(f"annotation.states contains invalid values: {invalid}")
    if states[ep["reference_id"]] == "visible_change":
        raise ValueError("the reference frame cannot be labelled visible_change")


def _base_target(ep, ann, answer, current_state):
    return {
        "episode_id": ep["id"],
        "annotator_id": ann["annotator_id"],
        "answer": answer,
        "first_change_id": None,
        "first_change_date": None,
        "last_clear_no_change_id": None,
        "last_clear_no_change_date": None,
        "evidence_ids": [],
        # These IDs are observations supporting this prefix-visible decision.
        # They do not prove a physical onset time or global absence of change.
        "evidence_ids_role": "supporting_observations_not_global_or_onset_proof",
        "current_state": current_state,
        "label_tier": "human_visible_single",
        "source": "annotation",
    }


def derive_target(ep, ann):
    """Derive one tri-state, visibility-grounded target from a completed annotation.

    The operational question is: "Compared with the first provided reference
    image, at which first observation is a new structural change clearly
    visible?"  It intentionally does not use a cumulative-building threshold.
    """

    validate_episode(ep)
    _validate_annotation(ep, ann)

    frames = ep["frames"]
    states = ann["states"]
    reference_id = ep["reference_id"]
    reference_state = states[reference_id]
    raw_current_state = states[frames[-1]["id"]]

    if reference_state in {"unreadable", "ambiguous"}:
        # A later frame may itself be readable, but without a readable baseline
        # its *comparative* current state is not grounded.  Preserve the raw
        # label separately rather than silently treating it as comparative gold.
        current_state = "unreadable" if raw_current_state == "unreadable" else "ambiguous"
        target = _base_target(ep, ann, "insufficient_evidence", current_state)
        target["reference_state"] = reference_state
        target["raw_current_state"] = raw_current_state
        target["evidence_ids"] = list(dict.fromkeys(
            [reference_id] + [
                frame["id"] for frame in frames
                if states[frame["id"]] in {"unreadable", "ambiguous"}
            ]
        ))
        return target

    current_state = raw_current_state

    first_change_index = next(
        (index for index, frame in enumerate(frames)
         if states[frame["id"]] == "visible_change"),
        None,
    )
    if first_change_index is not None:
        first_change = frames[first_change_index]
        clear_before = [
            frame for frame in frames[:first_change_index]
            if states[frame["id"]] == "no_visible_change"
        ]
        last_clear = clear_before[-1]
        target = _base_target(ep, ann, "change_supported", current_state)
        target.update({
            "first_change_id": first_change["id"],
            "first_change_date": first_change["date"],
            "last_clear_no_change_id": last_clear["id"],
            "last_clear_no_change_date": last_clear["date"],
            "evidence_ids": list(dict.fromkeys([reference_id, first_change["id"]])),
            "reference_state": reference_state,
            "raw_current_state": raw_current_state,
        })
        return target

    no_change_frames = [
        frame for frame in frames if states[frame["id"]] == "no_visible_change"
    ]
    last_clear = no_change_frames[-1]
    if len(no_change_frames) == len(frames):
        target = _base_target(ep, ann, "no_visible_change", current_state)
        target.update({
            "last_clear_no_change_id": last_clear["id"],
            "last_clear_no_change_date": last_clear["date"],
            "evidence_ids": [frame["id"] for frame in frames],
            "reference_state": reference_state,
            "raw_current_state": raw_current_state,
        })
        return target

    target = _base_target(ep, ann, "insufficient_evidence", current_state)
    target.update({
        "last_clear_no_change_id": last_clear["id"],
        "last_clear_no_change_date": last_clear["date"],
        "evidence_ids": list(dict.fromkeys(
            [reference_id] + [
                frame["id"] for frame in frames
                if states[frame["id"]] in {"unreadable", "ambiguous"}
            ]
        )),
        "reference_state": reference_state,
        "raw_current_state": raw_current_state,
    })
    return target


def consensus_target(ep, annotations):
    """Return an agreed target from at least two distinct annotators."""

    validate_episode(ep)
    if isinstance(annotations, (str, bytes, dict)):
        raise ValueError("annotations must be a sequence of annotation mappings")
    try:
        annotations = list(annotations)
    except TypeError as exc:
        raise ValueError("annotations must be iterable") from exc
    if len(annotations) < 2:
        raise ValueError("consensus requires at least two annotations")

    targets = [derive_target(ep, annotation) for annotation in annotations]
    annotator_ids = [target["annotator_id"] for target in targets]
    if len(set(annotator_ids)) != len(annotator_ids):
        raise ValueError("consensus annotations must come from distinct annotators")

    signature_keys = (
        "answer",
        "first_change_id",
        "last_clear_no_change_id",
        "current_state",
        "reference_state",
        "raw_current_state",
    )
    signatures = [
        tuple(target[key] for key in signature_keys) + (tuple(target["evidence_ids"]),)
        for target in targets
    ]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("annotators do not agree on the target")

    agreed = copy.deepcopy(targets[0])
    agreed.pop("annotator_id", None)
    agreed["label_tier"] = "human_visible_agreed"
    agreed["annotator_ids"] = sorted(annotator_ids)
    return agreed


def _validate_agreed_target(ep, target, name):
    _require_mapping(target, name)
    if target.get("label_tier") != "human_visible_agreed":
        raise ValueError(f"{name} must have label_tier='human_visible_agreed'")
    if target.get("episode_id") != ep["id"]:
        raise ValueError(f"{name}.episode_id does not match its episode")
    if target.get("answer") not in _ANSWERS:
        raise ValueError(f"{name}.answer is invalid")
    annotator_ids = target.get("annotator_ids")
    if (not isinstance(annotator_ids, list) or len(annotator_ids) < 2
            or any(not isinstance(value, str) or not value.strip()
                   for value in annotator_ids)
            or len(set(annotator_ids)) != len(annotator_ids)):
        raise ValueError(f"{name} must record at least two distinct annotators")

    by_id = {frame["id"]: frame for frame in ep["frames"]}
    first_id = target.get("first_change_id")
    first_date = target.get("first_change_date")
    if target["answer"] == "change_supported":
        if first_id not in by_id or first_date != by_id[first_id]["date"]:
            raise ValueError(f"{name} has an inconsistent first change")
    elif first_id is not None or first_date is not None:
        raise ValueError(f"{name} must not attach a first change to a non-change answer")

    last_id = target.get("last_clear_no_change_id")
    last_date = target.get("last_clear_no_change_date")
    if last_id is None:
        if last_date is not None:
            raise ValueError(f"{name} has a date without a last-clear frame")
    elif last_id not in by_id or last_date != by_id[last_id]["date"]:
        raise ValueError(f"{name} has an inconsistent last-clear frame")
    if target.get("current_state") not in _STATES:
        raise ValueError(f"{name}.current_state is invalid")


def make_control_pair(source_ep, source_target, donor_ep, donor_target):
    """Build a full-sequence, date-matched control while preserving both golds."""

    validate_episode(source_ep)
    validate_episode(donor_ep)
    _validate_agreed_target(source_ep, source_target, "source_target")
    _validate_agreed_target(donor_ep, donor_target, "donor_target")

    if source_ep["id"] == donor_ep["id"]:
        raise ValueError("source and donor episodes must be distinct")
    if source_ep["region"] == donor_ep["region"]:
        raise ValueError("source and donor must use different full-sequence regions")

    source_dates = [frame["date"] for frame in source_ep["frames"]]
    donor_dates = [frame["date"] for frame in donor_ep["frames"]]
    if source_dates != donor_dates:
        raise ValueError("source and donor frame dates must match exactly")

    source_signature = (source_target["answer"], source_target.get("first_change_date"))
    donor_signature = (donor_target["answer"], donor_target.get("first_change_date"))
    if source_signature == donor_signature:
        raise ValueError("source and donor outcome signatures must differ")

    frame_pairs = [
        {
            "date": source_frame["date"],
            "source_frame_id": source_frame["id"],
            "donor_frame_id": donor_frame["id"],
        }
        for source_frame, donor_frame in zip(source_ep["frames"], donor_ep["frames"])
    ]
    return {
        "source_episode_id": source_ep["id"],
        "donor_episode_id": donor_ep["id"],
        "frame_pairs": frame_pairs,
        "source_target": copy.deepcopy(source_target),
        "donor_target": copy.deepcopy(donor_target),
    }
