#!/usr/bin/env python3
"""Sen12Landslides G-P 입력·라벨·LOCO 계약을 실물 NetCDF에서 생성한다.

G-P는 "OLMoEarth를 쓸 수 있는가"를 판정하는 frozen-probe 게이트다. 모델을 돌리기 전에
아래를 동결하지 않으면 성능 차이가 split, 시점 선택, 구름 또는 라벨 반복 방식의 차이일 수 있다.

  - 입력: harmonized S2, 128x128, 15시점, B02--B12 10밴드
  - 라벨: MASK가 시점마다 같은 정적 event polygon인지 전 파일에서 확인
  - 시점: event/pre/post 인덱스와 SCL 품질을 기록하되 임의 보정하지 않음
  - split: Höhn 저자 고정 11지역 중 양성이 있는 10지역 outer-LOCO, 다음 지역을 validation으로 고정

이 스크립트는 학습을 하지 않는다. ``retrospective_contract_ready``가 참이어야 GPU smoke를
열고, ``operational_cutoff_ready``는 음성 샘플의 pseudo-cutoff 정책까지 동결하기 전에는
의도적으로 거짓으로 둔다.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


# M12의 annotation-process 감사에서 통과한 11지역 중 LanaoDelNorte는 실물 S2 MASK 양성이 0이다.
# 저자 고정 cohort에는 남지만 segmentation LOCO의 held-out region으로는 쓸 수 없다.
ANNOTATION_MATCHED_REGIONS = (
    "chimanimani",
    "china",
    "hiroshima",
    "hokkaido",
    "indonesia",
    "itogon",
    "kyrgyzstan1",
    "kyrgyzstan2",
    "lanaodelnorte",
    "newzealand",
    "thrissur",
)
HEADLINE_REGIONS = tuple(r for r in ANNOTATION_MATCHED_REGIONS if r != "lanaodelnorte")
NEGATIVE_ONLY_REGIONS = ("lanaodelnorte",)
BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
EXPECTED_VARS = set(BANDS) | {"SCL", "MASK", "DEM"}
CLEAR_SCL = {2, 4, 5, 6, 7}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def parse_pre_post_dates(value: Any) -> tuple[int | None, int | None]:
    """NetCDF attr의 dict/문자열 표현을 (pre, post) 정수 인덱스로 정규화한다."""
    if value is None:
        return None, None
    obj = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "nan", "{}"}:
            return None, None
        try:
            obj = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return None, None
    if not isinstance(obj, dict):
        return None, None

    def to_int(item: Any) -> int | None:
        try:
            return int(item) if item is not None else None
        except (TypeError, ValueError):
            return None

    return to_int(obj.get("pre")), to_int(obj.get("post"))


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def parse_float(value: Any) -> float | None:
    """빈 NetCDF attr를 결측으로 보존하고 숫자만 변환한다."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_index(path: Path, extracted: Path) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        name = str(row["file"])
        if name in rows:
            raise ValueError(f"region_index 중복 파일: {name}")
        rows[name] = {
            "sample_id": Path(name).stem,
            "file": name,
            "region": str(row["region"]).lower(),
            "part": row.get("part"),
            "path": str(extracted / name),
        }
    return [rows[k] for k in sorted(rows)]


def inspect_netcdf(base: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import xarray as xr

    path = Path(base["path"])
    record = dict(base)
    record["exists"] = path.is_file()
    if not record["exists"]:
        record["error"] = "missing_file"
        return record

    try:
        with xr.open_dataset(path, decode_times=True, cache=False) as ds:
            record["dims"] = {k: int(v) for k, v in ds.sizes.items()}
            record["data_vars"] = sorted(ds.data_vars)
            record["band_contract_ok"] = set(BANDS) <= set(ds.data_vars)
            record["expected_vars_ok"] = EXPECTED_VARS <= set(ds.data_vars)
            record["shape_ok"] = record["dims"] == {"time": 15, "x": 128, "y": 128}

            times = np.asarray(ds["time"].values)
            record["times"] = [str(np.datetime_as_string(t, unit="s")) for t in times]
            record["time_unique"] = len(np.unique(times)) == len(times)
            record["time_ordered"] = bool(np.all(times[1:] > times[:-1]))

            attrs = ds.attrs
            annotated = parse_bool(attrs.get("annotated"))
            pre, post = parse_pre_post_dates(attrs.get("pre_post_dates"))
            record.update({
                "annotated_attr": annotated,
                "ann_id": None if attrs.get("ann_id") is None else str(attrs.get("ann_id")),
                "event_date": None if attrs.get("event_date") in {None, "", "None"}
                else str(attrs.get("event_date")),
                "date_confidence": parse_float(attrs.get("date_confidence")),
                "pre_index": pre,
                "post_index": post,
            })

            valid_pair = (pre is not None and post is not None and 0 <= pre < post < len(times))
            record["pre_post_valid"] = valid_pair
            if valid_pair:
                record["pre_time"] = record["times"][pre]
                record["post_time"] = record["times"][post]

            mask = np.asarray(ds["MASK"].values)
            if mask.ndim == 2:
                mask = mask[None, ...]
            record["mask_values"] = [int(x) for x in np.unique(mask)]
            record["mask_binary"] = set(record["mask_values"]) <= {0, 1}
            record["mask_time_invariant"] = bool(np.all(mask == mask[0:1]))
            record["mask_positive_pixels"] = int(np.count_nonzero(mask[0] > 0))
            record["label_positive"] = record["mask_positive_pixels"] > 0
            record["annotation_consistent"] = (
                annotated is None or bool(annotated) == bool(record["label_positive"])
            )

            scl = np.asarray(ds["SCL"].values)
            if scl.ndim == 2:
                scl = scl[None, ...]
            record["scl_clear_fraction"] = [
                round(float(np.isin(s, tuple(CLEAR_SCL)).mean()), 6) for s in scl
            ]
            if valid_pair:
                record["pre_clear_fraction"] = record["scl_clear_fraction"][pre]
                record["post_clear_fraction"] = record["scl_clear_fraction"][post]

            center_lat, center_lon = attrs.get("center_lat"), attrs.get("center_lon")
            try:
                center_lat, center_lon = float(center_lat), float(center_lon)
                record["center_attrs_look_geographic"] = (
                    abs(center_lat) <= 90 and abs(center_lon) <= 180
                )
            except (TypeError, ValueError):
                record["center_attrs_look_geographic"] = None
            record["crs"] = None if attrs.get("crs") is None else str(attrs.get("crs"))
    except Exception as exc:  # noqa: BLE001 - 전수 감사는 오류를 기록하고 계속해야 함
        record["error"] = f"{type(exc).__name__}: {exc}"[:500]
    return record


def assign_task_eligibility(record: dict[str, Any]) -> dict[str, Any]:
    """감사 결과를 task cohort로 바꾼다. 원자료를 고치지 않고 애매한 표본은 fail-closed 제외."""
    row = dict(record)
    readable = not row.get("error")
    row["s15_eligible"] = bool(readable and row.get("annotation_consistent") is True)
    row["r_event_eligible"] = row["s15_eligible"]
    row["s_cutoff_positive_eligible"] = bool(
        row["s15_eligible"] and row.get("label_positive") and row.get("pre_post_valid")
    )
    return row


def build_loco_folds(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """outer test 지역과 inner validation 지역을 통째로 분리한 10-fold 계약."""
    by_region: dict[str, list[str]] = defaultdict(list)
    for row in records:
        if (row.get("region") in HEADLINE_REGIONS and not row.get("error")
                and row.get("s15_eligible", True)):
            by_region[row["region"]].append(row["sample_id"])

    folds = []
    regions = list(HEADLINE_REGIONS)
    for i, test_region in enumerate(regions):
        val_region = regions[(i + 1) % len(regions)]
        train_regions = [r for r in regions if r not in {test_region, val_region}]
        role_regions = {"train": train_regions, "val": [val_region], "test": [test_region]}
        role_samples = {
            role: sorted(s for region in chosen for s in by_region.get(region, []))
            for role, chosen in role_regions.items()
        }
        folds.append({
            "fold": f"holdout_{test_region}",
            "test_region": test_region,
            "val_region": val_region,
            "train_regions": train_regions,
            "sample_counts": {k: len(v) for k, v in role_samples.items()},
            "sample_sha256": {k: sha256_lines(v) for k, v in role_samples.items()},
        })
    return folds


def summarize(records: list[dict[str, Any]], index_sha256: str) -> dict[str, Any]:
    valid = [r for r in records if not r.get("error")]
    positives = [r for r in valid if r.get("label_positive")]
    headline = [r for r in valid if r.get("region") in HEADLINE_REGIONS]
    region_counts = Counter(r["region"] for r in valid)
    positive_counts = Counter(r["region"] for r in positives)

    def all_true(key: str, rows: list[dict[str, Any]] = valid) -> bool:
        return bool(rows) and all(r.get(key) is True for r in rows)

    positive_prepost = sum(bool(r.get("pre_post_valid")) for r in positives)
    schema_gates = {
        "all_indexed_files_readable": len(valid) == len(records),
        "uniform_15x128x128": all_true("shape_ok"),
        "s2_10band_contract": all_true("band_contract_ok"),
        "required_vars_present": all_true("expected_vars_ok"),
        "binary_static_mask": all_true("mask_binary") and all_true("mask_time_invariant"),
        "time_strictly_ordered": all_true("time_unique") and all_true("time_ordered"),
        "headline_10_regions_present": set(HEADLINE_REGIONS) <= set(region_counts),
        "headline_regions_have_both_classes": all(
            0 < positive_counts.get(region, 0) < region_counts.get(region, 0)
            for region in HEADLINE_REGIONS
        ),
    }
    label_anomalies = sorted(
        r["sample_id"] for r in valid if r.get("annotation_consistent") is not True
    )
    s15_eligible = [r for r in valid if r.get("s15_eligible")]
    retrospective_ready = all(schema_gates.values()) and (
        len(s15_eligible) + len(label_anomalies) == len(valid)
    )
    return {
        "schema": "sen12-gp-contract-v1",
        "source_index_sha256": index_sha256,
        "samples_indexed": len(records),
        "samples_readable": len(valid),
        "samples_positive": len(positives),
        "s15_eligible_samples": len(s15_eligible),
        "s15_excluded_label_anomalies": label_anomalies,
        "s15_excluded_label_anomalies_sha256": sha256_lines(label_anomalies),
        "headline_samples": len(headline),
        "headline_s15_eligible_samples": sum(r.get("s15_eligible") for r in headline),
        "annotation_matched_regions": list(ANNOTATION_MATCHED_REGIONS),
        "negative_only_regions": {
            region: region_counts.get(region, 0) for region in NEGATIVE_ONLY_REGIONS
        },
        "region_counts": dict(region_counts.most_common()),
        "positive_region_counts": dict(positive_counts.most_common()),
        "positive_prepost_valid": positive_prepost,
        "positive_prepost_coverage": round(positive_prepost / len(positives), 6)
        if positives else None,
        "center_attrs_geographic_count": sum(
            r.get("center_attrs_look_geographic") is True for r in valid
        ),
        "center_attrs_projected_or_invalid_count": sum(
            r.get("center_attrs_look_geographic") is not True for r in valid
        ),
        "schema_gates": schema_gates,
        "label_anomaly_policy": "exclude from S15 and R-event; preserve source unchanged",
        "retrospective_contract_ready": retrospective_ready,
        # 음성 patch에는 event date가 없으므로 pseudo-cutoff 표본추출 정책이 별도 필요하다.
        "negative_pseudo_cutoff_policy_frozen": False,
        "operational_cutoff_ready": False,
        "notes": [
            "MASK는 15개 시점의 독립 라벨이 아니라 동일 event polygon의 반복인지 전수 검사한다.",
            "center_lat/lon은 이름만으로 위경도로 쓰지 않고 CRS와 값 범위를 함께 검사한다.",
            "annotation-matched Höhn 11지역 중 LanaoDelNorte는 양성 MASK가 0이라 S15 headline에서 제외한다.",
            "G-P S15 headline은 Italy가 아니라 task-eligible Höhn 10지역 LOCO다.",
        ],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/home/work/data/sen12landslides"))
    p.add_argument("--out", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_gp_contract"))
    p.add_argument("--limit", type=int, default=None,
                   help="smoke용 앞 N개. full contract 봉인에는 사용하지 말 것")
    p.add_argument("--reuse-manifest", type=Path, default=None,
                   help="이미 전수 감사한 sample_contract.jsonl을 재분류할 때만 사용")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    index = args.root / "region_index.jsonl"
    args.out.mkdir(parents=True, exist_ok=True)
    if args.reuse_manifest is not None:
        records = [
            assign_task_eligibility(json.loads(line))
            for line in args.reuse_manifest.read_text(encoding="utf-8").splitlines()
            if line
        ]
        print(f"[{len(records)}] reused and reclassified", flush=True)
    else:
        rows = read_index(index, args.root / "extracted")
        if args.limit is not None:
            rows = rows[: args.limit]
        records = []
        for i, row in enumerate(rows, 1):
            records.append(assign_task_eligibility(inspect_netcdf(row)))
            if i % 500 == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}] audited", flush=True)

    manifest_tmp = args.out / "sample_contract.jsonl.tmp"
    manifest = args.out / "sample_contract.jsonl"
    manifest_tmp.write_text("".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records
    ), encoding="utf-8")
    manifest_tmp.replace(manifest)

    summary = summarize(records, sha256_file(index))
    summary["sample_contract_sha256"] = sha256_file(manifest)
    summary["limit"] = args.limit
    folds = {
        "schema": "sen12-gp-loco-v1",
        "headline_regions": list(HEADLINE_REGIONS),
        "validation_rule": "next region in the frozen HEADLINE_REGIONS order",
        "sample_contract_sha256": summary["sample_contract_sha256"],
        "folds": build_loco_folds(records),
    }
    (args.out / "loco_folds.json").write_text(
        json.dumps(folds, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
