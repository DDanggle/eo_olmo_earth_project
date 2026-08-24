#!/usr/bin/env python3
"""Major TOM 249k 임베딩 계약 감사.

답하는 질문 두 개:
  Q1. `249k-OlmoEarth-Base`와 `249k-Clay-v1_5`가 정말 같은 chip인가?
      (unique_id / grid_cell 조인이 1:1로 떨어지는가)
  Q2. 두 제품의 계약(차원·pooling·밴드·정규화)이 기계가 읽을 수 있는 형태로
      기록돼 있는가, 아니면 산문에만 있는가?

Q1이 깨지면 "paired cross-model 실험대"라는 전제가 죽는다 -> 먼저 안다.
Q2의 결과표는 embeddings-stac-specification v0.0.1 gap 분석에 그대로 들어간다.

GPU를 쓰지 않는다. 읽기 전용이며 원본을 수정하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOS = {
    "olmoearth_base": "Major-TOM/Core-S2L2A-249k-OlmoEarth-Base",
    "clay_v1_5": "Major-TOM/Core-S2L2A-249k-Clay-v1_5",
}

# 계약 필드: 우리가 문서화한 두 실패가 어느 필드의 부재에서 왔는지 대조한다.
CONTRACT_FIELDS = [
    ("model_weights_hash", "v1->v1.2 cross-release R@1 0.0000"),
    ("acquisition_dates", "jeju25<->jeju26r 184-day overlap"),
    ("temporal_recipe", "four-period season-confounded path"),
    ("band_order", "12-band vs 10-band contract"),
    ("normalization", "OlmoEarth normalizer vs Clay mean/std"),
    ("pooling", "unmasked-token mean vs CLS"),
    ("input_content_hash", "reproducibility check"),
    ("output_content_hash", "reproducibility check"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(repo: str, cache: Path):
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=repo, repo_type="dataset", cache_dir=str(cache),
        allow_patterns=["*.parquet"],
    )
    files = sorted(Path(local).rglob("*.parquet"))
    if not files:
        raise SystemExit(f"REFUSED: no parquet found in {repo}")
    return files


def describe(files, label):
    import pyarrow.parquet as pq

    total = 0
    schema = None
    for f in files:
        md = pq.ParquetFile(f)
        total += md.metadata.num_rows
        if schema is None:
            schema = md.schema_arrow
    dim = None
    # embedding 차원은 첫 행에서 확인
    tbl = pq.ParquetFile(files[0]).read_row_group(0, columns=["embedding"])
    first = tbl.column("embedding")[0].as_py()
    if first is not None:
        dim = len(first)
    return {
        "label": label,
        "parquet_files": [f.name for f in files],
        "file_sha256": {f.name: sha256_file(f) for f in files},
        "rows": total,
        "embedding_dim": dim,
        "columns": [n for n in schema.names],
    }


# 조인 키 후보를 순서대로 시험한다. 이름이 unique_id라고 공유 식별자인 것은 아니다.
JOIN_CANDIDATES = [
    ("unique_id",),
    ("grid_cell", "product_id"),
    ("grid_cell",),
]


def join_check(files_a, files_b, key):
    import pyarrow.parquet as pq

    cols = list(key)

    def keys(files):
        out = []
        for f in files:
            t = pq.read_table(f, columns=cols)
            out.extend(zip(*(t.column(c).to_pylist() for c in cols)))
        return out

    ka, kb = keys(files_a), keys(files_b)
    sa, sb = set(ka), set(kb)
    return {
        "key": list(key),
        "rows_a": len(ka), "rows_b": len(kb),
        "unique_a": len(sa), "unique_b": len(sb),
        "duplicates_a": len(ka) - len(sa), "duplicates_b": len(kb) - len(sb),
        "intersection": len(sa & sb),
        "only_a": len(sa - sb), "only_b": len(sb - sa),
        "is_one_to_one": (len(ka) == len(sa) == len(kb) == len(sb) == len(sa & sb)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="artifacts/external_data/majortom_cache")
    ap.add_argument("--out", default="artifacts/results/majortom_contract_audit.json")
    args = ap.parse_args()

    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    got = {}
    for label, repo in REPOS.items():
        print(f"[fetch] {repo}", flush=True)
        got[label] = load(repo, cache)

    desc = {k: describe(v, k) for k, v in got.items()}
    for k, d in desc.items():
        print(f"[desc] {k}: rows={d['rows']} dim={d['embedding_dim']}", flush=True)

    joins = {}
    working = None
    for key in JOIN_CANDIDATES:
        jc = join_check(got["olmoearth_base"], got["clay_v1_5"], key)
        joins["+".join(key)] = jc
        print(f"[join] {'+'.join(key)}: one_to_one={jc['is_one_to_one']} "
              f"intersection={jc['intersection']}", flush=True)
        if jc["is_one_to_one"] and working is None:
            working = "+".join(key)

    uid = joins.get("unique_id", {})
    uid_trap = bool(uid) and uid["intersection"] == 0 and uid["rows_a"] == uid["rows_b"]
    jc = joins[working] if working else joins["unique_id"]

    shared = sorted(set(desc["olmoearth_base"]["columns"])
                    & set(desc["clay_v1_5"]["columns"]))
    machine_readable = {
        field: {"present_in_schema": field in shared, "our_failure_evidence": why}
        for field, why in CONTRACT_FIELDS
    }

    result = {
        "schema": "majortom-contract-audit-v1",
        "question_1_paired": {
            "verdict": "PAIRED" if working else "NOT_PAIRED",
            "working_join_key": working,
            "all_join_attempts": joins,
            "unique_id_is_a_trap": uid_trap,
            "note": (
                "NOT_PAIRED이면 cross-model paired 실험대 전제가 죽는다. "
                "unique_id_is_a_trap=true는 두 데이터셋이 같은 이름·같은 스키마의 "
                "unique_id 컬럼을 갖고 행수도 같지만 교집합이 0이라는 뜻이다. "
                "즉 unique_id는 공유 chip 식별자가 아니라 데이터셋별 content hash이며, "
                "이를 조인 키로 쓰면 조용히 빈 결과가 나온다."
            ),
        },
        "question_2_contract_fields": {
            "shared_columns": shared,
            "contract_fields": machine_readable,
            "note": "present_in_schema=false는 그 계약이 데이터셋 카드 산문에만 있다는 뜻이다. "
                    "embeddings-stac-specification v0.0.1 gap 분석의 증거로 사용한다.",
        },
        "datasets": desc,
        "forbidden_claims": [
            "이 표로 모델 우열을 비교하지 않는다 (밴드 12 vs 10, pooling mean vs CLS, 정규화가 다름).",
            "이것은 cross-family이지 release pair가 아니므로 S1->S0 호환성에 답하지 않는다.",
            "chip당 벡터 하나뿐이므로 token/공간 수준 분석에 사용하지 않는다.",
        ],
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
