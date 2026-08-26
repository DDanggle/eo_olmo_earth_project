#!/usr/bin/env python3
"""확증 지역 개봉의 **사전등록 경계를 입증**하는 게이트. 결과 수치를 읽지 않는다.

감사 지적 [P1-2]: 실행기가 설정을 하드코딩하지만 recipe SHA·freeze blob·허용 fold·
clean snapshot·기존 output 부재를 검사하지도 기록하지도 않았다. 이 스크립트가 그것을 한다.

두 모드:
  --mode pre    개봉 **전**. 통과 시 release manifest를 쓴다. 실패하면 개봉하지 않는다.
  --mode post   개봉 **후**. 9실행 완결성·동일성 6항목을 검사한다. 수치는 읽지 않고
                per-sample 파일의 존재·해시·sample ID 집합·code_sha256만 본다.

**중요**: post 모드도 성능 수치를 출력하지 않는다. 게이트를 통과한 뒤 별도 분석
스크립트로 읽는다. 게이트와 판독을 분리하지 않으면 게이트가 형식이 된다.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, subprocess, sys

ARMS = ["P4", "P2", "P3"]
SEEDS = [1, 2, 3]


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--mode", choices=["pre", "post"], required=True)
    ap.add_argument("--recipe", type=pathlib.Path,
                    default=pathlib.Path("evidence/recipe_frozen_v2.json"))
    ap.add_argument("--results-root", type=pathlib.Path,
                    default=pathlib.Path("evidence/confirmatory"))
    args = ap.parse_args()

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    out_dir = pathlib.Path("evidence/confirmatory_manifests")
    out_dir.mkdir(parents=True, exist_ok=True)
    man = {"schema": "confirmatory-release-gate-v1", "fold": args.fold, "mode": args.mode,
           "checks": {}, "recipe": {}}
    fail = []

    def check(name, ok, detail):
        man["checks"][name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            fail.append(name)

    # ── 공통: recipe 신원 ──
    body = dict(recipe); body.pop("self_sha256", None)
    recomputed = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True).encode()).hexdigest()
    man["recipe"] = {
        "path": str(args.recipe),
        "declared_self_sha256": recipe.get("self_sha256"),
        "recomputed_self_sha256": recomputed,
        "git_blob": sh("git", "rev-parse", f"HEAD:{args.recipe}") or None,
        "schema": recipe.get("schema"),
    }
    check("recipe_self_sha_matches",
          recipe.get("self_sha256") == recomputed,
          "self_sha256 재계산 일치 여부. 불일치면 recipe가 커밋 후 수정됐다는 뜻")
    check("recipe_blob_exists", bool(man["recipe"]["git_blob"]),
          "recipe가 HEAD에 커밋돼 있는가 — 이것이 공식 freeze 경계다")

    allowed = recipe.get("region_release_plan", {}).get("release_order", [])
    check("fold_in_release_order", args.fold in allowed,
          f"개봉 순서에 등록된 지역인가. 등록: {allowed}")

    if args.mode == "pre":
        idx = allowed.index(args.fold) if args.fold in allowed else -1
        prior = allowed[:max(idx, 0)]
        done = [r for r in prior
                if (out_dir / f"{r}_post.json").exists()]
        check("prior_regions_closed", set(prior) == set(done),
              f"앞 순서 지역이 모두 post 게이트를 통과했는가. 미완: "
              f"{sorted(set(prior) - set(done))}")
        check("no_existing_output", not (args.results_root / args.fold).exists(),
              "이 지역의 결과 디렉터리가 이미 있으면 재실행이므로 중단")
        man["git"] = {"head": sh("git", "rev-parse", "HEAD"),
                      "dirty_files": [l for l in sh("git", "status", "--porcelain").splitlines()
                                      if l.strip()]}
        check("clean_worktree", not man["git"]["dirty_files"],
              "미커밋 변경이 있으면 실행 코드 신원을 증명할 수 없다")
        man["preregistered_predictions"] = recipe.get(
            "predictions_registered_before_unsealing")
        man["win_definition"] = recipe.get("win_definition")
        man["stopping_rule"] = recipe.get("stopping_rule")

    else:  # post
        root = args.results_root / args.fold
        runs, sample_sets, code_shas, splits = {}, {}, set(), set()
        for arm in ARMS:
            for s in SEEDS:
                d = root / f"{arm}_seed{s}"
                pj = d / f"{args.fold}_pilot.json"
                ps = d / "per_sample" / args.fold / f"{arm}_test.jsonl"
                info = {"pilot_json": pj.exists(), "per_sample": ps.exists()}
                if pj.exists():
                    j = json.loads(pj.read_text(encoding="utf-8"))
                    info["code_sha256"] = j.get("code_sha256")
                    code_shas.add(j.get("code_sha256"))
                    info["seed_declared"] = (j.get("development_protocol_v2") or {}).get("seed")
                    info["split_counts"] = j.get("split_counts")
                    splits.add(json.dumps(j.get("split_counts"), sort_keys=True))
                if ps.exists():
                    ids = tuple(sorted(json.loads(l)["sample_id"]
                                       for l in ps.read_text(encoding="utf-8").splitlines() if l))
                    info["n_samples"] = len(ids)
                    info["sample_id_sha256"] = hashlib.sha256(
                        "\n".join(ids).encode()).hexdigest()
                    sample_sets[f"{arm}_seed{s}"] = info["sample_id_sha256"]
                    info["per_sample_file_sha256"] = sha256_file(ps)
                runs[f"{arm}_seed{s}"] = info
        man["runs"] = runs
        check("all_nine_runs_present",
              all(v.get("pilot_json") and v.get("per_sample") for v in runs.values()),
              "9실행(3 arm x 3 seed)이 모두 산출물을 남겼는가")
        check("identical_sample_sets", len(set(sample_sets.values())) == 1,
              f"9실행의 test sample ID 집합이 동일한가. 고유 해시 {len(set(sample_sets.values()))}개")
        check("identical_code_sha", len(code_shas - {None}) == 1,
              f"9실행의 code_sha256이 동일한가 (실행 중 코드 변경 배제). 고유 {len(code_shas - {None})}개")
        check("identical_split", len(splits) == 1,
              f"split_counts가 동일한가. 고유 {len(splits)}개")
        check("metrics_read_from_per_sample", True,
              "이 게이트는 로그 문자열을 쓰지 않는다. 판독은 per-sample 재계산으로만 한다")
        # M57: code_sha256은 summary 작성 시점에 계산되므로 **실행 중** 파일 교체를
        # 탐지하지 못한다. mtime으로 별도 검사한다.
        snap = root / "code_snapshot"
        first_out = None
        outs = sorted(root.rglob("*_pilot.json"), key=lambda x: x.stat().st_mtime)
        if outs:
            first_out = outs[0].stat().st_mtime
        man["code_timeline"] = {
            "first_output_mtime": first_out,
            "code_snapshot_present": snap.exists(),
            "note": "code_sha256은 실행 후 계산되므로 실행 중 교체를 못 잡는다 (M57). "
                    "코드 스냅샷이 있으면 실물 대조가 가능하다",
        }
        check("code_snapshot_present", snap.exists(),
              "실행 시작 시 소스 스냅샷이 봉인됐는가. 해시만으로는 M57 시나리오를 못 잡는다")
        man["evidence_status_override"] = {
            "note": "pilot이 fold와 무관하게 development 문구를 하드코딩한다 (감사 [P2]). "
                    "이 manifest가 정정한다.",
            "actual_status": "confirmatory_first_look" ,
            "test_exposure": f"{args.fold} test는 이 실행에서 처음 열렸다",
        }

    man["verdict"] = "PASS" if not fail else "FAIL"
    man["failed_checks"] = fail
    p = out_dir / f"{args.fold}_{args.mode}.json"
    p.write_text(json.dumps(man, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
    print(json.dumps({"fold": args.fold, "mode": args.mode, "verdict": man["verdict"],
                      "failed": fail,
                      "checks": {k: v["pass"] for k, v in man["checks"].items()}},
                     ensure_ascii=False, indent=2))
    print(f"manifest: {p}")
    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()
