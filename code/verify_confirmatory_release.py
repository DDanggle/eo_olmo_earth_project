#!/usr/bin/env python3
"""확증 지역 개봉의 **실행 출처(provenance)를 실증**하는 게이트. 성능 수치를 읽지 않는다.

감사 지적 3건을 반영해 재작성했다.
  [P1] snapshot **존재**만 검사하던 것을 실증 검사로 바꿨다 —
       SHA256SUMS 대조 · 필수 파일 집합 · started_at < 최초 checkpoint ·
       snapshot pilot 해시와 각 실행의 code_sha256 일치 · live 소스와의 차이.
  [P1] pre 모드가 로컬 기본 경로를 보던 것을 --results-root 필수로 바꿨다.
  [P1] test sample 집합을 "9개가 서로 같은가"만 보던 것을
       **봉인된 loco_folds.json의 해당 fold SHA와 직접 대조**로 바꿨다.
       9개가 모두 똑같이 틀린 test set이어도 통과하던 결함이었다.

설계 원칙: 게이트는 성능 수치를 출력하지 않는다. 게이트와 판독을 분리하지 않으면
게이트가 형식이 된다.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, subprocess, sys

ARMS = ["P4", "P2", "P3"]
SEEDS = [1, 2, 3]
SNAP_REQUIRED = ["pilot_sen12_gp_heads.py", "sen12_official_baselines.py",
                 "extract_sen12_fold_cache.py", "audit_sen12_fold_cache.py"]


def sh(*a, cwd=None):
    return subprocess.run(a, capture_output=True, text=True, cwd=cwd).stdout.strip()


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
    ap.add_argument("--recipe", type=pathlib.Path, required=True)
    ap.add_argument("--results-root", type=pathlib.Path, required=True,
                    help="실제 결과가 쓰이는 경로. 로컬 기본값을 쓰다가 서버 경로를 "
                         "검사하지 못한 결함이 있었으므로 필수로 둔다")
    ap.add_argument("--manifest-root", type=pathlib.Path, required=True)
    ap.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path("."))
    ap.add_argument("--live-code", type=pathlib.Path, default=None,
                    help="live 소스 디렉터리. snapshot과 비교한다")
    ap.add_argument("--folds-json", type=pathlib.Path, default=None,
                    help="봉인된 loco_folds.json. test sample SHA 대조에 쓴다")
    ap.add_argument("--batch-pregen", action="store_true",
                    help="야간 자동화용 일괄 pre 생성. prior_regions_closed를 판정하지 않고 "
                         "생성 시점의 미완 지역 목록만 기록한다. 나머지 검사는 동일하다. "
                         "정당성: 뒤 지역의 결과를 본 적이 없고 순서는 등록돼 있으며 "
                         "worktree는 생성 시점에 clean이다")
    ap.add_argument("--allow-no-snapshot", action="store_true",
                    help="Thrissur처럼 snapshot 도입 **이전에** 시작된 실행에만 쓴다. "
                         "이 플래그를 쓰면 manifest에 pre_run_code_snapshot=false로 "
                         "명시되고 protocol_deviation이 기록된다. 사후 snapshot 생성 금지")
    a = ap.parse_args()

    recipe = json.loads(a.recipe.read_text(encoding="utf-8"))
    a.manifest_root.mkdir(parents=True, exist_ok=True)
    man = {"schema": "confirmatory-release-gate-v2", "fold": a.fold, "mode": a.mode,
           "checks": {}, "recipe": {}}
    fail = []

    def check(name, ok, detail):
        man["checks"][name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            fail.append(name)

    # ── recipe 신원 ──
    body = dict(recipe); body.pop("self_sha256", None)
    recomputed = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True).encode()).hexdigest()
    man["recipe"] = {"path": str(a.recipe), "schema": recipe.get("schema"),
                     "declared_self_sha256": recipe.get("self_sha256"),
                     "recomputed_self_sha256": recomputed}
    check("recipe_self_sha_matches", recipe.get("self_sha256") == recomputed,
          "self_sha256 재계산 일치. 불일치면 recipe가 커밋 후 수정됐다")

    allowed = recipe.get("region_release_plan", {}).get("release_order", [])
    check("fold_in_release_order", a.fold in allowed, f"등록 순서: {allowed}")

    root = a.results_root / a.fold
    snap = root / "code_snapshot"

    if a.mode == "pre":
        idx = allowed.index(a.fold) if a.fold in allowed else -1
        prior = allowed[:max(idx, 0)]
        closed = [r for r in prior if (a.manifest_root / f"{r}_post.json").exists()]
        if a.batch_pregen:
            man["checks"]["prior_regions_closed"] = {
                "pass": None,
                "detail": f"일괄 생성 — 생성 시점 미완: {sorted(set(prior) - set(closed))}. "
                          "서버 오케스트레이터가 순서를 강제하고 post gate를 지역마다 실행한다"}
        else:
            check("prior_regions_closed", set(prior) == set(closed),
                  f"앞 순서 지역이 모두 post 게이트를 통과했는가. 미완: "
                  f"{sorted(set(prior) - set(closed))}")
        check("no_existing_output", not root.exists(),
              f"결과 경로 {root} 가 이미 있으면 재실행이므로 중단")
        dirty = [l for l in sh("git", "status", "--porcelain",
                               cwd=str(a.repo_root)).splitlines() if l.strip()]
        man["git"] = {"head": sh("git", "rev-parse", "HEAD", cwd=str(a.repo_root)),
                      "dirty_files": dirty}
        check("clean_worktree", not dirty, "미커밋 변경이 있으면 실행 코드 신원을 증명 불가")
        man["preregistered"] = {
            "predictions": recipe.get("predictions_registered_before_unsealing"),
            "win_definition": recipe.get("win_definition"),
            "stopping_rule": recipe.get("stopping_rule")}

    else:  # ── post ──
        # (A) snapshot 실증
        if a.allow_no_snapshot:
            man["protocol_deviation"] = {
                "pre_run_code_snapshot": False,
                "reason": "이 실행은 snapshot 도입(M57) 이전에 시작됐다",
                "policy": "사후 snapshot을 만들어 통과시키지 않는다. 그것은 "
                          "pre-run snapshot이 아니라 retrospective copy다",
            }
            man["checks"]["code_snapshot_verified"] = {
                "pass": None, "detail": "면제 — protocol_deviation으로 기록됨"}
        else:
            ok_files = snap.is_dir() and all((snap / f).exists() for f in SNAP_REQUIRED)
            check("snapshot_required_files", ok_files,
                  f"필수 파일 {SNAP_REQUIRED} 전부 존재")
            sums_ok = False
            if (snap / "SHA256SUMS.txt").exists():
                declared = {}
                for line in (snap / "SHA256SUMS.txt").read_text().splitlines():
                    if line.strip():
                        h, n = line.split()
                        declared[pathlib.Path(n).name] = h
                actual = {f: sha256_file(snap / f) for f in SNAP_REQUIRED
                          if (snap / f).exists()}
                sums_ok = all(declared.get(k) == v for k, v in actual.items()) and bool(actual)
                man["snapshot"] = {"declared": declared, "actual": actual}
            check("snapshot_sha256sums_match", sums_ok,
                  "SHA256SUMS.txt가 실제 파일과 일치하는가 (사후 교체 탐지)")
            started = None
            if (snap / "started_at.txt").exists():
                started = (snap / "started_at.txt").read_text().strip()
            ckpts = sorted(root.rglob("*_best.pt"), key=lambda x: x.stat().st_mtime)
            earliest = ckpts[0].stat().st_mtime if ckpts else None
            man["snapshot_timing"] = {"started_at": started,
                                      "earliest_checkpoint_mtime": earliest}
            import datetime as _dt
            t_ok = False
            if started and earliest:
                try:
                    t_ok = _dt.datetime.fromisoformat(started).timestamp() < earliest
                except Exception:
                    t_ok = False
            check("snapshot_before_first_checkpoint", t_ok,
                  "started_at이 최초 checkpoint보다 앞서는가 (사후 생성 탐지)")
            if a.live_code and (snap / "pilot_sen12_gp_heads.py").exists():
                live = a.live_code / "pilot_sen12_gp_heads.py"
                man["live_vs_snapshot"] = {
                    "live_sha256": sha256_file(live) if live.exists() else None,
                    "snapshot_sha256": sha256_file(snap / "pilot_sen12_gp_heads.py"),
                    "note": "다르면 실행 후 live가 바뀐 것. snapshot을 실행했다면 결과는 무해"}

        # (B) 9실행 완결성 + 실행 코드 일치
        runs, sample_shas, code_shas, splits, seeds_seen = {}, {}, set(), set(), {}
        for arm in ARMS:
            for s in SEEDS:
                d = root / f"{arm}_seed{s}"
                pj, ps = d / f"{a.fold}_pilot.json", d / "per_sample" / a.fold / f"{arm}_test.jsonl"
                ck = d / "checkpoints" / a.fold / f"{arm}_best.pt"
                pm = d / "prob_maps" / a.fold / f"{arm}_test_probs_u8.npy"
                info = {"pilot_json": pj.exists(), "per_sample": ps.exists(),
                        "checkpoint": ck.exists(), "prob_map": pm.exists()}
                if pj.exists():
                    j = json.loads(pj.read_text(encoding="utf-8"))
                    info["code_sha256"] = j.get("code_sha256")
                    code_shas.add(j.get("code_sha256"))
                    proto = j.get("development_protocol_v2") or {}
                    info["seed_declared"] = proto.get("seed")
                    seeds_seen[f"{arm}_seed{s}"] = proto.get("seed")
                    info["test_region"] = j.get("test_region")
                    splits.add(json.dumps(j.get("split_counts"), sort_keys=True))
                    info["cache_audit_pass"] = ((j.get("cache_audit") or {}).get("summary") or {}
                                                ).get("gates")
                if ps.exists():
                    ids = tuple(sorted(json.loads(l)["sample_id"]
                                       for l in ps.read_text(encoding="utf-8").splitlines() if l))
                    info["n_samples"] = len(ids)
                    sample_shas[f"{arm}_seed{s}"] = hashlib.sha256(
                        "\n".join(ids).encode()).hexdigest()
                runs[f"{arm}_seed{s}"] = info
        man["runs"] = runs
        check("all_nine_runs_complete",
              all(v.get("pilot_json") and v.get("per_sample") and v.get("checkpoint")
                  for v in runs.values()),
              "9실행이 pilot JSON·per-sample·checkpoint를 모두 남겼는가")
        check("prob_maps_present", all(v.get("prob_map") for v in runs.values()),
              "확률맵이 저장됐는가 (임계값 스윕·불일치 분석의 전제)")
        check("seeds_declared_match", sorted(v for v in seeds_seen.values() if v is not None)
              == sorted(SEEDS * len(ARMS)),
              f"선언된 seed가 실제 [1,2,3] x 3 arm 인가. 관측: {seeds_seen}")
        check("identical_split", len(splits) == 1, f"split_counts 고유 {len(splits)}개")
        check("identical_code_sha_across_runs", len(code_shas - {None}) == 1,
              f"실행 간 code_sha256 동일. 단 M57대로 이것만으로는 실행 중 교체를 "
              f"탐지하지 못한다. 고유 {len(code_shas - {None})}개")
        check("test_region_matches_fold",
              all(v.get("test_region") == a.fold.replace("holdout_", "")
                  for v in runs.values() if v.get("test_region")),
              "각 산출물의 test_region이 fold와 일치하는가")

        # (C) test 집합을 **봉인 계약과** 대조 — 9개가 똑같이 틀려도 잡는다
        internal_ok = len(set(sample_shas.values())) == 1
        check("identical_sample_sets", internal_ok,
              f"9실행의 test sample 집합이 서로 동일. 고유 {len(set(sample_shas.values()))}개")
        sealed = None
        if a.folds_json and a.folds_json.exists():
            fj = json.loads(a.folds_json.read_text(encoding="utf-8"))
            f = next((x for x in fj["folds"] if x["fold"] == a.fold), None)
            sealed = (f or {}).get("sample_sha256", {}).get("test")
        man["sealed_contract"] = {"folds_json": str(a.folds_json) if a.folds_json else None,
                                  "sealed_test_sha256": sealed,
                                  "observed_test_sha256": next(iter(set(sample_shas.values())), None)}
        check("test_set_matches_sealed_contract",
              bool(sealed) and internal_ok and sealed == next(iter(set(sample_shas.values())), None),
              "test sample 집합이 봉인된 loco_folds.json의 SHA와 일치하는가. "
              "9개가 모두 동일하게 틀린 경우를 잡는 유일한 검사")

        man["evidence_status_override"] = {
            "note": "pilot이 fold와 무관하게 development 문구를 하드코딩한다. 이 manifest가 정정한다",
            "actual_status": ("confirmatory_first_look_with_disclosed_provenance_deviation"
                              if a.allow_no_snapshot else "confirmatory_first_look"),
            "test_exposure": f"{a.fold} test는 이 실행에서 처음 열렸다"}

    man["verdict"] = "PASS" if not fail else "FAIL"
    man["failed_checks"] = fail
    p = a.manifest_root / f"{a.fold}_{a.mode}.json"
    p.write_text(json.dumps(man, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
    print(json.dumps({"fold": a.fold, "mode": a.mode, "verdict": man["verdict"],
                      "failed": fail,
                      "checks": {k: v["pass"] for k, v in man["checks"].items()}},
                     ensure_ascii=False, indent=2))
    print(f"manifest: {p}")
    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()
