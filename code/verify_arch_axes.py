"""arch_axes_chain 진행/실패를 실물 산출물 기준으로 검증한다.

왜 필요한가: arch_axes_chain.sh의 `echo "$(date ...) name rc=$?"`는 $(date)가 먼저
실행되면서 $?를 덮어쓴다. 따라서 logs/arch_axes.log의 rc=0은 date의 종료코드이며
단계 성공의 증거가 아니다. 이 스크립트는 rc를 무시하고 산출물만 본다.
"""
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path("/home/work/data/olmoearth")
CACHES = ["olmo_nano", "olmo_tiny", "olmo_base_half", "galileo_nano",
          "galileo_tiny", "clay_in256_half", "galileo_base_half"]
FOLDS = ["hiroshima", "hokkaido", "indonesia", "itogon",
         "kyrgyzstan1", "kyrgyzstan2", "newzealand", "thrissur"]
XLOG = {"olmo_nano": "x_olmo_nano.log", "olmo_tiny": "x_olmo_tiny.log",
        "olmo_base_half": "x_olmo_base_half.log", "galileo_nano": "x_galileo_nano.log",
        "galileo_tiny": "x_galileo_tiny.log", "clay_in256_half": "x_clay_in256_half.log",
        "galileo_base_half": "x_galileo_base_half.log"}

def extraction_state(cache):
    """추출 상태를 로그의 게이트 JSON으로 판정. rc는 보지 않는다."""
    p = ROOT / "logs" / XLOG[cache]
    if not p.exists():
        return {"state": "not_started"}
    txt = p.read_text(errors="replace")
    gate = None
    for m in re.finditer(r'\{"all_gates_pass".*?\}', txt):
        gate = m.group(0)
    if gate:
        g = json.loads(gate)
        done = "CACHE DONE" in txt
        return {"state": "ok" if (g.get("all_gates_pass") and done) else "gate_fail",
                "gate": g, "done_marker": done}
    tail = [l for l in txt.strip().splitlines() if l.strip()][-3:]
    tb = "Traceback" in txt or "Error" in txt
    return {"state": "crashed" if tb else "running", "tail": tail}

report = {"extraction": {}, "decoders": {}, "missing": [], "invalid": [], "summary": {}}
for c in CACHES:
    report["extraction"][c] = extraction_state(c)

n_done = 0
for c in CACHES:
    got, files = [], {}
    for f in FOLDS:
        p = ROOT / "bv1_runs" / c / f"holdout_{f}_seed1.json"
        if not p.exists() or p.stat().st_size == 0:
            report["missing"].append(f"{c}/{f}")
            continue
        try:
            raw = p.read_bytes()
            d = json.loads(raw)
            metric = d["test"]["positive_patch_macro_iou"]
            valid = (
                d.get("cache") == c
                and d.get("fold") == f"holdout_{f}"
                and d.get("seed") == 1
                and isinstance(metric, (int, float))
                and math.isfinite(float(metric))
                and 0.0 <= float(metric) <= 1.0
                and isinstance(d.get("emb_shape"), list)
                and len(d["emb_shape"]) == 3
            )
            if not valid:
                raise ValueError("cache/fold/seed/metric/emb_shape contract mismatch")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            report["invalid"].append({"file": f"{c}/{f}", "reason": str(exc)})
            continue
        got.append(f)
        n_done += 1
        files[f] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "positive_patch_macro_iou": float(metric),
        }
    report["decoders"][c] = {
        "done": len(got), "of": len(FOLDS), "folds": got, "files": files,
    }

report["summary"] = {
    "extraction_ok": sum(1 for c in CACHES if report["extraction"][c]["state"] == "ok"),
    "extraction_of": len(CACHES),
    "extraction_bad": [c for c in CACHES
                       if report["extraction"][c]["state"] in ("gate_fail", "crashed")],
    "decoder_done": n_done, "decoder_of": len(CACHES) * len(FOLDS),
    "decoder_invalid": len(report["invalid"]),
    "chain_alive": None,
}
done_marker = (ROOT / "logs/arch_axes.log")
report["summary"]["ARCH_AXES_DONE"] = (done_marker.exists()
                                       and "ARCH_AXES_DONE" in done_marker.read_text())
report["note"] = ("logs/arch_axes.log의 rc= 값은 date의 종료코드다. 이 보고서는 rc를 "
                  "무시하고 게이트 JSON과 holdout_*.json 실물만 센다.")
(ROOT / "logs/arch_axes_verify.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
s = report["summary"]
print(f"추출 {s['extraction_ok']}/{s['extraction_of']} ok, 문제 {s['extraction_bad']}")
print(f"디코더 {s['decoder_done']}/{s['decoder_of']}  DONE마커={s['ARCH_AXES_DONE']}")
for c in CACHES:
    e = report["extraction"][c]["state"]; d = report["decoders"][c]
    print(f"  {c:20s} 추출={e:12s} 디코더={d['done']}/{d['of']}")
if report["missing"]:
    print(f"미완 {len(report['missing'])}건 (앞 8): {report['missing'][:8]}")
if report["invalid"]:
    print(f"무효 {len(report['invalid'])}건 (앞 8): {report['invalid'][:8]}")
