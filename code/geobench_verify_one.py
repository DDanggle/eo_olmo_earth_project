"""내려받은 GEO-Bench-2 데이터셋이 우리 계약에 맞는지 검증한다.

검사: 세 split 이 열리는가 / 샘플 텐서 모양·dtype / 요청한 밴드 순서가 지켜지는가 /
      split 간 크기가 0이 아닌가. 하나라도 실패하면 exit 1 (체인이 멈춘다).
"""
import argparse, json, sys
from pathlib import Path
from geobench_fetch_one import resolve

# 우리 Sen12/한국 캐시와 같은 10밴드 (extract_sen12_fold_cache.py MODEL_BANDS[:10] 순서)
OUR_S2 = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]

FAILS = []
def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + str(detail) if detail else ""))
    if not cond:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("--root", required=True)
    a = ap.parse_args()
    cls = resolve(a.dataset)
    root = Path(a.root) / a.dataset
    report = {"dataset": a.dataset, "class": cls.__name__, "splits": {}}

    # band_default_order 는 dict(모달리티별) 이거나 tuple(단일 모달리티) 이다.
    # dict 면 우리 10밴드 중 그 데이터셋이 가진 것만 요청하고, tuple 이면 그대로 쓴다.
    bdo = getattr(cls, "band_default_order", None)
    if isinstance(bdo, dict):
        band_order = {}
        for mod, bands in bdo.items():
            if mod == "s2":
                avail = list(bands)
                band_order[mod] = [b for b in OUR_S2 if b in avail] or avail
            else:
                band_order[mod] = list(bands)
        band_order = {"s2": band_order["s2"]} if "s2" in band_order else band_order
    else:
        band_order = list(bdo)          # 단일 모달리티 (예: fotw = red,green,blue,nir)
    report["requested_band_order"] = band_order
    for split in ("train", "val", "test"):
        try:
            kw = dict(root=root, split=split, download=False, band_order=band_order)
            ds = cls(**kw)
            n = len(ds)
            report["splits"][split] = {"n": n}
            check(f"{split} 열림 n>0", n > 0, n)
            if n:
                s = ds[0]
                shapes = {k: tuple(v.shape) for k, v in s.items() if hasattr(v, "shape")}
                report["splits"][split]["sample_shapes"] = {k: list(v) for k, v in shapes.items()}
                check(f"{split} 샘플에 텐서 존재", bool(shapes), shapes)
        except Exception as e:
            report["splits"][split] = {"error": f"{type(e).__name__}: {e}"}
            check(f"{split} 열림", False, f"{type(e).__name__}: {str(e)[:120]}")

    ns = [v.get("n", 0) for v in report["splits"].values()]
    check("세 split 모두 비어있지 않음", len(ns) == 3 and all(n > 0 for n in ns), ns)
    if isinstance(bdo, dict) and "s2" in bdo:
        want = [b for b in OUR_S2 if b in list(bdo["s2"])]
        check(f"우리 밴드 {len(want)}/10 로 열림", len(want) >= 10 and not FAILS, want)
    else:
        check("단일 모달리티 데이터셋 (다중밴드 아님 — 계약 주의)", True, list(bdo))

    report["ok"] = not FAILS
    report["failed_checks"] = FAILS
    out = Path("/home/work/data/olmoearth/artifacts") / f"geobench_verify_{a.dataset}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "splits"}, ensure_ascii=False))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
