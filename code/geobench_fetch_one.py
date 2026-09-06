"""GEO-Bench-2 데이터셋 하나를 내려받는다. 실패하면 0이 아닌 코드로 종료한다."""
import argparse, sys
from pathlib import Path

DATASETS = {
    "pastis":           ("geobench_v2.datasets.pastis", "GeoBenchPASTIS"),
    "fotw":             ("geobench_v2.datasets.fotw", "GeoBenchFieldsOfTheWorld"),
    "dynamic_earthnet": ("geobench_v2.datasets.dynamic_earthnet", "GeoBenchDynamicEarthNet"),
    "benv2":            ("geobench_v2.datasets.benv2", "GeoBenchBENV2"),            # classification
    "biomassters":      ("geobench_v2.datasets.biomassters", "GeoBenchBioMassters"),  # regression
    "cloudsen12":       ("geobench_v2.datasets.cloudsen12", "GeoBenchCloudSen12"),  # seg (S1+S2)
    "treesatai":        ("geobench_v2.datasets.treesatai", "GeoBenchTreeSatAI"),    # classification (time series)
}

# 각 클래스가 요구하는 기본 밴드 순서. band_order 는 필수 인자다.
DEFAULT_BAND_ORDER = {
    "pastis":           {"s2": ["B04", "B03", "B02"]},
    "fotw":             {"s2": ["B04", "B03", "B02"]},
    "dynamic_earthnet": {"s2": ["B04", "B03", "B02"]},
}


def resolve(name):
    import importlib, inspect
    mod_name, cls_name = DATASETS[name]
    mod = importlib.import_module(mod_name)
    if cls_name and hasattr(mod, cls_name):
        return getattr(mod, cls_name)
    # 베이스 클래스는 url 이 비어 있고 paths 가 없다 — 구체 클래스만 고른다
    cands = [o for n, o in vars(mod).items()
             if inspect.isclass(o) and n.startswith("GeoBench")
             and getattr(o, "url", "") and getattr(o, "paths", [])]
    if not cands:
        raise SystemExit(f"{name}: GeoBench* 데이터셋 클래스를 못 찾음")
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    cls = resolve(a.dataset)
    root = Path(a.root) / a.dataset
    root.mkdir(parents=True, exist_ok=True)
    print(f"{a.dataset}: {cls.__name__} -> {root}", flush=True)
    print(f"  url={getattr(cls,'url','?')}", flush=True)
    print(f"  paths={getattr(cls,'paths','?')}", flush=True)
    # download=True 로 생성하면 내려받고 sha256 을 검사한다.
    # band_order 는 필수 인자. 하드코딩 dict 우선, 없으면 클래스에서 유도(8개 데이터셋 견고).
    if a.dataset in DEFAULT_BAND_ORDER:
        bo = {k: list(v) for k, v in DEFAULT_BAND_ORDER[a.dataset].items()}
    else:
        bdo = getattr(cls, "band_default_order", None)
        if isinstance(bdo, dict):
            bo = {k: list(v) for k, v in bdo.items()}
        elif bdo is not None:
            bo = list(bdo)
        else:
            raise SystemExit(f"{a.dataset}: band_default_order 없음")
    ds = cls(root=root, split=a.split, band_order=bo, download=True)
    print(f"  OK len={len(ds)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
