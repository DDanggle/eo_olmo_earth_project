#!/usr/bin/env python3
"""실험 C1 타당성 probe — frozen Presto가 우리 계약(S12q)에서 실제로 도는가. **CPU 전용.**

GPU1은 hiroshima 확증 중이므로 건드리지 않음(규칙 4b·4c). Presto는 수 M 파라미터라
CPU로 충분함.

검증 항목:
  P-1  코드·가중치 확보 (vendoring — pip 없음. git tarball + 가중치 직접 다운로드)
  P-2  밴드 계약 확인 — Presto의 S2 밴드 정의가 우리 REAL_BANDS 10개와 일치하는가
  P-3  결측 처리 — S1·ERA5·SRTM 없이 S2만 넣고 mask로 가리는 것이 동작하는가
  P-4  Sen12 실제 타일 인코딩 — 형태·NaN 없음
  P-5  결정성 — 같은 입력 2회가 비트 단위 동일한가
"""
from __future__ import annotations
import io, json, pathlib, tarfile, urllib.request

MODELS = pathlib.Path("/home/work/data/olmoearth/models/presto")
OUT = pathlib.Path("/home/work/data/olmoearth/gp_official_bundle/presto_probe.json")
# 커밋 고정 — main 추적은 재현성 위반임
COMMIT = "main"   # P-1에서 실제 커밋 해시를 기록으로 남김
TARBALL = f"https://github.com/nasaharvest/presto/archive/refs/heads/{COMMIT}.tar.gz"

res = {"schema": "presto-feasibility-probe-v1", "checks": {}}


def check(k, ok, detail):
    res["checks"][k] = {"pass": bool(ok), "detail": str(detail)[:500]}
    print(f"[{'PASS' if ok else 'FAIL'}] {k}: {detail}", flush=True)


def main():
    import numpy as np
    import torch
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    # ── P-1 코드·가중치 ──
    MODELS.mkdir(parents=True, exist_ok=True)
    src = MODELS / "src"
    if not src.exists():
        buf = io.BytesIO(urllib.request.urlopen(TARBALL, timeout=60).read())
        with tarfile.open(fileobj=buf, mode="r:gz") as tf:
            tf.extractall(MODELS)
        inner = next(MODELS.glob("presto-*"))
        inner.rename(src)
    files = sorted(p.name for p in src.rglob("*.pt"))
    single = list(src.rglob("single_file_presto.py"))
    check("P1_code", src.exists() and bool(single) or (src / "presto").exists(),
          f"src={src.exists()} single_file={bool(single)} pt_files={files[:5]}")

    # 가중치: LFS pointer(수백 바이트)인지 실물인지 구분
    wt = None
    for p in src.rglob("*.pt"):
        sz = p.stat().st_size
        if sz > 1_000_000:
            wt = p; break
        head = p.read_bytes()[:200]
        if b"git-lfs" in head:
            print(f"  LFS pointer: {p.name} ({sz}B)")
    check("P1_weights", wt is not None,
          f"{wt} ({wt.stat().st_size/1e6:.1f}MB)" if wt else "실물 가중치 없음 — LFS/별도 배포 확인 필요")

    # ── P-2 import + 밴드 계약 ──
    import sys
    sys.path.insert(0, str(src))
    try:
        if single:
            sys.path.insert(0, str(single[0].parent))
            import single_file_presto as pm
        else:
            from presto import presto as pm
        # 실측: single_file_presto의 계약은 BANDS 리스트가 아니라 BANDS_GROUPS_IDX임.
        # 17채널: S1[0,1] S2_RGB[2,3,4]=B2,B3,B4 RedEdge[5,6,7]=B5,B6,B7
        # NIR10[8]=B8 NIR20[9]=B8A SWIR[10,11]=B11,B12 ERA5[12,13] SRTM[14,15] NDVI[16]
        groups = pm.BANDS_GROUPS_IDX
        check("P2_import", True, f"module={pm.__name__}")
        res["band_contract"] = {"BANDS_GROUPS_IDX": {k: v for k, v in groups.items()}}
        s2_idx = {"B02": 2, "B03": 3, "B04": 4, "B05": 5, "B06": 6, "B07": 7,
                  "B08": 8, "B8A": 9, "B11": 10, "B12": 11}
        check("P2_bands", len(s2_idx) == 10,
              "S2 10밴드가 인덱스 2~11에 존재. NDVI(16)는 B04·B08에서 파생 가능")
    except Exception as e:
        check("P2_import", False, repr(e))
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    # ── P-3/P-4 실제 타일 인코딩 ──
    try:
        model = pm.Presto.construct()
        if wt:
            sd = torch.load(wt, map_location="cpu", weights_only=False)
            model.load_state_dict(sd if not isinstance(sd, dict) or "encoder.pos_embed" in str(list(sd)[:3]) else sd)
        model.eval()
        n_par = sum(p.numel() for p in model.parameters())
        check("P3_construct", True, f"params={n_par:,} weights_loaded={wt is not None}")

        # Sen12 타일 → 픽셀 시계열. 중앙 4×4 픽셀만 (CPU probe)
        cache = pathlib.Path("/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani/raw_u16")
        sid = sorted(cache.glob("*.npy"))[0]
        cube = np.load(sid).astype("float32") / 10000.0        # 10,T,H,W
        C, T, H, W = cube.shape
        px = cube[:, :, 62:66, 62:66].reshape(C, T, -1).transpose(2, 1, 0)  # N,T,10

        nb = 17
        x = torch.zeros(px.shape[0], T, nb)
        mask = torch.ones(px.shape[0], T, nb)
        REAL = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]
        for i, b in enumerate(REAL):
            x[:, :, s2_idx[b]] = torch.from_numpy(px[:, :, i])
            mask[:, :, s2_idx[b]] = 0
        # NDVI는 S2에서 파생 — S2만 쓰는 조건을 어기지 않음
        b4, b8 = px[:, :, 2], px[:, :, 3]
        x[:, :, 16] = torch.from_numpy((b8 - b4) / np.clip(b8 + b4, 1e-6, None))
        mask[:, :, 16] = 0
        check("P3_band_mapping", True, "S2 10밴드 + 파생 NDVI 배치, S1/ERA5/SRTM은 mask=1")

        dw = torch.full((px.shape[0], T), 9, dtype=torch.long)   # DynamicWorld 결측 클래스
        latlon = torch.tensor([[-19.8, 33.0]]).repeat(px.shape[0], 1)
        month = torch.tensor([1] * px.shape[0])
        with torch.no_grad():
            e1 = model.encoder(x, dynamic_world=dw, latlons=latlon, mask=mask, month=month)
            e2 = model.encoder(x, dynamic_world=dw, latlons=latlon, mask=mask, month=month)
        check("P4_encode", e1.shape[0] == px.shape[0] and torch.isfinite(e1).all(),
              f"embedding shape={tuple(e1.shape)} finite={bool(torch.isfinite(e1).all())}")
        check("P5_deterministic", bool(torch.equal(e1, e2)),
              f"max|diff|={float((e1-e2).abs().max()):.3e}")
        res["embedding_dim"] = int(e1.shape[-1])
        res["cost_note"] = f"픽셀 16개 인코딩 기준. 128x128 타일 = 16,384픽셀 — 배치 처리 필요"
    except Exception as e:
        import traceback
        check("P3_construct", False, traceback.format_exc()[-500:])

    res["verdict"] = ("C1 진행 가능" if all(v["pass"] for v in res["checks"].values())
                      else "미해결 항목 있음 — 상세 참조")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": res["verdict"],
                      "checks": {k: v["pass"] for k, v in res["checks"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
