#!/usr/bin/env python3
"""v1 vs v1.2 토큰화 계약 확인 — M1의 R@1=0이 자작인지 판정하는 C1의 핵심.

배경:
- rslearn `_prepare_modality_inputs`는 `num_band_sets`를 `Modality.get(m).band_sets`로 계산한다.
  sentinel2_l2a는 3개(B02B03B04B08 / B05B06B07B8AB11B12 / B01B09)다. 릴리스와 무관하다.
- 그런데 v1.2 config는 `tokenization_config.overrides.sentinel2_l2a.band_groups`를
  12밴드 **단일 그룹**으로 선언한다.
- 즉 mask는 3-set, 모델은 1-group일 수 있다. 그렇다면 두 릴리스의 토큰 격자가 다르고
  '같은 토큰' 비교가 무의미해진다 → M1이 지표 정의 문제로 강등된다.

확인:
  1) model_path 로딩 시 override가 실제로 적용되는가 (v1 그룹 수 vs v1.2 그룹 수)
  2) 감사 config의 token_pooling 설정 — pooling이 켜져 있으면 band-set 차이가 흡수되고
     공간 patch 단위 비교는 유효하다
  3) 동일 합성 입력에서 두 릴리스의 출력 형태가 같은가
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/home/work/data/olmoearth/release_tokenization_probe")
SNAPS = {
    "v1": "/home/work/data/.cache/huggingface/hub/models--allenai--OlmoEarth-v1-Base/snapshots",
    "v1_2": "/home/work/data/.cache/huggingface/hub/models--allenai--OlmoEarth-v1_2-Base/snapshots",
}


def band_groups(obj, depth: int = 0) -> list | None:
    """로드된 모델에서 sentinel2_l2a band group 구조를 찾는다."""
    if depth > 4:
        return None
    tc = getattr(obj, "tokenization_config", None)
    if tc is not None:
        ov = getattr(tc, "overrides", None) or (tc.get("overrides") if isinstance(tc, dict) else None)
        if ov:
            s2 = ov.get("sentinel2_l2a") if isinstance(ov, dict) else getattr(ov, "sentinel2_l2a", None)
            if s2 is not None:
                bg = s2.get("band_groups") if isinstance(s2, dict) else getattr(s2, "band_groups", None)
                if bg:
                    return [list(g) for g in bg]
    for attr in ("encoder", "model", "backbone"):
        child = getattr(obj, attr, None)
        if child is not None and child is not obj:
            found = band_groups(child, depth + 1)
            if found:
                return found
    return None


def main() -> None:
    import glob

    import torch
    from rslearn.models.olmoearth_pretrain.model import OlmoEarth

    # 패키지 이름이 릴리스에 따라 다르다: rslearn 0.0.x → olmoearth_pretrain,
    # rslearn 0.1.x → olmoearth_pretrain_minimal. 어느 쪽이 쓰이는지도 기록한다.
    pretrain_module, Modality, ModelID = None, None, None
    for candidate in ("olmoearth_pretrain_minimal", "olmoearth_pretrain"):
        try:
            mod = __import__(candidate, fromlist=["*"])
        except ModuleNotFoundError:
            continue
        pretrain_module = candidate
        ModelID = getattr(mod, "ModelID", None)
        for path in (
            f"{candidate}.olmoearth_pretrain_v1.utils.constants",
            f"{candidate}.olmoearth_pretrain_v1.data.constants",
            f"{candidate}.data.constants",
        ):
            try:
                Modality = __import__(path, fromlist=["Modality"]).Modality
                break
            except (ModuleNotFoundError, ImportError):
                continue
        break
    if Modality is None:
        raise SystemExit(f"Modality를 찾지 못했다 (module={pretrain_module})")

    OUT.mkdir(parents=True, exist_ok=True)
    m = Modality.get("sentinel2_l2a")
    result = {
        "schema": "release-tokenization-probe-v1",
        "pretrain_package": pretrain_module,
        "model_ids_available": [x.name for x in ModelID] if ModelID else None,
        "rslearn_modality_band_sets": [list(bs.bands) for bs in m.band_sets],
        "rslearn_num_band_sets": len(m.band_sets),
        "releases": {},
    }

    for tag, snap_root in SNAPS.items():
        snap = sorted(glob.glob(f"{snap_root}/*"))[0]
        cfg = json.loads(Path(snap, "config.json").read_text())
        declared = (
            cfg["model"]["encoder_config"]
            .get("tokenization_config", {})
            .get("overrides", {})
            .get("sentinel2_l2a", {})
            .get("band_groups")
        )
        entry = {
            "snapshot": snap,
            "declared_band_groups_in_config": declared,
            "declared_group_count": len(declared) if declared else None,
        }
        model, how = None, None
        id_name = {"v1": "OLMOEARTH_V1_BASE", "v1_2": "OLMOEARTH_V1_2_BASE"}[tag]
        if ModelID is not None and id_name in [x.name for x in ModelID]:
            try:
                model = OlmoEarth(patch_size=4, model_id=id_name)
                how = f"model_id={id_name}"
            except Exception as exc:  # noqa: BLE001
                entry["model_id_error"] = repr(exc)[:300]
        if model is None:
            try:
                model = OlmoEarth(patch_size=4, model_path=snap)
                how = "model_path"
            except Exception as exc:  # noqa: BLE001
                entry["model_path_error"] = repr(exc)[:300]
                result["releases"][tag] = entry
                continue
        entry["loaded_via"] = how
        entry["loaded_band_groups"] = band_groups(model)
        entry["loaded_group_count"] = (
            len(entry["loaded_band_groups"]) if entry["loaded_band_groups"] else None
        )
        entry["token_pooling_default"] = bool(getattr(model, "token_pooling", None))
        entry["embedding_size"] = getattr(model, "embedding_size", None)
        entry["param_count_millions"] = round(
            sum(p.numel() for p in model.parameters()) / 1e6, 2
        )
        result["releases"][tag] = entry
        del model
        torch.cuda.empty_cache()

    v1, v12 = result["releases"]["v1"], result["releases"]["v1_2"]
    result["verdict"] = [
        f"config 선언 그룹 수: v1={v1['declared_group_count']} v1.2={v12['declared_group_count']}",
        f"로드된 그룹 수: v1={v1['loaded_group_count']} v1.2={v12['loaded_group_count']}",
        f"rslearn이 mask에 쓰는 band_set 수: {result['rslearn_num_band_sets']} (릴리스 무관)",
        f"파라미터: v1={v1['param_count_millions']}M v1.2={v12['param_count_millions']}M",
        f"token_pooling 기본값: v1={v1['token_pooling_default']} v1.2={v12['token_pooling_default']}"
        " — True면 band-set 차이가 patch 단위로 흡수되어 same-token 비교가 유효하다",
    ]
    (OUT / "release_tokenization_probe.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
