#!/usr/bin/env python3
"""Read-only probe rendering. Creates its own directory; never edits probe receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(args):
    # Matplotlib can create cache files: keep them in this renderer's output only.
    os.environ["MPLCONFIGDIR"] = str(args.out / "matplotlib_cache")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.monotonic()
    manifest = json.loads(args.manifest.read_text())
    cases = manifest["candidates"]
    if len(cases) != 20 or len({c["case_id"] for c in cases}) != 20:
        raise ValueError("Expected original 20 distinct candidates")
    status_path = args.result_dir / "status.json"
    metrics_path = args.result_dir / "scene_metrics.json"
    if json.loads(status_path.read_text()).get("status") != "completed":
        raise ValueError("Probe is not completed; do not render partial results")
    input_hashes = {str(p): sha(p) for p in (args.manifest, status_path, metrics_path)}
    rows = json.loads(metrics_path.read_text())
    chosen = cases[:4]  # The manifest was selected before inference, not from scores.
    fig, axes = plt.subplots(4, 2, figsize=(8.6, 12.8), constrained_layout=True)
    previews = []
    ratio_artist = None
    for r, c in enumerate(chosen):
        meta, receipt = c["original_patch_metadata"], c["raw_image"]
        rel = Path(meta["image_path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe image path")
        p = (args.data_dir / rel).resolve(strict=True)
        file_hash = sha(p)
        if file_hash != receipt["expected_npy_file_sha256"]:
            raise ValueError(f"Raw file SHA mismatch: {c['case_id']}")
        raw = np.load(p, allow_pickle=False)
        if list(raw.shape) != [12, 120, 120] or raw.dtype != np.uint16:
            raise ValueError("Unexpected raw array")
        if hashlib.sha256(raw.tobytes()).hexdigest() != receipt["expected_array_sha256"]:
            raise ValueError("Raw array SHA mismatch")
        bands = meta["bands"]
        if len(bands) != 12 or len(set(bands)) != 12:
            raise ValueError("Invalid band inventory")
        rgb = np.zeros((120, 120, 3), dtype=np.float64)
        stretches = {}
        for channel, band in enumerate(("B04", "B03", "B02")):
            signal = raw[bands.index(band)].astype(np.float64)
            valid = np.isfinite(signal) & (signal < 65535)
            if not valid.any():
                raise ValueError("RGB channel has no displayable pixels")
            lo, hi = np.percentile(signal[valid], [2, 98])
            if hi > lo:
                rgb[..., channel] = np.clip((signal - lo) / (hi - lo), 0, 1)
            rgb[..., channel][~valid] = 0
            stretches[band] = {"p02_DN": float(lo), "p98_DN": float(hi), "constant_channel": bool(hi <= lo)}
        red = raw[bands.index("B04")].astype(np.float64)
        nir = raw[bands.index("B08")].astype(np.float64)
        valid = np.isfinite(red) & np.isfinite(nir) & (red < 65535) & (nir < 65535) & ((red + nir) > 0)
        ratio = np.zeros_like(red)
        np.divide(nir - red, nir + red, out=ratio, where=valid)
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("#999999")
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"{meta['mgrs']} | {meta['timestamp_utc'][:10]}\nRGB: B04/B03/B02, per-band 2–98% stretch", fontsize=9)
        ratio_artist = axes[r, 1].imshow(np.ma.array(ratio, mask=~valid), vmin=-1, vmax=1, cmap=cmap)
        axes[r, 1].set_title("DN ratio (B08 − B04) / (B08 + B04)\nNot calibrated vegetation/health ground truth", fontsize=9)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
        previews.append({"case_id": c["case_id"], "mgrs": meta["mgrs"], "raw_path": str(p),
                         "file_sha256": file_hash, "array_sha256": receipt["expected_array_sha256"],
                         "rgb_stretch": stretches, "valid_ratio_pixels": int(valid.sum()),
                         "invalid_ratio_pixels": int((~valid).sum())})
    fig.colorbar(ratio_artist, ax=list(axes[:, 1]), fraction=.025, pad=.02, label="DN ratio")
    fig.suptitle("First four preselected train cases — display audit only", fontsize=13)
    preview_path = args.out / "first4_rgb_and_dn_ratio.png"
    fig.savefig(preview_path, dpi=135)
    plt.close(fig)

    methods = ("full", "pool4_nearest", "pool8_nearest", "coordinates", "train_mean")
    labels = ("Full spatial\nbandsets averaged", "16 spatial\nnearest restore", "64 spatial\nnearest restore", "Coordinates\nonly", "Train-scene\nmean")
    ids = [c["case_id"] for c in cases]
    lookup = {}
    for row in rows:
        key = (row["case_id"], row["method"])
        if key in lookup:
            raise ValueError("Duplicate scene/method metric")
        if not np.isfinite(row["mae"]) or row["mae"] < 0:
            raise ValueError("Invalid MAE")
        lookup[key] = row
    if set(lookup) != {(i, m) for i in ids for m in methods}:
        raise ValueError("Metrics do not cover exactly 20 scenes × 5 methods")
    for i in ids:
        if len({lookup[(i,m)]["valid_cells"] for m in methods}) != 1:
            raise ValueError("Method valid-cell support differs")
        if len({lookup[(i,m)]["fold"] for m in methods}) != 1:
            raise ValueError("Method scene-fold assignment differs")
    matrix = np.array([[lookup[(i,m)]["mae"] for m in methods] for i in ids])
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    for values in matrix:
        ax.plot(range(5), values, color="#a7a7a7", alpha=.5, linewidth=.7, marker="o", markersize=2)
    means = matrix.mean(axis=0)
    ax.plot(range(5), means, color="#154c79", linewidth=2.5, marker="o", markersize=7, label="Mean of 20 scene MAEs")
    for i, value in enumerate(means):
        ax.annotate(f"{value:.4f}", (i,value), xytext=(0,9), textcoords="offset points", ha="center", fontsize=9, color="#154c79")
    ax.set_xticks(range(5), labels)
    ax.set_ylabel("Held-out scene MAE of DN-derived ratio")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    ax.set_title("Paired descriptive audit: each gray line is one scene\n20 training-source scenes; no significance or semantic-capability claim")
    plot_path = args.out / "paired_scene_mae.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    if any(sha(Path(p)) != expected for p,expected in input_hashes.items()):
        raise RuntimeError("Probe inputs changed during rendering")
    receipt = {"status":"rendered", "selection":"first four manifest entries, fixed before model outcomes", "raw_files_read":4,
               "preview_cases":previews, "input_sha256":input_hashes, "renderer_sha256":sha(__file__),
               "outputs":{p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in (preview_path,plot_path)},
               "plot_methods":list(methods), "mean_scene_mae":dict(zip(methods,map(float,means))),
               "limitations":["RGB is a per-band percentile-stretched display, not a calibrated true-color product.",
                              "DN ratio is shown at source 120×120 resolution; regression targets used averages over native token cells.",
                              "Gray invalid ratio pixels are zero-denominator or uint16-saturated, not an independently verified cloud/quality mask.",
                              "Neither plot is semantic ground truth, a causal pooling-bottleneck proof, or a significance test.",
                              "Only four preselected train scenes are previewed; all 20 held-out-fold scene metrics are plotted.",
                              "Distinct MGRS is not proof of regional independence; no physical area is inferred from the inconsistent TIFF affine."],
               "python":sys.version,"numpy":np.__version__,"matplotlib":matplotlib.__version__,"elapsed_seconds":time.monotonic()-started}
    write_json(args.out / "render_receipt.json", receipt)
    print(json.dumps({"status":"rendered","out":str(args.out),"case_ids":[c["case_id"] for c in chosen]}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "data-dir", "result-dir", "out"):
        ap.add_argument("--"+name, type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        write_json(args.out / "render_failure.json", {"status":"failed", "error":repr(exc), "traceback":traceback.format_exc(),
                                                      "probe_outputs_modified":False})
        raise


if __name__ == "__main__":
    main()
