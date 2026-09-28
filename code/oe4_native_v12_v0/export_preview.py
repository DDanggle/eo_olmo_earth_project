#!/usr/bin/env python3
"""Export a CPU-only raw-observation gallery from an OE4 data manifest.

The first two train and two dev entries are shown by default. No model is loaded,
no examples are ranked by model loss, and no cloud/change interpretation is made.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import textwrap


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_official_palette(path):
    """Read the constant only; importing visualize.py would require cartopy."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "WORLDCOVER_LEGEND" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Official WORLDCOVER_LEGEND literal not found")


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def render_pillow(rows, footer_lines, legend_items, out_path):
    from PIL import Image, ImageDraw, ImageFont

    def font(size):
        paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"]
        for path in paths:
            if Path(path).exists():
                return ImageFont.truetype(path, size)
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    panel = 272
    columns = max(len(r["panels"]) for r in rows)
    width = 40 + columns * (panel + 24)
    row_height = panel + 136
    header_height = 100
    legend_rows = (len(legend_items) + 1) // 2
    footer_height = 24 * len(footer_lines) + 28 * legend_rows + 32
    image = Image.new("RGB", (width, header_height + len(rows) * row_height + footer_height), "#f6f7f9")
    draw = ImageDraw.Draw(image)
    heading_font, label_font, small_font = font(25), font(17), font(15)
    draw.text((20, 15), "OE4 | actual observation sanity gallery", fill="#17212f", font=heading_font)
    draw.text((20, 50), "First manifest train/dev entries | same input crops | no model predictions", fill="#304155", font=label_font)
    draw.text((20, 75), "RGB = B04/B03/B02, fixed linear DN 0..3000; WorldCover = raw static codes", fill="#304155", font=small_font)
    for i, row in enumerate(rows):
        y = header_height + i * row_height
        draw.text((20, y + 2), row["title"], fill="#17212f", font=label_font)
        draw.text((20, y + 29), row["subtitle"], fill="#445367", font=small_font)
        for j, p in enumerate(row["panels"]):
            x = 20 + j * (panel + 24)
            draw.text((x, y + 56), p["label"], fill="#17212f", font=small_font)
            draw.text((x, y + 78), p.get("detail", ""), fill="#304155", font=small_font)
            tile = Image.fromarray(p["rgb"], "RGB").resize((panel, panel), resample=Image.Resampling.NEAREST)
            image.paste(tile, (x, y + 106))
            draw.rectangle((x, y + 106, x + panel - 1, y + 106 + panel - 1), outline="#ccd3dc")
    base_y = header_height + len(rows) * row_height
    for i, line in enumerate(footer_lines):
        draw.text((20, base_y + i * 24), line, fill="#304155", font=small_font)
    legend_y = base_y + len(footer_lines) * 24 + 4
    for i, item in enumerate(legend_items):
        x = 20 + (i % 2) * ((width - 40) // 2)
        y = legend_y + (i // 2) * 28
        draw.rectangle((x, y, x + 18, y + 18), fill=item["color"], outline="#8b97a6")
        draw.text((x + 26, y), item["label"], fill="#304155", font=small_font)
    image.save(out_path)
    return {"renderer": "Pillow", "version": package_version("Pillow"), "size_pixels": list(image.size)}


def render_matplotlib(rows, footer_lines, legend_items, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    columns = max(len(r["panels"]) for r in rows)
    legend_rows = (len(legend_items) + 2) // 3
    footer_inches = .16 * len(footer_lines) + .23 * legend_rows + .3
    total_height = 4.0 * len(rows) + footer_inches + 1.0
    fig, axes = plt.subplots(len(rows), columns, figsize=(4.0 * columns, total_height), squeeze=False)
    for i, row in enumerate(rows):
        for j in range(columns):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(row["panels"]):
                p = row["panels"][j]
                ax.imshow(p["rgb"], interpolation="nearest")
                ax.set_title(p["label"] + "\n" + p.get("detail", ""), fontsize=10)
        axes[i, 0].text(0, 1.20, row["title"], transform=axes[i, 0].transAxes, fontsize=10)
        axes[i, 0].text(0, 1.12, row["subtitle"], transform=axes[i, 0].transAxes, fontsize=8)
    fig.suptitle("OE4 | actual observation sanity gallery\nRGB B04/B03/B02, fixed linear DN 0..3000; WorldCover raw static codes", fontsize=14)
    fig.text(.02, (.23 * legend_rows + .15) / total_height, "\n".join(footer_lines), fontsize=8, va="bottom")
    patches = [mpatches.Patch(facecolor=item["color"], edgecolor="#8b97a6", label=item["label"]) for item in legend_items]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.subplots_adjust(top=.9, bottom=footer_inches / total_height, hspace=.55, wspace=.04)
    fig.savefig(out_path, dpi=130, facecolor="#f6f7f9")
    size = [int(v * 130) for v in fig.get_size_inches()]
    plt.close(fig)
    return {"renderer": "matplotlib Agg", "version": package_version("matplotlib"), "size_pixels": size}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--deps-root", type=Path)
    p.add_argument("--out", type=Path, required=True, help="New output directory for gallery.png and input_provenance.json")
    p.add_argument("--per-split", type=int, choices=[2, 3], default=2, help="First 2 (4 total) or 3 (6 total) entries per split")
    args = p.parse_args()
    if args.deps_root:
        sys.path.insert(0, str(args.deps_root.resolve()))
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    import h5py
    import hdf5plugin  # noqa: F401 -- register official H5 compression filters.
    import olmoearth_pretrain
    from olmoearth_pretrain.data.constants import Modality, MISSING_VALUE
    from olmoearth_pretrain.data.normalize import Normalizer, Strategy
    if not Path(olmoearth_pretrain.__file__).resolve().is_relative_to(args.source_root.resolve()):
        raise RuntimeError("Imported model package does not come from --source-root")
    manifest = json.loads(args.manifest.read_text())
    selected = []
    for split in ["train_diagnostic", "dev_diagnostic"]:
        entries = [r for r in manifest["selected"] if r["split"] == split][:args.per_split]
        if len(entries) != args.per_split:
            raise ValueError(f"Manifest has fewer than {args.per_split} {split} entries")
        selected.extend(entries)
    palette_path = args.source_root / "olmoearth_pretrain/data/visualize.py"
    palette = read_official_palette(palette_path)
    s2_order = Modality.SENTINEL2_L2A.band_order
    rgb_names = ["B04", "B03", "B02"]
    rgb_indices = [s2_order.index(name) for name in rgb_names]
    normalizers = {"computed": Normalizer(Strategy.COMPUTED), "predefined": Normalizer(Strategy.PREDEFINED)}
    rows, records, observed_codes = [], [], set()

    def norm_summary(name, raw, strategy):
        norm = normalizers[strategy].normalize(Modality.get(name), raw)
        norm = np.where(raw == MISSING_VALUE, MISSING_VALUE, norm).astype(np.float32)
        return {"strategy": strategy, "shape": list(norm.shape), "dtype": str(norm.dtype),
                "min": float(norm.min()), "max": float(norm.max()), "mean": float(norm.mean()),
                "sha256_float32_values": hashlib.sha256(norm.tobytes()).hexdigest()}

    for entry in selected:
        path = Path(entry["file"])
        y, x, h, w = entry["crop_yxhw"]
        compact = entry["compact_timesteps"]
        timestamp_indices = entry["timestamp_indices"]
        with h5py.File(path, "r") as f:
            s2 = f["sentinel2_l2a"]
            if list(s2.shape) != entry["source_s2_shape"]:
                raise ValueError(f"Manifest/source S2 shape mismatch: {path}")
            raw_s2 = np.stack([np.asarray(s2[y:y+h, x:x+w, t, :], dtype=np.float32) for t in compact], axis=2)
            raw_wc = np.asarray(f["worldcover"][y:y+h, x:x+w], dtype=np.float32)
            if raw_wc.ndim == 3 and raw_wc.shape[-1] == 1:
                raw_wc = raw_wc[:, :, None, :]
            if raw_wc.shape != (h, w, 1, 1):
                raise ValueError(f"Invalid static WorldCover crop shape: {raw_wc.shape}")
            timestamps = np.asarray(f["timestamps"][()])[timestamp_indices].astype(np.int64)
            if timestamps.tolist() != entry["timestamps_day_month0_year"]:
                raise ValueError(f"Manifest/source timestamp mismatch: {path}")
            presence_key = "missing_timesteps_masks/sentinel2_l2a"
            if presence_key in f:
                present = np.flatnonzero(np.asarray(f[presence_key][()]))
                if present[compact].tolist() != timestamp_indices:
                    raise ValueError(f"Compact timestamp alignment mismatch: {path}")
            actual_hash = hashlib.sha256(raw_s2.tobytes() + raw_wc.tobytes() + timestamps.tobytes()).hexdigest()
            if actual_hash != entry["raw_crop_sha256"]:
                raise ValueError(f"Raw crop digest differs from training manifest: {path}")
            if not np.isfinite(raw_s2).all() or not np.isfinite(raw_wc).all():
                raise ValueError(f"Nonfinite selected raw crop: {path}")
            if np.any(raw_s2 == MISSING_VALUE) or np.any(raw_wc == MISSING_VALUE):
                raise ValueError("This first gallery expects the fully observed-crop runtime contract")
            panels, radiometry = [], []
            for k, t in enumerate(compact):
                raw_rgb = raw_s2[:, :, k, rgb_indices]
                # Advanced indexing may prepend the channel axis: normalize it
                # explicitly and fail rather than silently showing wrong axes.
                if raw_rgb.shape == (3, h, w):
                    raw_rgb = np.moveaxis(raw_rgb, 0, -1)
                if raw_rgb.shape != (h, w, 3):
                    raise ValueError(f"Unexpected RGB projection shape {raw_rgb.shape}")
                rgb = np.rint(np.clip(raw_rgb / 3000.0, 0, 1) * 255).astype(np.uint8)
                day, month0, year = timestamps[k]
                label = f"S2 {year:04d}-{month0+1:02d}-{day:02d} | observed t={t}"
                panels.append({"label": label, "detail": "Raw DN 0..3000, fixed stretch", "rgb": rgb})
                radiometry.append({"compact_timestep": t, "timestamp": timestamps[k].tolist(),
                                   "rgb_dn_min": raw_rgb.min(axis=(0, 1)).tolist(), "rgb_dn_max": raw_rgb.max(axis=(0, 1)).tolist(),
                                   "display_fraction_values_below_0": float(np.mean(raw_rgb < 0)),
                                   "display_fraction_values_above_3000": float(np.mean(raw_rgb > 3000))})
            codes = raw_wc[:, :, 0, 0]
            wc_rgb = np.full((h, w, 3), [255, 0, 255], dtype=np.uint8)
            counts = {}
            for code in np.unique(codes):
                if float(code) != int(code):
                    raise ValueError(f"Nonintegral WorldCover code {code}")
                code = int(code)
                observed_codes.add(code)
                counts[str(code)] = int((codes == code).sum())
                if code in palette:
                    color_hex, _ = palette[code]
                    wc_rgb[codes == code] = [int(color_hex[i:i+2], 16) for i in (1, 3, 5)]
            dominant_code = max(counts, key=counts.get)
            dominant_name = palette.get(int(dominant_code), ("#ff00ff", "UNKNOWN"))[1]
            unknown_pixels = sum(n for code, n in counts.items() if int(code) not in palette)
            panels.append({"label": f"WorldCover | unknown pixels: {unknown_pixels}",
                           "detail": f"{dominant_code} {dominant_name}: {100*counts[dominant_code]/codes.size:.1f}%", "rgb": wc_rgb})
            title = f"{entry['split']} | {path.name}"
            subtitle = f"Same array crop y={y}, x={x}, h={h}, w={w}; nearest-neighbor display; no reprojection"
            rows.append({"title": title, "subtitle": subtitle, "panels": panels})
            records.append({"manifest_entry": entry, "verified_raw_crop_sha256": actual_hash,
                            "radiometry": radiometry, "worldcover_pixel_counts": counts,
                            "worldcover_unknown_codes": [int(c) for c in np.unique(codes) if int(c) not in palette],
                            "model_input_normalization": {name: norm_summary(name, raw, entry["normalization"][name])
                                                          for name, raw in [("sentinel2_l2a", raw_s2), ("worldcover", raw_wc)]},
                            "h5_root_attribute_names": sorted(f.attrs.keys()),
                            "h5_keys": sorted(f.keys())})
    legend_items = [{"label": f"{code} {palette[code][1]}", "color": palette[code][0]} for code in sorted(observed_codes) if code in palette]
    unknown_total = sum(sum(r["worldcover_pixel_counts"].get(str(k), 0) for k in r["worldcover_unknown_codes"]) for r in records)
    legend_items.append({"label": f"Unknown code (#ff00ff): {unknown_total} pixels", "color": "#ff00ff"})
    footer = ["Display pixels are enlarged without interpolation. Fixed RGB stretch may clip values; exact fractions are in provenance.",
              "Model inputs use official computed/predefined normalization; this gallery shows raw DN, not those normalized tensors.",
              "Array orientation is retained. CRS/geotransform and WorldCover acquisition date are not inferred.",
              "These file-disjoint examples are a visual sanity check; no cloud labels, change conclusions or geographic-independence claim."]
    footer.append("WorldCover class 40 Cropland uses pale purple (#f096ff). Unknown codes use bright magenta (#ff00ff).")
    footer.append("WorldCover is a static target with an unspecified acquisition date. See exact code counts and normalization in provenance.")
    footer = [line for paragraph in footer for line in textwrap.wrap(paragraph, width=108)]
    args.out.mkdir(parents=True, exist_ok=False)
    image_path = args.out / "gallery.png"
    try:
        import PIL  # noqa: F401
        render = render_pillow(rows, footer, legend_items, image_path)
    except ModuleNotFoundError as error:
        if error.name not in ["PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont"]:
            raise
        render = render_matplotlib(rows, footer, legend_items, image_path)
    provenance = {
        "status": "raw_observation_gallery_exported_no_model_run",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
        "script_sha256": sha256(__file__), "source_root": str(args.source_root.resolve()),
        "source_palette_file_sha256": sha256(palette_path),
        "source_normalization_files": {name: sha256(args.source_root / "olmoearth_pretrain/data/norm_configs" / name)
                                       for name in ["computed.json", "predefined.json"]},
        "selection_rule": f"first {args.per_split} train_diagnostic + first {args.per_split} dev_diagnostic entries in existing manifest order",
        "rgb_projection": {"band_names": rgb_names, "zero_based_indices": rgb_indices, "source_band_order": s2_order},
        "rgb_display": {"formula": "round(255 * clip(raw_DN / 3000, 0, 1))", "gamma": 1,
                        "per_image_auto_stretch": False, "radiometric_offset_or_atmospheric_correction_added": False,
                        "physical_reflectance_calibration_claimed": False},
        "spatial_display": {"reprojection": False, "orientation": "original H5 array order", "interpolation": "nearest",
                            "crs_geotransform": "not inferred", "worldcover_date": "not inferred"},
        "model_normalization": "official Normalizer with manifest strategy, separately recomputed and hashed; not used as RGB display values",
        "palette": {str(k): {"color": v[0], "label": v[1]} for k, v in palette.items()},
        "display_legend_items": legend_items,
        "worldcover_unknown_pixels_total": unknown_total,
        "records": records, "render": render,
        "output_png": str(image_path.resolve()), "output_png_sha256": sha256(image_path),
        "versions": {p: package_version(p) for p in ["numpy", "h5py", "hdf5plugin"]},
    }
    (args.out / "input_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"png": str(image_path), "provenance": str(args.out / "input_provenance.json"), "examples": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
