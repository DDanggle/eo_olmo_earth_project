"""Plot the measured cache-vs-raw utility boundary for the MS-102/105/108 audit.

This is a characterization figure, not a significance plot.  OlmoEarth full/raw
come from the sealed confirmatory summary; the other bars are seed-1 diagnostics.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


reference = load_json(ARTIFACTS / "confirmatory_8region_summary.json")
architecture = load_json(ARTIFACTS / "arch_axes_summary.json")["verdict"]
diagnostics = load_json(ARTIFACTS / "bv1_diagnostics_summary.json")["verdict"]

raw = reference["headline"]["region_macro_primary_mean"]["raw_strong"]
olmo_full = reference["headline"]["region_macro_primary_mean"]["reuse"]


def arch(name: str) -> float:
    row = architecture[name]
    if not row["complete"] or row["n"] != 8:
        raise ValueError(f"incomplete architecture row: {name}")
    return row["macro"]


def diag(name: str) -> float:
    row = diagnostics[name]
    if row["n"] != 8:
        raise ValueError(f"incomplete diagnostic row: {name}")
    return row["macro"]


rows = [
    ("OlmoEarth base", "OlmoEarth", olmo_full, "confirmatory"),
    ("OlmoEarth tiny", "OlmoEarth", arch("olmo_tiny"), "diagnostic"),
    ("OlmoEarth base @ 50% depth", "OlmoEarth", arch("olmo_base_half"), "diagnostic"),
    ("OlmoEarth nano", "OlmoEarth", arch("olmo_nano"), "diagnostic"),
    ("Galileo base + group-concat", "Galileo", diag("galileo_cache_groupcat"), "diagnostic"),
    ("Galileo base", "Galileo", diag("galileo_cache"), "diagnostic"),
    ("Galileo base @ 50% depth", "Galileo", arch("galileo_base_half"), "diagnostic"),
    ("Galileo tiny", "Galileo", arch("galileo_tiny"), "diagnostic"),
    ("Galileo nano", "Galileo", arch("galileo_nano"), "diagnostic"),
    ("Clay input-256", "Clay", diag("clay_cache_in256"), "diagnostic"),
    ("Clay input-256 @ 50% depth", "Clay", arch("clay_in256_half"), "diagnostic"),
]

palette = {"OlmoEarth": "#2F6BFF", "Galileo": "#F59E0B", "Clay": "#7C3AED"}
labels = [row[0] for row in rows]
deltas = [row[2] - raw for row in rows]
colors = [palette[row[1]] for row in rows]

fig, ax = plt.subplots(figsize=(10.8, 6.8))
y = list(range(len(rows)))
bars = ax.barh(y, deltas, color=colors, alpha=0.9, height=0.7)
ax.axvline(0, color="#111827", linewidth=1.4)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlabel("Positive-patch macro IoU minus raw training")
ax.set_title("Cached Earth representations cross the raw-training boundary")
ax.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.7)
ax.set_axisbelow(True)

for bar, delta in zip(bars, deltas):
    if delta >= 0:
        x, align, text_color = delta + 0.003, "left", "#111827"
    elif abs(delta) >= 0.012:
        x, align, text_color = delta + 0.003, "left", "white"
    else:
        x, align, text_color = delta - 0.003, "right", "#111827"
    ax.text(
        x,
        bar.get_y() + bar.get_height() / 2,
        f"{delta:+.3f}",
        va="center",
        ha=align,
        fontsize=9,
        color=text_color,
    )

legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in palette.values()]
ax.legend(legend_handles, palette.keys(), loc="lower right", frameon=False, ncol=3)
fig.text(
    0.01,
    0.01,
    "Sen12Landslides, same 8 geographic folds. Raw baseline = "
    f"{raw:.3f}. OlmoEarth base/raw are sealed confirmatory values; other bars are seed-1 diagnostics.",
    fontsize=8.5,
    color="#4B5563",
)
fig.tight_layout(rect=(0, 0.045, 1, 1))

output = ARTIFACTS / "figures" / "cache_utility_boundary_ms108.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=180, bbox_inches="tight")
plt.close(fig)
print(output)
