#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from pathlib import Path

# ============================================================
# Fig. 2c | SOC Sz-projected band structure
# Final physical size: 55.3 mm x 55.3 mm
# ============================================================

# ============================================================
# 1. Final figure size
# ============================================================

# MM_TO_INCH = 1.0 / 25.4

# FIG_W_MM = 65
# FIG_H_MM = 55

# FIG_W = FIG_W_MM * MM_TO_INCH
# FIG_H = FIG_H_MM * MM_TO_INCH
# ============================================================
# 1. Final figure size
# ============================================================

MM_TO_INCH = 1.0 / 25.4

# Keep the same figure height as the layer-resolved NLH panel.
FIG_W_MM = 65.0
FIG_H_MM = 50.3

FIG_W = FIG_W_MM * MM_TO_INCH
FIG_H = FIG_H_MM * MM_TO_INCH


# ============================================================
# 2. Layout in millimetres
# ============================================================

# Shared vertical layout with the layer-resolved NLH panel.
LEFT_MM = 13.2
BOTTOM_MM = 8.5

# Main band axis.
# IMPORTANT: keep AXIS_H_MM identical to the NLH script.
AXIS_W_MM = 35.0
AXIS_H_MM = 35.0

# Right-side colorbar.
CBAR_GAP_MM = 2.3
CBAR_W_MM = 2.2

# Keep colorbar height exactly the same as the main axis height.
CBAR_H_MM = AXIS_H_MM*2/3

CBAR_LEFT_MM = LEFT_MM + AXIS_W_MM + CBAR_GAP_MM
CBAR_BOTTOM_MM = BOTTOM_MM

RIGHT_SAFE_MM = FIG_W_MM - (
    CBAR_LEFT_MM + CBAR_W_MM
)

TOP_SAFE_MM = FIG_H_MM - (
    BOTTOM_MM + AXIS_H_MM
)

if RIGHT_SAFE_MM < 1.0:
    raise ValueError(
        f"Right margin is too small: {RIGHT_SAFE_MM:.2f} mm."
    )

if TOP_SAFE_MM < 1.0:
    raise ValueError(
        f"Top margin is too small: {TOP_SAFE_MM:.2f} mm."
    )

AX_LEFT = LEFT_MM / FIG_W_MM
AX_BOTTOM = BOTTOM_MM / FIG_H_MM
AX_WIDTH = AXIS_W_MM / FIG_W_MM
AX_HEIGHT = AXIS_H_MM / FIG_H_MM

CBAR_LEFT = CBAR_LEFT_MM / FIG_W_MM
CBAR_BOTTOM = CBAR_BOTTOM_MM / FIG_H_MM
CBAR_WIDTH = CBAR_W_MM / FIG_W_MM
CBAR_HEIGHT = CBAR_H_MM / FIG_H_MM

# # ============================================================
# # 2. Layout in millimetres
# # ============================================================

# # Same vertical alignment as Fig. 2b.
# LEFT_MM = 13.2
# BOTTOM_MM = 8.5

# # Main band axis.
# AXIS_W_MM = 41.0
# AXIS_H_MM = 40.0

# # Right-side colorbar.
# CBAR_GAP_MM = 2.3
# CBAR_W_MM = 2.2
# CBAR_H_MM = AXIS_H_MM

# CBAR_LEFT_MM = LEFT_MM + AXIS_W_MM + CBAR_GAP_MM
# CBAR_BOTTOM_MM = BOTTOM_MM

# RIGHT_SAFE_MM = FIG_W_MM - (
#     CBAR_LEFT_MM + CBAR_W_MM
# )

# TOP_SAFE_MM = FIG_H_MM - (
#     BOTTOM_MM + AXIS_H_MM
# )

# if RIGHT_SAFE_MM < 1.0:
#     raise ValueError(
#         f"Right margin is too small: {RIGHT_SAFE_MM:.2f} mm."
#     )

# if TOP_SAFE_MM < 1.0:
#     raise ValueError(
#         f"Top margin is too small: {TOP_SAFE_MM:.2f} mm."
#     )

# AX_LEFT = LEFT_MM / FIG_W_MM
# AX_BOTTOM = BOTTOM_MM / FIG_H_MM
# AX_WIDTH = AXIS_W_MM / FIG_W_MM
# AX_HEIGHT = AXIS_H_MM / FIG_H_MM

# CBAR_LEFT = CBAR_LEFT_MM / FIG_W_MM
# CBAR_BOTTOM = CBAR_BOTTOM_MM / FIG_H_MM
# CBAR_WIDTH = CBAR_W_MM / FIG_W_MM
# CBAR_HEIGHT = CBAR_H_MM / FIG_H_MM


# ============================================================
# 3. Global plotting style
# ============================================================

config = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

    "mathtext.fontset": "dejavusans",
    "mathtext.rm": "DejaVu Sans",
    "mathtext.it": "DejaVu Sans:italic",
    "mathtext.bf": "DejaVu Sans:bold",

    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,

    "xtick.labelsize": 7.50,
    "ytick.labelsize": 7.50,

    "legend.fontsize": 7.50,

    "axes.linewidth": 0.7,
    "lines.linewidth": 1.0,

    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,

    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,


    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "savefig.bbox": None,
    "savefig.pad_inches": 0.0,
}
rcParams.update(config)


# ============================================================
# 4. Input files
# ============================================================

KLABELS_FILE = "KLABELS"

# Optional non-SOC reference bands.
NONSOC_UP_FILE = "REFORMATTED_BAND_UP.dat"
NONSOC_DW_FILE = "REFORMATTED_BAND_DW.dat"

# SOC total Sz-projected band file.
SOC_SZ_FILE = "PBAND_SUM_SOC_SZ.dat"


# ============================================================
# 5. Energy references and plotting range
# ============================================================

EFERMI_NONSOC = -0.02
EFERMI_SOC = 2.925 - 2.980

# Convert plotted energy from eV to meV after subtracting E_F.
ENERGY_SCALE = 1000.0
ENERGY_UNIT = "meV"

# Plotting range in meV.
YMIN = -110.0
YMAX = 10.0

MU_STAR = -38.0

HOTSPOTS = [
    # (0.35, r"$H_1$"),
    # (0.62, r"$H_2$"),
]


# ============================================================
# 6. Colors and line settings
# ============================================================

COLOR_AXIS = "#404040"
COLOR_GUIDE = "#9A9A9A"

COLOR_NONSOC_UP = "#C65D5D"
COLOR_NONSOC_DW = "#4D79B8"

# Print-friendly Sz color map with enhanced transition-region contrast.
# Around <Sz> ~ 0, add lightly tinted blue/red shoulders instead of a broad
# neutral plateau, so the near-degenerate region still shows the sign change.
CMAP_SZ = LinearSegmentedColormap.from_list(
    "spin_transition_enhanced",
    [
        (0.00, "#2F5DA8"),
        (0.30, "#6FA0E8"),
        (0.45, "#D9E8FF"),
        (0.50, "#E0DEDE"),
        (0.55, "#F9DDDD"),
        (0.70, "#E79090"),
        (1.00, "#B44242"),
    ],
    N=256,
)
SZ_VMIN = -1.0
SZ_VMAX = 1.0

# gamma < 1 enhances color separation near <Sz> = 0.
SZ_COLOR_GAMMA = 0.60

LW_NONSOC = 0.75

# Slightly thicker SOC bands improve readability in small printed panels.
LW_SOC = 1.20
ALPHA_SOC = 1.00

# Draw a neutral underlay below every SOC band.  This makes nearly zero-Sz
# segments visible even if the colored overlay is light or compressed by print.
PLOT_SOC_UNDERLAY = True
COLOR_SOC_UNDERLAY = "#4A4A4A"
LW_SOC_UNDERLAY = 1.35
ALPHA_SOC_UNDERLAY = 0.10

ALPHA_NONSOC = 0.95

# For printing or Illustrator placement, a white page background is safer than
# transparent output.  Set this to True only if you really need transparency.
SAVE_TRANSPARENT = False
FIG_FACE_COLOR = "white"


# ============================================================
# 7. Helper functions
# ============================================================

class SignedPowerNorm(Normalize):
    """
    Symmetric power-law normalization for signed spin projections.

    gamma < 1 expands the color contrast near <Sz> = 0, so the nearly
    spin-degenerate / transition region is easier to distinguish, while
    keeping 0 exactly at the center of the diverging colormap.
    """
    def __init__(self, gamma=0.60, vmin=-1.0, vmax=1.0, clip=False):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.gamma = gamma

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        data = result.data

        vmin = self.vmin if self.vmin is not None else -1.0
        vmax = self.vmax if self.vmax is not None else 1.0
        scale = max(abs(vmin), abs(vmax))
        if scale <= 0:
            scale = 1.0

        x = np.clip(data / scale, -1.0, 1.0)
        y = 0.5 + 0.5 * np.sign(x) * (np.abs(x) ** self.gamma)

        masked = np.ma.array(y, mask=np.ma.getmask(result), copy=False)
        if is_scalar:
            masked = masked[0]
        return masked

    def inverse(self, value):
        value = np.asarray(value)
        y = np.clip(2.0 * value - 1.0, -1.0, 1.0)
        x = np.sign(y) * (np.abs(y) ** (1.0 / self.gamma))
        scale = max(abs(self.vmin), abs(self.vmax))
        if scale <= 0:
            scale = 1.0
        return x * scale

def load_matrix_band(filename, efermi):
    """
    Matrix-format data:
    first column: k path
    remaining columns: band energies
    """
    data = np.loadtxt(filename, dtype=float)
    data[:, 1:] = (data[:, 1:] - efermi) * ENERGY_SCALE
    return data


def load_long_band(filename, efermi):
    """
    Long-format data:
    first column: k path
    second column: energy
    last column: Sz projection
    """
    data = np.loadtxt(filename, dtype=float)
    data[:, 1] = (data[:, 1] - efermi) * ENERGY_SCALE
    return data


def load_soc_sz_bands(filename, efermi):
    """
    Read VASPKIT PBAND_SUM_SOC_SZ.dat band-by-band.

    The file is divided by '# Band-Index:' headers.  In this output,
    adjacent bands can be stored with opposite k-path directions.
    Therefore, never identify bands by a decrease of k; use the
    explicit Band-Index headers.
    """
    bands = []
    rows = []

    with open(filename, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("# Band-Index:"):
                if rows:
                    band = np.asarray(rows, dtype=float)
                    band[:, 1] = (band[:, 1] - efermi) * ENERGY_SCALE
                    bands.append(band)
                    rows = []
                continue

            if line.startswith("#"):
                continue

            fields = line.split()
            rows.append([float(value) for value in fields])

    if rows:
        band = np.asarray(rows, dtype=float)
        band[:, 1] = (band[:, 1] - efermi) * ENERGY_SCALE
        bands.append(band)

    if not bands:
        raise ValueError(f"No band blocks were found in {filename}.")

    ncol = bands[0].shape[1]
    if any(band.shape[1] != ncol for band in bands):
        raise ValueError("Inconsistent column count among band blocks.")

    return bands


def split_long_segments(bands):
    """
    Compatibility wrapper.

    `bands` is already a list of complete band blocks parsed from the
    '# Band-Index:' headers. Do not split it again according to k direction.
    """
    return [band for band in bands if len(band) > 1]


def plot_matrix_bands(
    ax,
    data,
    color,
    lw,
    ls,
    alpha,
    zorder,
):
    """
    Plot non-SOC reference bands.
    """
    k = data[:, 0]

    for iband in range(1, data.shape[1]):
        ax.plot(
            k,
            data[:, iband],
            color=color,
            linewidth=lw,
            linestyle=ls,
            alpha=alpha,
            zorder=zorder,
        )


def plot_soc_band_underlay(
    ax,
    data,
    color,
    lw,
    alpha,
    zorder,
):
    """
    Draw a neutral guide line below the color-coded SOC bands.

    This is mainly for print: it preserves the band outline without
    overwhelming the red/blue contrast near the spin-transition region.
    """
    for seg in split_long_segments(data):
        k = seg[:, 0]
        energy = seg[:, 1]

        ax.plot(
            k,
            energy,
            color=color,
            linewidth=lw,
            alpha=alpha,
            zorder=zorder,
            solid_capstyle="round",
            solid_joinstyle="round",
        )


def plot_sz_colored_bands(
    ax,
    data,
    cmap,
    norm,
    lw,
    alpha,
    zorder,
):
    """
    Plot SOC bands as continuous line segments
    colored by the total Sz projection.
    """
    lc_for_colorbar = None

    for seg in split_long_segments(data):
        k = seg[:, 0]
        energy = seg[:, 1]
        sz = seg[:, -1]

        points = np.column_stack(
            (k, energy)
        ).reshape(-1, 1, 2)

        line_segments = np.concatenate(
            [points[:-1], points[1:]],
            axis=1,
        )

        sz_segments = 0.5 * (
            sz[:-1] + sz[1:]
        )

        lc = LineCollection(
            line_segments,
            cmap=cmap,
            norm=norm,
            linewidths=lw,
            alpha=alpha,
            zorder=zorder,
            capstyle="round",
            joinstyle="round",
        )

        lc.set_array(sz_segments)
        ax.add_collection(lc)

        lc_for_colorbar = lc

    return lc_for_colorbar


def read_high_symmetry_points(filename):
    """
    Read high-symmetry labels and positions from KLABELS.

    Ignore blank/comment/tip lines explicitly, so that the trailing
    VASPKIT note is never interpreted as tabular data.
    """
    labels = []
    positions = []

    with open(filename, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            fields = raw_line.split()

            if len(fields) < 2:
                continue

            try:
                position = float(fields[1])
            except ValueError:
                continue

            label = "Γ" if fields[0] == "GAMMA" else fields[0]
            labels.append(label)
            positions.append(position)

    if len(positions) < 2:
        raise ValueError(
            f"Could not read enough high-symmetry points from {filename}."
        )

    return positions, labels


def format_axis(ax, x_positions, x_labels):
    """
    Axis formatting.
    """
    ax.set_xlim(
        x_positions[0],
        x_positions[-1],
    )

    ax.set_ylim(YMIN, YMAX)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)

    ax.set_ylabel(
        r"Energy (meV)",
        color=COLOR_AXIS,
        labelpad=3.0,
    )

    for xpos in x_positions[1:-1]:
        ax.axvline(
            xpos,
            color=COLOR_GUIDE,
            linewidth=0.45,
            linestyle="--",
            zorder=0,
        )

    ax.axhline(
        0.0,
        color=COLOR_AXIS,
        linewidth=0.55,
        linestyle="-",
        zorder=0,
    )

    ax.axhline(
        MU_STAR,
        color=COLOR_GUIDE,
        linewidth=0.65,
        linestyle="--",
        zorder=0,
    )

    for xpos, label in HOTSPOTS:
        ax.axvline(
            xpos,
            color="#3B8C6E",
            linewidth=0.65,
            linestyle=":",
            zorder=0,
        )

        ax.text(
            xpos,
            0.96,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.2,
            color="#3B8C6E",
        )

    ax.yaxis.set_major_locator(
        ticker.MultipleLocator(50.0)
    )

    ax.yaxis.set_minor_locator(
        ticker.AutoMinorLocator(2)
    )

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=2.8,
        width=0.7,
        colors=COLOR_AXIS,
        pad=1.6,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=1.5,
        width=0.45,
        colors=COLOR_AXIS,
    )

    for spine in ax.spines.values():
        spine.set_color(COLOR_AXIS)
        spine.set_linewidth(0.7)


# ============================================================
# 8. Check files
# ============================================================

required_files = [
    KLABELS_FILE,
    SOC_SZ_FILE,
]

for filename in required_files:
    if not Path(filename).exists():
        raise FileNotFoundError(
            f"Missing required file: {filename}"
        )


# ============================================================
# 9. Read data
# ============================================================

x_positions, x_labels = read_high_symmetry_points(
    KLABELS_FILE
)

soc_sz_data = load_soc_sz_bands(
    SOC_SZ_FILE,
    EFERMI_SOC,
)

print(
    f"Loaded {len(soc_sz_data)} SOC band blocks; "
    f"{min(len(band) for band in soc_sz_data)}–"
    f"{max(len(band) for band in soc_sz_data)} k points per band."
)

# Optional non-SOC reference-band loading.
PLOT_NONSOC_REFERENCE = False

if PLOT_NONSOC_REFERENCE:
    band_up = load_matrix_band(
        NONSOC_UP_FILE,
        EFERMI_NONSOC,
    )

    band_dw = load_matrix_band(
        NONSOC_DW_FILE,
        EFERMI_NONSOC,
    )


# ============================================================
# 10. Create figure
# ============================================================

fig = plt.figure(figsize=(FIG_W, FIG_H))

ax = fig.add_axes([
    AX_LEFT,
    AX_BOTTOM,
    AX_WIDTH,
    AX_HEIGHT,
])

cax = fig.add_axes([
    CBAR_LEFT,
    CBAR_BOTTOM,
    CBAR_WIDTH,
    CBAR_HEIGHT,
])


# ============================================================
# 11. Plot bands
# ============================================================

if PLOT_NONSOC_REFERENCE:
    plot_matrix_bands(
        ax=ax,
        data=band_up,
        color=COLOR_NONSOC_UP,
        lw=LW_NONSOC,
        ls="--",
        alpha=ALPHA_NONSOC,
        zorder=1,
    )

    plot_matrix_bands(
        ax=ax,
        data=band_dw,
        color=COLOR_NONSOC_DW,
        lw=LW_NONSOC,
        ls="--",
        alpha=ALPHA_NONSOC,
        zorder=1,
    )

if PLOT_SOC_UNDERLAY:
    plot_soc_band_underlay(
        ax=ax,
        data=soc_sz_data,
        color=COLOR_SOC_UNDERLAY,
        lw=LW_SOC_UNDERLAY,
        alpha=ALPHA_SOC_UNDERLAY,
        zorder=2,
    )

sz_norm = SignedPowerNorm(
    gamma=SZ_COLOR_GAMMA,
    vmin=SZ_VMIN,
    vmax=SZ_VMAX,
)

line_collection = plot_sz_colored_bands(
    ax=ax,
    data=soc_sz_data,
    cmap=CMAP_SZ,
    norm=sz_norm,
    lw=LW_SOC,
    alpha=ALPHA_SOC,
    zorder=3,
)

format_axis(
    ax,
    x_positions,
    x_labels,
)


# ============================================================
# 12. In-panel labels
# ============================================================

# ax.text(
#     0.50,
#     0.96,
#     r"SOC $\langle S_z\rangle$",
#     transform=ax.transAxes,
#     ha="left",
#     va="top",
#     fontsize=6.8,
#     color=COLOR_AXIS,
# )

# ax.text(
#     0.99,
#     MU_STAR + 0.1,
#     r"$\mu^\star$",
#     transform=ax.get_yaxis_transform(),
#     ha="right",
#     va="bottom",
#     fontsize=8.5,
#     color=COLOR_AXIS,
# )


# ============================================================
# 13. Right-side colorbar
# ============================================================

cb = fig.colorbar(
    line_collection,
    cax=cax,
    orientation="vertical",
)

cb.set_ticks([-1.0,  0.0,  1.0])
cb.set_ticklabels([
    r"$-1$",
    r"$0$",
    r"$+1$",
])

cb.set_label(
    r"$\langle S_z\rangle$",
    fontsize=8,
    labelpad=2.4,
)

cb.ax.tick_params(
    direction="in",
    length=2.0,
    width=0.5,
    labelsize=7.5,
    pad=1.0,
)

cb.outline.set_linewidth(0.5)
cb.outline.set_edgecolor(COLOR_AXIS)


# ============================================================
# 14. Export
# ============================================================

out_base = "BAND_NC3_meV"

fig.savefig(
    f"{out_base}.pdf",
    transparent=SAVE_TRANSPARENT,
    facecolor=FIG_FACE_COLOR,
)

# fig.savefig(
#     f"{out_base}.svg",
#     transparent=SAVE_TRANSPARENT,
#     facecolor=FIG_FACE_COLOR,
# )

fig.savefig(
    f"{out_base}.png",
    transparent=SAVE_TRANSPARENT,
    facecolor=FIG_FACE_COLOR,
    dpi=600,
)

plt.close(fig)

print(f"Saved: {out_base}.pdf")
print(f"Saved: {out_base}.svg")
print(f"Saved: {out_base}.png")