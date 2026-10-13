#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from pathlib import Path

# ============================================================
# Fig. 2c | non-SOC spin-resolved band structure
# Final physical size: 65.0 mm x 55.3 mm
# ============================================================

# ============================================================
# 1. Final figure size
# ============================================================

MM_TO_INCH = 1.0 / 25.4

# Keep exactly the same figure size as the SOC version.
FIG_W_MM = 65.0
FIG_H_MM = 50.3

FIG_W = FIG_W_MM * MM_TO_INCH
FIG_H = FIG_H_MM * MM_TO_INCH


# ============================================================
# 2. Layout in millimetres
# ============================================================

# Keep exactly the same main-axis geometry as the SOC version.
LEFT_MM = 13.2
BOTTOM_MM = 8.5

AXIS_W_MM = 35.0
AXIS_H_MM = 35.0

RIGHT_SAFE_MM = FIG_W_MM - (LEFT_MM + AXIS_W_MM)
TOP_SAFE_MM = FIG_H_MM - (BOTTOM_MM + AXIS_H_MM)

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

# non-SOC spin-resolved bands from VASPKIT.
NONSOC_UP_FILE = "REFORMATTED_BAND_UP.dat"
NONSOC_DW_FILE = "REFORMATTED_BAND_DW.dat"


# ============================================================
# 5. Energy reference and plotting range
# ============================================================

EFERMI_NONSOC = -0.025

# Convert plotted energy from eV to meV after subtracting E_F.
ENERGY_SCALE = 1000.0
ENERGY_UNIT = "meV"

# Keep the same plotting window as the SOC version.
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

# Same non-SOC colors as the original script.
COLOR_NONSOC_UP = "#C65D5D"
COLOR_NONSOC_DW = "#4D79B8"

LW_NONSOC = 1.00
ALPHA_NONSOC = 1.00

# Use solid curves for the actual non-SOC band structure.
LS_NONSOC_UP = "-"
LS_NONSOC_DW = "--"

# Optional legend. Set False if the panel label already explains the colors.
SHOW_LEGEND = True

SAVE_TRANSPARENT = False
FIG_FACE_COLOR = "white"


# ============================================================
# 7. Helper functions
# ============================================================

def load_matrix_band(filename, efermi):
    """
    Matrix-format band data:
    first column: k path
    remaining columns: band energies
    """
    data = np.loadtxt(filename, dtype=float)
    data[:, 1:] = (data[:, 1:] - efermi) * ENERGY_SCALE
    return data


def plot_matrix_bands(
    ax,
    data,
    color,
    lw,
    ls,
    alpha,
    zorder,
    label=None,
):
    """Plot all bands in a matrix-format band file."""
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
            label=label if iband == 1 else None,
        )


def read_high_symmetry_points(filename):
    """Read high-symmetry labels and positions from KLABELS."""
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
    """Axis formatting: unchanged from the SOC version."""
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

    # ax.axhline(
    #     MU_STAR,
    #     color=COLOR_GUIDE,
    #     linewidth=0.65,
    #     linestyle="--",
    #     zorder=0,
    # )

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
    NONSOC_UP_FILE,
    NONSOC_DW_FILE,
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

band_up = load_matrix_band(
    NONSOC_UP_FILE,
    EFERMI_NONSOC,
)

band_dw = load_matrix_band(
    NONSOC_DW_FILE,
    EFERMI_NONSOC,
)

print(
    f"Loaded non-SOC bands: "
    f"up = {band_up.shape[1] - 1}, "
    f"down = {band_dw.shape[1] - 1}."
)


# ============================================================
# 10. Create figure
# ============================================================

fig = plt.figure(
    figsize=(FIG_W, FIG_H),
    facecolor=FIG_FACE_COLOR,
)

ax = fig.add_axes([
    AX_LEFT,
    AX_BOTTOM,
    AX_WIDTH,
    AX_HEIGHT,
])


# ============================================================
# 11. Plot non-SOC bands
# ============================================================

plot_matrix_bands(
    ax=ax,
    data=band_up,
    color=COLOR_NONSOC_UP,
    lw=LW_NONSOC,
    ls=LS_NONSOC_UP,
    alpha=ALPHA_NONSOC,
    zorder=2,
    label=r"up",
)

plot_matrix_bands(
    ax=ax,
    data=band_dw,
    color=COLOR_NONSOC_DW,
    lw=LW_NONSOC,
    ls=LS_NONSOC_DW,
    alpha=ALPHA_NONSOC,
    zorder=2,
    label=r"dn",
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
#     0.99,
#     MU_STAR + 0.1,
#     r"$\mu^\star$",
#     transform=ax.get_yaxis_transform(),
#     ha="right",
#     va="bottom",
#     fontsize=8.5,
#     color=COLOR_AXIS,
# )

if SHOW_LEGEND:
    leg = ax.legend(
        loc="best",
        frameon=False,
        handlelength=1.,
        handletextpad=0.45,
        borderaxespad=0.35,
        labelspacing=0.30,
    )

    for line in leg.get_lines():
        line.set_linewidth(1.15)


# ============================================================
# 13. Export
# ============================================================

out_base = "BAND_NONSOC_NC_meV"

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
print(f"Saved: {out_base}.png")
