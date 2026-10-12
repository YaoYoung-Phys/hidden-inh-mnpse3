#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path

# ============================================================
# Fig. 2b | Layer-resolved intrinsic nonlinear Hall response
# Final physical size: 55.3 mm x 55.3 mm
# ============================================================

# ============================================================
# 1. Final figure size
# ============================================================

# MM_TO_INCH = 1.0 / 25.4

# FIG_W_MM = 55.3
# FIG_H_MM = 55.3

# FIG_W = FIG_W_MM * MM_TO_INCH
# FIG_H = FIG_H_MM * MM_TO_INCH
# ============================================================
# 1. Final figure size
# ============================================================

MM_TO_INCH = 1.0 / 25.4

# Same figure height as the band-structure panel.
FIG_W_MM = 75.3
FIG_H_MM = 55.3

FIG_W = FIG_W_MM * MM_TO_INCH
FIG_H = FIG_H_MM * MM_TO_INCH


# ============================================================
# 2. Layout in millimetres
# ============================================================

# Shared vertical layout with the band-structure panel.
LEFT_MM = 13.2
BOTTOM_MM = 8.5

# Main plotting area.
# IMPORTANT: keep AXIS_H_MM identical to the band script.
AXIS_W_MM = 44
AXIS_H_MM = 28

RIGHT_SAFE_MM = FIG_W_MM - LEFT_MM - AXIS_W_MM
TOP_SAFE_MM = FIG_H_MM - BOTTOM_MM - AXIS_H_MM

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


# # ============================================================
# # 2. Layout in millimetres
# # ============================================================

# LEFT_MM = 13.2
# BOTTOM_MM = 8.5

# AXIS_W_MM = 39.1
# AXIS_H_MM = 40.0

# RIGHT_SAFE_MM = FIG_W_MM - LEFT_MM - AXIS_W_MM
# TOP_SAFE_MM = FIG_H_MM - BOTTOM_MM - AXIS_H_MM

# if RIGHT_SAFE_MM < 1.0:
#     raise ValueError(
#         f"Right margin is too small: {RIGHT_SAFE_MM:.2f} mm."
#     )

# if TOP_SAFE_MM < 5.0:
#     raise ValueError(
#         f"Top margin is too small: {TOP_SAFE_MM:.2f} mm."
#     )

# AX_LEFT = LEFT_MM / FIG_W_MM
# AX_BOTTOM = BOTTOM_MM / FIG_H_MM
# AX_WIDTH = AXIS_W_MM / FIG_W_MM
# AX_HEIGHT = AXIS_H_MM / FIG_H_MM


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

    "legend.frameon": False,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "savefig.bbox": None,
    "savefig.pad_inches": 0.0,
}
rcParams.update(config)


# ============================================================
# 4. Colors
# ============================================================

COLOR_AXIS = "#3F3F3F"
COLOR_GUIDE = "#9A9A9A"

COLOR_TOP = "#D46A5A"
COLOR_BOTTOM = "#0072B2"
COLOR_TOTAL = "#303030"


# ENERGY_COLOR = "#D46A5A"
# PZ_COLOR = "#5B7FA3"

# ============================================================
# 5. Input parameters
# ============================================================

component = "xyy"

fermiE0 = -2.980608

# Vertical axis: chemical potential
# Convert plotted chemical potential from eV to meV.
ENERGY_SCALE = 1000.0
ENERGY_UNIT = "meV"

x_min = -110.0
x_max = 10.0

# Horizontal axis: chi_INH
# Effective-thickness correction enlarges the response by about 4.421.
# Adjust these values if needed.
y_min = -55.0
y_max = 55.0

xticks = [-100.0, -50.0, 0.0]

# Manual chi-axis ticks.
# Use None for automatic ticks.
CHI_TICKS = [-40,  0,  40]
# CHI_TICKS = None

# Selected peak energy.
mu_star = -38.0

# ============================================================
# Effective-thickness correction
# ============================================================
# Original response is first multiplied by scale_factor.
# Then it is multiplied by the effective-thickness factor L_EFF.
#
# plotted_chi = raw_chi * scale_factor * L_EFF
#
# If you do not want the effective-thickness correction,
# set USE_EFFECTIVE_THICKNESS = False.
USE_EFFECTIVE_THICKNESS = True

scale_factor = 1.0e3
L_EFF = 1.0 / (0.61309 - 0.38690)

if USE_EFFECTIVE_THICKNESS:
    applied_scale_factor = scale_factor * L_EFF
else:
    applied_scale_factor = scale_factor

# ylabel_unit = r"mA V$^{-2}$"
YLABEL_UNIT = r"mA V$^{-2}$"
Y_LABEL = r"$\chi_{yxx}$ (" + YLABEL_UNIT + ")"
files_and_labels = [
    ("wannier90-gyrotropic-LGM-layer1.dat", "top"),
    ("wannier90-gyrotropic-LGM-layer2.dat", "bottom"),
    ("wannier90-gyrotropic-LGM.dat", "total"),
]

out_base = "Fig2b_NC4_meV"


# ============================================================
# 6. Data reading
# ============================================================

def read_blocks(filename):
    """
    Read LGM data files containing several blocks.
    Each block corresponds to one current direction.
    """
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_blocks = []
    current_block = []

    for line in lines:
        line = line.strip()

        if (not line) or line.startswith("#"):
            if current_block:
                data_blocks.append(
                    np.array(current_block, dtype=float)
                )
                current_block = []
            continue

        try:
            current_block.append(
                [float(x) for x in line.split()]
            )
        except ValueError:
            continue

    if current_block:
        data_blocks.append(
            np.array(current_block, dtype=float)
        )

    return data_blocks


def extract_INH_component(
    filename,
    component,
    fermiE0,
    applied_scale_factor
):
    """
    Extract layer-resolved intrinsic nonlinear Hall component.

    yxx:
        j_y response under E_x E_x

    xyy:
        j_x response under E_y E_y

    The returned chi has already been multiplied by:
        scale_factor * L_EFF
    """
    blocks = read_blocks(filename)

    if component == "yxx":
        if len(blocks) < 2:
            raise RuntimeError(
                f"{filename}: cannot extract yxx."
            )
        block = blocks[1]
        col_idx = 2

    elif component == "xyy":
        if len(blocks) < 1:
            raise RuntimeError(
                f"{filename}: cannot extract xyy."
            )
        block = blocks[0]
        col_idx = 3

    else:
        raise ValueError(
            "component must be 'xyy' or 'yxx'."
        )

    mu_rel = (block[:, 0] - fermiE0) * ENERGY_SCALE
    chi = block[:, col_idx] * applied_scale_factor

    return mu_rel, chi


# ============================================================
# 7. Read data
# ============================================================

curves = {}

for filename, label in files_and_labels:
    mu_rel, chi = extract_INH_component(
        filename=filename,
        component=component,
        fermiE0=fermiE0,
        applied_scale_factor=applied_scale_factor,
    )
    curves[label] = (mu_rel, chi)

mu_top, chi_top = curves["top"]
mu_bottom, chi_bottom = curves["bottom"]
mu_total, chi_total = curves["total"]


# ============================================================
# 8. Create figure
# ============================================================

fig = plt.figure(figsize=(FIG_W, FIG_H))

ax = fig.add_axes([
    AX_LEFT,
    AX_BOTTOM,
    AX_WIDTH,
    AX_HEIGHT,
])


# ============================================================
# 9. Plot curves
# x axis: mu - E_F^0
# y axis: chi_INH^{xyy}
# ============================================================

ax.plot(
    mu_top,
    chi_top,
    color=COLOR_TOP,
    linewidth=1.2,
    label="Top",
    zorder=3,
)

ax.plot(
    mu_bottom,
    chi_bottom,
    color=COLOR_BOTTOM,
    linewidth=1.2,
    label="Bottom",
    zorder=3,
)

ax.plot(
    mu_total,
    chi_total,
    color=COLOR_TOTAL,
    linewidth=0.95,
    linestyle="-",
    label="Total",
    zorder=4,
)

# Zero-response reference line: chi_INH = 0.
ax.axhline(
    0.0,
    color=COLOR_GUIDE,
    linewidth=0.65,
    linestyle="--",
    zorder=1,
)

# Selected chemical potential.
ax.axvline(
    mu_star,
    color=COLOR_GUIDE,
    linewidth=0.75,
    linestyle="--",
    zorder=1,
)


# ============================================================
# 10. Axis formatting
# ============================================================

# Horizontal axis: chemical potential.
ax.set_xlim(x_min, x_max)

# Vertical axis: chi_INH.
ax.set_ylim(y_min, y_max)

# x ticks for chemical potential.
ax.set_xticks(xticks)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

# y ticks for chi_INH.
if CHI_TICKS is not None:
    ax.set_yticks(CHI_TICKS)
else:
    ax.yaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=4,
            symmetric=True,
        )
    )

ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.set_xlabel(
    r"Energy (meV)",
    color=COLOR_AXIS,
    labelpad=2.5,
)

ax.set_ylabel(
    Y_LABEL,
    color=COLOR_AXIS,
    labelpad=3.0,
)

ax.tick_params(
    axis="both",
    which="major",
    colors=COLOR_AXIS,
    length=3.0,
    width=0.7,
    pad=1.6,
)

ax.tick_params(
    axis="both",
    which="minor",
    colors=COLOR_AXIS,
    length=1.8,
    width=0.45,
)

for spine in ax.spines.values():
    spine.set_color(COLOR_AXIS)
    spine.set_linewidth(0.7)


# ============================================================
# 11. Legend
# ============================================================

ax.legend(
    loc="upper left",
    ncol=1,
    handlelength=1.15,
    handletextpad=0.45,
    borderaxespad=0.25,
    labelspacing=0.20,
)


# ============================================================
# 12. Export
# ============================================================

fig.savefig(
    f"{out_base}.pdf",
    transparent=True,
)

# fig.savefig(
#     f"{out_base}.svg",
#     transparent=True,
# )

fig.savefig(
    f"{out_base}.png",
    transparent=True,
    dpi=600,
)

plt.close(fig)

print("=" * 70)
print(f"USE_EFFECTIVE_THICKNESS = {USE_EFFECTIVE_THICKNESS}")
print(f"scale_factor = {scale_factor:.12e}")
print(f"L_EFF = {L_EFF:.12e}")
print(f"applied_scale_factor = {applied_scale_factor:.12e}")
print("-" * 70)
print(f"Saved: {out_base}.pdf")
print(f"Saved: {out_base}.svg")
print(f"Saved: {out_base}.png")
print("=" * 70)