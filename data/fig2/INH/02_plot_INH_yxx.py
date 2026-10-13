#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot new-coordinate yxx using the extracted old-coordinate xyy data."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "INH_yxx.dat"
OUT_BASE = BASE_DIR / "Fig2b_NC5_meV"

# Dimensions and layout copied from the original Fig. 2b script.
MM_TO_INCH = 1.0 / 25.4
FIG_W_MM, FIG_H_MM = 75.3, 55.3
LEFT_MM, BOTTOM_MM = 13.2, 8.5
AXIS_W_MM, AXIS_H_MM = 44.0, 28.0
if FIG_W_MM - LEFT_MM - AXIS_W_MM < 1 or FIG_H_MM - BOTTOM_MM - AXIS_H_MM < 1:
    raise ValueError("Figure margins too small")

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "mathtext.rm": "DejaVu Sans",
    "mathtext.it": "DejaVu Sans:italic",
    "mathtext.bf": "DejaVu Sans:bold",
    "font.size": 8.0,
    "axes.labelsize": 8.0, "axes.titlesize": 8.0,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.7,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.bbox": None, "savefig.pad_inches": 0.0,
})

COLOR_AXIS = "#3F3F3F"
COLOR_GUIDE = "#9A9A9A"
COLOR_TOP = "#D46A5A"
COLOR_BOTTOM = "#0072B2"
COLOR_TOTAL = "#303030"

X_MIN, X_MAX = -110.0, 10.0
Y_MIN, Y_MAX = -55.0, 55.0
XTICKS = [-100.0, -50.0, 0.0]
CHI_TICKS = [-40.0, 0.0, 40.0]  # None for automatic ticks
MU_STAR = -38.0
Y_LABEL = r"$\chi_{yxx}$ (mA V$^{-2}$)"  # new-coordinate label


def main() -> None:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Run 01_extract_INH_xyy.py first: {DATA_FILE}")
    data = np.loadtxt(DATA_FILE, comments="#", ndmin=2)
    if data.shape[1] != 4 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("Expected finite N x 4 data: energy, top, bottom, total")
    mu, top, bottom, total = data.T

    fig = plt.figure(figsize=(FIG_W_MM * MM_TO_INCH, FIG_H_MM * MM_TO_INCH))
    ax = fig.add_axes([
        LEFT_MM / FIG_W_MM, BOTTOM_MM / FIG_H_MM,
        AXIS_W_MM / FIG_W_MM, AXIS_H_MM / FIG_H_MM,
    ])
    ax.plot(mu, top, color=COLOR_TOP, linewidth=1.2, label="Top", zorder=3)
    ax.plot(mu, bottom, color=COLOR_BOTTOM, linewidth=1.2, label="Bottom", zorder=3)
    ax.plot(mu, total, color=COLOR_TOTAL, linewidth=0.95, label="Total", zorder=4)
    ax.axhline(0.0, color=COLOR_GUIDE, linewidth=0.65, linestyle="--", zorder=1)
    ax.axvline(MU_STAR, color=COLOR_GUIDE, linewidth=0.75, linestyle="--", zorder=1)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks(XTICKS)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if CHI_TICKS is not None:
        ax.set_yticks(CHI_TICKS)
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, symmetric=True))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel("Energy (meV)", color=COLOR_AXIS, labelpad=2.5)
    ax.set_ylabel(Y_LABEL, color=COLOR_AXIS, labelpad=3.0)
    ax.tick_params(axis="both", which="major", colors=COLOR_AXIS,
                   length=3.0, width=0.7, pad=1.6)
    ax.tick_params(axis="both", which="minor", colors=COLOR_AXIS,
                   length=1.8, width=0.45)
    for spine in ax.spines.values():
        spine.set_color(COLOR_AXIS)
        spine.set_linewidth(0.7)
    ax.legend(loc="upper left", ncol=1, handlelength=1.15,
              handletextpad=0.45, borderaxespad=0.25, labelspacing=0.20)

    fig.savefig(f"{OUT_BASE}.pdf", transparent=True)
    fig.savefig(f"{OUT_BASE}.png", transparent=True, dpi=600)
    plt.close(fig)
    print(f"Saved {OUT_BASE}.pdf and {OUT_BASE}.png")


if __name__ == "__main__":
    main()
