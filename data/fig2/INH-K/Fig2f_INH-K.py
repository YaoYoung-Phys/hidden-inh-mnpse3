#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BCP-dipole k-space map in an orthogonal top view.

Choose one output:
    PLOT_TARGET = "top"
    PLOT_TARGET = "bottom"
    PLOT_TARGET = "colorbar"

When PLOT_TARGET = "colorbar", choose its source with:
    COLORBAR_TARGET = "top"
    COLORBAR_TARGET = "bottom"

Input file format:
    k1   k2   value

The script:
    1. Reads POSCAR and constructs reciprocal lattice vectors.
    2. Constructs the Wigner-Seitz first Brillouin zone.
    3. Finds Cartesian kx/ky extrema of the first BZ.
    4. Enlarges the displayed reciprocal-space window by VIEW_SCALE.
    5. Uses the original regular k grid directly.
    6. Periodically tiles the data to fill the full enlarged plane.
    7. Preserves the Cartesian reciprocal-space aspect ratio in a
       non-tilted rectangular plotting plane.
    8. Draws the first BZ as a thin dashed line only.

No numerical interpolation or triangulation is used.
"""

import os
import numpy as np

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt

from matplotlib import rcParams
# from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon, ConnectionPatch
from matplotlib.transforms import Affine2D
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ============================================================
# 1. USER SETTINGS
# ============================================================

# ---------------- Input / output ----------------

DATA_DIR = "./"
POSCAR_FILE = "POSCAR"
OUT_DIR = "Fig2d_BCP"

TOP_FILE = "gyro_bcp_yxx_L1.dat"
BOTTOM_FILE = "gyro_bcp_yxx_L2.dat"

# Choose: "top", "bottom", or "colorbar".
PLOT_TARGET = "top"

# "fractional": k1, k2 are reduced coordinates in b1, b2 basis.
# "cartesian" : k1, k2 are Cartesian kx, ky in Angstrom^-1.
K_COORD_TYPE = "fractional"

# ---------------- Reciprocal-space range ----------------

# Display range = VIEW_SCALE × first-BZ Cartesian kx/ky extent.
VIEW_SCALE = 1.330

# Number of periodic replicas on each side.
# TILE_RANGE = 1 is usually sufficient for VIEW_SCALE around 1.2-1.4.
# Increase to 2 only if the plane edge is not fully filled.
TILE_RANGE = 1

# ---------------- Data transform ----------------

# |v| <= threshold: v / threshold
# |v| > threshold : sign(v) * log10(|v|)
LOG_THRESHOLD = 10.0

# Unified Hidden-INH palette. Blue/red encode negative/positive response;
# the layer/state identity colours are kept here for consistency with Fig. 1-4.
COLORS = {
    "top": "#C44E52",
    "bottom": "#0072B2",
    "D1": "#009E73",
    "D2": "#8B5AA5",
    "total": "#30343B",
    "A": "#30343B",
    "text": "#202124",
    "axis": "#30343B",
    "guide": "#AAB2BB",
    "background": "#FFFFFF",
    "zero": "#F7F7F7",
}

# SIGNED_ANCHORS = [
#     (0.00, COLORS["bottom"]),
#     (0.25, "#80B9D5"),
#     (0.50, COLORS["zero"]),
#     (0.75, "#DFA0A3"),
#     (1.00, COLORS["top"]),
# ]

# CMAP_SIGNED = LinearSegmentedColormap.from_list(
#     "inh_blue_white_red",
#     SIGNED_ANCHORS,
#     N=257,
# )
USE_CUSTOM_CMAP = True
if USE_CUSTOM_CMAP:
    CMAP_SIGNED = LinearSegmentedColormap.from_list(
        "spin_transition_enhanced",
        [
            (0.00, "#2F5DA8"),
            (0.30, "#6FA0E8"),
            (0.45, "#D9E8FF"),
            (0.50, "#FFFFFF"),
            (0.55, "#F9DDDD"),
            (0.70, "#E79090"),
            (1.00, "#B44242"),
        ],
        N=256,
    )
else:
    CMAP_SIGNED = "RdBu_r"

# K-space heatmap opacity for the two layers:
# 1.0 = fully opaque; smaller values = more transparent.
# These settings affect only the colored tensor maps, not the BZ/plane boundaries.
TOP_MAP_ALPHA = 1.00
BOTTOM_MAP_ALPHA = 1.00

# Options: "both_layers", "selected_layer".
COLOR_SCALE_MODE = "both_layers"

# Optional colorbar limits for each K-space distribution.
#
# IMPORTANT:
# These limits are values after the piecewise logarithmic transform:
#   |v| <= LOG_THRESHOLD: v / LOG_THRESHOLD
#   |v| >  LOG_THRESHOLD: sign(v) * log10(|v|)
#
# Use None for automatic symmetric limits from COLOR_SCALE_MODE.
# Use a positive number to set the symmetric limit independently, for example:
#     "top":    4.0,   -> [-4.0, 4.0]
#     "bottom": 3.0,   -> [-3.0, 3.0]
# Values outside the selected range are saturated at the endpoint colours.
COLORBAR_LIMITS = {
    "top": 18.0,
    "bottom": 18.0,
}

# Select which distribution is used when PLOT_TARGET = "colorbar".
COLORBAR_TARGET = "bottom"

# 1 = use all original grid points.
# 2 = use every second original grid point.
DISPLAY_STRIDE = 1

# ---------------- Output format ----------------

SAVE_PDF = True
SAVE_PNG = True
DPI = 1200


# Plane-map canvas. The total width is enlarged to 10 cm so that
# zoomed K-space regions can be placed on both sides of the main map.
MAP_FIG_W_MM = 70.0
MAP_FIG_H_MM = 30.0

# Main orthogonal K-space plane region inside the 120 mm-wide canvas.
# The original map width is retained and centered.
MAIN_MAP_W_MM = 35.0
MAIN_MAP_H_MM = 30.0
MAIN_MAP_LEFT_MM = 0.5 * (MAP_FIG_W_MM - MAIN_MAP_W_MM)
MAIN_MAP_BOTTOM_MM = 0.5 * (MAP_FIG_H_MM - MAIN_MAP_H_MM)

# ---------------- K-space zoom region ----------------

SHOW_ZOOM_REGIONS = True

# Coordinate convention used by ZOOM_REGIONS:
#   "fractional": xmin/xmax and ymin/ymax are k1/k2 coordinates.
#   "cartesian" : xmin/xmax and ymin/ymax are kx/ky in Angstrom^-1.
#
# Only one zoom region is used here, and it is forced to be displayed
# on the right side of the main map.
#
# Adjust the following range to select the exact region of interest.
# ZOOM_COORD_TYPE = "fractional"
ZOOM_FORCE_SIDE = "right"  # "left" or "right"
ZOOM_COORD_TYPE = "cartesian"
ZOOM_REGIONS = [
    # {
    #     "xmin": -0.825,
    #     "xmax": -0.545,
    #     "ymin": -0.14,
    #     "ymax":  0.14,
    #     # "label": r"$k_x<0$",
    # },
    {
        "xmin":  0.545,
        "xmax":  0.825,
        "ymin": -0.14,
        "ymax":  0.14,
        # "label": r"$k_x>0$",
    },
]

# Zoom-panel dimensions and styling.
ZOOM_PANEL_W_MM = 15.0
ZOOM_PANEL_H_MM = 15.0
ZOOM_PANEL_SIDE_MARGIN_MM = 1.5
ZOOM_PANEL_VERTICAL_GAP_MM = 2.0

ZOOM_BOX_COLOR = "#202020"
ZOOM_BOX_LINE_WIDTH = 0.60
ZOOM_BOX_DASH_PATTERN = (2.0, 1.2)

ZOOM_PANEL_EDGE_COLOR = "#303030"
ZOOM_PANEL_EDGE_WIDTH = 0.60
ZOOM_CONNECTOR_COLOR = "#505050"
ZOOM_CONNECTOR_WIDTH = 0.45
ZOOM_LABEL_FONTSIZE = 6.0

# Standalone colorbar canvas.
CBAR_FIG_W_MM = 30.0
CBAR_FIG_H_MM = 40.0

# ---------------- Orthogonal plane geometry ----------------

# Safe empty margin around the automatically generated plane.
PLANE_MARGIN_MM = 1.0

# ---------------- BZ styling ----------------

SHOW_BZ_BOUNDARY = True
BZ_LINE_WIDTH = 0.35
BZ_DASH_PATTERN = (1.5, 1.2)
BZ_COLOR = "#202020"

PLANE_EDGE_WIDTH = 0.55
PLANE_EDGE_COLOR = "#4A4A4A"

# Optional Gamma point.
SHOW_GAMMA = False

# Optional hotspot coordinates.
# Coordinates use the same convention as input:
# fractional if K_COORD_TYPE = "fractional";
# Cartesian if K_COORD_TYPE = "cartesian".
HOTSPOTS_TOP = [
    # (0.12, -0.18, r"$H_1$"),
]

HOTSPOTS_BOTTOM = [
    # (-0.12, 0.18, r"$H_2$"),
]

HOTSPOT_COLOR = "#2D8A68"
HOTSPOT_SIZE = 30.0

# ---------------- Colorbar layout ----------------

CBAR_LEFT_MM = 4.0
CBAR_BOTTOM_MM = 5.0
CBAR_WIDTH_MM = 2.2
CBAR_HEIGHT_MM = 25.0

CBAR_LABEL = r"$\Lambda^{yxx}$(a.u.)"
CBAR_TICK_COUNT = 3


# ============================================================
# 2. GLOBAL STYLE
# ============================================================

MM_TO_INCH = 1.0 / 25.4

config = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

    "mathtext.fontset": "dejavusans",
    "mathtext.rm": "DejaVu Sans",
    "mathtext.it": "DejaVu Sans:italic",
    "mathtext.bf": "DejaVu Sans:bold",


    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,

    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,

    "legend.fontsize": 7.0,

    "axes.linewidth": 0.7,
    "lines.linewidth": 1.0,

    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,

    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,

    "xtick.direction": "out",
    "ytick.direction": "out",


    "font.size": 7.0,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "savefig.bbox": None,
    "savefig.pad_inches": 0.0,
}

rcParams.update(config)


# ============================================================
# 3. POSCAR AND RECIPROCAL LATTICE
# ============================================================

def read_poscar_lattice(poscar_file):
    """Read real-space lattice vectors from POSCAR."""
    with open(poscar_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip() for line in f if line.strip()]

    scale = float(lines[1].split()[0])

    a1 = np.array(
        [float(x) for x in lines[2].split()[:3]],
        dtype=float,
    ) * scale

    a2 = np.array(
        [float(x) for x in lines[3].split()[:3]],
        dtype=float,
    ) * scale

    a3 = np.array(
        [float(x) for x in lines[4].split()[:3]],
        dtype=float,
    ) * scale

    return a1, a2, a3


def reciprocal_lattice(a1, a2, a3):
    """Return b1, b2, b3 with b_i · a_j = 2πδ_ij."""
    volume = np.dot(a1, np.cross(a2, a3))

    if abs(volume) < 1.0e-14:
        raise ValueError("Lattice volume is too small.")

    b1 = 2.0 * np.pi * np.cross(a2, a3) / volume
    b2 = 2.0 * np.pi * np.cross(a3, a1) / volume
    b3 = 2.0 * np.pi * np.cross(a1, a2) / volume

    return b1, b2, b3


def get_reciprocal_2d(poscar_file):
    """Return in-plane reciprocal lattice vectors."""
    a1, a2, a3 = read_poscar_lattice(poscar_file)
    b1, b2, b3 = reciprocal_lattice(a1, a2, a3)

    return a1, a2, a3, b1[:2], b2[:2]


def frac_to_cart_2d(k1, k2, b1_2d, b2_2d):
    """Convert reduced reciprocal coordinates to Cartesian kx, ky."""
    kx = k1 * b1_2d[0] + k2 * b2_2d[0]
    ky = k1 * b1_2d[1] + k2 * b2_2d[1]

    return kx, ky


def cart_to_frac_2d(kx, ky, b1_2d, b2_2d):
    """Convert Cartesian kx, ky to reduced reciprocal coordinates."""
    matrix = np.column_stack([b1_2d, b2_2d])

    frac = np.linalg.solve(
        matrix,
        np.column_stack([kx, ky]).T,
    ).T

    return frac[:, 0], frac[:, 1]


# ============================================================
# 4. FIRST BRILLOUIN ZONE
# ============================================================

def polygon_area(poly):
    """Return signed polygon area."""
    x = poly[:, 0]
    y = poly[:, 1]

    return 0.5 * np.sum(
        x * np.roll(y, -1)
        - y * np.roll(x, -1)
    )


def sort_polygon_ccw(poly):
    """Sort polygon vertices counterclockwise."""
    center = np.mean(poly, axis=0)

    angle = np.arctan2(
        poly[:, 1] - center[1],
        poly[:, 0] - center[0],
    )

    poly = poly[np.argsort(angle)]

    if polygon_area(poly) < 0:
        poly = poly[::-1]

    return poly


def deduplicate_polygon(poly, tol=1.0e-10):
    """Remove duplicate neighbouring vertices."""
    if len(poly) == 0:
        return poly

    output = [poly[0]]

    for point in poly[1:]:
        if np.linalg.norm(point - output[-1]) > tol:
            output.append(point)

    if len(output) > 1:
        if np.linalg.norm(output[0] - output[-1]) < tol:
            output.pop()

    return np.array(output, dtype=float)


def clip_polygon_with_halfplane(poly, G, c, tol=1.0e-12):
    """Clip polygon using k · G <= c."""
    if len(poly) == 0:
        return poly

    output = []

    def f(point):
        return np.dot(point, G) - c

    for i in range(len(poly)):
        P = poly[i]
        Q = poly[(i + 1) % len(poly)]

        fP = f(P)
        fQ = f(Q)

        inside_P = fP <= tol
        inside_Q = fQ <= tol

        if inside_P and inside_Q:
            output.append(Q)

        elif inside_P and not inside_Q:
            t = fP / (fP - fQ)
            output.append(P + t * (Q - P))

        elif not inside_P and inside_Q:
            t = fP / (fP - fQ)
            output.append(P + t * (Q - P))
            output.append(Q)

    if len(output) == 0:
        return np.empty((0, 2), dtype=float)

    return sort_polygon_ccw(
        deduplicate_polygon(
            np.array(output, dtype=float)
        )
    )


def build_first_bz(b1_2d, b2_2d, nG=4):
    """Construct the 2D Wigner-Seitz first BZ."""
    radius = 4.0 * max(
        np.linalg.norm(b1_2d),
        np.linalg.norm(b2_2d),
    )

    polygon = np.array(
        [
            [-radius, -radius],
            [ radius, -radius],
            [ radius,  radius],
            [-radius,  radius],
        ],
        dtype=float,
    )

    reciprocal_vectors = []

    for m in range(-nG, nG + 1):
        for n in range(-nG, nG + 1):
            if m == 0 and n == 0:
                continue

            reciprocal_vectors.append(
                m * b1_2d + n * b2_2d
            )

    reciprocal_vectors = sorted(
        reciprocal_vectors,
        key=np.linalg.norm,
    )

    for G in reciprocal_vectors:
        polygon = clip_polygon_with_halfplane(
            polygon,
            G,
            0.5 * np.dot(G, G),
        )

    if len(polygon) == 0:
        raise RuntimeError("Failed to construct the first BZ.")

    return sort_polygon_ccw(
        deduplicate_polygon(polygon)
    )


def get_view_bounds_from_bz(bz_poly, scale):
    """Use Cartesian kx/ky extrema of first BZ and enlarge both ranges."""
    kx_min = np.min(bz_poly[:, 0])
    kx_max = np.max(bz_poly[:, 0])

    ky_min = np.min(bz_poly[:, 1])
    ky_max = np.max(bz_poly[:, 1])

    kx_center = 0.5 * (kx_min + kx_max)
    ky_center = 0.5 * (ky_min + ky_max)

    kx_half = 0.5 * (kx_max - kx_min) * scale
    ky_half = 0.5 * (ky_max - ky_min) * scale

    return {
        "xmin": kx_center - kx_half,
        "xmax": kx_center + kx_half,
        "ymin": ky_center - ky_half,
        "ymax": ky_center + ky_half,
    }


# ============================================================
# 5. INPUT GRID
# ============================================================

def load_tensor_file(filename):
    """Read data file with columns: k1, k2, value."""
    data = np.loadtxt(filename, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError(
            f"{filename} requires at least three columns."
        )

    return data[:, 0], data[:, 1], data[:, 2]


def canonical_fractional(values):
    """Convert reduced coordinates to [-0.5, 0.5)."""
    values = np.asarray(values, dtype=float)

    return values - np.floor(values + 0.5)


def reconstruct_regular_grid(fx, fy, values, decimals=10):
    """
    Reconstruct original regular k mesh without interpolation.

    Returns:
        fx_unique, fy_unique, Z[fy, fx]
    """
    fx_round = np.round(fx, decimals)
    fy_round = np.round(fy, decimals)

    fx_unique = np.unique(fx_round)
    fy_unique = np.unique(fy_round)

    nx = len(fx_unique)
    ny = len(fy_unique)

    ix = np.searchsorted(fx_unique, fx_round)
    iy = np.searchsorted(fy_unique, fy_round)

    z_sum = np.zeros((ny, nx), dtype=float)
    z_count = np.zeros((ny, nx), dtype=int)

    np.add.at(z_sum, (iy, ix), values)
    np.add.at(z_count, (iy, ix), 1)

    if np.any(z_count == 0):
        missing = np.count_nonzero(z_count == 0)

        raise RuntimeError(
            "Input file is not a complete regular k mesh. "
            f"Expected {ny} × {nx} points, but {missing} are missing."
        )

    Z = z_sum / z_count

    return fx_unique, fy_unique, Z


def get_fractional_grid(k1, k2, b1_2d, b2_2d):
    """Return canonical reduced coordinates from input data."""
    if K_COORD_TYPE.lower() == "fractional":
        fx = np.asarray(k1, dtype=float)
        fy = np.asarray(k2, dtype=float)

    elif K_COORD_TYPE.lower() == "cartesian":
        fx, fy = cart_to_frac_2d(
            np.asarray(k1, dtype=float),
            np.asarray(k2, dtype=float),
            b1_2d,
            b2_2d,
        )

    else:
        raise ValueError(
            "K_COORD_TYPE must be 'fractional' or 'cartesian'."
        )

    return canonical_fractional(fx), canonical_fractional(fy)


# ============================================================
# 6. DATA TRANSFORM AND PERIODIC EXTENSION
# ============================================================

def signed_piecewise_log10_transform(values, threshold):
    """
    Piecewise signed transform:
        |v| <= threshold: v / threshold
        |v| > threshold : sign(v) * log10(|v|)
    """
    values = np.asarray(values, dtype=float)

    abs_values = np.abs(values)
    output = np.zeros_like(values)

    small = abs_values <= threshold
    large = abs_values > threshold

    output[small] = values[small] / threshold
    output[large] = (
        np.sign(values[large])
        * np.log10(abs_values[large])
    )

    return output


def make_norm_from_raw(values_raw, threshold):
    """Create a zero-centred symmetric normalization from raw values."""
    absmax_raw = np.nanmax(np.abs(values_raw))

    if absmax_raw < 1.0e-30:
        absmax_raw = 1.0

    absmax_plot = signed_piecewise_log10_transform(
        np.array([absmax_raw]),
        threshold,
    )[0]

    if abs(absmax_plot) < 1.0e-12:
        absmax_plot = 1.0

    norm = TwoSlopeNorm(
        vmin=-absmax_plot,
        vcenter=0.0,
        vmax=absmax_plot,
    )

    return norm, absmax_plot


def make_colorbar_ticks(vmin, vmax, nticks):
    """Return transformed-value colorbar ticks."""
    ticks = np.linspace(vmin, vmax, nticks)

    if vmin < 0.0 < vmax:
        zero_index = np.argmin(np.abs(ticks))
        ticks[zero_index] = 0.0

    return ticks


def apply_target_colorbar_limit(plot_target, automatic_norm):
    """
    Apply an optional target-specific symmetric colorbar range.

    COLORBAR_LIMITS values are positive limits in transformed plotting
    coordinates. The colour scale is always [-limit, 0, +limit], so zero
    maps exactly to COLORS["zero"].
    """
    if plot_target not in COLORBAR_LIMITS:
        raise ValueError(
            f"COLORBAR_LIMITS is missing target '{plot_target}'."
        )

    limits = COLORBAR_LIMITS[plot_target]

    if limits is None:
        return automatic_norm

    limit = float(limits)

    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError(
            f"COLORBAR_LIMITS['{plot_target}'] must be positive and finite."
        )

    return TwoSlopeNorm(
        vmin=-limit,
        vcenter=0.0,
        vmax=limit,
    )


def extend_periodic_grid(fx, fy, Z, tile_range):
    """Periodically tile a regular fractional-coordinate grid."""
    shifts = np.arange(-tile_range, tile_range + 1)

    fx_extended = np.concatenate([
        fx + shift
        for shift in shifts
    ])

    fy_extended = np.concatenate([
        fy + shift
        for shift in shifts
    ])

    Z_x = np.concatenate(
        [Z] * len(shifts),
        axis=1,
    )

    Z_extended = np.concatenate(
        [Z_x] * len(shifts),
        axis=0,
    )

    return fx_extended, fy_extended, Z_extended


def crop_grid_to_cartesian_view(
    fx,
    fy,
    Z,
    b1_2d,
    b2_2d,
    view_bounds,
):
    """Retain periodically repeated data covering the displayed range."""
    FX, FY = np.meshgrid(
        fx,
        fy,
        indexing="xy",
    )

    KX, KY = frac_to_cart_2d(
        FX,
        FY,
        b1_2d,
        b2_2d,
    )

    dx = np.median(np.abs(np.diff(KX, axis=1)))
    dy = np.median(np.abs(np.diff(KY, axis=0)))

    if not np.isfinite(dx) or dx <= 0.0:
        dx = 0.0

    if not np.isfinite(dy) or dy <= 0.0:
        dy = 0.0

    pad_x = 2.0 * max(dx, 1.0e-8)
    pad_y = 2.0 * max(dy, 1.0e-8)

    keep = (
        (KX >= view_bounds["xmin"] - pad_x)
        & (KX <= view_bounds["xmax"] + pad_x)
        & (KY >= view_bounds["ymin"] - pad_y)
        & (KY <= view_bounds["ymax"] + pad_y)
    )

    rows = np.where(np.any(keep, axis=1))[0]
    cols = np.where(np.any(keep, axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        raise RuntimeError(
            "No periodic k-grid points overlap the requested display range."
        )

    row_min = max(rows.min() - 1, 0)
    row_max = min(rows.max() + 2, len(fy))

    col_min = max(cols.min() - 1, 0)
    col_max = min(cols.max() + 2, len(fx))

    fx_crop = fx[col_min:col_max]
    fy_crop = fy[row_min:row_max]
    Z_crop = Z[row_min:row_max, col_min:col_max]

    FX_crop, FY_crop = np.meshgrid(
        fx_crop,
        fy_crop,
        indexing="xy",
    )

    KX_crop, KY_crop = frac_to_cart_2d(
        FX_crop,
        FY_crop,
        b1_2d,
        b2_2d,
    )

    return KX_crop, KY_crop, Z_crop


# ============================================================
# 7. AUTOMATIC PARALLELOGRAM PROJECTION
# ============================================================

def get_plane_vectors(view_bounds):
    """
    Determine an orthogonal top-view plane automatically.

    The horizontal and vertical directions remain perpendicular, and the
    original Cartesian reciprocal-space aspect ratio is preserved.
    """
    delta_kx = view_bounds["xmax"] - view_bounds["xmin"]
    delta_ky = view_bounds["ymax"] - view_bounds["ymin"]

    if delta_kx <= 0.0 or delta_ky <= 0.0:
        raise ValueError(
            "Invalid reciprocal-space display range."
        )

    cartesian_ratio = delta_ky / delta_kx

    available_width = MAIN_MAP_W_MM - 2.0 * PLANE_MARGIN_MM
    available_height = MAIN_MAP_H_MM - 2.0 * PLANE_MARGIN_MM

    max_width_from_canvas_width = available_width
    max_width_from_canvas_height = available_height / cartesian_ratio

    plane_width_mm = min(
        max_width_from_canvas_width,
        max_width_from_canvas_height,
    )

    plane_height_mm = plane_width_mm * cartesian_ratio

    u_vec_mm = np.array([
        plane_width_mm,
        0.0,
    ])

    v_vec_mm = np.array([
        0.0,
        plane_height_mm,
    ])

    total_width_mm = u_vec_mm[0] + v_vec_mm[0]
    total_height_mm = v_vec_mm[1]

    origin_mm = np.array([
        MAIN_MAP_LEFT_MM
        + 0.5 * (MAIN_MAP_W_MM - total_width_mm),
        MAIN_MAP_BOTTOM_MM
        + 0.5 * (MAIN_MAP_H_MM - total_height_mm),
    ])

    return origin_mm, u_vec_mm, v_vec_mm


def get_plane_corners(origin_mm, u_vec_mm, v_vec_mm):
    """Return rectangle corners in normalized figure coordinates."""
    corners_mm = np.array([
        origin_mm,
        origin_mm + u_vec_mm,
        origin_mm + u_vec_mm + v_vec_mm,
        origin_mm + v_vec_mm,
    ])

    corners_norm = np.column_stack([
        corners_mm[:, 0] / MAP_FIG_W_MM,
        corners_mm[:, 1] / MAP_FIG_H_MM,
    ])

    return corners_norm


def make_cartesian_to_plane_transform(
    view_bounds,
    origin_mm,
    u_vec_mm,
    v_vec_mm,
    ax,
):
    """Build affine transform from Cartesian k space to the plane."""
    delta_kx = view_bounds["xmax"] - view_bounds["xmin"]
    delta_ky = view_bounds["ymax"] - view_bounds["ymin"]

    a = u_vec_mm[0] / delta_kx / MAP_FIG_W_MM
    b = u_vec_mm[1] / delta_kx / MAP_FIG_H_MM

    c = v_vec_mm[0] / delta_ky / MAP_FIG_W_MM
    d = v_vec_mm[1] / delta_ky / MAP_FIG_H_MM

    e = (
        origin_mm[0]
        - u_vec_mm[0] * view_bounds["xmin"] / delta_kx
        - v_vec_mm[0] * view_bounds["ymin"] / delta_ky
    ) / MAP_FIG_W_MM

    f = (
        origin_mm[1]
        - u_vec_mm[1] * view_bounds["xmin"] / delta_kx
        - v_vec_mm[1] * view_bounds["ymin"] / delta_ky
    ) / MAP_FIG_H_MM

    return Affine2D().from_values(
        a,
        b,
        c,
        d,
        e,
        f,
    ) + ax.transData


def project_cartesian_points(
    kx,
    ky,
    view_bounds,
    origin_mm,
    u_vec_mm,
    v_vec_mm,
):
    """Project Cartesian points to normalized figure coordinates."""
    u = (
        (kx - view_bounds["xmin"])
        / (view_bounds["xmax"] - view_bounds["xmin"])
    )

    v = (
        (ky - view_bounds["ymin"])
        / (view_bounds["ymax"] - view_bounds["ymin"])
    )

    x_mm = origin_mm[0] + u * u_vec_mm[0] + v * v_vec_mm[0]
    y_mm = origin_mm[1] + u * u_vec_mm[1] + v * v_vec_mm[1]

    return (
        x_mm / MAP_FIG_W_MM,
        y_mm / MAP_FIG_H_MM,
    )


# ============================================================
# 8. DATA PREPARATION
# ============================================================

def prepare_layer_data(filename, b1_2d, b2_2d, view_bounds):
    """Read one layer and prepare periodic direct-grid data."""
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Input tensor file not found: {file_path}"
        )

    k1, k2, values_raw = load_tensor_file(file_path)

    fx, fy = get_fractional_grid(
        k1,
        k2,
        b1_2d,
        b2_2d,
    )

    fx_grid, fy_grid, Z_raw = reconstruct_regular_grid(
        fx,
        fy,
        values_raw,
    )

    print(
        f"{filename}: "
        f"regular grid = {len(fy_grid)} × {len(fx_grid)}"
    )

    fx_tile, fy_tile, Z_tile_raw = extend_periodic_grid(
        fx_grid,
        fy_grid,
        Z_raw,
        TILE_RANGE,
    )

    KX_tile, KY_tile, Z_view_raw = crop_grid_to_cartesian_view(
        fx_tile,
        fy_tile,
        Z_tile_raw,
        b1_2d,
        b2_2d,
        view_bounds,
    )

    Z_view_plot = signed_piecewise_log10_transform(
        Z_view_raw,
        LOG_THRESHOLD,
    )

    return {
        "raw_original": Z_raw,
        "KX": KX_tile,
        "KY": KY_tile,
        "Z_plot": Z_view_plot,
    }


def get_shared_norm(top_data, bottom_data, selected_data):
    """Return color normalization according to COLOR_SCALE_MODE."""
    if COLOR_SCALE_MODE == "both_layers":
        values_raw = np.concatenate([
            top_data["raw_original"].ravel(),
            bottom_data["raw_original"].ravel(),
        ])

    elif COLOR_SCALE_MODE == "selected_layer":
        values_raw = selected_data["raw_original"].ravel()

    else:
        raise ValueError(
            "COLOR_SCALE_MODE must be 'both_layers' or 'selected_layer'."
        )

    return make_norm_from_raw(
        values_raw,
        LOG_THRESHOLD,
    )


# ============================================================
# 9. DRAWING
# ============================================================

def resolve_zoom_regions(b1_2d, b2_2d):
    """
    Convert user-defined zoom regions to Cartesian K-space polygons.

    For this version, only one zoom region is expected and the side
    panel can be forced by ZOOM_FORCE_SIDE. If ZOOM_FORCE_SIDE is None,
    the side is chosen automatically from the sign of the Cartesian kx
    coordinate at the region center.
    """
    if not SHOW_ZOOM_REGIONS:
        return []

    coord_type = ZOOM_COORD_TYPE.lower()

    if coord_type not in {"fractional", "cartesian"}:
        raise ValueError(
            "ZOOM_COORD_TYPE must be 'fractional' or 'cartesian'."
        )

    resolved = []

    for index, region in enumerate(ZOOM_REGIONS):
        required = {"xmin", "xmax", "ymin", "ymax"}
        missing = required.difference(region)

        if missing:
            raise ValueError(
                f"ZOOM_REGIONS[{index}] is missing: "
                + ", ".join(sorted(missing))
            )

        xmin = float(region["xmin"])
        xmax = float(region["xmax"])
        ymin = float(region["ymin"])
        ymax = float(region["ymax"])

        if not xmin < xmax or not ymin < ymax:
            raise ValueError(
                f"ZOOM_REGIONS[{index}] requires "
                "xmin < xmax and ymin < ymax."
            )

        # Counterclockwise corners in the coordinates selected by the user.
        input_polygon = np.array([
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ])

        if coord_type == "fractional":
            kx, ky = frac_to_cart_2d(
                input_polygon[:, 0],
                input_polygon[:, 1],
                b1_2d,
                b2_2d,
            )

            cart_polygon = np.column_stack([kx, ky])

        else:
            cart_polygon = input_polygon.copy()

        kx_min = float(np.min(cart_polygon[:, 0]))
        kx_max = float(np.max(cart_polygon[:, 0]))
        ky_min = float(np.min(cart_polygon[:, 1]))
        ky_max = float(np.max(cart_polygon[:, 1]))

        center_kx = float(np.mean(cart_polygon[:, 0]))
        center_ky = float(np.mean(cart_polygon[:, 1]))

        forced_side = ZOOM_FORCE_SIDE
        if forced_side is not None:
            forced_side = str(forced_side).lower()
            if forced_side not in {"left", "right"}:
                raise ValueError(
                    "ZOOM_FORCE_SIDE must be None, 'left', or 'right'."
                )
            side = forced_side
        else:
            # kx = 0 is assigned to the right by convention.
            side = "left" if center_kx < 0.0 else "right"

        resolved.append({
            "index": index,
            "label": region.get("label", ""),
            "side": side,
            "center_kx": center_kx,
            "center_ky": center_ky,
            "cart_polygon": cart_polygon,
            "cart_bounds": {
                "xmin": kx_min,
                "xmax": kx_max,
                "ymin": ky_min,
                "ymax": ky_max,
            },
        })

    return resolved


def assign_zoom_panel_positions(regions):
    """Assign non-overlapping left/right panel positions in millimetres."""
    output = [dict(region) for region in regions]

    for side in ("left", "right"):
        indices = [
            i for i, region in enumerate(output)
            if region["side"] == side
        ]

        count = len(indices)

        if count == 0:
            continue

        max_panel_height = (
            MAP_FIG_H_MM
            - 2.0 * ZOOM_PANEL_SIDE_MARGIN_MM
            - (count - 1) * ZOOM_PANEL_VERTICAL_GAP_MM
        ) / count

        panel_h_mm = min(
            ZOOM_PANEL_H_MM,
            max_panel_height,
        )

        if panel_h_mm <= 0.0:
            raise ValueError(
                "Too many zoom panels for the current canvas height."
            )

        if side == "left":
            side_width_mm = MAIN_MAP_LEFT_MM
            panel_w_mm = min(
                ZOOM_PANEL_W_MM,
                side_width_mm - 2.0 * ZOOM_PANEL_SIDE_MARGIN_MM,
            )
            panel_left_mm = ZOOM_PANEL_SIDE_MARGIN_MM

        else:
            main_right_mm = MAIN_MAP_LEFT_MM + MAIN_MAP_W_MM
            side_width_mm = MAP_FIG_W_MM - main_right_mm
            panel_w_mm = min(
                ZOOM_PANEL_W_MM,
                side_width_mm - 2.0 * ZOOM_PANEL_SIDE_MARGIN_MM,
            )
            panel_left_mm = (
                MAP_FIG_W_MM
                - ZOOM_PANEL_SIDE_MARGIN_MM
                - panel_w_mm
            )

        if panel_w_mm <= 0.0:
            raise ValueError(
                "The side space is too narrow for zoom panels. "
                "Increase MAP_FIG_W_MM or reduce MAIN_MAP_W_MM."
            )

        total_h_mm = (
            count * panel_h_mm
            + (count - 1) * ZOOM_PANEL_VERTICAL_GAP_MM
        )

        first_bottom_mm = 0.5 * (MAP_FIG_H_MM - total_h_mm)

        # Higher center ky is placed above lower center ky.
        indices = sorted(
            indices,
            key=lambda i: output[i]["center_ky"],
            reverse=True,
        )

        for order, region_index in enumerate(indices):
            panel_bottom_mm = (
                first_bottom_mm
                + (count - 1 - order)
                * (panel_h_mm + ZOOM_PANEL_VERTICAL_GAP_MM)
            )

            output[region_index]["panel_rect"] = [
                panel_left_mm / MAP_FIG_W_MM,
                panel_bottom_mm / MAP_FIG_H_MM,
                panel_w_mm / MAP_FIG_W_MM,
                panel_h_mm / MAP_FIG_H_MM,
            ]

    return output


def draw_zoom_regions(
    fig,
    main_ax,
    data,
    norm,
    view_bounds,
    b1_2d,
    b2_2d,
    map_alpha,
):
    """Draw boxed K-space regions and corresponding side zoom panels."""
    regions = resolve_zoom_regions(
        b1_2d,
        b2_2d,
    )

    if not regions:
        return

    regions = assign_zoom_panel_positions(regions)

    origin_mm, u_vec_mm, v_vec_mm = get_plane_vectors(
        view_bounds
    )

    for region in regions:
        cart_polygon = region["cart_polygon"]

        box_x, box_y = project_cartesian_points(
            cart_polygon[:, 0],
            cart_polygon[:, 1],
            view_bounds,
            origin_mm,
            u_vec_mm,
            v_vec_mm,
        )

        projected_polygon = np.column_stack([box_x, box_y])

        # Region box on the orthogonal main map.
        main_ax.add_patch(
            Polygon(
                projected_polygon,
                closed=True,
                fill=False,
                edgecolor=ZOOM_BOX_COLOR,
                linewidth=ZOOM_BOX_LINE_WIDTH,
                linestyle=(0, ZOOM_BOX_DASH_PATTERN),
                zorder=12,
            )
        )

        # Enlarged Cartesian K-space panel.
        inset_ax = fig.add_axes(region["panel_rect"])

        stride = max(int(DISPLAY_STRIDE), 1)

        Z_zoom = np.clip(
            data["Z_plot"][::stride, ::stride],
            norm.vmin,
            norm.vmax,
        )

        inset_ax.pcolormesh(
            data["KX"][::stride, ::stride],
            data["KY"][::stride, ::stride],
            Z_zoom,
            shading="nearest",
            cmap=CMAP_SIGNED,
            norm=norm,
            alpha=map_alpha,
            rasterized=True,
            zorder=2,
        )

        bounds = region["cart_bounds"]
        inset_ax.set_xlim(bounds["xmin"], bounds["xmax"])
        inset_ax.set_ylim(bounds["ymin"], bounds["ymax"])

        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.tick_params(length=0.0)

        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_color(ZOOM_PANEL_EDGE_COLOR)
            spine.set_linewidth(ZOOM_PANEL_EDGE_WIDTH)

        label = region["label"]

        if label:
            inset_ax.text(
                0.04,
                0.96,
                label,
                transform=inset_ax.transAxes,
                fontsize=ZOOM_LABEL_FONTSIZE,
                color=ZOOM_PANEL_EDGE_COLOR,
                ha="left",
                va="top",
                zorder=5,
            )

        # Connect the nearest two corners of the main box to the inset.
        if region["side"] == "left":
            order = np.argsort(projected_polygon[:, 0])[:2]
            panel_points = [(1.0, 0.0), (1.0, 1.0)]
        else:
            order = np.argsort(projected_polygon[:, 0])[-2:]
            panel_points = [(0.0, 0.0), (0.0, 1.0)]

        selected_main_points = projected_polygon[order]
        selected_main_points = selected_main_points[
            np.argsort(selected_main_points[:, 1])
        ]

        for main_point, panel_point in zip(
            selected_main_points,
            panel_points,
        ):
            connector = ConnectionPatch(
                xyA=tuple(main_point),
                coordsA=main_ax.transData,
                axesA=main_ax,
                xyB=panel_point,
                coordsB=inset_ax.transAxes,
                axesB=inset_ax,
                color=ZOOM_CONNECTOR_COLOR,
                linewidth=ZOOM_CONNECTOR_WIDTH,
                zorder=6,
                clip_on=False,
            )

            fig.add_artist(connector)

        print(
            "Zoom region "
            f"{region['index']}: center = "
            f"({region['center_kx']:.6f}, "
            f"{region['center_ky']:.6f}) Angstrom^-1, "
            f"placed on {region['side']}"
        )

def get_selected_hotspots():
    """Return hotspots corresponding to PLOT_TARGET."""
    if PLOT_TARGET == "top":
        return HOTSPOTS_TOP

    if PLOT_TARGET == "bottom":
        return HOTSPOTS_BOTTOM

    return []


def get_selected_map_alpha():
    """Return the heatmap opacity corresponding to PLOT_TARGET."""
    if PLOT_TARGET == "top":
        return TOP_MAP_ALPHA

    if PLOT_TARGET == "bottom":
        return BOTTOM_MAP_ALPHA

    raise ValueError(
        "Heatmap opacity is only defined for PLOT_TARGET='top' or 'bottom'."
    )


def draw_one_plane(
    ax,
    data,
    norm,
    bz_poly,
    view_bounds,
    b1_2d,
    b2_2d,
    hotspots,
    map_alpha,
):
    """
    Draw one direct-grid heatmap in an orthogonal top view.

    The heatmap is clipped only to the outer rectangle.
    It is not clipped to the first BZ.
    """
    origin_mm, u_vec_mm, v_vec_mm = get_plane_vectors(
        view_bounds
    )

    plane_corners = get_plane_corners(
        origin_mm,
        u_vec_mm,
        v_vec_mm,
    )

    clip_polygon = Polygon(
        plane_corners,
        closed=True,
        facecolor="none",
        edgecolor="none",
        transform=ax.transData,
    )

    ax.add_patch(clip_polygon)

    plane_transform = make_cartesian_to_plane_transform(
        view_bounds,
        origin_mm,
        u_vec_mm,
        v_vec_mm,
        ax,
    )

    stride = max(int(DISPLAY_STRIDE), 1)

    KX = data["KX"][::stride, ::stride]
    KY = data["KY"][::stride, ::stride]
    Z = np.clip(
        data["Z_plot"][::stride, ::stride],
        norm.vmin,
        norm.vmax,
    )

    mesh = ax.pcolormesh(
        KX,
        KY,
        Z,
        shading="nearest",
        cmap=CMAP_SIGNED,
        norm=norm,
        alpha=map_alpha,
        transform=plane_transform,
        rasterized=True,
        zorder=2,
    )

    mesh.set_clip_path(clip_polygon)

    # Outer rectangle edge.
    ax.add_patch(
        Polygon(
            plane_corners,
            closed=True,
            fill=False,
            edgecolor=PLANE_EDGE_COLOR,
            linewidth=PLANE_EDGE_WIDTH,
            zorder=7,
        )
    )

    # First-BZ boundary only; no BZ clipping.
    if SHOW_BZ_BOUNDARY:
        bz_x, bz_y = project_cartesian_points(
            bz_poly[:, 0],
            bz_poly[:, 1],
            view_bounds,
            origin_mm,
            u_vec_mm,
            v_vec_mm,
        )

        ax.plot(
            np.append(bz_x, bz_x[0]),
            np.append(bz_y, bz_y[0]),
            color=BZ_COLOR,
            linewidth=BZ_LINE_WIDTH,
            linestyle=(0, BZ_DASH_PATTERN),
            zorder=8,
        )

    # Optional Gamma marker.
    if SHOW_GAMMA:
        gx, gy = project_cartesian_points(
            0.0,
            0.0,
            view_bounds,
            origin_mm,
            u_vec_mm,
            v_vec_mm,
        )

        ax.scatter(
            [gx],
            [gy],
            s=10,
            facecolors="white",
            edgecolors=BZ_COLOR,
            linewidths=0.4,
            zorder=9,
        )

        ax.text(
            gx + 0.008,
            gy + 0.008,
            r"$\Gamma$",
            fontsize=6.0,
            color=BZ_COLOR,
            ha="left",
            va="bottom",
            zorder=10,
        )

    # Optional hotspots.
    for k1, k2, label in hotspots:
        if K_COORD_TYPE.lower() == "fractional":
            kx, ky = frac_to_cart_2d(
                np.array([k1]),
                np.array([k2]),
                b1_2d,
                b2_2d,
            )

            kx = kx[0]
            ky = ky[0]

        else:
            kx = k1
            ky = k2

        px, py = project_cartesian_points(
            kx,
            ky,
            view_bounds,
            origin_mm,
            u_vec_mm,
            v_vec_mm,
        )

        ax.scatter(
            [px],
            [py],
            s=HOTSPOT_SIZE,
            facecolors="none",
            edgecolors=HOTSPOT_COLOR,
            linewidths=0.75,
            zorder=10,
        )

        if label:
            ax.text(
                px + 0.008,
                py + 0.008,
                label,
                fontsize=6.0,
                color=HOTSPOT_COLOR,
                ha="left",
                va="bottom",
                zorder=11,
            )


def plot_map(
    selected_data,
    norm,
    bz_poly,
    view_bounds,
    b1_2d,
    b2_2d,
    map_alpha,
):
    """Generate a top or bottom BCP-dipole plane map."""
    fig_w = MAP_FIG_W_MM * MM_TO_INCH
    fig_h = MAP_FIG_H_MM * MM_TO_INCH

    fig = plt.figure(figsize=(fig_w, fig_h))

    ax = fig.add_axes([
        0.0,
        0.0,
        1.0,
        1.0,
    ])

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()

    draw_one_plane(
        ax=ax,
        data=selected_data,
        norm=norm,
        bz_poly=bz_poly,
        view_bounds=view_bounds,
        b1_2d=b1_2d,
        b2_2d=b2_2d,
        hotspots=get_selected_hotspots(),
        map_alpha=map_alpha,
    )

    draw_zoom_regions(
        fig=fig,
        main_ax=ax,
        data=selected_data,
        norm=norm,
        view_bounds=view_bounds,
        b1_2d=b1_2d,
        b2_2d=b2_2d,
        map_alpha=map_alpha,
    )

    out_base = f"BCP_dipole_{PLOT_TARGET}_plane_zoom_right"

    if SAVE_PDF:
        fig.savefig(
            os.path.join(OUT_DIR, out_base + ".pdf"),
            transparent=True,
        )

    if SAVE_PNG:
        fig.savefig(
            os.path.join(OUT_DIR, out_base + ".png"),
            transparent=True,
            dpi=DPI,
        )

    plt.close(fig)

    print(f"Written: {out_base}.pdf/png")


# def plot_colorbar(norm, plot_target):
#     """Generate a target-specific standalone horizontal colorbar."""
#     fig_w = CBAR_FIG_W_MM * MM_TO_INCH
#     fig_h = CBAR_FIG_H_MM * MM_TO_INCH

#     fig = plt.figure(figsize=(fig_w, fig_h))

#     cax = fig.add_axes([
#         CBAR_LEFT_MM / CBAR_FIG_W_MM,
#         CBAR_BOTTOM_MM / CBAR_FIG_H_MM,
#         CBAR_WIDTH_MM / CBAR_FIG_W_MM,
#         CBAR_HEIGHT_MM / CBAR_FIG_H_MM,
#     ])

#     scalar_map = ScalarMappable(
#         norm=norm,
#         cmap=CMAP_SIGNED,
#     )

#     scalar_map.set_array([])

#     cb = fig.colorbar(
#         scalar_map,
#         cax=cax,
#         orientation="horizontal",
#     )

#     cb.set_ticks(
#         make_colorbar_ticks(
#             norm.vmin,
#             norm.vmax,
#             CBAR_TICK_COUNT,
#         )
#     )

#     cb.ax.xaxis.set_major_formatter(
#         ticker.FormatStrFormatter("%.1f")
#     )

#     cb.ax.tick_params(
#         which="major",
#         direction="in",
#         length=1.8,
#         width=0.5,
#         labelsize=7.50,
#         pad=1.0,
#     )

#     cb.set_label(
#         CBAR_LABEL,
#         fontsize=8,
#         labelpad=1.2,
#     )

#     cb.outline.set_linewidth(0.5)
#     cb.outline.set_edgecolor("#404040")

#     out_base = f"BCP_dipole_colorbar_{plot_target}"

#     if SAVE_PDF:
#         fig.savefig(
#             os.path.join(OUT_DIR, out_base + ".pdf"),
#             transparent=True,
#         )

#     if SAVE_PNG:
#         fig.savefig(
#             os.path.join(OUT_DIR, out_base + ".png"),
#             transparent=True,
#             dpi=DPI,
#         )

#     plt.close(fig)

#     print(f"Written: {out_base}.pdf/png")

def plot_colorbar(norm, plot_target):
    """Generate a target-specific standalone vertical colorbar."""
    fig_w = CBAR_FIG_W_MM * MM_TO_INCH
    fig_h = CBAR_FIG_H_MM * MM_TO_INCH

    fig = plt.figure(figsize=(fig_w, fig_h))

    cax = fig.add_axes([
        CBAR_LEFT_MM / CBAR_FIG_W_MM,
        CBAR_BOTTOM_MM / CBAR_FIG_H_MM,
        CBAR_WIDTH_MM / CBAR_FIG_W_MM,
        CBAR_HEIGHT_MM / CBAR_FIG_H_MM,
    ])

    scalar_map = ScalarMappable(
        norm=norm,
        cmap=CMAP_SIGNED,
    )
    scalar_map.set_array([])

    cb = fig.colorbar(
        scalar_map,
        cax=cax,
        orientation="vertical",
    )

    cb.set_ticks(
        make_colorbar_ticks(
            norm.vmin,
            norm.vmax,
            CBAR_TICK_COUNT,
        )
    )

    # Vertical colorbar uses the y axis
    cb.ax.yaxis.set_major_formatter(
        ticker.FormatStrFormatter("%.1f")
    )

    cb.ax.tick_params(
        axis="y",
        which="major",
        direction="out",
        length=2.0,
        width=0.5,
        labelsize=7.5,
        pad=1.5,
    )

    cb.set_label(
        CBAR_LABEL,
        fontsize=8.0,
        rotation=90,
        labelpad=3.0,
    )

    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(COLORS["axis"])

    out_base = f"BCP_dipole_colorbar_vertical_{plot_target}"

    if SAVE_PDF:
        fig.savefig(
            os.path.join(OUT_DIR, out_base + ".pdf"),
            transparent=True,
        )

    if SAVE_PNG:
        fig.savefig(
            os.path.join(OUT_DIR, out_base + ".png"),
            transparent=True,
            dpi=DPI,
        )

    plt.close(fig)

    print(f"Written: {out_base}.pdf/png")

# ============================================================
# 10. MAIN
# ============================================================

def main():
    """Run selected map or standalone colorbar."""
    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    if PLOT_TARGET not in {"top", "bottom", "colorbar"}:
        raise ValueError(
            "PLOT_TARGET must be 'top', 'bottom', or 'colorbar'."
        )

    if COLORBAR_TARGET not in {"top", "bottom"}:
        raise ValueError(
            "COLORBAR_TARGET must be 'top' or 'bottom'."
        )

    for alpha_name, alpha_value in {
        "TOP_MAP_ALPHA": TOP_MAP_ALPHA,
        "BOTTOM_MAP_ALPHA": BOTTOM_MAP_ALPHA,
    }.items():
        if not 0.0 <= alpha_value <= 1.0:
            raise ValueError(
                f"{alpha_name} must be between 0.0 and 1.0."
            )

    _, _, _, b1_2d, b2_2d = get_reciprocal_2d(
        POSCAR_FILE
    )

    bz_poly = build_first_bz(
        b1_2d,
        b2_2d,
        nG=4,
    )

    view_bounds = get_view_bounds_from_bz(
        bz_poly,
        VIEW_SCALE,
    )

    origin_mm, u_vec_mm, v_vec_mm = get_plane_vectors(
        view_bounds
    )

    print("b1_2d =", b1_2d)
    print("b2_2d =", b2_2d)

    print("\nFirst-BZ Cartesian limits:")
    print(
        f"kx: {np.min(bz_poly[:, 0]): .8f} "
        f"to {np.max(bz_poly[:, 0]): .8f} Angstrom^-1"
    )
    print(
        f"ky: {np.min(bz_poly[:, 1]): .8f} "
        f"to {np.max(bz_poly[:, 1]): .8f} Angstrom^-1"
    )

    print("\nExpanded Cartesian display limits:")
    print(
        f"kx: {view_bounds['xmin']: .8f} "
        f"to {view_bounds['xmax']: .8f} Angstrom^-1"
    )
    print(
        f"ky: {view_bounds['ymin']: .8f} "
        f"to {view_bounds['ymax']: .8f} Angstrom^-1"
    )

    print("\nAutomatic orthogonal-plane geometry:")
    print(f"origin = {origin_mm}")
    print(f"U vector = {u_vec_mm}")
    print(f"V vector = {v_vec_mm}")

    top_data = prepare_layer_data(
        TOP_FILE,
        b1_2d,
        b2_2d,
        view_bounds,
    )

    bottom_data = prepare_layer_data(
        BOTTOM_FILE,
        b1_2d,
        b2_2d,
        view_bounds,
    )

    if PLOT_TARGET == "top":
        selected_data = top_data

    elif PLOT_TARGET == "bottom":
        selected_data = bottom_data

    else:
        selected_data = (
            top_data
            if COLORBAR_TARGET == "top"
            else bottom_data
        )

    selected_target = (
        COLORBAR_TARGET
        if PLOT_TARGET == "colorbar"
        else PLOT_TARGET
    )

    automatic_norm, _ = get_shared_norm(
        top_data,
        bottom_data,
        selected_data,
    )

    norm = apply_target_colorbar_limit(
        selected_target,
        automatic_norm,
    )

    limit_source = (
        "manual"
        if COLORBAR_LIMITS[selected_target] is not None
        else "automatic"
    )

    print("\nFinal transformed colorbar limits:")
    print(
        f"{selected_target}: "
        f"[{norm.vmin:.6g}, {norm.vmax:.6g}] "
        f"({limit_source})"
    )

    if PLOT_TARGET == "colorbar":
        plot_colorbar(
            norm,
            selected_target,
        )

    else:
        map_alpha = get_selected_map_alpha()

        print(
            f"{PLOT_TARGET} heatmap opacity: "
            f"{map_alpha:.3f}"
        )

        plot_map(
            selected_data,
            norm,
            bz_poly,
            view_bounds,
            b1_2d,
            b2_2d,
            map_alpha,
        )


if __name__ == "__main__":
    main()
