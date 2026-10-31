#!/usr/bin/env python3
"""Table II as small-multiple bar charts (poster figure).

Reads the numbers straight from main.tex, so the chart cannot drift from the paper.
Usage: python scripts/make_table2_barchart.py [--compact]
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path as MPath
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]

GUTTER = 1.62  # inches for model names
ROW_PITCH_PT, BAR_FRACTION = 14.0, 0.78  # row spacing in points; share of it taken by the bar
OURS, CONTEXT = "#2a78d6", "#b3b2a9"
INK, INK2, MUTED, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#eef3fb"

METRICS = [  # key, header, unit line, decimals
    ("depth", "Depth MSE", "mm²", 4),
    ("normal", "Normal MSE", "", 4),
    ("rot", "Rotation L₁", "", 4),
    ("angle", "Angular error", "degrees", 3),
    ("trans", "Translation L₁", "normalized", 4),
]
GROUPS = [("(a) Vis only", "Vision only"), ("(b) Tac only", "Tactile only"), ("(c) Both", "Vision + Tactile")]


def read_table():
    tex = "\n".join(l for l in (ROOT / "main.tex").read_text().splitlines() if not l.lstrip().startswith("%"))
    groups = []
    for marker, title in GROUPS:
        block = re.split(r"\\(?:midrule|bottomrule)", tex.split(marker, 1)[1], maxsplit=1)[0]
        rows = []
        for row in block.split(r"\\"):
            if " & " not in row:
                continue
            row = re.sub(r"\\addlinespace\[[^]]*\]|\\rowcolor\{[^}]*\}", "", row)
            cells = [c.strip() for c in row.split("&")]
            name = re.sub(r"\\cite\{[^}]*\}|\$\^\{\\dag\}\$|~", "", cells[0])
            name = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", name.replace(r"\method{}", "VisTacFusion")).strip()
            name = re.sub(r"\s*\+\s*", "+", name)
            values = [None if "---" in c else float(re.sub(r"\\textbf\{([^{}]*)\}", r"\1", c)) for c in cells[1:]]
            rows.append({"name": name, "ours": name == "VisTacFusion", "values": values})
        groups.append((title, rows))
    assert [len(r) for _, r in groups] == [9, 11, 5], "Table II structure changed"
    return groups


def rounded_barh(ax, y, width, height, color, radius_pt=3.0):
    """Horizontal bar, square at the baseline, rounded at the data end."""
    fig = ax.figure
    box = ax.get_position()
    w_pt, h_pt = box.width * fig.get_figwidth() * 72, box.height * fig.get_figheight() * 72
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    rx = min(radius_pt * (x1 - x0) / w_pt, width)
    ry = min(radius_pt * (y1 - y0) / h_pt, height / 2)
    lo, hi = y - height / 2, y + height / 2
    verts = [(0, lo), (width - rx, lo), (width, lo), (width, lo + ry), (width, hi - ry), (width, hi), (width - rx, hi), (0, hi), (0, lo)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true", help="depth, angle, translation only")
    parser.add_argument("--panel-width", type=float, help="inches per metric panel (default 2.25, compact 1.35)")
    args = parser.parse_args()
    panel_width = args.panel_width or (1.35 if args.compact else 2.25)
    keep = [0, 3, 4] if args.compact else list(range(5))
    groups = read_table()

    plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    # Layout in inches: a fixed row pitch keeps bar spacing identical in every group.
    top_in, group_gap_in, bottom_in = 1.30, 0.60, 0.30
    group_heights = [(len(rows) + 0.2) * ROW_PITCH_PT / 72 for _, rows in groups]
    fig_w = GUTTER + 0.12 + panel_width * len(keep)
    fig_h = top_in + sum(group_heights) + group_gap_in * (len(groups) - 1) + bottom_in
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    axes_w = (fig_w - GUTTER - 0.09) / (len(keep) + 0.16 * (len(keep) - 1))

    for gi, (title, rows) in enumerate(groups):
        n = len(rows)
        group_top = top_in + sum(group_heights[:gi]) + group_gap_in * gi
        for ci, mi in enumerate(keep):
            _, header, unit, decimals = METRICS[mi]
            ax = fig.add_axes([(GUTTER + ci * axes_w * 1.16) / fig_w, 1 - (group_top + group_heights[gi]) / fig_h,
                               axes_w / fig_w, group_heights[gi] / fig_h])
            values = [row["values"][mi] for row in rows]
            present = [v for v in values if v is not None]
            ax.set_xlim(0, max(present) * 1.32)
            ax.set_ylim(-0.6, n - 0.4)
            best = min(present)
            for ri, (row, value) in enumerate(zip(rows, values)):
                y = n - 1 - ri
                if row["ours"]:
                    ax.axhspan(y - 0.5, y + 0.5, color=BAND, zorder=0, lw=0)
                if value is None:
                    ax.text(0, y, " n/a", va="center", ha="left", fontsize=9, color=MUTED)
                    continue
                rounded_barh(ax, y, value, BAR_FRACTION, OURS if row["ours"] else CONTEXT)
                if row["ours"] or value == best:
                    ax.annotate(f"{value:.{decimals}f}", (value, y), xytext=(4, 0), textcoords="offset points",
                                va="center", ha="left", fontsize=9.5, color=INK if row["ours"] else INK2,
                                fontweight="bold" if row["ours"] and value == best else "normal", zorder=4)
            ax.set_yticks(range(n))
            if ci == 0:
                ax.set_yticklabels([row["name"] for row in reversed(rows)], fontsize=10.5, color=INK2)
                for label, row in zip(ax.get_yticklabels(), reversed(rows)):
                    if row["ours"]:
                        label.set_fontweight("bold")
                        label.set_color(INK)
                group_label = ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(-6, 7), textcoords="offset points",
                                          ha="right", va="bottom", fontsize=12, fontweight="bold", color=INK)
                if gi == 0:
                    first_group_label = group_label
            else:
                ax.set_yticklabels([])
            if gi == 0:
                ax.annotate(header, (0, 1), xycoords="axes fraction", xytext=(0, 24), textcoords="offset points",
                            ha="left", va="bottom", fontsize=12, fontweight="bold", color=INK)
                ax.annotate(unit, (0, 1), xycoords="axes fraction", xytext=(0, 9), textcoords="offset points",
                            ha="left", va="bottom", fontsize=9, color=MUTED)
            ax.tick_params(axis="y", length=0, pad=6)
            ax.tick_params(axis="x", length=0, labelsize=8.5, labelcolor=MUTED, pad=3)
            x_max = ax.get_xlim()[1]
            ticks = MaxNLocator(3, min_n_ticks=2).tick_values(0, x_max)
            ax.set_xticks([t for t in ticks if 0 <= t <= 0.93 * x_max])  # a tick at the far edge collides with the next panel's 0
            ax.set_xlim(0, x_max)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.grid(axis="x", color=GRID, lw=0.8, zorder=1)
            ax.set_axisbelow(True)
            for side in ("top", "right", "bottom"):
                ax.spines[side].set_visible(False)
            ax.spines["left"].set_color(AXIS)
            ax.spines["left"].set_linewidth(1.0)

    # Title and legend share the left edge of the first group label.
    fig.canvas.draw()
    head_x = first_group_label.get_window_extent().x0 / fig.bbox.width
    fig.legend(handles=[Patch(color=OURS, label="VisTacFusion (ours)"), Patch(color=CONTEXT, label="Baselines")],
               loc="upper left", bbox_to_anchor=(head_x, 1 - 0.34 / fig_h), ncol=2, frameon=False, borderpad=0, borderaxespad=0, fontsize=10.5,
               handlelength=1.1, handleheight=0.9, columnspacing=1.6, labelcolor=INK2)
    fig.text(head_x, 1 - 0.07 / fig_h, "Error on real validation data (lower is better)", ha="left", va="top",
             fontsize=13, fontweight="bold", color=INK)

    stem = "table2_barchart_compact" if args.compact else "table2_barchart"
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT / f"figures/{stem}.{suffix}", dpi=300, facecolor="white")
    print(f"wrote figures/{stem}.pdf/.png")


if __name__ == "__main__":
    main()
