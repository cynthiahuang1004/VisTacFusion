"""Rebuild the three-metric ratio figure from a frozen, traceable data snapshot.

Run from any directory. The 1:1.27 point uses the main-table Ours run;
all other points use the completed ratio-sweep runs in data/ratio_sweep.json.
Geometry uses the best-depth epoch; both pose metrics use the best-angle epoch.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "data/ratio_sweep.json").read_text())
    selected = data["main_reference"]
    rows = [selected if row["sim_per_real"] == selected["sim_per_real"] else row
            for row in data["sweep"]]
    assert all(row["epochs"] == (100 if row["sim_per_real"] == 0 else 150)
               for row in rows)
    assert selected["rot_deg"] == 0.652 and selected["trans_l1"] == 0.0053
    xs = np.arange(len(rows))
    selected_idx = next(i for i, row in enumerate(rows)
                        if row["run"] == selected["run"])
    labels = ["Real only" if row["sim_per_real"] == 0 else row["ratio_label"]
              for row in rows]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    fig, axes = plt.subplots(3, 1, figsize=(3.45, 2.65), sharex=True)
    panels = [
        ("depth_mse", 1000, "(a) Depth MSE", r"$10^{-3}$ mm$^2$", "#2876BB", "o"),
        ("rot_deg", 1, "(b) Angular error", "Degrees", "#CA5A28", "s"),
        ("trans_l1", 1000, r"(c) Translation $L_1$", r"$10^{-3}$ (normalized)", "#27815B", "D"),
    ]
    for ax, (key, scale, title, unit, color, marker) in zip(axes, panels):
        ys = np.array([row[key] * scale for row in rows])
        ax.axvspan(selected_idx - .32, selected_idx + .32, color="#ECF2E7", zorder=0)
        ax.plot(xs, ys, color=color, marker=marker, ms=3.4, lw=1.15, zorder=3)
        ax.plot(selected_idx, ys[selected_idx], "*", color=color, ms=9, zorder=4)
        ax.set_ylabel(unit.replace(" (normalized)", "\n(normalized)"),
                      fontsize=6.8, labelpad=5)
        ax.text(.98, .91, title, transform=ax.transAxes, fontsize=7,
                va="top", ha="right")
        ax.set_xticks(xs, labels, rotation=40, ha="right", fontsize=6.5)
        ax.tick_params(axis="both", length=2, width=.5, labelsize=6.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f" if key == "rot_deg" else "%.1f"))
        ax.margins(x=.06, y=.16)
        ax.grid(axis="y", color="#DDDDDD", lw=.45)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Real : Sim ratio", fontsize=7, y=.005)
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=.23, right=.93, top=.98, bottom=.23, hspace=.30)
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT / f"figures/ratio_sweep_panel.{suffix}", dpi=300)
    plt.close(fig)
    print("Plotted seven completed configurations; 1:1.27 uses the main-table run.")


if __name__ == "__main__":
    main()
