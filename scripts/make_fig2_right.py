"""Compact qualitative panel for the workshop Fig. 2 (right): rows = samples,
cols = Transparent-only | Opaque-only | Both | GT, showing predicted depth with the
predicted rotation (solid arrow) vs ground truth (dashed arrow).

Usage:
  python scripts/make_fig2_right.py --src eval_results/vis_real_modes_sim348 \
      --indices 182,375 --out workshop_abstract/figures/preds_real.pdf
"""
from __future__ import annotations

import argparse
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLS = [("rgb", "Transparent"), ("tactile", "Opaque"), ("both", "Both"), ("gt", "GT")]


def theta_deg(p):
    return float(np.degrees(np.arctan2(p[1], p[0])))


def draw_arrow(ax, th_deg, H, W, color, ls, lw):
    c = np.array([W / 2, H / 2])
    L = 0.36 * min(H, W)
    d = np.array([np.cos(np.radians(th_deg)), -np.sin(np.radians(th_deg))]) * L
    ax.annotate("", xy=c + d, xytext=c - 0.35 * d,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                mutation_scale=7, shrinkA=0, shrinkB=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--indices", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width-in", type=float, default=2.15, help="figure width (in)")
    args = ap.parse_args()

    idxs = [int(x) for x in args.indices.split(",")]
    n = len(idxs)
    fs = args.width_in / 2.64          # font scale relative to the 2.64in reference layout
    cell = args.width_in / 4
    fig, axes = plt.subplots(n, 4, figsize=(args.width_in, cell * n + 0.12),
                             gridspec_kw=dict(wspace=0.04, hspace=0.04,
                                              left=0, right=1, bottom=0,
                                              top=1 - 0.12 / (cell * n + 0.12)))
    axes = np.atleast_2d(axes)

    for r, k in enumerate(idxs):
        d = np.load(osp.join(args.src, f"sample_{k:04d}.npz"))
        gt_d = d["gt_depth"]
        vmax = float(gt_d.max()) if gt_d.max() > 0 else None
        H, W = gt_d.shape
        gt_th = theta_deg(d["gt_pose"])
        for c, (key, title) in enumerate(COLS):
            ax = axes[r, c]
            dep = gt_d if key == "gt" else d[f"{key}_depth"]
            ax.imshow(dep, cmap="viridis", vmin=0, vmax=vmax, interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.3); s.set_edgecolor("0.6")
            if key == "gt":
                txt, col = f"θ={gt_th:.0f}°", "#7CFC00"
            else:
                th = theta_deg(d[f"{key}_pose"])
                err = abs((th - gt_th + 180) % 360 - 180)
                txt = f"θ={th:.0f}° ({err:.0f}° off)" if err >= 3 else f"θ={th:.0f}°"
                col = "white"
            ax.text(0.03, 0.04, txt, transform=ax.transAxes, fontsize=5.2 * fs, color=col,
                    ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.45))
            if r == 0:
                ax.set_title(title, fontsize=6.2 * fs, pad=1.5 * fs)

    fig.savefig(args.out, dpi=400)
    fig.savefig(osp.splitext(args.out)[0] + ".png", dpi=400)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
