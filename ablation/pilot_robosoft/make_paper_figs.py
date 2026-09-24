"""Paper figures for the RoboSoft draft, from the pilot results.

  fig_kshot.pdf      depth MSE / contact MAE / rotation vs K real samples per object
  fig_fidelity.pdf   renderer fidelity (contact L1 on real val pairs) vs downstream depth MSE
  fig_loo.pdf        leave-object-out: depth and rotation on the three unseen objects
  fig_renders.png    same simulated contact rendered by Blender / GAN-K / GAN-full, plus a real frame

usage: python ablation/pilot_robosoft/make_paper_figs.py <out_dir>
"""
import glob
import json
import os
import os.path as osp
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_RUNS = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"
MASKED = json.load(open("ablation/pilot_robosoft/masked_results.json"))
SIM = "/media/hdd2/ihsuan/gs_blender/renders_v3"
REAL = "/media/hdd2/ihsuan/gs_blender/real_filtered"

# series -> (color, marker, linestyle); gray baseline + categorical slots 1-3
STY = {
    "Real only": ("#52514e", "o", "-"),
    "+ Blender (physical)": ("#2a78d6", "s", "-"),
    "+ GAN-K (learned, strict)": ("#eb6834", "^", "-"),
    "+ GAN-full (leaky)": ("#1baf7a", "D", "--"),
}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9a9995", "axes.grid": True, "grid.color": "#e6e5e1",
                     "grid.linewidth": 0.5, "pdf.fonttype": 42})


CROSS = json.load(open("ablation/pilot_robosoft/cross_session_results.json"))
SB = json.load(open("ablation/pilot_robosoft/sensorB_results.json"))


def prefix_of(run):
    """pilot150_ if a complete 150-epoch run exists, else pilot_."""
    p = f"{OUT_RUNS}/pilot150_{run}/history.json"
    if osp.exists(p) and len(json.load(open(p))) >= 50:
        return "pilot150_"
    return "pilot_"


def best(run, key="depth_mse", select="depth_mse", split="val"):
    h = json.load(open(f"{OUT_RUNS}/{prefix_of(run)}{run}/history.json"))
    g = lambda e, s: next(iter(e[s].values()))
    b = min(h, key=lambda e: g(e, "val")[select])
    return g(b, split)[key]


def masked(run, split="val", key="c_mae"):
    return MASKED[f"{prefix_of(run)}{run}/{split}"][key]


def fig_kshot(out):
    Ks = [0, 10, 25, 50, 100]
    series = {
        "Real only": [None, "B_k10_realonly", "B_k25_realonly", "B_k50_realonly", "B_k100_realonly"],
        "+ Blender (physical)": ["B_k0_blender", "B_k10_blender", "B_k25_blender", "B_k50_blender", "B_k100_blender"],
        "+ GAN-K (learned, strict)": [None, "B_k10_gank", "B_k25_gank", "B_k50_gank", "B_k100_gank"],
        "+ GAN-full (leaky)": ["B_k0_ganfull", "B_k10_ganfull", "B_k25_ganfull", None, "B_k100_ganfull"],
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0))
    panels = [("depth_mse", "val", "Depth MSE [mm$^2$]"), ("c_mae", "masked", "Contact MAE [mm]"),
              ("pose_rot_deg", "rot", "Rotation error [deg]")]
    for ax, (key, src, lab) in zip(axes, panels):
        for name, runs in series.items():
            c, m, ls = STY[name]
            xs, ys = [], []
            for K, r in zip(Ks, runs):
                if r is None:
                    continue
                if src == "masked":
                    y = masked(r)
                elif src == "rot":
                    y = best(r, "pose_rot_deg", select="pose_rot_deg")
                else:
                    y = best(r)
                xs.append(K); ys.append(y)
            ax.plot(xs, ys, color=c, marker=m, ls=ls, lw=1.6, ms=4.5, mec="white", mew=0.6, label=name)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks])
        ax.set_xlabel("Real samples per object, $K$"); ax.set_title(lab, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.08, 1, 1))
    fig.savefig(osp.join(out, "fig_kshot.pdf")); plt.close(fig)


def fig_fidelity(out):
    fid = {}
    for line in open("outputs/pilot_logs/renderer_fidelity_all.log"):
        p = line.split()
        if len(p) == 5 and p[0] != "generator":
            fid[p[0]] = float(p[3])
    pts = [  # (generator, downstream run, family, K)
        ("k10", "B_k10_gank", "GAN-K", 10), ("k25", "B_k25_gank", "GAN-K", 25), ("k50", "B_k50_gank", "GAN-K", 50),
        ("k100", "B_k100_gank", "GAN-K", 100),
        ("hyb_k10", "B_k10_hyb", "Blender-pretrained", 10), ("hyb_k25", "B_k25_hyb", "Blender-pretrained", 25),
        ("hyb_k100", "B_k100_hyb", "Blender-pretrained", 100),
        ("diff_k10", "B_k10_diff", "difference image", 10), ("diff_k25", "B_k25_diff", "difference image", 25),
        ("diff_k100", "B_k100_diff", "difference image", 100),
        ("bgcond_k25", "B_k25_bgcond", "bg-conditioned", 25), ("bgcond_k100", "B_k100_bgcond", "bg-conditioned", 100),
        ("full", "B_k10_ganfull", "GAN-full (leaky)", 10), ("full", "B_k25_ganfull", "GAN-full (leaky)", 25),
        ("full", "B_k100_ganfull", "GAN-full (leaky)", 100),
    ]
    sty = {"GAN-K": ("#eb6834", "^"), "Blender-pretrained": ("#2a78d6", "s"), "difference image": ("#8e5bd1", "v"),
           "bg-conditioned": ("#d6a52a", "P"), "GAN-full (leaky)": ("#1baf7a", "D")}
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    seen = set(); xs, ys = [], []
    for g, r, lab, K in pts:
        if g not in fid or not osp.exists(f"{OUT_RUNS}/{prefix_of(r)}{r}/history.json"):
            continue
        c, m = sty[lab]; x, y = fid[g], best(r); xs.append(x); ys.append(y)
        ax.scatter(x, y, color=c, marker=m, s=26, edgecolor="white", linewidth=0.6, label=None if lab in seen else lab, zorder=3)
        seen.add(lab)
        if lab == "GAN-K":
            ax.annotate(f"K={K}", (x, y), textcoords="offset points", xytext=(4, 2), fontsize=6, color="#52514e")
    from scipy.stats import spearmanr
    rho = spearmanr(xs, ys).correlation if len(xs) > 2 else float("nan")
    ax.text(0.98, 0.04, f"Spearman $\\rho$ = {rho:.2f} (n={len(xs)})", transform=ax.transAxes, ha="right", fontsize=6.5, color="#52514e")
    ax.set_xlabel("Renderer error: contact-region L1 on real pairs")
    ax.set_ylabel("Downstream depth MSE [mm$^2$]")
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    fig.tight_layout()
    fig.savefig(osp.join(out, "fig_fidelity.pdf")); plt.close(fig)


def fig_loo(out):
    rows = [("Real only", "A_realonly", "#52514e"),
            ("+ Blender", "A_blender", "#2a78d6"),
            ("+ Blender, 1 rotation/object", "A_blender_cov1", "#8fb8ea"),
            ("+ Blender, 2 rotations/object", "A_blender_cov2", "#8fb8ea"),
            ("+ GAN-LOO", "A_ganloo", "#eb6834"),
            ("+ GAN-LOO, sim w/o\nheld-out geometry", "A_ganloo_9obj", "#eb6834"),
            ("+ bg-cond. GAN-LOO", "A_bgcondloo", "#d6a52a"),
            ("+ GAN-full (leaky)", "A_ganfull", "#1baf7a")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    y = np.arange(len(rows))[::-1]
    for ax, (key, src, lab) in zip(axes, [("c_mae", "masked", "Contact MAE on unseen objects [mm]"),
                                          ("pose_rot_deg", "rot", "Rotation error on unseen objects [deg]")]):
        vals = []
        for name, r, c in rows:
            v = masked(r, "TEST") if src == "masked" else \
                best(r, "pose_rot_deg", select="pose_rot_deg", split="test")
            vals.append(v)
        ax.barh(y, vals, color=[c for _, _, c in rows], height=0.62)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.2f}" if src == "masked" else f" {v:.1f}", va="center", fontsize=6.5, color="#0b0b0b")
        ax.set_yticks(y); ax.set_yticklabels([n for n, _, _ in rows])
        ax.set_title(lab, loc="left"); ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.18)
    axes[1].set_yticklabels([])
    fig.tight_layout(w_pad=0.6)
    fig.savefig(osp.join(out, "fig_loo.pdf")); plt.close(fig)


def fig_renders(out):
    picks = [("pattern_31_rod", "session_003", 37), ("pattern_04_3_lines_angle_1", "session_002", 61)]
    cols = [("Blender + BO", "samples"), ("GAN-K, K=25", "samples_g_k25"),
            ("GAN-K, K=100", "samples_g_k100"), ("GAN-full", "samples_g")]
    tiles = []
    for obj, sess, idx in picks:
        u = f"{SIM}/{obj}/{sess}/sensor_0000"
        d = np.load(f"{u}/raw_data/{idx:04d}_gt.npy")
        dimg = (255 * (1 - np.clip(d / max(d.max(), 1e-6), 0, 1))).astype(np.uint8)
        row = [np.stack([dimg] * 3, -1)] + [np.asarray(Image.open(f"{u}/{s}/{idx:04d}.png").convert("RGB")) for _, s in cols]
        rf = sorted(glob.glob(f"{REAL}/{obj}/session_000/sensor_0000/samples/*.png"))[idx % 200]
        row.append(np.asarray(Image.open(rf).convert("RGB")))
        tiles.append(row)
    names = ["GT depth"] + [c for c, _ in cols] + ["Real (same object)"]
    fig, axes = plt.subplots(len(tiles), len(names), figsize=(7.0, 1.2 * len(tiles) + 0.25))
    for i, row in enumerate(tiles):
        for j, im in enumerate(row):
            ax = axes[i, j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(names[j], fontsize=7)
    fig.tight_layout(h_pad=0.15, w_pad=0.15)
    fig.savefig(osp.join(out, "fig_renders.png"), dpi=220); plt.close(fig)


def fig_sensorb(out):
    Ks = [10, 25, 50, 100]
    series = {"Real only (sensor B)": ("realonly", "#52514e", "o"), "+ Blender (sensor-A calibration)": ("blender", "#2a78d6", "s"),
              "+ GAN-K (sensor B, strict)": ("gan40", "#eb6834", "^")}
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9))
    for ax, (key, lab) in zip(axes, [("c_mae", "Contact MAE [mm]"), ("rot_deg_calib", "Rotation error [deg]")]):
        for name, (suf, c, m) in series.items():
            xs, ys = [], []
            for K in Ks:
                r = SB.get(f"pilot_S_k{K}_{suf}/sensorB_test")
                if r:
                    xs.append(K); ys.append(r[key])
            ax.plot(xs, ys, color=c, marker=m, lw=1.6, ms=4.5, mec="white", mew=0.6, label=name)
        ax.set_xscale("log"); ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks])
        ax.set_xlabel("Sensor-B real samples / object, $K$"); ax.set_title(lab, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=1, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=6)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.22, 1, 1))
    fig.savefig(osp.join(out, "fig_sensorb.pdf")); plt.close(fig)


def fig_cross(out):
    """Cross-session (curated 68 frames): contact IoU and rotation vs the amount of session-1 real data."""
    pts = [("Blender only", "B_k0_blender", 0), ("K=25", "B_k25_realonly", 25), ("K=50", "B_k50_realonly", 50),
           ("K=100", "B_k100_realonly", 100), ("all real", "A_realonly", 400)]
    pts_sim = [("K=25 + GAN", "B_k25_gank", 25), ("K=50 + GAN", "B_k50_gank", 50), ("K=100 + GAN", "B_k100_gank", 100), ("all + Blender", "A_blender", 400)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0))
    for ax, (key, lab) in zip(axes, [("iou", "Cross-session contact IoU"), ("rot_deg", "Cross-session rotation error [deg]")]):
        for series, c, m, name in ((pts, "#52514e", "o", "real only"), (pts_sim, "#eb6834", "^", "+ simulation")):
            xs, ys = [], []
            for _, r, K in series:
                e = CROSS.get(f"{prefix_of(r)}{r}/cross_session_curated") or CROSS.get(f"pilot_{r}/cross_session_curated")
                if e:
                    xs.append(K); ys.append(e[key])
            ax.plot(xs, ys, color=c, marker=m, lw=1.6, ms=4.5, mec="white", mew=0.6, label=name)
        ax.set_xscale("symlog", linthresh=10); ax.set_xticks([0, 25, 50, 100, 400]); ax.set_xticklabels(["0", "25", "50", "100", "all"])
        ax.set_xlabel("Session-1 real samples per object"); ax.set_title(lab, loc="left")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(osp.join(out, "fig_cross.pdf")); plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    fig_kshot(out); fig_loo(out); fig_renders(out); fig_sensorb(out); fig_cross(out)
    try:
        fig_fidelity(out)
    except Exception as e:  # noqa: BLE001
        print("fig_fidelity skipped:", e)
    print("figures written to", out)
