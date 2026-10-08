"""Paper figures for the RoboSoft draft, from the pilot results.

  fig_kshot.pdf      depth MSE / contact MAE / rotation vs K real samples per object
  fig_fidelity.pdf   renderer fidelity (contact L1 on real val pairs) vs downstream depth MSE
  fig_loo.pdf        leave-object-out: depth and rotation on the three unseen objects
  fig_renders.png    same simulated contact rendered by Blender / GAN-K / GAN-full, plus a real frame

usage: python paper_RoboSoft/scripts/make_paper_figs.py <out_dir>
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

LW, MS, MS_STAR = 0.9, 2.6, 4.4   # shared line width / marker size of every line plot
# series -> (color, marker, linestyle); gray baseline + categorical slots 1-3
STY = {
    "Real only": ("#52514e", "o", "-"),
    "+ Blender (physical)": ("#2a78d6", "s", "-"),
    "+ GAN-K (learned, strict)": ("#eb6834", "^", "-"),
    "+ GAN-full (leaky)": ("#1baf7a", "D", "--"),
    "+ self-training": ("#c9a227", "v", "-"),
    "+ MeshTac (ours, strict)": ("#8a4fd3", "*", "-"),
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


HALF_MM = 41.0   # pose normalization of both sensors: translations are fractions of half the 82 mm object frame


def pose_ckpt(run, key, split="val"):
    """metric at the checkpoint with the lowest validation rotation error (= best_pose.pt); used for the pose metrics, as in the tables."""
    h = json.load(open(f"{OUT_RUNS}/{prefix_of(run)}{run}/history.json"))
    e = min(h, key=lambda e: next(iter(e["val"].values()))["pose_rot_deg"])
    return next(iter(e[split].values()))[key]


def masked(run, split="val", key="c_mae"):
    return MASKED[f"{prefix_of(run)}{run}/{split}"][key]


def fig_kshot(out):
    Ks = [10, 25, 50, 100]
    series = {
        "Real only": ["B_k10_realonly", "B_k25_realonly", "B_k50_realonly", "B_k100_realonly"],
        "+ Blender (physical)": ["B_k10_blender", "B_k25_blender", "B_k50_blender", "B_k100_blender"],
        "+ GAN-K (learned, strict)": ["B_k10_gank", "B_k25_gank", "B_k50_gank", "B_k100_gank"],
        "+ self-training": ["B_k10_semipm_pseudo", "B_k25_semipm_pseudo", "B_k50_semipm_pseudo", "B_k100_semipm_pseudo"],
        "+ GAN-full (leaky)": ["B_k10_ganfull", "B_k25_ganfull", "B_k50_ganfull", "B_k100_ganfull"],
        "+ MeshTac (ours, strict)": ["B_k10_rf", "B_k25_rf", "B_k50_rf", "B_k100_rf"],
    }
    names = {"+ Blender (physical)": "PBR", "+ GAN-K (learned, strict)": "Learned renderer ($K$)", "+ self-training": "Self-training",
             "+ GAN-full (leaky)": "Learned renderer (full)", "+ MeshTac (ours, strict)": "MeshTac (ours)"}
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 6, figsize=(7.1, 1.75))         # one row: narrow, tall panels so that nearby lines separate
    panels = [("c_mae", "masked", "Contact MAE"), ("pose_rot_deg", "rot", "Rotation"), ("pose_trans", "trans", "Translation")]
    letters = "abcdef"
    for i, (ax, (key, src, lab)) in enumerate(zip(axes[:3], panels)):          # sensor A
        for name, runs in series.items():
            c, m, ls = STY[name]
            xs, ys = [], []
            for K, r in zip(Ks, runs):
                if r is None or f"{prefix_of(r)}{r}/val" not in MASKED:
                    continue
                y = masked(r) if src == "masked" else pose_ckpt(r, "pose_rot_deg") if src == "rot" else pose_ckpt(r, "pose_trans") * HALF_MM if src == "trans" else best(r)
                xs.append(K); ys.append(y)
            ax.plot(xs, ys, color=c, marker=m, ls=ls, lw=LW, ms=MS_STAR if m == "*" else MS, mec="white", mew=0.5, label=names.get(name, name))
        ax.set_title(f"({letters[i]}) {lab} $\\downarrow$", loc="left", fontsize=7, pad=3)
        ax.text(0.96, 0.96, "Sensor A", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="#6b6a66")
    # sensor B: dense metrics from the depth checkpoint (sensorB_mesh), pose from best_pose.pt (sensorB_mesh_bp); trans_l1 is summed over x,y -> /2
    sbkey = {"c_mae": ("c_mae", 1.0, "sensorB_mesh"), "pose_rot_deg": ("rot_deg", 1.0, "sensorB_mesh_bp"), "pose_trans": ("trans_l1", HALF_MM / 2, "sensorB_mesh_bp")}
    for i, (ax, (key, src, lab)) in enumerate(zip(axes[3:], panels)):          # sensor B (mesh-registered labels), 50 epochs
        for name, suf in (("Real only", "realonly_m"), ("+ GAN-K (learned, strict)", "gan40m"), ("+ self-training", "semipm_m"), ("+ MeshTac (ours, strict)", "maplnr_m")):
            c, m, ls = STY[name]
            ax.plot(Ks, [SB[f"pilot_S_k{k}_{suf}/{sbkey[key][2]}"][sbkey[key][0]] * sbkey[key][1] for k in Ks], color=c, marker=m, ls=ls, lw=LW,
                    ms=MS_STAR if m == "*" else MS, mec="white", mew=0.5)
        ax.set_title(f"({letters[i + 3]}) {lab} $\\downarrow$", loc="left", fontsize=7, pad=3)
        ax.text(0.96, 0.96, "Sensor B", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="#6b6a66")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks], fontsize=6.5); ax.minorticks_off()
        ax.yaxis.set_major_locator(MaxNLocator(5)); ax.tick_params(axis="y", labelsize=6.5, pad=1.5); ax.tick_params(axis="x", pad=1.5)
        ax.set_xlabel("Labels per object, $K$", fontsize=6.8, labelpad=1.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.015), columnspacing=1.0, handletextpad=0.4, fontsize=6.8)
    fig.tight_layout(w_pad=0.45, rect=(0, 0.075, 1, 1))
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
        ("ref_k25", "B_k25_ref", "refiner (0 labels)", 25), ("ref_k100", "B_k100_ref", "refiner (0 labels)", 100),
        ("ref_ufull", "B_k25_refufull", "refiner (0 labels)", 25), ("ref_ufull", "B_k100_refufull", "refiner (0 labels)", 100),
        ("pseudo_k25", "B_k25_pseudo", "self-trained GAN-K", 25), ("pseudo_k100", "B_k100_pseudo", "self-trained GAN-K", 100),
        ("full", "B_k10_ganfull", "GAN-full (all pairs)", 10), ("full", "B_k25_ganfull", "GAN-full (all pairs)", 25),
        ("full", "B_k100_ganfull", "GAN-full (all pairs)", 100),
    ]
    sty = {"GAN-K": ("#eb6834", "^"), "Blender-pretrained": ("#2a78d6", "s"), "difference image": ("#8e5bd1", "v"),
           "bg-conditioned": ("#d6a52a", "P"), "refiner (0 labels)": ("#c2372f", "*"),
           "self-trained GAN-K": ("#3a9ad9", "X"), "GAN-full (all pairs)": ("#1baf7a", "D")}
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    seen = set(); xs, ys = [], []
    for g, r, lab, K in pts:
        if g not in fid or not osp.exists(f"{OUT_RUNS}/{prefix_of(r)}{r}/history.json"):
            continue
        c, m = sty[lab]; x, y = fid[g], best(r)
        # rho over strict depth->image renderers; the leaky one has one fidelity for many budgets, and the refiner is
        # scored against the +-3 px gel-warped physics render (misalignment floor 0.18), not against its own input
        if "all pairs" not in lab and "refiner" not in lab:
            xs.append(x); ys.append(y)
        ax.scatter(x, y, color=c, marker=m, s=34 if m == "*" else 26, edgecolor="white", linewidth=0.6, label=None if lab in seen else lab, zorder=3)
        seen.add(lab)
        if lab == "GAN-K":
            ax.annotate(f"K={K}", (x, y), textcoords="offset points", xytext=(4, 2), fontsize=6, color="#52514e")
    from scipy.stats import spearmanr
    rho = spearmanr(xs, ys).correlation if len(xs) > 2 else float("nan")
    ax.text(0.98, 0.04, f"Spearman $\\rho$ = {rho:.2f} (strict paired, n={len(xs)})", transform=ax.transAxes, ha="right", fontsize=6.5, color="#52514e")
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
            ("+ GAN-full (all pairs)", "A_ganfull", "#1baf7a")]
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


def fig_renders_small(out):
    """Column-width version for the 6-page paper: one simulated contact as GT depth | physical renderer | learned renderer (K=25) |
    all-pairs renderer, next to a real frame of the same object; two objects."""
    picks = [("pattern_31_rod", "session_003", 37), ("pattern_04_3_lines_angle_1", "session_002", 61)]
    cols = [("GT depth", None), ("PBR", "samples"), ("Learned ($K$=25)", "samples_g_k25"), ("Learned (full)", "samples_g"), ("Real frame", None)]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(3.45, 0.74 * len(picks) + 0.2))
    for i, (obj, sess, idx) in enumerate(picks):
        u = f"{SIM}/{obj}/{sess}/sensor_0000"
        d = np.load(f"{u}/raw_data/{idx:04d}_gt.npy")
        rf = sorted(glob.glob(f"{REAL}/{obj}/session_000/sensor_0000/samples/*.png"))[idx % 200]
        for j, (name, sub) in enumerate(cols):
            if sub is None and j == 0:
                im = plt.cm.viridis(np.clip(d / max(d.max(), 1e-6), 0, 1))[..., :3]
            elif sub is None:
                im = np.asarray(Image.open(rf).convert("RGB"))
            else:
                im = np.asarray(Image.open(f"{u}/{sub}/{idx:04d}.png").convert("RGB"))
            ax = axes[i, j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(name, fontsize=6.5, pad=2)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.86, bottom=0.01, wspace=0.03, hspace=0.04)
    fig.savefig(osp.join(out, "fig_renders_small.pdf"), dpi=300); plt.close(fig)


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
            ax.plot(xs, ys, color=c, marker=m, lw=LW, ms=MS_STAR if m == "*" else MS, mec="white", mew=0.5, label=name)
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
            ax.plot(xs, ys, color=c, marker=m, lw=LW, ms=MS_STAR if m == "*" else MS, mec="white", mew=0.5, label=name)
        ax.set_xscale("symlog", linthresh=10); ax.set_xticks([0, 25, 50, 100, 400]); ax.set_xticklabels(["0", "25", "50", "100", "all"])
        ax.set_xlabel("Session-1 real samples per object"); ax.set_title(lab, loc="left")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(osp.join(out, "fig_cross.pdf")); plt.close(fig)


def fig_align(out, K=25):
    """Qualitative: unlabeled frame | teacher depth | mesh at the teacher pose | mesh at the aligned pose (MeshTac label) | reference."""
    sys.path.insert(0, "scripts")
    import build_real_filtered_new as B
    B.VIEW = 0.017502
    G = "/media/hdd2/ihsuan/gs_blender"
    fits = json.load(open(f"{G}/real_pseudo_k{K}_ro_al/rc_fits.json"))
    objs = ["pattern_01_2_lines_angle_1_2", "pattern_06_5_lines_angle_1", "pattern_36"]
    rows = []
    for o in objs:
        cand = sorted(t for t in fits if f"/{o}/" in t)
        def iou_of(t):
            f = fits[t]; gt = np.load(t.replace("samples", "raw_data").replace(".png", "_gt.npy")) * 1000
            m = B.gt_depth_mm(o, *f["pose"], f["p"]); a, b = m > 0.05, gt > 0
            return (a & b).sum() / max(1, (a | b).sum())
        sc = sorted((iou_of(t), t) for t in cand[::7])
        rows.append(sc[len(sc) // 2][1])                       # the object's median-quality frame
    fig, axes = plt.subplots(len(rows), 5, figsize=(3.45, 0.64 * len(rows) + 0.24))   # vertical colorbar on the right
    titles = ["unlabeled\nframe", "teacher\ndepth", "mesh at\nteacher pose", "mesh at\naligned pose", "GT mesh\ndepth"]
    for r, t in enumerate(rows):
        f = fits[t]; o = t.split("/")[-5]
        gt = np.load(t.replace("samples", "raw_data").replace(".png", "_gt.npy")) * 1000
        td = np.load(t.replace(REAL, f"{G}/real_pseudo_k{K}_ro_ng").replace("samples", "raw_data").replace(".png", "_gt.npy")) * 1000
        tp = f["tp"]; m0 = B.gt_depth_mm(o, tp[0], tp[1], tp[2], f["p"]); m1 = B.gt_depth_mm(o, *f["pose"], f["p"])
        for c, im in enumerate([np.array(Image.open(t).convert("RGB")), td, m0, m1, gt]):
            ax = axes[r, c]
            ax.imshow(im) if c == 0 else ax.imshow(im, cmap="viridis", vmin=0, vmax=1.2)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], fontsize=5.5, pad=2, linespacing=1.0)
    fig.subplots_adjust(left=0.005, right=0.885, top=0.885, bottom=0.01, wspace=0.03, hspace=0.03)
    from matplotlib.cm import ScalarMappable
    cax = fig.add_axes([0.905, 0.05, 0.016, 0.78])
    cb = fig.colorbar(ScalarMappable(norm=plt.Normalize(0, 1.2), cmap="viridis"), cax=cax,
                      orientation="vertical", ticks=[0, 0.4, 0.8, 1.2], extend="max")
    cb.ax.tick_params(labelsize=5.5, pad=1, length=2)
    cb.set_label("Depth [mm]", fontsize=5.5, labelpad=2)
    fig.savefig(osp.join(out, "fig_align.pdf"), dpi=300, bbox_inches="tight", pad_inches=0.01); plt.close(fig)


def _label_quality():
    """(teacher IoU, aligned IoU, teacher pose err, aligned pose err) per budget, from the labeling logs (--eval-gt)."""
    import re
    def read(path):
        t = open(path).read()
        g = lambda k: [float(v) for v in re.search(rf"{re.escape(k)}\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", t).groups()]
        te, me = g("teacher (all)"), g("mesh (kept)")
        return te[0], me[0], None, te[1], me[1], None, te[2], me[2]
    L = "outputs/pilot_logs"
    A = {K: read(f"{L}/pseudo_k{K}_ro_al.log") for K in (10, 25, 50, 100)}             # real-only teacher + real-only aligner (the MeshTac pipeline)
    Bq = {K: read(f"{L}/pseudo_S_k{K}_m_al.log") for K in (10, 25, 50, 100)}
    return A, Bq


def fig_labels(out):
    """Label quality on the unlabeled frames: the K-shot teacher's own prediction vs the mesh at the aligned pose, both sensors."""
    A, Bq = _label_quality(); Ks = [10, 25, 50, 100]
    series = [("Teacher, sensor A", A, 0, "#52514e", "o", 4.2), ("Aligned mesh, sensor A", A, 1, "#8a4fd3", "*", 6.8),
              ("Teacher, sensor B", Bq, 0, "#eb6834", "s", 4.0), ("Aligned mesh, sensor B", Bq, 1, "#1baf7a", "D", 3.8)]
    fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.48))
    for ax, (i, lab) in zip(axes, [(0, "Label IoU $\\uparrow$"), (3, "Contact MAE [mm] $\\downarrow$"), (6, "Pose error [deg] $\\downarrow$")]):
        for name, D, off, c, m, ms in series:
            ax.plot(Ks, [D[k][i + off] for k in Ks], color=c, marker=m, ls="-", lw=LW, ms=MS_STAR if m == "*" else MS, mec="white", mew=0.5, label=name)
        ax.set_xscale("log"); ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks]); ax.minorticks_off()
        ax.set_xlabel("$K$"); ax.set_title(lab, loc="left", fontsize=6.6, pad=3)
    axes[1].set_ylim(0, None); axes[2].set_ylim(0, None)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.03), columnspacing=1.2, handletextpad=0.4,
               handlelength=2.0, fontsize=6.5)
    fig.tight_layout(w_pad=0.6, rect=(0, 0.19, 1, 1))
    fig.savefig(osp.join(out, "fig_labels.pdf")); plt.close(fig)


def fig_conv(out):
    """Aligner convergence (sensor A, K=25, aligner trained on real examples only; data: scripts/aligner_convergence.py).
    (a) rotation error vs. iteration from the K-shot teacher's predicted pose on the 472 validation frames;
    (b) rotation error after six iterations vs. the magnitude of a rotation perturbation of the GT pose (plus +-1.5 mm shift)."""
    d = json.load(open("ablation/pilot_robosoft/aligner_convergence.json")); N = d["iters"]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.28))
    ax = axes[0]; t = d["teacher"]["rot"]
    ax.plot(range(N + 1), t["mean"], color="#8a4fd3", marker="*", ms=MS_STAR, mec="white", mew=0.5, lw=LW, label="mean")
    ax.plot(range(N + 1), t["median"], color="#52514e", marker="o", ms=MS, mec="white", mew=0.5, lw=LW, label="median")
    ax.axvline(6, color="#b5b3ad", lw=0.8, ls=":"); ax.set_xticks([0, 2, 4, 6, 8, 10]); ax.set_ylim(0, None)
    ax.set_xlabel("Aligner iterations"); ax.set_title("(a) From teacher pose [deg]", loc="left", fontsize=6.6, pad=3); ax.legend(frameon=False, fontsize=6)
    ax = axes[1]; R = sorted(int(r) for r in d["basin"])
    ax.plot(R, [d["basin"][str(r)]["rot"]["mean"][6] for r in R], color="#8a4fd3", marker="*", ms=MS_STAR, mec="white", mew=0.5, lw=LW, label="mean")
    ax.plot(R, [d["basin"][str(r)]["rot"]["median"][6] for r in R], color="#52514e", marker="o", ms=MS, mec="white", mew=0.5, lw=LW, label="median")
    ax.axvspan(0, 8, color="#8a4fd3", alpha=0.08, lw=0); ax.set_ylim(0, None); ax.set_xticks([0, 8, 15, 20, 25, 30])
    ax.set_xlabel("Initial rotation error [deg]"); ax.set_title("(b) After 6 iterations [deg]", loc="left", fontsize=6.6, pad=3)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(osp.join(out, "fig_conv.pdf")); plt.close(fig)


def fig_poseref(out):
    """Test-time pose refinement (replaces the pose-refinement table): paired bars pred. / +align per K for real-only and MeshTac
    (renderer-free) predictors, rotation and translation, session-1 validation and session-2 test (data: eval_pose_refine.py)."""
    PRJ = json.load(open("ablation/pilot_robosoft/pose_refine_results.json"))
    Ks = [10, 25, 50, 100]
    meth = [("Real only", "B_k{K}_realonly", "#52514e"), ("MeshTac (ours)", "B_k{K}_rf_realonly", "#8a4fd3")]
    panels = [("rot", "Rotation [deg]", "val", "session 1"), ("rot", "Rotation [deg]", "session2", "session 2"),
              ("trans", "Translation [mm]", "val", "session 1"), ("trans", "Translation [mm]", "session2", "session 2")]
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 2.0)); axes = axes.ravel()
    w = 0.19; x = np.arange(len(Ks))
    for i, (ax, (key, lab, split, sname)) in enumerate(zip(axes, panels)):
        sc = HALF_MM if key == "trans" else 1.0
        for j, (name, run, c) in enumerate(meth):
            pre = [PRJ[f"pilot150_{run.format(K=K)}/{split}"][key] * sc for K in Ks]
            post = [PRJ[f"pilot150_{run.format(K=K)}/{split}"][key + "_al"] * sc for K in Ks]
            off = (j - 0.5) * 2 * w
            ax.bar(x + off - w / 2, pre, w, color="white", edgecolor=c, lw=0.8, label=name if i == 0 else None)
            ax.bar(x + off + w / 2, post, w, color=c, edgecolor=c, lw=0.8, label=f"{name} + align" if i == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels(Ks)
        if i >= 2:
            ax.set_xlabel("Labels per object, $K$")
        ax.set_title(f"({'abcd'[i]}) {lab}, {sname}", loc="left", fontsize=6.6, pad=3); ax.set_ylim(0, None)
        ax.grid(axis="y", color="#e6e4df", lw=0.5); ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.02), handlelength=1.3, columnspacing=1.0)
    fig.tight_layout(w_pad=0.5, h_pad=0.6, rect=(0, 0.1, 1, 1))
    fig.savefig(osp.join(out, "fig_poseref.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_qual(out):
    """Held-out frames of both sensors at K=25: frame | real only | MeshTac (renderer-free) | GT.  Per object, the frame with the
    object's median MeshTac contact MAE (data: paper_RoboSoft/scripts/dump_qual_preds.py)."""
    sets = [("A", "B_k25_realonly", "B_k25_rf", ["pattern_01_2_lines_angle_2", "pattern_32", "pattern_36"], "Sensor A"),
            ("B", "S_k25_realonly_m", "S_k25_maplnr_m", ["pattern_04_3_lines_angle_1", "pattern_33", "pattern_06_5_lines_angle_1"], "Sensor B (GelSlim 4.0)")]
    fig = plt.figure(figsize=(3.45, 1.85))                      # cells square -> no gaps between the tiles of a group
    outer = fig.add_gridspec(1, 2, left=0.003, right=0.997, top=0.875, bottom=0.2, wspace=0.05)
    for g, (name, r0, r1, objs, title) in zip(outer, sets):
        d = np.load(f"outputs/robosoft_qual/qual_{name}.npz")
        inner = g.subgridspec(3, 4, wspace=0.012, hspace=0.02)
        err = d[f"cmae_{r1}"]                                       # select by MeshTac's own error, not by its gain over the baseline
        for r, o in enumerate(objs):
            ids = [i for i, k in enumerate(d["keys"]) if f"/{o}/" in str(k)]
            i = sorted(ids, key=lambda j: err[j])[len(ids) // 2]
            vmax = 1.2  # Fixed depth scale across all examples and both sensors.
            for c, im in enumerate([d["img"][i], d[f"pred_{r0}"][i].astype(np.float32), d[f"pred_{r1}"][i].astype(np.float32), d["gt"][i].astype(np.float32)]):
                ax = fig.add_subplot(inner[r, c])
                ax.imshow(im) if c == 0 else ax.imshow(im, cmap="viridis", vmin=0, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0:
                    ax.set_title(["frame", "real only", "MeshTac", "GT"][c], fontsize=5.2, pad=1.5)
                if c in (1, 2):
                    e = float(d[f"cmae_{(r0, r1)[c - 1]}"][i])
                    import matplotlib.patheffects as pe
                    ax.text(0.96, 0.05, f"{e:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=4.6, color="white",
                            path_effects=[pe.withStroke(linewidth=0.9, foreground="black")])
        bb = g.get_position(fig)
        fig.text((bb.x0 + bb.x1) / 2, 0.995, title, ha="center", va="top", fontsize=6, fontweight="bold")
    from matplotlib.cm import ScalarMappable
    cax = fig.add_axes([0.33, 0.115, 0.34, 0.022])
    cb = fig.colorbar(ScalarMappable(norm=plt.Normalize(0, 1.2), cmap="viridis"), cax=cax,
                      orientation="horizontal", ticks=[0, 0.4, 0.8, 1.2], extend="max")
    cb.ax.tick_params(labelsize=5, pad=1, length=1.5)
    cb.set_label("Depth [mm]", fontsize=5, labelpad=0.5)
    fig.savefig(osp.join(out, "fig_qual.pdf"), dpi=300); plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    fig_kshot(out); fig_loo(out); fig_renders(out); fig_sensorb(out); fig_cross(out); fig_align(out); fig_labels(out); fig_qual(out); fig_conv(out); fig_poseref(out)
    try:
        fig_fidelity(out)
    except Exception as e:  # noqa: BLE001
        print("fig_fidelity skipped:", e)
    print("figures written to", out)
