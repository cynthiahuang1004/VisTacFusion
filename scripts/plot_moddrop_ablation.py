"""Modality-dropout / DPT-injection ablation figures (SiTR+MAE, sim148, crop 0.816, rule B).

Produces (eval_results/):
  moddrop_inject_sweep.png        - p_dpt_inject sweep at the default mix, 3 modes x 4 metrics
  moddrop_vs_default_heatmap.png  - % change vs default for every setting, all modes
  moddrop_effective_injection.png - both/tactile dense vs the two governing fractions
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
ns = {}
exec(open(os.path.join(ROOT, "scripts/plot_ratio_ladder.py")).read().split("def gather")[0], ns)
load = ns["load_best_per_mode"]

OUT = "/media/hdd/ihsuan/VisTacFusion_outputs"
OUT2 = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"
FIG = os.path.join(ROOT, "eval_results")

# palette (dataviz reference instance): categorical slots 1-3, diverging blue<->red, gray midpoint
C_BOTH, C_TAC, C_RGB = "#2a78d6", "#eb6834", "#1baf7a"
DIV_BLUE, DIV_GRAY, DIV_RED = "#1c5cab", "#f0efec", "#d03b3b"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
MODE_COLOR = {"both": C_BOTH, "tactile": C_TAC, "rgb": C_RGB}
MODE_LABEL = {"both": "Both (RGB + tactile)", "tactile": "Tactile-only", "rgb": "RGB-only"}
METRICS = [("depth_mse", "Depth MSE"), ("normal_mse", "Normal MSE"),
           ("rot_deg", "Rot error (deg)"), ("trans_l1", "Trans L1")]


def run_path(n):
    return f"{OUT}/ablation_c816_sim148_sitr_mae" if n == "default" else f"{OUT2}/ablation_c816_sim148_sitr_mae_{n}"


def get(n):
    return load(f"{run_path(n)}/history.json")


# label, p_both, p_tac, p_rgb, inject, [runs]
SETTINGS = [
    ("no dropout, inject 1.0",   1.00, 0.00, 0.00, 1.00, ["md_none"]),
    ("no dropout, inject 0.5",   1.00, 0.00, 0.00, 0.50, ["md_none_inj05"]),
    ("0.90/0/0.10  (no tactile)", 0.90, 0.00, 0.10, 0.50, ["md_notac"]),
    ("0.80/0.15/0.05",           0.80, 0.15, 0.05, 0.50, ["md_both80", "md_both80_seed1"]),
    ("0.55/0.45/0  (no rgb)",    0.55, 0.45, 0.00, 0.50, ["md_norgb", "md_norgb_seed1"]),
    ("0.40/0.30/0.30",           0.40, 0.30, 0.30, 0.50, ["md_rgbheavy"]),
    ("0.34/0.33/0.33",           0.34, 0.33, 0.33, 0.50, ["md_uniform"]),
    ("0.30/0.60/0.10",           0.30, 0.60, 0.10, 0.50, ["md_tacheavy", "md_tacheavy_seed1"]),
    ("default, inject 0",        0.55, 0.35, 0.10, 0.00, ["md_inject0", "md_inject0_seed1"]),
    ("default, inject 0.25",     0.55, 0.35, 0.10, 0.25, ["md_inject025", "md_inject025_seed1"]),
    ("default  (0.55/0.35/0.10, inject 0.5)", 0.55, 0.35, 0.10, 0.50, ["default", "seed1"]),
    ("default, inject 0.75",     0.55, 0.35, 0.10, 0.75, ["md_inject075", "md_inject075_seed1"]),
    ("default, inject 1.0",      0.55, 0.35, 0.10, 1.00, ["md_inject1", "md_inject1_seed1"]),
    ("0.70/0.25/0.05, inject 0.75", 0.70, 0.25, 0.05, 0.75, ["md_b70_inj075", "md_b70_inj075_seed1"]),
    ("0.70/0.25/0.05, inject 1.0", 0.70, 0.25, 0.05, 1.00, ["md_b70_inj1"]),
]
MODES = ["both", "tactile", "rgb"]


def stats(runs, mode, key):
    v = np.array([get(r)[mode][key] for r in runs])
    return v.mean(), v.min(), v.max(), len(v)


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ Fig 1: inject sweep
def fig_inject_sweep():
    sweep = [(0.0, ["md_inject0", "md_inject0_seed1"]), (0.25, ["md_inject025", "md_inject025_seed1"]), (0.5, ["default", "seed1"]),
             (0.75, ["md_inject075", "md_inject075_seed1"]), (1.0, ["md_inject1", "md_inject1_seed1"])]
    fig, axes = plt.subplots(3, 4, figsize=(12.5, 8.2))
    for r, mode in enumerate(MODES):
        for c, (key, lab) in enumerate(METRICS):
            ax = axes[r, c]
            xs = [s[0] for s in sweep]
            mu, lo, hi, n = zip(*[stats(runs, mode, key) for _, runs in sweep])
            mu, lo, hi = map(np.array, (mu, lo, hi))
            col = MODE_COLOR[mode]
            # default reference: dashed mean + shaded seed range
            d_mu, d_lo, d_hi, _ = stats(["default", "seed1"], mode, key)
            ax.axhspan(d_lo, d_hi, color=col, alpha=0.10, lw=0)
            ax.axhline(d_mu, color=INK2, lw=1, ls=(0, (4, 3)))
            ax.plot(xs, mu, color=col, lw=2, zorder=3)
            ax.errorbar(xs, mu, yerr=[mu - lo, hi - mu], fmt="none", ecolor=col, elinewidth=1.2,
                        capsize=3, zorder=3)
            ax.scatter(xs, mu, s=38, color=col, zorder=4, edgecolor="white", linewidth=1.2)
            # highlight the recommended point
            i75 = xs.index(0.75)
            ax.scatter([0.75], [mu[i75]], s=150, facecolor="none", edgecolor=INK, linewidth=1.4, zorder=5)
            for x, m in zip(xs, mu):
                ax.annotate(f"{m:.4f}" if key != "rot_deg" else f"{m:.3f}", (x, m),
                            textcoords="offset points", xytext=(0, 9), ha="center",
                            fontsize=7, color=INK2)
            ax.set_xticks(xs)
            ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"])
            style_ax(ax)
            if r == 0:
                ax.set_title(lab, fontsize=10, color=INK, pad=8)
            if r == 2:
                ax.set_xlabel("p_dpt_inject", fontsize=9, color=INK2)
            if c == 0:
                ax.set_ylabel(MODE_LABEL[mode], fontsize=10, color=col, fontweight="bold")
            ymin, ymax = min(lo.min(), d_lo), max(hi.max(), d_hi)
            pad = (ymax - ymin) * 0.35 + 1e-9
            ax.set_ylim(ymin - pad * 0.4, ymax + pad)
    fig.suptitle("DPT RGB-injection sweep at the default dropout mix 0.55 / 0.35 / 0.10   "
                 "(dashed = default inject 0.5, band = its seed range; ring = inject 0.75; bars = seed range)",
                 fontsize=10.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(FIG, "moddrop_inject_sweep.png")
    fig.savefig(p, dpi=170)
    print("Saved:", p)


# ------------------------------------------------------------------ Fig 2: heatmap vs default
def fig_heatmap():
    cols = [("both", "depth_mse", "Both\nDepth"), ("both", "normal_mse", "Both\nNormal"),
            ("both", "rot_deg", "Both\nRot"), ("both", "trans_l1", "Both\nTrans"),
            ("tactile", "depth_mse", "Tactile\nDepth"), ("tactile", "normal_mse", "Tactile\nNormal"),
            ("tactile", "rot_deg", "Tactile\nRot"), ("tactile", "trans_l1", "Tactile\nTrans"),
            ("rgb", "depth_mse", "RGB\nDepth"), ("rgb", "rot_deg", "RGB\nRot"), ("rgb", "trans_l1", "RGB\nTrans")]
    base = {(m, k): stats(["default", "seed1"], m, k) for m, k, _ in cols}
    rows = [s for s in SETTINGS if not s[0].startswith("default  (")]
    M = np.zeros((len(rows), len(cols)))
    within = np.zeros_like(M, dtype=bool)
    nseed = []
    for i, (lab, pb, pt, pr, pi, runs) in enumerate(rows):
        nseed.append(len(runs))
        for j, (m, k, _) in enumerate(cols):
            mu, lo, hi, n = stats(runs, m, k)
            b_mu, b_lo, b_hi, _ = base[(m, k)]
            M[i, j] = (mu - b_mu) / b_mu * 100
            # "within noise": the two seed ranges overlap (or |delta| < default half-range)
            within[i, j] = (lo <= b_hi and hi >= b_lo) or abs(mu - b_mu) <= (b_hi - b_lo) / 2

    cmap = LinearSegmentedColormap.from_list("div", [DIV_BLUE, DIV_GRAY, DIV_RED])
    norm = TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)
    fig, ax = plt.subplots(figsize=(12.5, 7.6))
    ax.imshow(np.clip(M, -50, 50), cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            if v > 100:
                txt, col = f"+{v:.0f}%", "white"
            else:
                txt = f"{v:+.0f}%"
                col = "white" if abs(v) > 32 else INK
            if within[i, j]:
                txt += "\n≈"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.6, color=col,
                    fontweight="bold" if (abs(v) > 15 and not within[i, j]) else "normal")
    # gridlines between cells (2px surface gap)
    for i in range(len(rows) + 1):
        ax.axhline(i - 0.5, color="white", lw=2)
    for j in range(len(cols) + 1):
        ax.axvline(j - 0.5, color="white", lw=2)
    for j in (4, 8):
        ax.axvline(j - 0.5, color=INK2, lw=1.2)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[2] for c in cols], fontsize=8.5, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[0]}  ({n} seed{'s' if n > 1 else ''})" for r, n in zip(rows, nseed)],
                       fontsize=8.5, color=INK)
    # highlight the recommended row
    i75 = [r[0] for r in rows].index("default, inject 0.75")
    ax.add_patch(plt.Rectangle((-0.5, i75 - 0.5), len(cols), 1, fill=False, edgecolor=INK, lw=2.2, zorder=5))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Change vs. default (0.55 / 0.35 / 0.10, inject 0.5)  —  blue = better (lower error), "
                 "red = worse;  ≈ = within seed noise;  color clipped at ±50%",
                 fontsize=10, color=INK, pad=34)
    fig.tight_layout()
    p = os.path.join(FIG, "moddrop_vs_default_heatmap.png")
    fig.savefig(p, dpi=170)
    print("Saved:", p)


# ------------------------------------------------------------------ Fig 3: governing fractions
def fig_effective():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    groups = {"inject sweep (default mix)": ("md_inject0", "md_inject0_seed1", "md_inject025", "md_inject025_seed1", "default", "seed1", "md_inject075",
                                             "md_inject075_seed1", "md_inject1", "md_inject1_seed1"),
              "dropout-mix variants": ("md_both80", "md_both80_seed1", "md_norgb", "md_norgb_seed1", "md_rgbheavy",
                                       "md_uniform", "md_tacheavy", "md_tacheavy_seed1", "md_notac", "md_b70_inj1", "md_b70_inj075", "md_b70_inj075_seed1"),
              "no dropout": ("md_none", "md_none_inj05")}
    cfg = {"md_none": (1, 0, 0, 1), "md_none_inj05": (1, 0, 0, .5), "md_notac": (.9, 0, .1, .5),
           "md_both80": (.8, .15, .05, .5), "md_both80_seed1": (.8, .15, .05, .5), "md_norgb": (.55, .45, 0, .5),
           "md_norgb_seed1": (.55, .45, 0, .5), "md_rgbheavy": (.4, .3, .3, .5), "md_uniform": (.34, .33, .33, .5),
           "md_tacheavy": (.3, .6, .1, .5), "md_tacheavy_seed1": (.3, .6, .1, .5), "md_inject0": (.55, .35, .1, 0), "md_inject0_seed1": (.55, .35, .1, 0), "md_inject025_seed1": (.55, .35, .1, .25), "md_b70_inj075": (.7, .25, .05, .75), "md_b70_inj075_seed1": (.7, .25, .05, .75),
           "md_inject025": (.55, .35, .1, .25), "default": (.55, .35, .1, .5), "seed1": (.55, .35, .1, .5),
           "md_inject075": (.55, .35, .1, .75), "md_inject075_seed1": (.55, .35, .1, .75),
           "md_inject1": (.55, .35, .1, 1), "md_inject1_seed1": (.55, .35, .1, 1), "md_b70_inj1": (.7, .25, .05, 1)}
    gcol = {"inject sweep (default mix)": C_BOTH, "dropout-mix variants": C_TAC, "no dropout": C_RGB}
    gmk = {"inject sweep (default mix)": "o", "dropout-mix variants": "s", "no dropout": "^"}
    for ax, (mode, xf, xlabel, title) in zip(axes, [
            ("both", lambda pb, pt, pr, pi: pb * pi, "effective injection rate  =  p_both × p_dpt_inject",
             "Both-mode depth vs. how often the DPT trains WITH RGB injection"),
            ("tactile", lambda pb, pt, pr, pi: pt + pb * (1 - pi), "un-injected fraction  =  p_tac + p_both × (1 − p_dpt_inject)",
             "Tactile-mode depth vs. how often the DPT trains WITHOUT injection")]):
        xs_all, ys_all = [], []
        for g, runs in groups.items():
            xs = [xf(*cfg[r]) for r in runs]
            ys = [get(r)[mode]["depth_mse"] for r in runs]
            ax.scatter(xs, ys, s=46, color=gcol[g], marker=gmk[g], edgecolor="white", linewidth=1,
                       label=g, zorder=3)
            xs_all += xs; ys_all += ys
        from scipy.stats import spearmanr
        rho = spearmanr(xs_all, ys_all)[0]
        ax.text(0.98 if mode == "both" else 0.02, 0.95 if mode == "both" else 0.80,
                f"Spearman ρ = {rho:+.2f}  (n = {len(xs_all)})", transform=ax.transAxes,
                ha="right" if mode == "both" else "left", va="top", fontsize=9, color=INK2)
        offs = {"both": {"md_inject075": (6, 4), "md_none": (-8, 8), "md_b70_inj1": (-70, 8), "md_b70_inj075": (6, 4),
                         "default": (6, -2), "md_inject0": (6, 4)},
                "tactile": {"md_inject075": (-14, 12), "md_none": (6, 0), "md_b70_inj1": (6, 6), "md_b70_inj075": (-30, -16),
                            "default": (8, 8), "md_inject0": (-40, 10)}}[mode]
        for r, lab in (("md_inject075", "inject 0.75"), ("md_none", "no dropout"), ("md_b70_inj1", "0.70/0.25/0.05@1.0"), ("md_b70_inj075", "0.70/0.25/0.05@0.75"),
                       ("default", "default"), ("md_inject0", "inject 0")):
            x, y = xf(*cfg[r]), get(r)[mode]["depth_mse"]
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=offs[r], fontsize=7.5, color=INK2)
        ax.set_xlabel(xlabel, fontsize=9, color=INK2)
        ax.set_ylabel(f"{MODE_LABEL[mode]} depth MSE", fontsize=9, color=INK2)
        ax.set_title(title, fontsize=9.5, color=INK)
        style_ax(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower left" if mode == "both" else "center right")
    axes[1].set_yscale("log")
    fig.tight_layout()
    p = os.path.join(FIG, "moddrop_effective_injection.png")
    fig.savefig(p, dpi=170)
    print("Saved:", p)


# ------------------------------------------------------------------ Fig 4: inject 0.75 on a second encoder pair
def fig_encoders():
    pairs = {"SiTR + MAE": {0.5: ["default", "seed1"], 0.75: ["md_inject075", "md_inject075_seed1"]},
             "DAv2 + MAE": {0.5: [f"{OUT2}/ablation_c816_sim148_dav2_mae"],
                            0.75: [f"{OUT2}/ablation_c816_sim148_dav2_mae_md_inject075"]}}
    def st(runs, mode, key):
        v = np.array([(get(r) if not r.startswith("/") else load(f"{r}/history.json"))[mode][key] for r in runs])
        return v.mean(), v.min(), v.max()
    fig, axes = plt.subplots(3, 4, figsize=(12.5, 8.0))
    for r, mode in enumerate(MODES):
        for c, (key, lab) in enumerate(METRICS):
            ax = axes[r, c]; col = MODE_COLOR[mode]
            for xi, (enc, d) in enumerate(pairs.items()):
                m5, l5, h5 = st(d[0.5], mode, key); m7, l7, h7 = st(d[0.75], mode, key)
                ax.plot([xi - 0.18, xi + 0.18], [m5, m7], color=col, lw=2, zorder=2)
                ax.errorbar([xi - 0.18], [m5], yerr=[[m5 - l5], [h5 - m5]], fmt="o", color=col, mfc="white",
                            mew=2, ms=8, capsize=3, zorder=3)
                ax.errorbar([xi + 0.18], [m7], yerr=[[m7 - l7], [h7 - m7]], fmt="o", color=col, ms=8,
                            capsize=3, zorder=3)
                ax.annotate(f"{(m7 - m5) / m5 * 100:+.0f}%", (xi, max(m5, m7)), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=8.5, color=INK,
                            fontweight="bold" if abs(m7 - m5) / m5 > 0.10 else "normal")
            ax.set_xticks([0, 1]); ax.set_xticklabels(list(pairs), fontsize=8.5); ax.set_xlim(-0.6, 1.6)
            style_ax(ax)
            lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + (hi_ - lo_) * 0.25)
            if r == 0: ax.set_title(lab, fontsize=10, color=INK, pad=8)
            if c == 0: ax.set_ylabel(MODE_LABEL[mode], fontsize=10, color=col, fontweight="bold")
    fig.suptitle("p_dpt_inject 0.5 (hollow) → 0.75 (filled) on two encoder pairs, dropout mix 0.55 / 0.35 / 0.10   "
                 "(bars = seed range where 2 seeds; label = relative change)", fontsize=10.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(FIG, "moddrop_inject075_encoders.png"); fig.savefig(p, dpi=170); print("Saved:", p)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_inject_sweep()
    fig_heatmap()
    fig_effective()
    fig_encoders()
