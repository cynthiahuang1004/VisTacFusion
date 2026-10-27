"""Video versions of the paper's result figures: dark background, large fonts, and short animations rendered from the same
data as make_paper_figs.py / fig_qual_plus4.py (no paper-figure screenshots).  Writes paper_RoboSoft/video/plots/:
  kshot_anim.mp4 / kshot_final.png   Fig. 4: contact MAE vs K on both sensors, one method revealed at a time
  labels_anim.mp4 / labels_final.png Fig. 5: teacher vs aligned-mesh label quality, teacher first, aligned curves added
  align_toggle.mp4                   Fig. 3: mesh at the teacher pose <-> mesh at the aligned pose (IoU) on three frames
  qual_cycle.mp4                     Fig. 6: held-out samples (two at a time) with depth and normal rows and their errors
  conv_live_it{0..6}.png, align_anim_plot.mp4   Fig. 7(a) growing next to the alignment strips (video_align_anim.py)
  conv_b.png, poseref.png            Fig. 7(b) and Fig. 8 restyled
usage (cwd = VisTacFusion): python paper_RoboSoft/scripts/video_plots.py"""
import json, os, os.path as osp, subprocess, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "paper_RoboSoft/scripts"); sys.path.insert(0, "scripts")
import make_paper_figs as M          # noqa: E402  (loads the result JSONs at import)
import fig_qual_plus4 as F           # noqa: E402

OUT = "paper_RoboSoft/video/plots"; os.makedirs(OUT, exist_ok=True)
import matplotlib.font_manager as _fm
BOLD = _fm.findfont(_fm.FontProperties(family="Arial", weight="bold")); REG = _fm.findfont(_fm.FontProperties(family="Arial"))
BG = "#ffffff"; FG = "#000000"; GRID = "#e4e2dd"; DPI = 200
COL = {k: v[0] for k, v in M.STY.items()}                     # the paper's series colours (light background)
NAMES = {"Real only": "Real only", "+ Blender (physical)": "PBR", "+ GAN-K (learned, strict)": "Learned renderer (K)", "+ self-training": "Self-training",
         "+ GAN-full (leaky)": "Learned renderer (full)", "+ AlignTac (ours, strict)": "AlignTac (ours)"}
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": "#9a9995", "grid.color": GRID, "axes.grid": True,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "legend.frameon": False})
Ks = [10, 25, 50, 100]
MAE = os.environ.get("MAE_NAME", "Contact MAE")        # metric name in the video plots (the paper says contact MAE)
QUAL_NOHDR = bool(os.environ.get("QUAL_NOHDR"))      # qual_cycle without its own column-header row (headers placed on the slide)


def concat_video(frames, durs, out, fps=30, gif=None):
    """Still frames (paths) shown for the given seconds (concat demuxer, like video_align_anim.py)."""
    lst = out + ".txt"
    with open(lst, "w") as f:
        for p, s in zip(frames, durs): f.write(f"file '{osp.abspath(p)}'\nduration {s}\n")
        f.write(f"file '{osp.abspath(frames[-1])}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
                    "-r", str(fps), out], check=True); os.remove(lst)


def style(ax, xlab, title, lab=15, tk=13, tt=16):
    ax.set_xscale("log"); ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks]); ax.minorticks_off()
    ax.tick_params(labelsize=tk); ax.set_xlabel(xlab, fontsize=lab); ax.set_title(title, loc="left", fontsize=tt, pad=8)
    ax.yaxis.set_major_locator(MaxNLocator(5))


# ------------------------------------------------------------------ Fig. 4: K-shot curves, one method at a time
def kshot():
    series = {"Real only": "realonly", "+ Blender (physical)": "blender", "+ GAN-K (learned, strict)": "gank", "+ self-training": "semipm",
              "+ GAN-full (leaky)": "ganfull", "+ AlignTac (ours, strict)": "rf"}
    A = {}
    for name, suf in series.items():
        xs, ys = [], []
        for K in Ks:
            r = f"T_k{K}_{suf}"; y = M.ta(r, "c_mae", "pilot_" if r == "T_k10_ganfull" else "pilot150_")
            if y is not None: xs.append(K); ys.append(y)
        A[name] = (xs, ys)
    Bq = {name: (Ks, [M.SB[f"pilot_S_k{k}_{suf}/sensorB_mesh"]["c_mae"] for k in Ks]) for name, suf in
          (("Real only", "realonly_m"), ("+ GAN-K (learned, strict)", "gan40m"), ("+ self-training", "semipm_m"), ("+ AlignTac (ours, strict)", "maplnr_m"))}
    rotA = {name: M.ta(f"T_k25_{suf}", "pose_rot_deg") for name, suf in (("Real only", "realonly"), ("+ AlignTac (ours, strict)", "rf"))}
    print("K=25 contact MAE A:", {NAMES[n]: round(A[n][1][1], 3) for n in A}, " B:", {NAMES[n]: round(Bq[n][1][1], 3) for n in Bq})
    print("K=25 rotation A real-only / AlignTac:", {NAMES[n]: round(v, 2) for n, v in rotA.items()})
    order = list(series); frames = []
    for step in range(1, len(order) + 1):
        shown = order[:step]
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.1), dpi=DPI)
        for ax, D, t in zip(axes, [A, Bq], ["Sensor A: marker-free GelSlim 4.0", "Sensor B: dot-marker GelSlim 4.0"]):
            for name in order:
                if name not in D: continue
                c = COL[name]; m, ls = M.STY[name][1], M.STY[name][2]; on = name in shown; last = name == "+ AlignTac (ours, strict)"
                ax.plot(*D[name], color=c, marker=m, ls=ls, lw=3.2 if last else 2.2, ms=15 if m == "*" else 8, mec=BG, mew=0.8, label=NAMES[name],
                        alpha=1.0 if on else 0.0)
            style(ax, "K  (labeled contacts per object)", t); ax.set_ylabel(MAE + " [mm]  ↓", fontsize=15); ax.set_ylim(0, None)
        h, l = axes[0].get_legend_handles_labels()
        leg = fig.legend(h, l, loc="lower center", ncol=6, fontsize=13, columnspacing=1.4, handletextpad=0.5, handlelength=2.2, bbox_to_anchor=(0.5, 0.0))
        for txt, hd, name in zip(leg.get_texts(), leg.legend_handles, order):
            on = name in shown; txt.set_color(FG if on else "#c8c6c1"); hd.set_alpha(1.0 if on else 0.15)
        fig.tight_layout(w_pad=2.0, rect=(0, 0.09, 1, 1)); p = f"{OUT}/kshot_{step}.png"; fig.savefig(p); plt.close(fig); frames.append(p)
    concat_video(frames, [1.4] * (len(frames) - 1) + [5.0], f"{OUT}/kshot_anim.mp4"); Image.open(frames[-1]).save(f"{OUT}/kshot_final.png")


# ------------------------------------------------------------------ Fig. 5: label quality, teacher then aligned mesh
def labels():
    A, Bq = M._label_quality()
    series = [("Teacher (A)", A, 0, "#52514e", "o"), ("Aligned mesh (A)", A, 1, "#8a4fd3", "*"), ("Teacher (B)", Bq, 0, "#eb6834", "s"), ("Aligned mesh (B)", Bq, 1, "#1baf7a", "D")]
    frames = []
    for step in (1, 2):
        fig, axes = plt.subplots(1, 3, figsize=(8.8, 4.0), dpi=DPI)
        for ax, (i, lab) in zip(axes, [(0, "Label IoU  ↑"), (3, MAE + " [mm]  ↓"), (6, "Rotation error [deg]  ↓")]):
            for name, D, off, c, m in series:
                on = step == 2 or off == 0
                ax.plot(Ks, [D[k][i + off] for k in Ks], color=c, marker=m, lw=3.0 if off else 2.2, ms=15 if m == "*" else 8, mec=BG, mew=0.8, label=name, alpha=1.0 if on else 0.0)
            style(ax, "K", lab, lab=14, tk=12, tt=14)
        axes[1].set_ylim(0, None); axes[2].set_ylim(0, None)
        h, l = axes[0].get_legend_handles_labels()
        leg = fig.legend(h, l, loc="lower center", ncol=4, fontsize=12.5, columnspacing=1.2, handletextpad=0.4, bbox_to_anchor=(0.5, 0.0))
        for txt, hd, (name, _, off, _, _) in zip(leg.get_texts(), leg.legend_handles, series):
            on = step == 2 or off == 0; txt.set_color(FG if on else "#c8c6c1"); hd.set_alpha(1.0 if on else 0.15)
        fig.tight_layout(w_pad=1.5, rect=(0, 0.1, 1, 1)); p = f"{OUT}/labels_{step}.png"; fig.savefig(p); plt.close(fig); frames.append(p)
    concat_video(frames, [2.5, 6.0], f"{OUT}/labels_anim.mp4"); Image.open(frames[-1]).save(f"{OUT}/labels_final.png")


# ------------------------------------------------------------------ Fig. 3: mesh at teacher pose <-> aligned pose
def vir(d, vmax=1.2):
    return (cm.viridis(np.clip(np.asarray(d, np.float32) / vmax, 0, 1))[..., :3] * 255).astype(np.uint8)


def tile(im, size):
    return np.array(Image.fromarray(im).resize((size, size), Image.NEAREST if im.shape[0] >= size else Image.BICUBIC))


def put(draw, xy, txt, size=26, color=(0, 0, 0), bold=True, anchor="la"):
    draw.text(xy, txt, fill=color, font=ImageFont.truetype(BOLD if bold else REG, size), anchor=anchor)


def align_toggle(K=25):
    import build_real_filtered_new as B
    B.VIEW = 0.017502; G = "/media/hdd2/ihsuan/gs_blender"; fits = json.load(open(f"{G}/real_pseudo_k{K}_ro_al/rc_fits.json"))
    rows = []
    for o in ["pattern_01_2_lines_angle_1_2", "pattern_06_5_lines_angle_1", "pattern_36"]:      # the same frames as paper Fig. 3
        cand = sorted(t for t in fits if f"/{o}/" in t)
        def iou_of(t):
            f = fits[t]; gt = np.load(t.replace("samples", "raw_data").replace(".png", "_gt.npy")) * 1000
            m = B.gt_depth_mm(o, *f["pose"], f["p"]); a, b = m > 0.05, gt > 0
            return (a & b).sum() / max(1, (a | b).sum())
        sc = sorted((iou_of(t), t) for t in cand[::7]); rows.append(sc[len(sc) // 2][1])
    INTILE = bool(os.environ.get("ALIGN_TOGGLE_INTILE"))   # no header / footer rows; the phase label sits inside the mesh tile
    S, gap, top = 300, 10, (0 if INTILE else 70); cols = ["real image", "teacher depth", "mesh at the", "GT mesh depth"]
    W = 4 * S + 3 * gap; H = top + 3 * S + 2 * gap + (0 if INTILE else 50)
    frames = []
    for phase in (0, 1):
        im = Image.new("RGB", (W, H), (255, 255, 255)); dr = ImageDraw.Draw(im)
        if not INTILE:
            for c, t in enumerate(cols):
                put(dr, (c * (S + gap) + S / 2, 8), t, 24, anchor="ma")
            put(dr, (2 * (S + gap) + S / 2, 38), "teacher pose" if phase == 0 else "aligned pose", 24, (255, 90, 96) if phase == 0 else (60, 230, 90), anchor="ma")
        for r, t in enumerate(rows):
            f = fits[t]; o = t.split("/")[-5]
            gt = np.load(t.replace("samples", "raw_data").replace(".png", "_gt.npy")) * 1000
            td = np.load(t.replace(M.REAL, f"{G}/real_pseudo_k{K}_ro_ng").replace("samples", "raw_data").replace(".png", "_gt.npy")) * 1000
            tp = f["tp"]; m0 = B.gt_depth_mm(o, tp[0], tp[1], tp[2], f["p"]); m1 = B.gt_depth_mm(o, *f["pose"], f["p"]); mm = m0 if phase == 0 else m1
            iou = ((mm > 0.05) & (gt > 0)).sum() / max(1, ((mm > 0.05) | (gt > 0)).sum())
            y = top + r * (S + gap)
            for c, a in enumerate([np.array(Image.open(t).convert("RGB")), vir(td), vir(mm), vir(gt)]):
                im.paste(Image.fromarray(tile(a, S)), (c * (S + gap), y))
            put(dr, (2 * (S + gap) + S - 8, y + S - 8), f"IoU {iou:.2f}", 26, (255, 90, 96) if phase == 0 else (60, 230, 90), anchor="rd")
            if INTILE: put(dr, (2 * (S + gap) + 8, y + 8), "teacher pose" if phase == 0 else "aligned pose", 24, (255, 90, 96) if phase == 0 else (60, 230, 90), anchor="la")
        if not INTILE: put(dr, (W / 2, H - 40), "same teacher, K = 25;  IoU against the withheld GT", 22, (90, 90, 90), False, anchor="ma")
        p = f"{OUT}/align_toggle_{phase}.png"; im.save(p); frames.append(p)
    concat_video(frames * 3, [1.6, 1.6] * 3, f"{OUT}/align_toggle.mp4")


# ------------------------------------------------------------------ Fig. 6: held-out samples, two at a time, depth + normals
def qual_cycle():
    os.environ.setdefault("FIG6_SELECT", "gain")
    sets = {"A": "sensor A (marker-free)", "B": "sensor B (dot-marker)", "P4": "sensor A, additional objects"}
    pretty = lambda o: o.replace("_v2", "").replace("_", " ").replace("lines angle", "lines, angle")
    samples = [(name, obj) for grp in F.LAYOUTS["wide"][0] for name, obj in grp]
    S, gap, hdr, blk_gap = 230, 8, 46, 80; cols = ["real image", "real only", "AlignTac", "ground truth"]
    bw = 4 * S + 3 * gap; W = 2 * bw + blk_gap; nh = 1 if QUAL_NOHDR else 2; H = nh * hdr + 2 * S + gap + 20
    vis = lambda nm: (((nm.astype(np.float32) + 1) / 2).clip(0, 1) * 255).astype(np.uint8)
    frames = []
    for k in range(0, len(samples), 2):
        im = Image.new("RGB", (W, H), (255, 255, 255)); dr = ImageDraw.Draw(im)
        for b, (name, obj) in enumerate(samples[k:k + 2]):
            d = F.data(name); i = F.median_frame(name, obj, "cmae"); r0, r1 = F.RUNS[name]; x0 = b * (bw + blk_gap)
            put(dr, (x0 + bw / 2, 6), f"{sets[name]}:  {pretty(obj)}", 24, (106, 43, 184), anchor="ma")
            if not QUAL_NOHDR:
                for c, t in enumerate(cols): put(dr, (x0 + c * (S + gap) + S / 2, hdr + 4), t, 19, anchor="ma")
            masks = [d["gt"][i] > 0, d[f"pred_{r0}"][i] > 0.05, d[f"pred_{r1}"][i] > 0.05]
            ha, va, fx, fy = F.free_corner(masks)
            for r, ims in enumerate([[d["img"][i], vir(d[f"pred_{r0}"][i]), vir(d[f"pred_{r1}"][i]), vir(d["gt"][i])],
                                     [d["img"][i], vis(d[f"normal_{r0}"][i]), vis(d[f"normal_{r1}"][i]), vis(d["gt_normal"][i])]]):
                y = nh * hdr + r * (S + gap)
                for c, a in enumerate(ims):
                    x = x0 + c * (S + gap); im.paste(Image.fromarray(tile(a, S)), (x, y))
                    if c in (1, 2):
                        run = r0 if c == 1 else r1
                        txt = f"{float(d[f'cmae_{run}'][i]):.2f} mm" if r == 0 else f"{float(d[f'nerr_{run}'][i]):.0f}°"
                        px = x + (S - 8 if ha == "right" else 8); py = y + (S - 8 if va == "bottom" else 8)
                        put(dr, (px, py), txt, 24, (255, 255, 255), anchor=("r" if ha == "right" else "l") + ("d" if va == "bottom" else "a"))
            put(dr, (x0 + bw + 2, nh * hdr + S / 2), "depth", 19, (90, 90, 90), False, anchor="lm") if b == 0 else None
            put(dr, (x0 + bw + 2, nh * hdr + S + gap + S / 2), "normals", 19, (90, 90, 90), False, anchor="lm") if b == 0 else None
        p = f"{OUT}/qual_{k // 2}{'_nohdr' if QUAL_NOHDR else ''}.png"; im.save(p); frames.append(p)
    concat_video(frames, [2.2] * len(frames), f"{OUT}/qual_cycle{'_nohdr' if QUAL_NOHDR else ''}.mp4")


# ------------------------------------------------------------------ Fig. 7(a) growing beside the alignment strips; Fig. 7(b); Fig. 8
CONV_VIDEO = "ablation/pilot_robosoft/aligner_convergence_video.json"   # same evaluation as the paper's json plus mesh-label IoU / MAE


def conv_live(T=6):
    d = json.load(open("ablation/pilot_robosoft/aligner_convergence.json")); t = d["teacher"]["rot"]; strips = []; plots = []   # the paper's numbers
    for it in range(T + 1):
        fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=DPI)
        ax.plot(range(T + 1), t["mean"][:T + 1], color="#dedcd7", lw=2.0); ax.plot(range(T + 1), t["median"][:T + 1], color="#dedcd7", lw=2.0)
        ax.plot(range(it + 1), t["mean"][:it + 1], color="#8a4fd3", marker="*", ms=13, mec=BG, lw=3.0, label="mean")
        ax.plot(range(it + 1), t["median"][:it + 1], color="#52514e", marker="o", ms=7, mec=BG, lw=2.2, label="median")
        ax.set_xlim(-0.3, T + 0.3); ax.set_ylim(0, 6.5); ax.set_xticks(range(T + 1)); ax.tick_params(labelsize=12)
        ax.set_xlabel("aligner updates", fontsize=13); ax.set_ylabel("rotation error [deg]", fontsize=13)
        ax.set_title(f"{d['n']} held-out frames, from the teacher pose", loc="left", fontsize=13, pad=6); ax.legend(fontsize=12, loc="upper right")
        ax.text(it, t["mean"][it] + 0.35, f"{t['mean'][it]:.1f}°", color="#8a4fd3", fontsize=13, ha="center", fontweight="bold")
        fig.tight_layout(); p = f"{OUT}/conv_live_it{it}.png"; fig.savefig(p); plt.close(fig); plots.append(p)
        strip = Image.open(f"paper_RoboSoft/video/align_anim/strip_it{it}.png"); plot = Image.open(p).convert("RGB")
        plot = plot.resize((int(plot.width * strip.height / plot.height), strip.height), Image.LANCZOS)
        out = Image.new("RGB", (strip.width + 30 + plot.width, strip.height), (255, 255, 255)); out.paste(strip, (0, 0)); out.paste(plot, (strip.width + 30, 0))
        q = f"{OUT}/align_plot_it{it}.png"; out.save(q); strips.append(q)
    frames = [strips[0]] + strips[1:] + [strips[-1]]; durs = [1.5] + [0.8] * T + [1.5]       # same timing as align_anim.mp4 (video_align_anim.py)
    concat_video(frames, durs, f"{OUT}/align_anim_plot.mp4"); Image.open(strips[0]).save(f"{OUT}/align_anim_plot_poster.png")
    concat_video([plots[0]] + plots[1:] + [plots[-1]], durs, f"{OUT}/conv_live.mp4"); Image.open(plots[0]).save(f"{OUT}/conv_live_poster.png")


def conv_b():
    d = json.load(open("ablation/pilot_robosoft/aligner_convergence.json")); R = sorted(int(r) for r in d["basin"])
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=DPI)
    curves = [(1, "#b57bee", "-"), (2, "#8f4fd6", "-"), (3, "#6a2bb8", "-"), (6, "#3d1778", "-")]      # video: no n = 0 diagonal
    for it, c, ls in curves:
        y = [d["basin"][str(r)]["rot"]["mean"][it] for r in R]
        ax.plot(R, y, color=c, ls=ls, lw=2.8, marker="*", ms=12, mec=BG)
        ax.text(31.0, y[-1] * (1.25 if it == 3 else 0.8 if it == 6 else 1.0), f"n = {it}", color=c, fontsize=13, va="center", ha="left", fontweight="bold", clip_on=False)
    ax.axvspan(0, 8, color="#8a4fd3", alpha=0.08, lw=0); ax.text(4, 33, "training\nrange", color="#6a2bb8", fontsize=11, ha="center", va="top")
    ax.set_yscale("log"); ax.set_ylim(0.3, 40); ax.set_xlim(-0.5, 30.5); ax.set_yticks([0.5, 1, 3, 10, 30]); ax.set_yticklabels(["0.5", "1", "3", "10", "30"]); ax.minorticks_off()
    ax.set_xticks([0, 8, 15, 20, 25, 30]); ax.tick_params(labelsize=13); ax.set_xlabel("initial rotation error [deg]", fontsize=14); ax.set_ylabel("rotation error after n updates [deg]", fontsize=14)
    ax.set_title("From a perturbed GT pose (+ random ±1.5 mm shift)", loc="left", fontsize=14, pad=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/conv_b.png", bbox_inches="tight", pad_inches=0.05); plt.close(fig)


def poseref():
    PRJ = json.load(open("ablation/pilot_robosoft/pose_refine_results.json"))
    meth = [("Real only", "T_k{K}_realonly_realonly", "#52514e"), ("AlignTac (ours)", "T_k{K}_rf_realonly", "#8a4fd3")]
    panels = [("rot", "Rotation error [deg]", "testA", "session 1"), ("rot", "Rotation error [deg]", "session2", "session 2"),
              ("trans", "Translation error [mm]", "testA", "session 1"), ("trans", "Translation error [mm]", "session2", "session 2")]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), dpi=DPI); axes = axes.ravel(); w = 0.19; x = np.arange(len(Ks))
    for i, (ax, (key, lab, split, sname)) in enumerate(zip(axes, panels)):
        sc = M.HALF_MM if key == "trans" else 1.0
        for j, (name, run, c) in enumerate(meth):
            pre = [PRJ[f"pilot150_{run.format(K=K)}/{split}"][key] * sc for K in Ks]; post = [PRJ[f"pilot150_{run.format(K=K)}/{split}"][key + "_al"] * sc for K in Ks]
            off = (j - 0.5) * 2 * w
            ax.bar(x + off - w / 2, pre, w, color=BG, edgecolor=c, lw=1.5, label=name if i == 0 else None)
            ax.bar(x + off + w / 2, post, w, color=c, edgecolor=c, lw=1.5, label=f"{name} + align" if i == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels(Ks); ax.tick_params(labelsize=11); ax.set_title(f"{lab}, {sname}", loc="left", fontsize=12, pad=5); ax.set_ylim(0, None)
        ax.grid(axis="x", visible=False)
        if i >= 2: ax.set_xlabel("K", fontsize=12, labelpad=1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=11, bbox_to_anchor=(0.5, 0.0), handlelength=1.4, columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(w_pad=1.0, h_pad=0.8, rect=(0, 0.06, 1, 1)); fig.savefig(f"{OUT}/poseref.png", bbox_inches="tight", pad_inches=0.05); plt.close(fig)


# ------------------------------------------------------------------ Fig. 4, all six panels, video size
def kshot6():
    series = {"Real only": "realonly", "+ Blender (physical)": "blender", "+ GAN-K (learned, strict)": "gank", "+ self-training": "semipm",
              "+ GAN-full (leaky)": "ganfull", "+ AlignTac (ours, strict)": "rf"}
    panels = [("c_mae", MAE + " [mm]"), ("pose_rot_deg", "Rotation [deg]"), ("pose_trans", "Translation [mm]")]
    sbkey = {"c_mae": ("c_mae", 1.0, "sensorB_mesh"), "pose_rot_deg": ("rot_deg", 1.0, "sensorB_mesh_bp"), "pose_trans": ("trans_l1", M.HALF_MM / 2, "sensorB_mesh_bp")}
    fig, axes = plt.subplots(1, 6, figsize=(12.6, 3.9), dpi=DPI)
    for i, (ax, (key, lab)) in enumerate(zip(axes[:3], panels)):
        for name, suf in series.items():
            xs, ys = [], []
            for K in Ks:
                r = f"T_k{K}_{suf}"; y = M.ta(r, key, "pilot_" if r == "T_k10_ganfull" else "pilot150_")
                if y is None: continue
                xs.append(K); ys.append(y * (M.HALF_MM if key == "pose_trans" else 1.0))
            c, m, ls = M.STY[name]; last = suf == "rf"
            ax.plot(xs, ys, color=c, marker=m, ls=ls, lw=2.6 if last else 1.8, ms=12 if m == "*" else 6.5, mec=BG, mew=0.7, label=NAMES[name])
        ax.set_title(f"({'abc'[i]}) {lab}", loc="left", fontsize=12.5, pad=6); ax.text(0.96, 0.96, "Sensor A", transform=ax.transAxes, ha="right", va="top", fontsize=12, color="#6b6a66")
    for i, (ax, (key, lab)) in enumerate(zip(axes[3:], panels)):
        for name, suf in (("Real only", "realonly_m"), ("+ GAN-K (learned, strict)", "gan40m"), ("+ self-training", "semipm_m"), ("+ AlignTac (ours, strict)", "maplnr_m")):
            c, m, ls = M.STY[name]; last = suf == "maplnr_m"
            ax.plot(Ks, [M.SB[f"pilot_S_k{k}_{suf}/{sbkey[key][2]}"][sbkey[key][0]] * sbkey[key][1] for k in Ks], color=c, marker=m, ls=ls,
                    lw=2.6 if last else 1.8, ms=12 if m == "*" else 6.5, mec=BG, mew=0.7)
        ax.set_title(f"({'def'[i]}) {lab}", loc="left", fontsize=12.5, pad=6); ax.text(0.96, 0.96, "Sensor B", transform=ax.transAxes, ha="right", va="top", fontsize=12, color="#6b6a66")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks]); ax.minorticks_off(); ax.tick_params(labelsize=11.5)
        ax.yaxis.set_major_locator(MaxNLocator(5)); ax.set_xlabel("K", fontsize=13, labelpad=2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=13, columnspacing=1.6, handletextpad=0.5, handlelength=2.4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(w_pad=1.0, rect=(0, 0.1, 1, 1)); fig.savefig(f"{OUT}/kshot6.png"); plt.close(fig)


# ------------------------------------------------------------------ the four additional objects, one at a time, depth + normals
def qual_p4():
    os.environ.setdefault("FIG6_SELECT", "gain"); nice = {"button_v2": "button", "hex_key_v2": "hex key", "edge_v2": "block edge", "marble_v2": "marble"}
    S, gap, hdr = 240, 8, 46; cols = ["real image", "real only", "AlignTac", "ground truth"]
    W = 4 * S + 3 * gap + 110; H = 2 * hdr + 2 * S + gap + 10
    vis = lambda nm: (((nm.astype(np.float32) + 1) / 2).clip(0, 1) * 255).astype(np.uint8); frames = []
    d = F.data("P4"); r0, r1 = F.RUNS["P4"]
    for obj, name in nice.items():
        i = F.median_frame("P4", obj, "cmae"); im = Image.new("RGB", (W, H), (255, 255, 255)); dr = ImageDraw.Draw(im)
        put(dr, ((4 * S + 3 * gap) / 2, 6), f"{name}   (K = 25, trained on all 16 objects)", 24, (106, 43, 184), anchor="ma")
        for c, t in enumerate(cols): put(dr, (c * (S + gap) + S / 2, hdr + 4), t, 20, anchor="ma")
        ha, va, fx, fy = F.free_corner([d["gt"][i] > 0, d[f"pred_{r0}"][i] > 0.05, d[f"pred_{r1}"][i] > 0.05])
        for r, ims in enumerate([[d["img"][i], vir(d[f"pred_{r0}"][i]), vir(d[f"pred_{r1}"][i]), vir(d["gt"][i])],
                                 [d["img"][i], vis(d[f"normal_{r0}"][i]), vis(d[f"normal_{r1}"][i]), vis(d["gt_normal"][i])]]):
            y = 2 * hdr + r * (S + gap)
            for c, a in enumerate(ims):
                x = c * (S + gap); im.paste(Image.fromarray(tile(a, S)), (x, y))
                if c in (1, 2):
                    run = r0 if c == 1 else r1; txt = f"{float(d[f'cmae_{run}'][i]):.2f} mm" if r == 0 else f"{float(d[f'nerr_{run}'][i]):.0f}°"
                    px = x + (S - 8 if ha == "right" else 8); py = y + (S - 8 if va == "bottom" else 8)
                    put(dr, (px, py), txt, 25, (255, 255, 255), anchor=("r" if ha == "right" else "l") + ("d" if va == "bottom" else "a"))
            put(dr, (4 * S + 3 * gap + 10, y + S / 2), "depth" if r == 0 else "normals", 20, (90, 90, 90), False, anchor="lm")
        p = f"{OUT}/qual_p4_{obj}.png"; im.save(p); frames.append(p)
    concat_video(frames, [2.5] * 4, f"{OUT}/qual_p4.mp4"); Image.open(frames[0]).save(f"{OUT}/qual_p4_poster.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["kshot", "kshot6", "labels", "align_toggle", "qual_cycle", "qual_p4", "conv_live", "conv_b", "poseref"]
    for w in which:
        globals()[w](); print("done", w, flush=True)
