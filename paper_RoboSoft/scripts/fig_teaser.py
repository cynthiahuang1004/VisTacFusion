"""Intro teaser (Fig. 1 of the 8-page version): real registration vs. simulated appearance vs. AlignTac.
usage: conda run -n vistacfusion python paper_RoboSoft/scripts/fig_teaser.py paper_RoboSoft/figures   (cwd = VisTacFusion)"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import FancyArrowPatch
sys.path.insert(0, "scripts")
import build_real_filtered_new as B
B.VIEW = 0.017502
G = "/media/hdd2/ihsuan/gs_blender"; REAL = f"{G}/real_filtered"; K = 25
OBJ = "pattern_06_5_lines_angle_1"; SIM = f"{G}/renders_v3/{OBJ}/session_000/sensor_0000"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 5.6, "font.size": 5.6})

fits = json.load(open(f"{G}/real_pseudo_k{K}_ro_al/rc_fits.json"))
cand = sorted(t for t in fits if f"/{OBJ}/" in t)
def iou_of(t, pose):
    f = fits[t]; gt = np.load(t.replace("samples", "raw_data").replace(".png", "_gt.npy")) * 1000
    m = B.gt_depth_mm(OBJ, *pose, f["p"]); a, b = m > 0.05, gt > 0
    return (a & b).sum() / max(1, (a | b).sum())
sc = [(iou_of(t, fits[t]["pose"]) - iou_of(t, fits[t]["tp"][:3]), iou_of(t, fits[t]["pose"]), t) for t in cand[::5]]
sc = sorted(x for x in sc if x[1] > 0.7); t = sc[-1][2]; f = fits[t]   # a frame where alignment visibly matters (largest IoU gain, good final label)
real = np.array(Image.open(t).convert("RGB"))
gt = np.load(t.replace("samples", "raw_data").replace(".png", "_gt.npy")) * 1000
tp = f["tp"]; m0 = B.gt_depth_mm(OBJ, tp[0], tp[1], tp[2], f["p"]); m1 = B.gt_depth_mm(OBJ, *f["pose"], f["p"])
# panel (b): the SAME contact rendered from its registered state: PBR via Blender at the real pose (render_real_pose_tactile.py ->
# real_filtered_phys), learned renderer (K=25 pairs) applied to the registered depth (scratchpad/gan_realpose.py -> figures/teaser/gan_realpose.png)
sim_d = gt
pbr = np.array(Image.open(t.replace("/real_filtered/", "/real_filtered_phys/")).convert("RGB"))
gan = np.array(Image.open(os.path.join(sys.argv[1], "teaser", "gan_realpose.png")).convert("RGB"))

W, H = 3.45, 1.70
fig = plt.figure(figsize=(W, H))
tw = 0.097; th = tw * W / H                                   # thumbnail size in figure fraction
y0 = 0.27                                                     # bottom of thumbnails
def thumb(x, im, label, depth=False, box=None):
    ax = fig.add_axes([x, y0, tw, th]); ax.imshow(im, cmap="viridis", vmin=0, vmax=1.2) if depth else ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(box is not None); sp.set_edgecolor(box or "k"); sp.set_linewidth(1.1)
    fig.text(x + tw / 2, y0 - 0.035, label, ha="center", va="top", fontsize=5.0, linespacing=1.0)
    return ax
def arrow(x1, x2, text=None, color="#444", where="above"):
    fig.patches.append(FancyArrowPatch((x1, y0 + th / 2), (x2, y0 + th / 2), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=5, lw=0.7, color=color))
    if text and where == "above": fig.text((x1 + x2) / 2, y0 + th + 0.012, text, ha="center", va="bottom", fontsize=4.6, color=color, linespacing=1.0)
    if text and where == "below": fig.text((x1 + x2) / 2, y0 + th / 2 - 0.02, text, ha="center", va="top", fontsize=4.4, color=color)
def header(xc, title, sub, color):
    fig.text(xc, 0.975, title, ha="center", va="top", fontsize=6.0, fontweight="bold", color=color)
    fig.text(xc, 0.885, sub, ha="center", va="top", fontsize=4.9, linespacing=1.05, color="#333")
gap, gw = 0.02, 0.048                                        # thumb gap, arrow width
# (a) real: image -> registered depth (label)
xa = 0.03; thumb(xa, real, "real image"); arrow(xa + tw + 0.004, xa + tw + gw - 0.004, "register\n(robot pose)", "#b03a2e"); thumb(xa + tw + gw, gt, "mesh depth\n(label)", True)
header(xa + tw + gw / 2, "(a) Real contacts", "appearance is right;\neach label needs a registered pose", "#b03a2e")
# (b) simulation: depth -> PBR / learned renderer images
xb = xa + 2 * tw + gw + 0.05; thumb(xb, sim_d, "mesh depth\n(free)", True); arrow(xb + tw + 0.004, xb + tw + gw - 0.004, "render", "#1f6fb2")
thumb(xb + tw + gw, pbr, "PBR"); thumb(xb + 2 * tw + gw + gap, gan, "learned\nrenderer")
header(xb + (3 * tw + gw + gap) / 2, "(b) Simulated contacts", "labels are free; appearance gap,\nrenderer fitted on real pairs", "#1f6fb2")
# (c) AlignTac: unlabeled real image + mesh at teacher pose -> aligned mesh (label)
xc = xb + 3 * tw + gw + gap + 0.05; thumb(xc, m0, "mesh at\nteacher pose", True); arrow(xc + tw + 0.004, xc + tw + gw - 0.004, "align", "#6a3fb5", where="below")
thumb(xc + tw + gw, m1, "aligned mesh\n(label)", True, box="#6a3fb5")
header(xc + tw + gw / 2, "(c) AlignTac (ours)", "real image for appearance,\naligned mesh geometry for labels", "#6a3fb5")
# small real image inset above the aligner arrow, feeding the alignment
axr = fig.add_axes([xc + tw + gw / 2 - tw * 0.33, y0 + th + 0.03, tw * 0.66, th * 0.66]); axr.imshow(real); axr.set_xticks([]); axr.set_yticks([])
for sp in axr.spines.values(): sp.set_visible(False)
fig.text(xc + tw + gw / 2, y0 + th + 0.03 + th * 0.66 + 0.005, "unlabeled real image", ha="center", va="bottom", fontsize=4.4, color="#6a3fb5")
fig.patches.append(FancyArrowPatch((xc + tw + gw / 2, y0 + th + 0.03), (xc + tw + gw / 2, y0 + th + 0.004), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=4, lw=0.6, color="#6a3fb5"))
# bottom strip: cost summary
fig.text(0.5, 0.04, "cost:  a pose per label   |   renderer fit on real pairs   |   K registered contacts, then unlabeled images", ha="center", va="bottom", fontsize=4.8, color="#555", style="italic")
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig_teaser.pdf")); fig.savefig(os.path.join(out, "fig_teaser.png"), dpi=300); print("teaser written", t)

# individual thumbnails for the TikZ teaser (paper_RoboSoft/teaser_diagram.tex)
from matplotlib import cm
td = os.path.join(sys.argv[1], "teaser"); os.makedirs(td, exist_ok=True)
def save(name, im, depth=False):
    if depth:
        im = (cm.viridis(np.clip(im / 1.2, 0, 1))[..., :3] * 255).astype(np.uint8)
    Image.fromarray(np.asarray(im).astype(np.uint8)).resize((224, 224), Image.BILINEAR).save(os.path.join(td, name + ".png"))
save("real", real); save("gt", gt, True); save("sim_depth", sim_d, True); save("pbr", pbr); save("gan", gan); save("mesh_teacher", m0, True); save("mesh_aligned", m1, True)
sim_real = np.array(Image.open(t).convert("RGB")); save("real2", sim_real)
print("thumbnails in", td)
