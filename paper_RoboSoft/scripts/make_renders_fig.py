"""Fig. 'Synthetic training images' of the 6-page paper: the SAME real validation frame rendered by every renderer.
Columns: real frame | GT depth | physical renderer (Blender at the real pose, real_filtered_phys) | learned renderer
trained on K=25 real pairs | all-pairs renderer.  Rows: validation frames of multi-ridge objects with the largest contact area.
usage: python paper_RoboSoft/scripts/make_renders_fig.py [--device cuda:0]"""
import argparse, glob, os.path as osp, sys
import numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))); sys.path.insert(0, osp.join(ROOT, "scripts", "tactile_render"))
from train_tactile_gan import REAL_ROOT, PHYS_ROOT, DEPTH_SCALE, UNetG, with_coords  # noqa: E402

OBJS = ["pattern_32", "pattern_36"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", default=osp.join(ROOT, "paper_RoboSoft/figures/fig_renders_small.pdf")); a = ap.parse_args()
    dev = a.device
    G = {}
    for tag, d in (("k25", "tactile_gan_k25"), ("full", "tactile_gan")):
        g = UNetG(in_ch=3).to(dev).eval(); g.load_state_dict(torch.load(f"{ROOT}/outputs/{d}/G_final.pt", map_location=dev)); G[tag] = g
    rows = []
    for obj in OBJS:
        unit = f"{REAL_ROOT}/{obj}/session_000/sensor_0000"
        cands = []
        for p in sorted(glob.glob(f"{PHYS_ROOT}/{obj}/session_000/sensor_0000/samples/*.png")):
            idx = int(osp.basename(p)[:4])
            if idx % 10:                                     # validation frames only
                continue
            d = np.load(f"{unit}/raw_data/{idx:04d}_gt.npy")
            cands.append(((d > 0).mean(), idx))
        _, idx = max(cands)                                  # the frame with the largest contact area (several ridges visible)
        d = np.load(f"{unit}/raw_data/{idx:04d}_gt.npy").astype(np.float32)
        real = np.asarray(Image.open(f"{unit}/samples/{idx:04d}.png").convert("RGB"))
        phys = np.asarray(Image.open(f"{PHYS_ROOT}/{obj}/session_000/sensor_0000/samples/{idx:04d}.png").convert("RGB"))
        x = with_coords(torch.from_numpy(d / DEPTH_SCALE * 2 - 1)[None])[None].to(dev)
        with torch.no_grad():
            ren = {t: ((g(x)[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8) for t, g in G.items()}
        depth = (plt.cm.viridis(np.clip(d / 1.2e-3, 0, 1))[..., :3] * 255).astype(np.uint8) if d.max() < 0.01 else (plt.cm.viridis(np.clip(d / 1.2, 0, 1))[..., :3] * 255).astype(np.uint8)
        rows.append([real, depth, phys, ren["k25"], ren["full"]]); print(obj, idx, "contact frac %.3f" % max(cands)[0], "depth max", d.max())
    names = ["Real frame", "GT depth", "PBR", "Learned ($K$=25)", "Learned (full)"]
    fig, axes = plt.subplots(len(rows), 5, figsize=(3.45, 0.74 * len(rows) + 0.2))
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            ax = axes[i, j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(names[j], fontsize=5.6, pad=2)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.86, bottom=0.01, wspace=0.03, hspace=0.04)
    fig.savefig(a.out, dpi=300); print("wrote", a.out)


if __name__ == "__main__":
    main()
