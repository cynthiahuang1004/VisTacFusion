"""Thumbnails of the MeshTac architecture figure (paper_RoboSoft/architecture_diagram.tex), all from one real unlabeled frame
(K=25 run): the frame, the object height map, the mesh depth at the teacher pose and at the aligned pose, its normals,
and the student's prediction on that frame.
usage: python paper_RoboSoft/scripts/make_arch_assets.py
"""
import json
import os
import os.path as osp
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image

sys.path.insert(0, "scripts")
import build_real_filtered_new as B  # noqa: E402
from vistacfusion.data.dataset import depth_to_normal  # noqa: E402

B.VIEW = 0.017502
G = "/media/hdd2/ihsuan/gs_blender"; OUT = "paper_RoboSoft/figures/arch"; OBJ = "pattern_06_5_lines_angle_1"; K = 25
fits = json.load(open(f"{G}/real_pseudo_k{K}_al/rc_fits.json"))


def iou_of(t):
    f = fits[t]; gt = np.load(t.replace("samples", "raw_data").replace(".png", "_gt.npy")) * 1000
    m = B.gt_depth_mm(OBJ, *f["pose"], f["p"]); a, b = m > 0.05, gt > 0
    return (a & b).sum() / max(1, (a | b).sum())


cand = sorted(t for t in fits if f"/{OBJ}/" in t)
sc = sorted((iou_of(t), t) for t in cand[::7]); t = sc[len(sc) // 2][1]          # the frame of Fig. "labels of unlabeled frames"
f = fits[t]


def depth_png(d, name):
    Image.fromarray((cm.viridis(np.clip(d / 1.2, 0, 1))[..., :3] * 255).astype(np.uint8)).save(osp.join(OUT, name))


def normal_png(d, name):
    n = depth_to_normal((d / 1000).astype(np.float32), B.VIEW / 224, B.VIEW / 224)
    Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8)).save(osp.join(OUT, name))


Image.open(t).convert("RGB").save(osp.join(OUT, "frame.png"))
Image.open(cand[len(cand) // 3]).convert("RGB").save(osp.join(OUT, "frame2.png"))
m0 = B.gt_depth_mm(OBJ, *f["tp"], f["p"]); m1 = B.gt_depth_mm(OBJ, *f["pose"], f["p"])
depth_png(m0, "mesh_q0.png"); depth_png(m1, "label_depth.png"); normal_png(m1, "label_normal.png")
gt = np.load(t.replace("samples", "raw_data").replace(".png", "_gt.npy")) * 1000
depth_png(gt, "out_depth.png"); normal_png(gt, "out_normal.png")
# object height map (top view of the mesh), lightly shaded
zmap, LO, BB, PPM, RS = B.get_zmap(OBJ)
h = np.where(zmap > 900, np.nan, zmap); h = np.nanmax(h) - h; h = np.nan_to_num(h, nan=0.0)
ys, xs = np.where(h > 0); c0 = slice(max(0, ys.min() - 20), ys.max() + 20), slice(max(0, xs.min() - 20), xs.max() + 20)
h = h[c0]; side = min(h.shape); h = h[(h.shape[0] - side) // 2:(h.shape[0] - side) // 2 + side, (h.shape[1] - side) // 2:(h.shape[1] - side) // 2 + side]
gy, gx = np.gradient(h); shade = np.clip(0.75 + 0.5 * (gx - gy) / (np.abs(gx).max() + 1e-6), 0, 1)
img = (np.clip(0.35 + 0.6 * h / h.max(), 0, 1) * shade)
Image.fromarray((np.stack([img * 0.92, img * 0.95, img], -1) * 255).astype(np.uint8)).resize((224, 224), Image.LANCZOS).save(osp.join(OUT, "mesh.png"))
print("frame:", t, "| label IoU", round(iou_of(t), 3), "| assets ->", OUT)


# ---- the student's actual prediction on this frame (pilot150_B_k25_al2, best_depth.pt) ----
# The model sees the central 0.816 crop; its output is pasted back into the full-frame canvas so that the
# thumbnail is aligned with the other ones (outside the crop the canvas stays empty).
import contextlib, copy, io
import torch
from vistacfusion.data.dataset import SimVisuoTactileDataset
from vistacfusion.engine.train import load_checkpoint
from vistacfusion.models.model import build_model
from vistacfusion.utils.config import merge_configs

RUN = "pilot150_B_k25_al2"; OUTD = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"
cfg = merge_configs("ablation/encoder/tac_sitr_single.yaml", "ablation/pilot_robosoft/train_pilot_e50.yaml", "ablation/pilot_robosoft/data_B_k25_al2.yaml")
with contextlib.redirect_stdout(io.StringIO()):
    ref = SimVisuoTactileDataset(cfg, cfg.image_size, augment=False, split="val", data_section="real")
    c2 = copy.deepcopy(cfg); c2["real"]["root"] = "/media/hdd2/ihsuan/gs_blender/real_filtered"; c2["real"]["val_every"] = 1
    ds = SimVisuoTactileDataset(c2, cfg.image_size, augment=False, split="all", data_section="real", shared_obj_map=ref._obj_to_id)
    model = build_model(cfg).to("cuda:0").eval(); load_checkpoint(osp.join(OUTD, RUN, "best_depth.pt"), model, device="cuda:0")
unit = osp.dirname(osp.dirname(os.path.realpath(t))); sidx = int(osp.basename(t)[:4])
i = [k for k, (u, s) in enumerate(ds.samples) if os.path.realpath(u) == unit and s == sidx][0]
b = ds[i]
with torch.no_grad():
    o = model(b["rgb"][None].cuda(), b["tactile"][None].cuda(), config="tactile", object_ids=torch.tensor([b["object"]]).cuda())
pd = o["depth"][0, 0].float().cpu().numpy(); pn = o["normal"][0].float().cpu().numpy().transpose(1, 2, 0)
pn = pn / np.clip(np.linalg.norm(pn, axis=-1, keepdims=True), 1e-6, None)
crop = ds.fixed_crop; side = int(round(224 * crop)); o0 = (224 - side) // 2
from PIL import Image as _I
canvas_d = np.zeros((224, 224), np.float32); canvas_d[o0:o0 + side, o0:o0 + side] = np.array(_I.fromarray(pd).resize((side, side), _I.BILINEAR))
canvas_n = np.zeros((224, 224, 3), np.float32); canvas_n[..., 2] = 1.0
canvas_n[o0:o0 + side, o0:o0 + side] = np.stack([np.array(_I.fromarray(pn[..., c]).resize((side, side), _I.BILINEAR)) for c in range(3)], -1)
depth_png(np.clip(canvas_d, 0, None), "out_depth.png")
_I.fromarray(((canvas_n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)).save(osp.join(OUT, "out_normal.png"))
m = b["depth"][0].numpy() > 0
print(f"student prediction written: contact MAE on this frame {np.abs(pd - b['depth'][0].numpy())[m].mean():.3f} mm (crop {crop})")
