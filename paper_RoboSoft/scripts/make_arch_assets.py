"""Thumbnails of the MeshTac overview figure (paper_RoboSoft/architecture_diagram.tex), all from one real unlabeled
frame of the K=25 run: the frame, the object height map, the mesh depth at the teacher pose and at the aligned pose,
its normals, and the student's ACTUAL prediction on that frame (pilot150_B_k25_al2, best_depth.pt).
Frame choice: among every 7th unlabeled frame of OBJ, the one with the lowest student contact MAE whose aligned
label has IoU >= 0.75 against the withheld GT (a clear illustration; not a typical-case figure).
usage: python paper_RoboSoft/scripts/make_arch_assets.py
"""
import contextlib, copy, io, json, os, os.path as osp, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np, torch
from PIL import Image

sys.path.insert(0, "scripts")
import build_real_filtered_new as B  # noqa: E402
from vistacfusion.data.dataset import SimVisuoTactileDataset, depth_to_normal  # noqa: E402
from vistacfusion.engine.train import load_checkpoint  # noqa: E402
from vistacfusion.models.model import build_model  # noqa: E402
from vistacfusion.utils.config import merge_configs  # noqa: E402

B.VIEW = 0.017502
G = "/media/hdd2/ihsuan/gs_blender"; OUT = "paper_RoboSoft/figures/arch"; OBJ = "pattern_06_5_lines_angle_1"; K = 25
RUN = "pilot150_B_k25_al2"; OUTD = "/media/hdd2/ihsuan/VisTacFusion_outputs_hdd2"; dev = "cuda:0"
fits = json.load(open(f"{G}/real_pseudo_k{K}_al/rc_fits.json"))


def iou_of(t):
    f = fits[t]; gt = np.load(t.replace("samples", "raw_data").replace(".png", "_gt.npy")) * 1000
    m = B.gt_depth_mm(OBJ, *f["pose"], f["p"]); a, b = m > 0.05, gt > 0
    return (a & b).sum() / max(1, (a | b).sum())


# ---- student + dataset over all real frames ----
cfg = merge_configs("ablation/encoder/tac_sitr_single.yaml", "ablation/pilot_robosoft/train_pilot_e50.yaml", f"ablation/pilot_robosoft/data_B_k{K}_al2.yaml")
with contextlib.redirect_stdout(io.StringIO()):
    ref = SimVisuoTactileDataset(cfg, cfg.image_size, augment=False, split="val", data_section="real")
    c2 = copy.deepcopy(cfg); c2["real"]["root"] = f"{G}/real_filtered"; c2["real"]["val_every"] = 1
    ds = SimVisuoTactileDataset(c2, cfg.image_size, augment=False, split="all", data_section="real", shared_obj_map=ref._obj_to_id)
    model = build_model(cfg).to(dev).eval(); load_checkpoint(osp.join(OUTD, RUN, "best_depth.pt"), model, device=dev)
index = {(os.path.realpath(u), s): k for k, (u, s) in enumerate(ds.samples)}


@torch.no_grad()
def predict(t):
    i = index[(os.path.realpath(osp.dirname(osp.dirname(t))), int(osp.basename(t)[:4]))]; b = ds[i]
    o = model(b["rgb"][None].to(dev), b["tactile"][None].to(dev), config="tactile", object_ids=torch.tensor([b["object"]]).to(dev))
    pd = o["depth"][0, 0].float().cpu().numpy(); pn = o["normal"][0].float().cpu().numpy().transpose(1, 2, 0)
    gt = b["depth"][0].numpy(); m = gt > 0
    return pd, pn / np.clip(np.linalg.norm(pn, axis=-1, keepdims=True), 1e-6, None), float(np.abs(pd - gt)[m].mean())


# ---- frame choice ----
cand = sorted(t for t in fits if f"/{OBJ}/" in t)[::7]
scored = sorted((predict(t)[2], t) for t in cand if iou_of(t) >= 0.75)
err, t = scored[0]
f = fits[t]
print(f"frame: {t} | label IoU {iou_of(t):.3f} | student contact MAE {err:.3f} mm")


def depth_png(d, name):
    Image.fromarray((cm.viridis(np.clip(d / 1.2, 0, 1))[..., :3] * 255).astype(np.uint8)).save(osp.join(OUT, name))


def normal_png(d, name):
    n = depth_to_normal((d / 1000).astype(np.float32), B.VIEW / 224, B.VIEW / 224)
    Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8)).save(osp.join(OUT, name))


Image.open(t).convert("RGB").save(osp.join(OUT, "frame.png"))
Image.open(cand[len(cand) // 3]).convert("RGB").save(osp.join(OUT, "frame2.png"))
m0 = B.gt_depth_mm(OBJ, *f["tp"], f["p"]); m1 = B.gt_depth_mm(OBJ, *f["pose"], f["p"])
depth_png(m0, "mesh_q0.png"); depth_png(m1, "label_depth.png"); normal_png(m1, "label_normal.png")
# student prediction, pasted back into the full-frame canvas (the model sees the central 0.816 crop)
pd, pn, _ = predict(t)
side = int(round(224 * ds.fixed_crop)); o0 = (224 - side) // 2
cd = np.zeros((224, 224), np.float32); cd[o0:o0 + side, o0:o0 + side] = np.array(Image.fromarray(pd).resize((side, side), Image.BILINEAR))
cn = np.zeros((224, 224, 3), np.float32); cn[..., 2] = 1.0
cn[o0:o0 + side, o0:o0 + side] = np.stack([np.array(Image.fromarray(pn[..., c]).resize((side, side), Image.BILINEAR)) for c in range(3)], -1)
depth_png(np.clip(cd, 0, None), "out_depth.png")
Image.fromarray(((cn * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)).save(osp.join(OUT, "out_normal.png"))
# object height map (top view of the mesh), lightly shaded
zmap, LO, BB, PPM, RS = B.get_zmap(OBJ)
h = np.where(zmap > 900, np.nan, zmap); h = np.nanmax(h) - h; h = np.nan_to_num(h, nan=0.0)
ys, xs = np.where(h > 0); c0 = slice(max(0, ys.min() - 20), ys.max() + 20), slice(max(0, xs.min() - 20), xs.max() + 20)
h = h[c0]; s_ = min(h.shape); h = h[(h.shape[0] - s_) // 2:(h.shape[0] - s_) // 2 + s_, (h.shape[1] - s_) // 2:(h.shape[1] - s_) // 2 + s_]
gy, gx = np.gradient(h); shade = np.clip(0.75 + 0.5 * (gx - gy) / (np.abs(gx).max() + 1e-6), 0, 1)
img = (np.clip(0.35 + 0.6 * h / h.max(), 0, 1) * shade)
Image.fromarray((np.stack([img * 0.92, img * 0.95, img], -1) * 255).astype(np.uint8)).resize((224, 224), Image.LANCZOS).save(osp.join(OUT, "mesh.png"))
print("assets ->", OUT)
