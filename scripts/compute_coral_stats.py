"""Precompute CORAL statistics for sim->real feature alignment.

For each branch (tactile/T3, rgb/MAE) computes, over the TRAINING distribution:
  - global patch-token stats:  mu_s, mu_r [D], A = Sigma_s^{-1/2} Sigma_r^{1/2} [D, D]
  - global CLS stats:          mu_s, mu_r [D], A [D, D]
  - per-object mean shifts:    obj_patch_mu_s/r [n_obj, D], obj_cls_mu_s/r [n_obj, D]
  - obj_has_real [n_obj] bool  (objects without real data fall back to global means)

Object id order matches SimVisuoTactileDataset: sorted(sim root object dirs).
Sim sessions are filtered by the per-object rotation windows (same as training).
Real uses the train split only (idx % val_every != 0, val_every=10).

Usage:
    python scripts/compute_coral_stats.py --out pretrained_encoders/coral_stats.pt
"""
import argparse
import glob
import json
import math
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from vistacfusion.models.encoders import MAEEncoder, T3Encoder

SIM_ROOT = "/media/hdd2/ihsuan/gs_blender/renders_v3"
REAL_ROOT = "/media/hdd2/ihsuan/gs_blender/real_filtered"
WINDOWS = "ablation/simqty_filtered/real_rotation_windows.json"
CROP = 1.0 / math.sqrt(2.0)
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
N_PER_OBJ = 400
REAL_VAL_EVERY = 10
EPS_REL = 0.05  # covariance shrinkage relative to mean eigenvalue


def preprocess(path):
    img = np.array(Image.open(path).convert("RGB"))
    H, W = img.shape[:2]
    s = int(min(H, W) * CROP)
    oy, ox = (H - s) // 2, (W - s) // 2
    img = img[oy:oy + s, ox:ox + s]
    img = np.array(Image.fromarray(img).resize((224, 224), Image.BILINEAR), np.float32)
    return torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1))


def in_window(base_deg, lo, hi):
    center = (lo + hi) / 2.0
    halfwidth = (hi - lo) / 2.0
    d = (base_deg - center) % 360.0
    return min(d, 360.0 - d) <= halfwidth


def collect_paths(windows):
    """Return {obj: {"sim": {"samples": [...], "rgb": [...]}, "real": {...}}}."""
    rng = random.Random(0)
    out = {}
    for obj in sorted(os.listdir(SIM_ROOT)):
        if not osp.isdir(osp.join(SIM_ROOT, obj)):
            continue
        entry = {"sim": None, "real": None}
        # ---- sim: sessions inside the rotation window (training distribution)
        idxs = []
        for sdir in sorted(glob.glob(f"{SIM_ROOT}/{obj}/session_*")):
            sj = json.load(open(osp.join(sdir, "session.json")))
            base_deg = math.degrees(sj["base_rotation"][2])
            if obj in windows and not in_window(base_deg, *windows[obj]):
                continue
            for tac in glob.glob(f"{sdir}/sensor_*/samples/*.png"):
                rgb = tac.replace("/samples/", "/rgb/")
                if osp.exists(rgb):
                    idxs.append((tac, rgb))
        if idxs:
            picks = rng.sample(idxs, min(N_PER_OBJ, len(idxs)))
            entry["sim"] = {"samples": [p[0] for p in picks],
                            "rgb": [p[1] for p in picks]}
        # ---- real: train split only
        idxs = []
        for sdir in sorted(glob.glob(f"{REAL_ROOT}/{obj}/session_*/sensor_*")):
            pngs = sorted(glob.glob(f"{sdir}/samples/*.png"))
            for i, tac in enumerate(pngs):
                if i % REAL_VAL_EVERY == 0:
                    continue  # val split
                rgb = tac.replace("/samples/", "/rgb/")
                if osp.exists(rgb):
                    idxs.append((tac, rgb))
        if idxs:
            picks = rng.sample(idxs, min(N_PER_OBJ, len(idxs)))
            entry["real"] = {"samples": [p[0] for p in picks],
                             "rgb": [p[1] for p in picks]}
        out[obj] = entry
    return out


class StatAcc:
    """Streaming mean/covariance accumulator (float64 CPU)."""

    def __init__(self, dim):
        self.n = 0
        self.s = torch.zeros(dim, dtype=torch.float64)
        self.ss = torch.zeros(dim, dim, dtype=torch.float64)

    def add(self, X):  # X: [N, D] float32 (gpu or cpu)
        X = X.double().cpu()
        self.n += X.shape[0]
        self.s += X.sum(0)
        self.ss += X.T @ X

    def mean(self):
        return self.s / self.n

    def cov(self):
        mu = self.mean()
        return self.ss / self.n - torch.outer(mu, mu)


def mat_pow(C, p, eps_rel=EPS_REL):
    d = C.shape[0]
    eps = eps_rel * torch.diagonal(C).mean().clamp(min=1e-8)
    vals, vecs = torch.linalg.eigh(C + eps * torch.eye(d, dtype=C.dtype))
    vals = vals.clamp(min=1e-10)
    return vecs @ torch.diag(vals ** p) @ vecs.T


@torch.no_grad()
def branch_stats(enc, paths_by_obj, subdir, device, batch=32):
    """Compute all stats for one branch. subdir: 'samples' or 'rgb'."""
    D = enc.embed_dim
    acc = {("global", dom, kind): StatAcc(D)
           for dom in ("sim", "real") for kind in ("patch", "cls")}
    obj_mean = {}   # (obj, dom, kind) -> (sum [D], n)

    for obj, entry in paths_by_obj.items():
        for dom in ("sim", "real"):
            if entry[dom] is None:
                continue
            paths = entry[dom][subdir]
            for i in range(0, len(paths), batch):
                x = torch.stack([preprocess(p) for p in paths[i:i + batch]]).to(device)
                patch, cls = enc(x)                       # [B,196,D], [B,1,D]
                pt = patch.reshape(-1, D).float()
                ct = cls.squeeze(1).float()
                acc[("global", dom, "patch")].add(pt)
                acc[("global", dom, "cls")].add(ct)
                for kind, t in (("patch", pt), ("cls", ct)):
                    key = (obj, dom, kind)
                    if key not in obj_mean:
                        obj_mean[key] = [torch.zeros(D, dtype=torch.float64), 0]
                    obj_mean[key][0] += t.double().cpu().sum(0)
                    obj_mean[key][1] += t.shape[0]
        print(f"    {obj}: sim={'y' if entry['sim'] else '-'} real={'y' if entry['real'] else '-'}", flush=True)

    out = {}
    for kind in ("patch", "cls"):
        mu_s = acc[("global", "sim", kind)].mean()
        mu_r = acc[("global", "real", kind)].mean()
        A = mat_pow(acc[("global", "sim", kind)].cov(), -0.5) @ \
            mat_pow(acc[("global", "real", kind)].cov(), 0.5)
        out[f"{kind}_mu_s"] = mu_s.float()
        out[f"{kind}_mu_r"] = mu_r.float()
        out[f"{kind}_A"] = A.float()

    objects = sorted(paths_by_obj.keys())
    n_obj = len(objects)
    for kind in ("patch", "cls"):
        oms = torch.zeros(n_obj, D)
        omr = torch.zeros(n_obj, D)
        for i, obj in enumerate(objects):
            ks, kr = (obj, "sim", kind), (obj, "real", kind)
            oms[i] = (obj_mean[ks][0] / obj_mean[ks][1]).float() \
                if ks in obj_mean else out[f"{kind}_mu_s"]
            omr[i] = (obj_mean[kr][0] / obj_mean[kr][1]).float() \
                if kr in obj_mean else out[f"{kind}_mu_r"]
        out[f"obj_{kind}_mu_s"] = oms
        out[f"obj_{kind}_mu_r"] = omr
    out["obj_has_real"] = torch.tensor(
        [paths_by_obj[o]["real"] is not None for o in objects])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pretrained_encoders/coral_stats.pt")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    windows = json.load(open(WINDOWS))
    paths = collect_paths(windows)
    objects = sorted(paths.keys())
    n_sim = sum(len(e["sim"]["rgb"]) for e in paths.values() if e["sim"])
    n_real = sum(len(e["real"]["rgb"]) for e in paths.values() if e["real"])
    print(f"objects={len(objects)}  sim imgs={n_sim}  real imgs={n_real}")

    stats = {"objects": objects}

    print("  [tactile / T3]")
    t3 = T3Encoder("pretrained_encoders/t3_large").to(args.device).eval()
    stats["tactile"] = branch_stats(t3, paths, "samples", args.device)
    del t3
    torch.cuda.empty_cache()

    print("  [rgb / MAE]")
    mae = MAEEncoder("pretrained_encoders/mae_vitl16.pth").to(args.device).eval()
    stats["rgb"] = branch_stats(mae, paths, "rgb", args.device)

    torch.save(stats, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
